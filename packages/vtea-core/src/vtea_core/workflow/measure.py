"""Measuring every segmentation a protocol produces, not just the last one.

A protocol rarely has one segmentation. A nucleus is segmented, a ring is
derived from it for the cytoplasm, a second channel gives lysosomes - and
each of those is a population of objects with its own features. Until now
the builder measured whichever segmentation a measurement step happened to
be pointed at, so measuring three of them meant adding three measurement
steps by hand and re-pointing each one, and forgetting to do that for the
ring quietly produced an analysis with a hole in it.

So the measurement steps are derived from the segmentations instead. Every
named segmentation gets one, named after it (`measure_<segmentation>`), and
the pairing is recorded on the step itself (`Step.auto_for`) rather than
guessed at from the wiring afterwards. That record is what makes the rest
follow:

- Rename the segmentation and its measurement step follows the new name,
  through the GUI, along with everything pointed at either of them.
- Delete the segmentation and its measurement step goes with it, rather
  than staying behind to fail with "needs context key(s) ['nuclei_2']".
- Add a measurement step by hand for a segmentation and nothing is raised
  for it - a hand-made step is the answer to "is this one measured?", and
  a second automatic one would double every row of the table.

Nothing here runs anything: it edits the two Pipelines the builder holds,
and the builder runs them. That keeps it testable headlessly and usable
from a script that builds a protocol without a GUI at all.
"""

from __future__ import annotations

from collections.abc import Iterable

from vtea_core.workflow.pipeline import Pipeline, Step, unique_step_name

# What a segmentation step writes, and what a measurement step writes.
SEGMENTATION_OUTPUT = "labels"
MEASUREMENT_OUTPUT = "measurements"

# The measurement to raise for a segmentation nobody has measured. The
# multi-channel one, for the same reason the builder's Add-step menu
# defaults to it: it measures the segmentation against every channel and
# tags each feature with the channel it came from, which is what a
# multi-channel acquisition wants and what the single-channel form cannot
# say.
DEFAULT_MEASUREMENT = "extract_measurements_by_channel"

# Prefix for the raised steps' names, so a glance at the analysis pane says
# which of its steps came from a segmentation and which were added by hand.
MEASUREMENT_PREFIX = "measure_"


def measurement_name_for(segmentation: str, taken: Iterable[str] = ()) -> str:
    """The name a segmentation's measurement step should have.

    `measure_nuclei` for a segmentation called `nuclei` - the segmentation's
    own name, default or user-entered, so the table a person is looking at
    says which segmentation it measures without them having to trace the
    wiring. Falls back to a numbered variant if that name is somehow taken.
    """
    preferred = f"{MEASUREMENT_PREFIX}{segmentation}"
    taken = set(taken)
    if preferred not in taken:
        return preferred
    return unique_step_name(preferred, taken)


def segmentation_names(pipeline: Pipeline) -> list[str]:
    """Every named segmentation in `pipeline`, in protocol order."""
    return [
        step.name
        for step in pipeline.steps
        if step.name and step.output_key == SEGMENTATION_OUTPUT
    ]


def measured_segmentations(pipeline: Pipeline) -> set[str]:
    """Which segmentations the measurement steps of `pipeline` are pointed at."""
    return {
        step.input_keys.get(SEGMENTATION_OUTPUT, "")
        for step in pipeline.steps
        if step.output_key == MEASUREMENT_OUTPUT
    } - {""}


def sync_measurement_steps(
    processing: Pipeline,
    analysis: Pipeline,
    *,
    function_name: str = DEFAULT_MEASUREMENT,
    available: set[str] | None = None,
    taken_names: Iterable[str] | None = None,
    channel: int | None = None,
) -> tuple[list[Step], list[Step]]:
    """Give every segmentation in `processing` a measurement step in `analysis`.

    Returns `(added, removed)` - the steps this call raised and the ones it
    retired, so a caller can report what changed rather than silently
    rewriting somebody's protocol.

    Idempotent: calling it again with nothing changed adds and removes
    nothing. `available` and `taken_names` come from the caller because the
    two pipelines share one run context and one namespace, and only the
    caller knows both.
    """
    known = set(segmentation_names(processing))
    taken = set(taken_names) if taken_names is not None else set(
        processing.step_names() + analysis.step_names()
    )

    removed = [
        step
        for step in list(analysis.steps)
        if step.auto_for and step.auto_for not in known
    ]
    for step in removed:
        analysis.remove_step(analysis.steps.index(step))
        taken.discard(step.name)

    already = measured_segmentations(analysis) | {
        step.auto_for for step in analysis.steps if step.auto_for
    }
    seed = set(available or {"labels", "intensity", "channel_axis", "spacing"})

    added: list[Step] = []
    for segmentation in segmentation_names(processing):
        if segmentation in already:
            continue
        step = Step.for_function(
            "measurements",
            function_name,
            available=seed,
            taken_names=taken,
            name=measurement_name_for(segmentation, taken),
            channel=channel,
            auto_for=segmentation,
        )
        step.input_keys[SEGMENTATION_OUTPUT] = segmentation
        analysis.add_step(step)
        taken.add(step.name)
        already.add(segmentation)
        added.append(step)
    return added, removed


def rename_segmentation(
    analysis: Pipeline,
    old_name: str,
    new_name: str,
    *,
    taken_names: Iterable[str] = (),
) -> list[tuple[Step, str]]:
    """Follow a renamed segmentation through the steps raised for it.

    Returns `(step, previous_name)` for each step touched, so the caller can
    re-point anything that referred to the step by its old name and rename
    the table it published.

    A step still carrying the name this module gave it is renamed to match
    the segmentation's new one - that is the whole point of raising it. A
    step somebody has since named themselves keeps that name: they have said
    what they want this table called, and a rename elsewhere in the protocol
    is not a reason to overrule them.
    """
    taken = set(taken_names)
    updated: list[tuple[Step, str]] = []
    for step in analysis.steps:
        if step.auto_for != old_name:
            continue
        previous = step.name
        step.auto_for = new_name
        if step.name == measurement_name_for(old_name):
            step.name = measurement_name_for(new_name, taken - {previous})
        updated.append((step, previous))
    return updated
