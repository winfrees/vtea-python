"""Step/Pipeline: the headless engine behind the napari protocol builder.

Replaces vtea.protocol/vtea.workflow's Java classes (ProtocolManagerMulti,
the datastructure/Protocol hierarchy, Workflow/AbstractWorkflow). Runs the
same whether driven from the GUI or a script/notebook, matching vtea-core's
headless-usable design goal - the napari widget (Phase 4) is a thin layer
that renders a Pipeline's steps as cards and edits them via magicgui forms,
it doesn't own execution.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vtea_core.workflow.registry import get_step_function


def unique_step_name(base: str, taken: Iterable[str]) -> str:
    """`base_1`, or the next free `base_N`.

    Steps are named after the function that produced them, numbered so that
    two watershed segmentations in one protocol stay distinguishable
    (`watershed_split_1`, `watershed_split_2`) without the user having to
    invent names before the protocol even runs.
    """
    used = set(taken)
    index = 1
    while f"{base}_{index}" in used:
        index += 1
    return f"{base}_{index}"


@dataclass
class Step:
    """One configured pipeline step.

    `input_keys` maps the target function's argument names to keys in the
    Pipeline's run() context (e.g. {"labels": "labels", "intensity":
    "volume"} for extract_measurements, which needs both a prior step's
    output and the original image). `params` are additional fixed keyword
    arguments (thresholds, n_clusters, ...). The function's return value is
    stored in the context under `output_key`.

    Rather than filling these in by hand, build a step with
    `Step.for_function(...)`, which derives them from
    vtea_core.workflow.wiring.

    `channel` picks one channel out of a multi-channel image for this step
    to work on (None = use the array as-is). See run() for exactly which
    inputs that applies to.

    `name` identifies this step's result on its own. `output_key` is
    semantic and shared - every segmentation writes "labels" - which is what
    makes a chain wire itself up, but it also means a protocol with two
    segmentations can't say *which* one a measurement step should measure.
    Each step's result is therefore also published under its (unique) name,
    so `input_keys["labels"] = "watershed_split_1"` picks one of them.
    """

    category: str
    function_name: str
    params: dict[str, Any] = field(default_factory=dict)
    input_keys: dict[str, str] = field(default_factory=dict)
    output_key: str = "result"
    comment: str = ""
    channel: int | None = None
    name: str = ""
    # Which measured features this step's feature-table input is built from.
    # Empty means every numeric feature. Recorded on the step rather than
    # applied by the caller so a protocol carries its own answer to "which
    # of the forty measured features did this clustering actually use?".
    features: list[str] = field(default_factory=list)
    # The step this one was created *for*, when it was created automatically
    # rather than added by hand - see vtea_core.workflow.measure. A
    # measurement step raised because a segmentation exists knows which
    # segmentation that is, which is what lets it follow that segmentation's
    # name and disappear with it. Empty for a step somebody added.
    auto_for: str = ""

    @classmethod
    def for_function(
        cls,
        category: str,
        function_name: str,
        *,
        available: set[str] | None = None,
        taken_names: Iterable[str] = (),
        **kwargs: Any,
    ) -> Step:
        """A Step with input_keys/output_key derived from the function's
        declared I/O (see vtea_core.workflow.wiring), so it can actually run
        without the caller wiring the data arguments themselves, and a
        default name that doesn't collide with `taken_names`."""
        from vtea_core.workflow.wiring import default_wiring

        input_keys, output_key = default_wiring(category, function_name, available)
        kwargs.setdefault("input_keys", input_keys)
        kwargs.setdefault("output_key", output_key)
        kwargs.setdefault("name", unique_step_name(function_name, taken_names))
        return cls(category=category, function_name=function_name, **kwargs)

    @property
    def function(self):
        return get_step_function(self.category, self.function_name)

    @property
    def result_key(self) -> str:
        """Where this step's result is looked up from when it's referred to
        individually - its name, or the shared output key if unnamed."""
        return self.name or self.output_key

    @property
    def channel_mode(self) -> str:
        """How this step relates to the image's channel axis - see
        vtea_core.workflow.wiring's CHANNEL_* constants."""
        from vtea_core.workflow.wiring import CHANNEL_SLICE, STEP_IO

        spec = STEP_IO.get((self.category, self.function_name))
        return CHANNEL_SLICE if spec is None else spec.channel_mode

    @property
    def channel_aware(self) -> bool:
        """True when the function handles the channel axis itself (it takes
        `channel_axis`/`channel` arguments) and so must be handed the whole
        multi-channel array rather than a pre-sliced single channel."""
        from vtea_core.workflow.wiring import CHANNEL_ARGUMENT

        return self.channel_mode == CHANNEL_ARGUMENT

    @property
    def produces_image(self) -> bool:
        """Whether this step's result is an image to show as a layer, rather
        than per-object numbers to add to the table - see
        vtea_core.workflow.wiring.produces_image."""
        from vtea_core.workflow.wiring import produces_image

        return produces_image(self.category, self.function_name)

    @property
    def settings_signature(self) -> tuple:
        """Everything about this step that changes what it computes.

        Its name is deliberately not in here: renaming a segmentation
        changes what its result is *called*, not what the result is, and
        re-running an hour of watershed because somebody typed a better name
        would be its own bug. Its parameters, its channel, its wiring and
        its feature selection all are - change any of them and the last
        result no longer describes this step.
        """
        return (
            self.category,
            self.function_name,
            tuple(sorted((key, repr(value)) for key, value in self.params.items())),
            tuple(sorted(self.input_keys.items())),
            self.channel,
            tuple(self.features),
        )

    @property
    def feature_input(self) -> str | None:
        """The argument, if any, that takes a feature table rather than a
        plain array - see vtea_core.workflow.wiring.StepIO."""
        from vtea_core.workflow.wiring import STEP_IO

        spec = STEP_IO.get((self.category, self.function_name))
        return None if spec is None else spec.feature_input

    def selected_features(self, frame) -> list[str]:
        """The columns of `frame` this step will actually be built from.

        An empty `features` means every numeric feature. A selection is
        filtered against the table rather than trusted: a feature chosen
        before a re-run that dropped it should quietly fall out, not abort
        the step with a KeyError on a column that no longer exists.
        """
        from vtea_core.measurements import feature_matrix

        if not self.features:
            return feature_matrix(frame)[1]
        return [name for name in self.features if name in frame.columns]

    @property
    def channel_applies(self) -> bool:
        """Whether a channel choice means anything for this step.

        It doesn't for the steps that consume per-object tables - clustering,
        reduction, gating - whose input is the measured feature matrix, not
        an image. Offering them a channel picker says they work on one
        channel of the image when they work on every feature that has been
        measured, from every channel.
        """
        from vtea_core.workflow.wiring import CHANNEL_NONE

        return self.channel_mode != CHANNEL_NONE

    def run(
        self,
        context: dict[str, Any],
        *,
        channel_axis: int | None = None,
        full_ndim: int | None = None,
    ) -> Any:
        """Calls this step's function with its data inputs resolved from
        `context`.

        Channel selection applies only to inputs that still carry the
        channel axis, identified by having the same ndim as the image the
        pipeline started with (`full_ndim`). Anything produced by an earlier
        step has already been reduced to a single channel and has one fewer
        dimension, so it is passed through untouched - which is what keeps a
        second channel-selecting step from slicing a spatial axis by
        mistake.

        A channel-aware function (see `channel_aware`) is the exception: it
        gets the whole multi-channel array and `self.channel` as an argument,
        because it needs to see every channel at once to label its output
        columns by the channel each came from.
        """
        missing = [key for key in self.input_keys.values() if key not in context]
        if missing:
            raise KeyError(
                f"step '{self.category}.{self.function_name}' needs context key(s) {missing}, "
                f"available: {list(context)}"
            )
        if not self.channel_applies:
            # A per-object table has no channel axis to slice.
            kwargs = {arg: context[key] for arg, key in self.input_keys.items()}
            self._build_feature_matrix(kwargs)
        elif self.channel_aware:
            kwargs = {arg: context[key] for arg, key in self.input_keys.items()}
            parameters = inspect.signature(self.function).parameters
            if "channel" in parameters and "channel" not in self.params:
                kwargs["channel"] = self.channel
        else:
            kwargs = {
                arg: self._select_channel(context[key], channel_axis, full_ndim)
                for arg, key in self.input_keys.items()
            }
        self._fill_source_names(kwargs)
        kwargs.update(self.params)
        return self.function(**kwargs)

    def _fill_source_names(self, kwargs: dict[str, Any]) -> None:
        """Pass the *name* of an input's source to the parameters that want
        it - see vtea_core.workflow.wiring.StepIO.names_from.

        An association records which segmentation each object came from, and
        that name has to be the one the step is actually reading. Deriving it
        from `input_keys` is what stops an ObjectRef saying `child#7` (the
        function's own default) or naming a segmentation the step is not
        pointed at.
        """
        from vtea_core.workflow.wiring import STEP_IO

        spec = STEP_IO.get((self.category, self.function_name))
        if spec is None:
            return
        for parameter, argument in spec.names_from:
            source = self.input_keys.get(argument)
            if source:
                kwargs[parameter] = source

    def _build_feature_matrix(self, kwargs: dict[str, Any]) -> None:
        """Turn a feature *table* wired to this step's feature input into the
        matrix its function expects, keeping only the selected features.

        A plain array is left alone, so a script that hands a clustering step
        a ready-made matrix keeps working; only a DataFrame is narrowed, and
        only for a step that declares a feature input.
        """
        import pandas as pd

        from vtea_core.measurements import feature_matrix

        argument = self.feature_input
        if argument is None or argument not in kwargs:
            return
        value = kwargs[argument]
        if not isinstance(value, pd.DataFrame):
            return
        columns = self.selected_features(value)
        if not columns:
            raise ValueError(
                f"step '{self.category}.{self.function_name}' has no features to work on - "
                f"none of {self.features or 'the table columns'} are numeric features of the "
                f"measurement table (columns: {list(value.columns)})"
            )
        kwargs[argument] = feature_matrix(value, columns)[0]

    def _select_channel(self, value: Any, channel_axis: int | None, full_ndim: int | None) -> Any:
        if self.channel is None or channel_axis is None:
            return value
        if not isinstance(value, np.ndarray) or value.ndim != full_ndim:
            return value
        if not -value.ndim <= channel_axis < value.ndim:
            raise IndexError(
                f"step '{self.category}.{self.function_name}': channel axis {channel_axis} "
                f"is out of range for an array of shape {value.shape}"
            )
        n_channels = value.shape[channel_axis]
        if not 0 <= self.channel < n_channels:
            raise IndexError(
                f"step '{self.category}.{self.function_name}': channel {self.channel} is out of "
                f"range - axis {channel_axis} of a {value.shape} array has {n_channels} channel(s)"
            )
        return np.take(value, self.channel, axis=channel_axis)


class Pipeline:
    """An ordered, editable sequence of Steps.

    `channel_axis` says which axis of the input image is the channel axis
    (None = the image isn't multi-channel). It's a property of the data, so
    it lives here rather than on each Step; which *channel* to use is a
    per-step choice and lives on Step.channel.
    """

    def __init__(self, steps: list[Step] | None = None, channel_axis: int | None = None):
        self.steps: list[Step] = list(steps) if steps else []
        self.channel_axis = channel_axis

    def available_keys(self, seed_keys: set[str] | None = None) -> set[str]:
        """Context keys that will exist once every current step has run -
        i.e. what a step appended next could wire itself to."""
        keys = set() if seed_keys is None else set(seed_keys)
        for step in self.steps:
            keys.add(step.output_key)
            if step.name:
                keys.add(step.name)
        return keys

    def step_names(self) -> list[str]:
        return [step.name for step in self.steps if step.name]

    def names_producing(self, output_key: str) -> list[str]:
        """Names of the steps that produce `output_key` - the choices for
        "which segmentation should this measurement step measure?"."""
        return [step.name for step in self.steps if step.name and step.output_key == output_key]

    def add_step(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    def insert_step(self, index: int, step: Step) -> Step:
        self.steps.insert(index, step)
        return step

    def remove_step(self, index: int) -> Step:
        return self.steps.pop(index)

    def move_step(self, from_index: int, to_index: int) -> None:
        step = self.steps.pop(from_index)
        self.steps.insert(to_index, step)

    def run(
        self,
        context: dict[str, Any] | None = None,
        *,
        channel_axis: int | None = None,
        progress: Any = None,
    ) -> dict[str, Any]:
        """Runs every step in order, threading results through a shared context.

        Returns the final context (the input context plus every step's
        output_key, and its name where it has one); the input dict itself
        isn't mutated.

        `channel_axis` overrides the pipeline's own setting for this run.
        The widest array in the starting context defines "still has a
        channel axis" for per-step channel selection - see Step.run.

        `progress(step, done, total)` is called before each step starts and
        once more when the last one finishes, so a caller can say which step
        a run is on. It is called on whatever thread `run` is on - which for
        the napari builder is a worker thread, so what it is given has to be
        carried to the GUI rather than drawn from inside it.
        """
        context = dict(context) if context else {}
        axis = self.channel_axis if channel_axis is None else channel_axis
        seeded = [value for value in context.values() if isinstance(value, np.ndarray)]
        full_ndim = max((value.ndim for value in seeded), default=None)
        total = len(self.steps)
        for done, step in enumerate(self.steps):
            if progress is not None:
                progress(step, done, total)
            result = step.run(context, channel_axis=axis, full_ndim=full_ndim)
            context[step.output_key] = result
            if step.name:
                # Also under its own name, so a later step can name the one
                # segmentation it wants instead of taking whichever ran last.
                context[step.name] = result
        if progress is not None:
            progress(None, total, total)
        return context

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)
