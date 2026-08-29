"""Declarative I/O description for each registered step function, and the
default wiring derived from it.

A Step needs to know two things that its function's signature alone doesn't
say: which of its parameters are *data* (arrays/tables threaded through the
Pipeline's run context) rather than configuration literals, and what the
thing it returns should be called so the next step can pick it up.

Without this, a Step built by just naming a function - which is all the
napari protocol-builder widget can do from a menu - has an empty
`input_keys` and so calls its function with no data at all:

    TypeError: cellpose_segmentation() missing 1 required positional
    argument: 'volume'

and every step defaults to the same `output_key`, so steps overwrite each
other instead of chaining. STEP_IO plus default_wiring() below are what let
a step be added by name and still run.

The output names are deliberately semantic ("mask", "labels") rather than
positional ("step2_out"): a step's data inputs are matched to context keys
by name, so `threshold_mask -> mask` feeding `label_components(mask)` wires
itself up with no extra bookkeeping.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from vtea_core.workflow.registry import get_step_function

# How a step relates to the image's channel axis.
#
# SLICE: its inputs are images that still carry the channel axis, so
#   Step.channel selects one before the function is called.
# ARGUMENT: the function handles channels itself and needs to see all of
#   them at once (to label its output columns by channel, say), so it is
#   handed the whole array and the channel choice as an argument.
# NONE: its inputs are per-object tables - a feature matrix, a column of
#   values - which have no channel axis at all. A channel choice is
#   meaningless for these, and offering one in the GUI implies the step
#   works on one channel of the image when it works on the measured data.
CHANNEL_SLICE = "slice"
CHANNEL_ARGUMENT = "argument"
CHANNEL_NONE = "none"


@dataclass(frozen=True)
class StepIO:
    """`inputs`: parameter names resolved from the run context.
    `output`: context key this step's return value is stored under.
    `channel_mode`: one of CHANNEL_SLICE / CHANNEL_ARGUMENT / CHANNEL_NONE
    above.
    `feature_input`: the name of the input that is a *feature table* rather
    than a plain array. When a DataFrame is wired to it, Step.run turns it
    into the (n_objects, n_features) matrix the function expects, using the
    step's own `features` selection - which is how a clustering step is told
    to use six of the forty measured features and no others.
    """

    inputs: tuple[str, ...]
    output: str
    channel_mode: str = CHANNEL_SLICE
    feature_input: str | None = None

    @property
    def channel_aware(self) -> bool:
        return self.channel_mode == CHANNEL_ARGUMENT


# Preprocessing writes back to "volume" on purpose, so a blur/background
# step feeds the segmentation that follows it. The untouched original stays
# available as "intensity", which is what measurement steps want.
STEP_IO: dict[tuple[str, str], StepIO] = {
    ("imageprocessing", "gaussian_blur"): StepIO(("volume",), "volume"),
    ("imageprocessing", "median_filter"): StepIO(("volume",), "volume"),
    ("imageprocessing", "enhance_contrast"): StepIO(("volume",), "volume"),
    ("imageprocessing", "subtract_background"): StepIO(("volume",), "volume"),
    ("segmentation", "threshold_mask"): StepIO(("volume",), "mask"),
    ("segmentation", "label_components"): StepIO(("mask",), "labels"),
    ("segmentation", "watershed_split"): StepIO(("intensity", "mask"), "labels"),
    ("segmentation", "filter_by_size"): StepIO(("labels",), "labels"),
    ("segmentation", "labels_from_points"): StepIO(("points", "shape"), "labels"),
    ("segmentation", "cellpose_segmentation"): StepIO(("volume", "model"), "labels"),
    ("measurements", "extract_measurements"): StepIO(("labels", "intensity"), "measurements"),
    ("measurements", "extract_measurements_by_channel"): StepIO(
        ("labels", "intensity", "channel_axis"), "measurements", channel_mode=CHANNEL_ARGUMENT
    ),
    # Clustering and reduction consume "data" - the per-object feature
    # matrix built from the measurement table, which already carries every
    # channel's features as separate columns (mean_ch0, mean_ch2, ...) plus
    # any earlier clustering/reduction output. There is no channel axis left
    # to choose from at that point.
    ("clustering", "kmeans"): StepIO(
        ("data",), "clusters", channel_mode=CHANNEL_NONE, feature_input="data"
    ),
    ("clustering", "gaussian_mixture"): StepIO(
        ("data",), "clusters", channel_mode=CHANNEL_NONE, feature_input="data"
    ),
    ("clustering", "hierarchical"): StepIO(
        ("data",), "clusters", channel_mode=CHANNEL_NONE, feature_input="data"
    ),
    ("clustering", "auto_k_kmeans"): StepIO(
        ("data",), "clusters", channel_mode=CHANNEL_NONE, feature_input="data"
    ),
    ("reduction", "pca"): StepIO(
        ("data",), "reduced", channel_mode=CHANNEL_NONE, feature_input="data"
    ),
    ("reduction", "isomap"): StepIO(
        ("data",), "reduced", channel_mode=CHANNEL_NONE, feature_input="data"
    ),
    ("reduction", "laplacian_eigenmap"): StepIO(
        ("data",), "reduced", channel_mode=CHANNEL_NONE, feature_input="data"
    ),
    ("reduction", "tsne"): StepIO(
        ("data",), "reduced", channel_mode=CHANNEL_NONE, feature_input="data"
    ),
    ("gates", "polygon_gate"): StepIO(("x", "y", "vertices"), "gate", channel_mode=CHANNEL_NONE),
    ("gates", "rectangle_gate"): StepIO(("x", "y"), "gate", channel_mode=CHANNEL_NONE),
    ("classification", "class_map"): StepIO(
        ("labels", "object_ids", "class_labels"), "class_map", channel_mode=CHANNEL_NONE
    ),
    ("classification", "train_classifier"): StepIO(
        ("model", "crops", "labels"), "model", channel_mode=CHANNEL_NONE
    ),
    ("classification", "predict"): StepIO(
        ("model", "crops"), "predictions", channel_mode=CHANNEL_NONE
    ),
}

# Kept as a name-only view for callers that just need "which parameters are
# data, not form fields" - the napari Edit-step form uses this to decide
# what to render.
DATA_PARAMETERS: dict[tuple[str, str], set[str]] = {
    key: set(step_io.inputs) for key, step_io in STEP_IO.items()
}


def step_io(category: str, function_name: str) -> StepIO:
    try:
        return STEP_IO[(category, function_name)]
    except KeyError as exc:
        raise KeyError(f"no I/O description for step '{category}.{function_name}'") from exc


def _required_inputs(category: str, function_name: str) -> set[str]:
    """Data inputs the function can't be called without. Optional ones (a
    default is defined, e.g. cellpose's `model=None`) are only wired when
    something upstream actually produced them."""
    signature = inspect.signature(get_step_function(category, function_name))
    return {
        name
        for name in step_io(category, function_name).inputs
        if name in signature.parameters
        and signature.parameters[name].default is inspect.Parameter.empty
    }


def default_wiring(
    category: str, function_name: str, available: set[str] | None = None
) -> tuple[dict[str, str], str]:
    """(input_keys, output_key) for a step added by name alone.

    Data inputs map to the context key of the same name. `available` is the
    set of keys the pipeline will already hold when this step runs; optional
    inputs are wired only if present there. Required inputs are always
    wired, even when nothing produces them yet - that way running the
    pipeline reports the clear "needs context key(s) [...], available [...]"
    error from Step.run instead of a bare TypeError from deep inside the
    function.
    """
    available = set() if available is None else available
    required = _required_inputs(category, function_name)
    spec = step_io(category, function_name)
    input_keys = {name: name for name in spec.inputs if name in required or name in available}
    return input_keys, spec.output
