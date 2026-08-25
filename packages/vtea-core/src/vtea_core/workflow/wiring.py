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


@dataclass(frozen=True)
class StepIO:
    """`inputs`: parameter names resolved from the run context.
    `output`: context key this step's return value is stored under."""

    inputs: tuple[str, ...]
    output: str


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
    ("clustering", "kmeans"): StepIO(("data",), "clusters"),
    ("clustering", "gaussian_mixture"): StepIO(("data",), "clusters"),
    ("clustering", "hierarchical"): StepIO(("data",), "clusters"),
    ("clustering", "auto_k_kmeans"): StepIO(("data",), "clusters"),
    ("reduction", "pca"): StepIO(("data",), "reduced"),
    ("reduction", "isomap"): StepIO(("data",), "reduced"),
    ("reduction", "laplacian_eigenmap"): StepIO(("data",), "reduced"),
    ("reduction", "tsne"): StepIO(("data",), "reduced"),
    ("gates", "polygon_gate"): StepIO(("x", "y", "vertices"), "gate"),
    ("gates", "rectangle_gate"): StepIO(("x", "y"), "gate"),
    ("classification", "class_map"): StepIO(("labels", "object_ids", "class_labels"), "class_map"),
    ("classification", "train_classifier"): StepIO(("model", "crops", "labels"), "model"),
    ("classification", "predict"): StepIO(("model", "crops"), "predictions"),
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
