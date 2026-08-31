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

from vtea_core.blocked.contract import (
    ACCUMULATE,
    APPROXIMATE,
    DEFAULT_SCALING,
    ELEMENTWISE,
    EXACT_WITH_HALO,
    GLOBAL_STAT,
    NEIGHBORHOOD,
    OBJECT_LOCAL,
    TABLE,
    HaloSpec,
    Scaling,
)
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
    `names_from`: parameters whose value is the *name of another input's
    source*, mapped parameter -> input. An association step records which
    segmentation each object came from, and that name is the wiring - the
    step the input is pointed at - not something for the user to retype and
    get wrong. Step.run fills these in, and they are excluded from the Edit
    form for the same reason data inputs are.
    `scaling`: what this step does to memory, and how far beyond a tile it
    reaches - see vtea_core.blocked.contract.Scaling. Declared here for the
    same reason the rest of this table is: it cannot be recovered from the
    function's signature, and without it a dataset larger than RAM cannot be
    divided up without guessing. It changes nothing about how a step runs in
    memory; the tile planner is its only reader today.
    """

    inputs: tuple[str, ...]
    output: str
    channel_mode: str = CHANNEL_SLICE
    feature_input: str | None = None
    names_from: tuple[tuple[str, str], ...] = ()
    scaling: Scaling = DEFAULT_SCALING

    @property
    def channel_aware(self) -> bool:
        return self.channel_mode == CHANNEL_ARGUMENT


# Scaling contracts shared by several steps. Spelling the common shapes out
# once keeps the table below about I/O, with only the steps that are
# genuinely interesting about memory saying anything unusual.
#
# `bytes_per_voxel` is peak live bytes per voxel of the tile - inputs,
# output, and whatever the library allocates in between - assuming a uint16
# image. They are upper bounds on purpose: too large costs smaller tiles,
# too small costs an out-of-memory kill hours into a run.
_TABLE_STEP = Scaling(
    mode=TABLE,
    bytes_per_voxel=0,
    notes="scales with object count, not image size",
)
_LABEL_ELEMENTWISE = Scaling(mode=ELEMENTWISE, bytes_per_voxel=12)

# Preprocessing writes back to "volume" on purpose, so a blur/background
# step feeds the segmentation that follows it. The untouched original stays
# available as "intensity", which is what measurement steps want.
STEP_IO: dict[tuple[str, str], StepIO] = {
    ("imageprocessing", "gaussian_blur"): StepIO(
        ("volume",),
        "volume",
        scaling=Scaling(
            mode=NEIGHBORHOOD,
            halo=HaloSpec(param="sigma", scale=4.0, minimum=1),
            bytes_per_voxel=12,
            exactness=EXACT_WITH_HALO,
            notes="scipy truncates the Gaussian kernel at 4 sigma",
        ),
    ),
    ("imageprocessing", "median_filter"): StepIO(
        ("volume",),
        "volume",
        scaling=Scaling(
            mode=NEIGHBORHOOD,
            halo=HaloSpec(param="radius", minimum=1),
            bytes_per_voxel=10,
            exactness=EXACT_WITH_HALO,
        ),
    ),
    # Two different steps wearing one name. Rescaling to the image's own
    # intensity range needs one streaming pass for the min and max and is
    # then exact; CLAHE is tile-local by construction and cannot be made
    # exact by any halo - the best that can be done is to align the tiles to
    # its kernel grid.
    ("imageprocessing", "enhance_contrast"): StepIO(
        ("volume",),
        "volume",
        scaling=Scaling(
            mode=GLOBAL_STAT,
            bytes_per_voxel=16,
            notes="one streaming pass for the global min and max",
            variant_param="method",
            variants={
                "normalize": Scaling(mode=GLOBAL_STAT, bytes_per_voxel=16),
                "equalize": Scaling(
                    mode=NEIGHBORHOOD,
                    halo=HaloSpec(param="kernel_size", minimum=1),
                    bytes_per_voxel=24,
                    exactness=APPROXIMATE,
                    notes=(
                        "CLAHE adapts to each kernel window; a tiled run matches a whole-image "
                        "one only if kernel_size is set explicitly and divides the tile shape"
                    ),
                ),
            },
        ),
    ),
    ("imageprocessing", "subtract_background"): StepIO(
        ("volume",),
        "volume",
        scaling=Scaling(
            mode=NEIGHBORHOOD,
            halo=HaloSpec(param="radius", minimum=1),
            bytes_per_voxel=32,
            exactness=APPROXIMATE,
            notes=(
                "a rolling ball is not strictly local; estimating the background on a "
                "coarse pyramid level and upsampling is both closer and faster at scale"
            ),
        ),
    ),
    # Fixed thresholding is per voxel; Otsu and percentile need to see every
    # voxel before they can classify one. For integer data the streaming
    # pass is a full histogram, so the threshold is exact rather than
    # sampled.
    ("segmentation", "threshold_mask"): StepIO(
        ("volume",),
        "mask",
        scaling=Scaling(
            mode=ELEMENTWISE,
            bytes_per_voxel=6,
            variant_param="method",
            variants={
                "fixed": Scaling(mode=ELEMENTWISE, bytes_per_voxel=6),
                "otsu": Scaling(
                    mode=GLOBAL_STAT,
                    bytes_per_voxel=6,
                    notes="exact for integer data: a full histogram, not a sample",
                ),
                "percentile": Scaling(
                    mode=GLOBAL_STAT,
                    bytes_per_voxel=6,
                    notes="exact for integer data; a float image needs an approximate sketch",
                ),
            },
        ),
    ),
    ("segmentation", "label_components"): StepIO(
        ("mask",),
        "labels",
        scaling=Scaling(
            mode=NEIGHBORHOOD,
            halo=HaloSpec(voxels=1),
            bytes_per_voxel=10,
            needs_reconciliation=True,
            notes="dask_image.ndmeasure.label already merges components across chunks",
        ),
    ),
    # The distance transform is global in principle: how far a voxel is from
    # background is only bounded by how large the objects are. Hence a halo
    # from the expected object size, and a check afterwards that it held.
    ("segmentation", "watershed_split"): StepIO(
        ("intensity", "mask"),
        "labels",
        scaling=Scaling(
            mode=NEIGHBORHOOD,
            halo=HaloSpec(object_extent=True, minimum=8),
            bytes_per_voxel=35,
            exactness=EXACT_WITH_HALO,
            needs_reconciliation=True,
            notes="float64 distance transform plus markers; the heaviest step in a typical protocol",
        ),
    ),
    # Sizes come from the reconciled object ledger rather than from a tile,
    # so the blocked form is a lookup-table remap - per voxel, and exact.
    ("segmentation", "filter_by_size"): StepIO(
        ("labels",),
        "labels",
        scaling=Scaling(
            mode=ELEMENTWISE,
            bytes_per_voxel=12,
            needs_reconciliation=True,
            notes="the remap is per voxel, but the sizes it filters on are per whole object",
        ),
    ),
    ("segmentation", "labels_from_points"): StepIO(
        ("points", "shape"),
        "labels",
        scaling=Scaling(
            mode=OBJECT_LOCAL,
            halo=HaloSpec(param="radius", minimum=1),
            bytes_per_voxel=8,
            notes="points are sparse; each is drawn into its own window",
        ),
    ),
    # The one step whose tile size is set by the GPU rather than by RAM, and
    # the one whose blocked answer genuinely differs from a whole-image one:
    # the flow field near a tile edge is computed from truncated context.
    ("segmentation", "cellpose_segmentation"): StepIO(
        ("volume", "model"),
        "labels",
        scaling=Scaling(
            mode=NEIGHBORHOOD,
            halo=HaloSpec(param="diameter", scale=1.5, minimum=32),
            bytes_per_voxel=96,
            exactness=APPROXIMATE,
            needs_reconciliation=True,
            notes=(
                "not translation-invariant, so seam objects are re-segmented in a "
                "seam-centred window rather than owned by one tile; sized against the "
                "GPU budget, not the CPU one. do_3D grows faster than the volume - "
                "stitch_threshold segments plane-wise and links z with cellpose's own "
                "stitch3D instead"
            ),
        ),
    ),
    # Derived segmentations. They consume a label image, so a channel choice
    # is meaningless; thicknesses are physical when a spacing is available -
    # and so are their halos, which is why a 5 um dilation is a different
    # number of voxels along z than in x.
    ("segmentation", "expand_labels"): StepIO(
        ("labels", "spacing"),
        "labels",
        channel_mode=CHANNEL_NONE,
        scaling=Scaling(
            mode=NEIGHBORHOOD,
            halo=HaloSpec(param="distance", physical=True, minimum=1),
            bytes_per_voxel=40,
            exactness=EXACT_WITH_HALO,
            notes="growth is capped by `distance`, so a halo above it is provably enough",
        ),
    ),
    ("segmentation", "label_ring"): StepIO(
        ("labels", "spacing"),
        "labels",
        channel_mode=CHANNEL_NONE,
        scaling=Scaling(
            mode=NEIGHBORHOOD,
            halo=HaloSpec(param="thickness", physical=True, minimum=1),
            bytes_per_voxel=48,
            exactness=EXACT_WITH_HALO,
        ),
    ),
    # Reaches outward by a parameter and inward by however deep the object
    # is, so its halo is bounded by object size as well.
    ("segmentation", "label_shell"): StepIO(
        ("labels", "spacing"),
        "labels",
        channel_mode=CHANNEL_NONE,
        scaling=Scaling(
            mode=NEIGHBORHOOD,
            halo=HaloSpec(param="outward", physical=True, object_extent=True, minimum=1),
            bytes_per_voxel=48,
            exactness=EXACT_WITH_HALO,
        ),
    ),
    ("segmentation", "subtract_labels"): StepIO(
        ("labels", "other"), "labels", channel_mode=CHANNEL_NONE, scaling=_LABEL_ELEMENTWISE
    ),
    ("segmentation", "restrict_labels_to"): StepIO(
        ("labels", "mask"), "labels", channel_mode=CHANNEL_NONE, scaling=_LABEL_ELEMENTWISE
    ),
    # Divides a region among the markers inside it, keeping their ids, so the
    # territories associate back to their owners by identity. Scheduled per
    # connected region of the mask rather than by grid - a region is
    # cell-sized, and a window that holds one fits in memory by definition.
    ("segmentation", "watershed_ownership"): StepIO(
        ("labels", "mask", "spacing"),
        "labels",
        channel_mode=CHANNEL_NONE,
        scaling=Scaling(
            mode=OBJECT_LOCAL,
            halo=HaloSpec(object_extent=True, minimum=8),
            bytes_per_voxel=40,
            needs_reconciliation=True,
        ),
    ),
    # `child_name`/`parent_name` are filled from the wiring: the segmentation
    # an object came from is the step the input points at.
    ("association", "associate_by_identity"): StepIO(
        ("child_labels", "parent_labels"),
        "associations",
        channel_mode=CHANNEL_NONE,
        names_from=(("child_name", "child_labels"), ("parent_name", "parent_labels")),
        scaling=_TABLE_STEP,
    ),
    # Centroid scoring is a table operation over a kd-tree; boundary-distance
    # scoring reads each candidate pair's bounding boxes. Either way the
    # candidates are already restricted to what is nearby, which is what
    # keeps this tractable at ten million objects.
    ("association", "associate_objects"): StepIO(
        ("child_labels", "parent_labels", "spacing"),
        "associations",
        channel_mode=CHANNEL_NONE,
        names_from=(("child_name", "child_labels"), ("parent_name", "parent_labels")),
        scaling=Scaling(
            mode=OBJECT_LOCAL,
            halo=HaloSpec(param="max_distance", physical=True, minimum=1),
            bytes_per_voxel=12,
        ),
    ),
    ("association", "merge_associations"): StepIO(
        ("associations", "other"),
        "associations",
        channel_mode=CHANNEL_NONE,
        scaling=_TABLE_STEP,
    ),
    # `root` names the segmentation that identifies a cell, so it is filled
    # from the wiring like the association names are.
    ("cells", "build_cells"): StepIO(
        ("associations", "root_labels"),
        "cells",
        channel_mode=CHANNEL_NONE,
        names_from=(("root", "root_labels"),),
        scaling=_TABLE_STEP,
    ),
    # One row per cell, from one measurement table per segmentation. The
    # tables are seeded by the caller (the builder gathers them from its
    # measurement steps), the way `data` is for clustering.
    ("cells", "cell_features"): StepIO(
        ("cells", "measurement_tables"),
        "cell_table",
        channel_mode=CHANNEL_NONE,
        scaling=_TABLE_STEP,
    ),
    # A posterior over owners per voxel rather than one answer. `segmentation`
    # names the markers the ids belong to, taken from the wiring. Its output
    # is several times the size of the label image, which is why the large
    # form of it is restricted to the mask rather than dense.
    ("ownership", "distance_ownership"): StepIO(
        ("labels", "mask", "spacing"),
        "ownership",
        channel_mode=CHANNEL_NONE,
        names_from=(("segmentation", "labels"),),
        scaling=Scaling(
            mode=OBJECT_LOCAL,
            halo=HaloSpec(param="reach", physical=True, minimum=2),
            bytes_per_voxel=40,
        ),
    ),
    # Voxels in, one row per object out. Count, sum, sum of squares, min,
    # max and the coordinate sums all compose across tiles exactly; the
    # features that do not (threshold_mean, anything about shape) are
    # recomputed for the few objects a seam actually cut.
    ("measurements", "extract_measurements"): StepIO(
        ("labels", "intensity", "spacing"),
        "measurements",
        scaling=Scaling(
            mode=ACCUMULATE,
            bytes_per_voxel=32,
            notes=(
                "the accumulators are per object, but the pass needs float64 copies of "
                "the tile: the intensities, their squares, and one axis of coordinates"
            ),
        ),
    ),
    ("measurements", "extract_measurements_by_channel"): StepIO(
        ("labels", "intensity", "channel_axis", "spacing"),
        "measurements",
        channel_mode=CHANNEL_ARGUMENT,
        scaling=Scaling(mode=ACCUMULATE, bytes_per_voxel=40),
    ),
    ("measurements", "weighted_measurements_by_channel"): StepIO(
        ("ownership", "intensity", "channel_axis", "spacing"),
        "measurements",
        channel_mode=CHANNEL_ARGUMENT,
        scaling=Scaling(
            mode=ACCUMULATE,
            bytes_per_voxel=48,
            notes="probability-weighted sums are additive, so this needs no second pass",
        ),
    ),
    # Clustering and reduction consume "data" - the per-object feature
    # matrix built from the measurement table, which already carries every
    # channel's features as separate columns (mean_ch0, mean_ch2, ...) plus
    # any earlier clustering/reduction output. There is no channel axis left
    # to choose from at that point, and no image either.
    ("clustering", "kmeans"): StepIO(
        ("data",),
        "clusters",
        channel_mode=CHANNEL_NONE,
        feature_input="data",
        scaling=_TABLE_STEP,
    ),
    ("clustering", "gaussian_mixture"): StepIO(
        ("data",),
        "clusters",
        channel_mode=CHANNEL_NONE,
        feature_input="data",
        scaling=_TABLE_STEP,
    ),
    # The one analysis step with no scalable form: agglomerative clustering
    # is O(n^2) in both time and memory, so above roughly 1e5 objects it has
    # to be fitted on a subsample and extended, and should say so.
    ("clustering", "hierarchical"): StepIO(
        ("data",),
        "clusters",
        channel_mode=CHANNEL_NONE,
        feature_input="data",
        scaling=Scaling(
            mode=TABLE,
            bytes_per_voxel=0,
            exactness=APPROXIMATE,
            notes="O(n^2); above ~1e5 objects only a subsample can be fitted",
        ),
    ),
    ("clustering", "auto_k_kmeans"): StepIO(
        ("data",),
        "clusters",
        channel_mode=CHANNEL_NONE,
        feature_input="data",
        scaling=_TABLE_STEP,
    ),
    ("reduction", "pca"): StepIO(
        ("data",),
        "reduced",
        channel_mode=CHANNEL_NONE,
        feature_input="data",
        scaling=_TABLE_STEP,
    ),
    ("reduction", "isomap"): StepIO(
        ("data",),
        "reduced",
        channel_mode=CHANNEL_NONE,
        feature_input="data",
        scaling=Scaling(
            mode=TABLE,
            bytes_per_voxel=0,
            exactness=APPROXIMATE,
            notes="fitted on a subsample and extended with Isomap.transform",
        ),
    ),
    ("reduction", "laplacian_eigenmap"): StepIO(
        ("data",),
        "reduced",
        channel_mode=CHANNEL_NONE,
        feature_input="data",
        scaling=Scaling(
            mode=TABLE,
            bytes_per_voxel=0,
            exactness=APPROXIMATE,
            notes="sklearn's SpectralEmbedding has no transform; needs a Nystrom extension",
        ),
    ),
    ("reduction", "tsne"): StepIO(
        ("data",),
        "reduced",
        channel_mode=CHANNEL_NONE,
        feature_input="data",
        scaling=Scaling(
            mode=TABLE,
            bytes_per_voxel=0,
            exactness=APPROXIMATE,
            notes="sklearn's TSNE has no transform; openTSNE would be needed to extend a fit",
        ),
    ),
    ("gates", "polygon_gate"): StepIO(
        ("x", "y", "vertices"), "gate", channel_mode=CHANNEL_NONE, scaling=_TABLE_STEP
    ),
    ("gates", "rectangle_gate"): StepIO(
        ("x", "y"), "gate", channel_mode=CHANNEL_NONE, scaling=_TABLE_STEP
    ),
    ("classification", "class_map"): StepIO(
        ("labels", "object_ids", "class_labels"),
        "class_map",
        channel_mode=CHANNEL_NONE,
        scaling=_LABEL_ELEMENTWISE,
    ),
    # Both take per-object crops, not the volume, so they stream naturally
    # once the crops are read from the store by bounding box instead of
    # being materialized as one array.
    ("classification", "train_classifier"): StepIO(
        ("model", "crops", "labels"),
        "model",
        channel_mode=CHANNEL_NONE,
        scaling=_TABLE_STEP,
    ),
    ("classification", "predict"): StepIO(
        ("model", "crops"), "predictions", channel_mode=CHANNEL_NONE, scaling=_TABLE_STEP
    ),
}

# Kept as a name-only view for callers that just need "which parameters are
# data, not form fields" - the napari Edit-step form uses this to decide
# what to render. The `names_from` parameters join them: they are derived
# from the wiring too, so putting them on the form would invite a typed
# segmentation name that disagrees with the step the input is pointed at.
DATA_PARAMETERS: dict[tuple[str, str], set[str]] = {
    key: set(step_io.inputs) | {parameter for parameter, _input in step_io.names_from}
    for key, step_io in STEP_IO.items()
}


def step_io(category: str, function_name: str) -> StepIO:
    try:
        return STEP_IO[(category, function_name)]
    except KeyError as exc:
        raise KeyError(f"no I/O description for step '{category}.{function_name}'") from exc


def scaling_for(category: str, function_name: str) -> Scaling:
    """A step's scaling contract, or a neutral default for one that has none.

    Unlike `step_io`, this does not raise on an unknown step. A third-party
    step registered through the entry-point groups is a step nobody has
    characterised for memory yet, and the right response is to plan
    conservatively around it rather than to refuse to plan at all. Call
    `Scaling.resolve(params)` on the result to apply any parameter-dependent
    variant.
    """
    spec = STEP_IO.get((category, function_name))
    return DEFAULT_SCALING if spec is None else spec.scaling


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
