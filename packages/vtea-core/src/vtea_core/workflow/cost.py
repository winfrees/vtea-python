"""How long a step will take, worked out before it is run.

A progress bar needs to know how far along a run is, and the only honest
source for that before the first voxel is touched is the size of the work:
how many voxels the tile holds, how many objects the table has. That is the
same kind of fact `vtea_core.blocked.contract.Scaling` already declares
about memory - it cannot be recovered from the function's signature - so it
is declared the same way, per step, in one table.

Two things this deliberately does *not* try to be:

- **Accurate.** The coefficients below are order-of-magnitude, measured on
  ordinary CPU hardware, and a machine with twice the memory bandwidth will
  beat them. They are a scale for a progress bar and a "this will take about
  four minutes" before someone commits an afternoon to it, not a promise.
  `Calibration` closes the gap where it matters: it watches what the steps
  actually took on *this* machine and scales later estimates by it.

- **Universal.** Some steps have no a priori estimate worth showing. t-SNE's
  runtime depends on how the optimisation converges, a Leiden partition on
  how the graph is shaped, agglomerative clustering grows as the square of
  the object count with a constant nobody can predict. Those are marked
  `superlinear`, `estimate_seconds` returns None for them, and a caller
  showing a progress bar should show a continuous/indeterminate one instead
  of a fraction it would have to invent.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from vtea_core.blocked.contract import TABLE

# Below this, a duration is not worth reporting as anything but "instant" -
# and a progress bar that appears and vanishes inside one frame is noise.
NEGLIGIBLE_SECONDS = 0.05


@dataclass(frozen=True)
class StepCost:
    """What one step costs, per unit of the work it does.

    `per_voxel_ns` is for the steps that walk the image; `per_object_ns` and
    `per_object_feature_ns` for the ones that walk the feature table (an
    O(n*d) pass like k-means costs per object *and* per feature, which is
    why both terms exist). Nanoseconds because a per-voxel cost in seconds
    is all leading zeros.

    `superlinear` marks a step whose runtime cannot be predicted from the
    size of its input - see the module docstring. Nothing else about the
    entry is read when it is set.
    """

    per_voxel_ns: float = 0.0
    per_object_ns: float = 0.0
    per_object_feature_ns: float = 0.0
    superlinear: bool = False
    notes: str = ""


# Categories whose steps all consume the per-object table rather than the
# image, used to cost a step nobody has timed - see cost_for.
TABLE_CATEGORIES = frozenset(
    {"clustering", "reduction", "classes", "gates", "association", "cells"}
)

# A step nobody has timed: assume it walks its input once, cheaply, and be
# wrong by a small factor rather than by an order of magnitude.
DEFAULT_VOXEL_COST = StepCost(per_voxel_ns=20.0, notes="untimed step; assumed one cheap pass")
DEFAULT_TABLE_COST = StepCost(per_object_ns=1000.0, per_object_feature_ns=100.0)

# Measured roughly, on one CPU core, over uint16 volumes of a few hundred
# megavoxels. Where a step's cost depends strongly on a parameter (a median
# filter's radius, cellpose's diameter) the entry assumes that parameter's
# default - which is why `Calibration` exists.
STEP_COSTS: dict[tuple[str, str], StepCost] = {
    ("imageprocessing", "gaussian_blur"): StepCost(per_voxel_ns=25.0),
    ("imageprocessing", "median_filter"): StepCost(
        per_voxel_ns=250.0, notes="grows with the cube of the radius"
    ),
    ("imageprocessing", "enhance_contrast"): StepCost(per_voxel_ns=40.0),
    ("imageprocessing", "subtract_background"): StepCost(
        per_voxel_ns=400.0, notes="a rolling ball is the most expensive preprocessing step"
    ),
    ("segmentation", "threshold_mask"): StepCost(per_voxel_ns=8.0),
    ("segmentation", "label_components"): StepCost(per_voxel_ns=60.0),
    ("segmentation", "watershed_split"): StepCost(
        per_voxel_ns=450.0, notes="a float64 distance transform plus the flood; the heaviest step"
    ),
    ("segmentation", "filter_by_size"): StepCost(per_voxel_ns=30.0),
    ("segmentation", "labels_from_points"): StepCost(per_voxel_ns=10.0),
    # On a GPU. On CPU it is several times this, which is exactly the kind
    # of gap the calibration below is for.
    ("segmentation", "cellpose_segmentation"): StepCost(
        per_voxel_ns=2500.0, notes="assumes a GPU; CPU inference is several times slower"
    ),
    ("segmentation", "expand_labels"): StepCost(per_voxel_ns=200.0),
    ("segmentation", "label_ring"): StepCost(per_voxel_ns=250.0),
    ("segmentation", "label_shell"): StepCost(per_voxel_ns=300.0),
    ("segmentation", "subtract_labels"): StepCost(per_voxel_ns=6.0),
    ("segmentation", "restrict_labels_to"): StepCost(per_voxel_ns=6.0),
    ("segmentation", "watershed_ownership"): StepCost(per_voxel_ns=400.0),
    ("ownership", "distance_ownership"): StepCost(
        per_voxel_ns=600.0, notes="one distance transform per candidate owner"
    ),
    ("measurements", "extract_measurements"): StepCost(per_voxel_ns=120.0),
    ("measurements", "extract_measurements_by_channel"): StepCost(
        per_voxel_ns=120.0, notes="per voxel of the whole multi-channel array"
    ),
    ("measurements", "weighted_measurements_by_channel"): StepCost(per_voxel_ns=150.0),
    ("association", "associate_by_identity"): StepCost(per_object_ns=2000.0),
    ("association", "associate_objects"): StepCost(
        per_object_ns=20000.0, notes="a kd-tree query per child"
    ),
    ("association", "merge_associations"): StepCost(per_object_ns=1500.0),
    ("cells", "build_cells"): StepCost(per_object_ns=3000.0),
    ("cells", "cell_features"): StepCost(per_object_ns=4000.0, per_object_feature_ns=200.0),
    ("clustering", "kmeans"): StepCost(per_object_ns=3000.0, per_object_feature_ns=400.0),
    ("clustering", "gaussian_mixture"): StepCost(per_object_ns=6000.0, per_object_feature_ns=900.0),
    # Every one of these is either quadratic in the object count or an
    # iterative optimisation that stops when it stops improving. Neither can
    # be turned into "42% done" honestly.
    ("clustering", "hierarchical"): StepCost(superlinear=True, notes="O(n^2) in time and memory"),
    ("clustering", "auto_k_kmeans"): StepCost(
        superlinear=True, notes="one k-means fit per candidate k"
    ),
    ("clustering", "louvain"): StepCost(
        superlinear=True, notes="a kNN graph, then an iterative modularity optimisation"
    ),
    ("clustering", "leiden"): StepCost(
        superlinear=True, notes="a kNN graph, then an iterative modularity optimisation"
    ),
    ("reduction", "pca"): StepCost(per_object_ns=800.0, per_object_feature_ns=150.0),
    ("reduction", "isomap"): StepCost(superlinear=True, notes="a geodesic distance matrix"),
    ("reduction", "laplacian_eigenmap"): StepCost(superlinear=True, notes="an eigendecomposition"),
    ("reduction", "tsne"): StepCost(
        superlinear=True, notes="runtime depends on how the optimisation converges"
    ),
    ("reduction", "umap"): StepCost(
        superlinear=True, notes="a kNN graph, then a stochastic layout optimisation"
    ),
    # A class is a comparison per object, or a handful of them; a label set
    # is a column stack. Cheap enough that the bar is gone before it is
    # seen, which is itself the right thing to report.
    ("classes", "class_from_range"): StepCost(per_object_ns=120.0),
    ("classes", "class_from_values"): StepCost(per_object_ns=150.0),
    ("classes", "class_from_expression"): StepCost(per_object_ns=300.0),
    ("classes", "label_set"): StepCost(per_object_ns=200.0),
    ("classes", "combine_labels"): StepCost(per_object_ns=400.0),
    ("classification", "class_map"): StepCost(per_voxel_ns=10.0),
    ("classification", "train_classifier"): StepCost(superlinear=True, notes="epochs over crops"),
    ("classification", "predict"): StepCost(per_object_ns=200000.0, notes="a forward pass per crop"),
}


def cost_for(category: str, function_name: str) -> StepCost:
    """A step's cost, or a neutral default for one nobody has timed.

    Like `vtea_core.workflow.wiring.scaling_for`, this does not raise on an
    unknown step: a third-party step registered through the entry points is
    one nobody has characterised, and guessing cheaply beats refusing to
    show a progress bar at all. Which default it gets follows the step's
    declared block mode - a table step is not costed per voxel.
    """
    known = STEP_COSTS.get((category, function_name))
    if known is not None:
        return known
    from vtea_core.workflow.wiring import scaling_for

    # An unknown step has no declared scaling either - `scaling_for` hands
    # back the neutral elementwise default - so the category is what is left
    # to go on, and it is a good enough answer: everything in these
    # categories consumes the feature table.
    is_table = (
        category in TABLE_CATEGORIES or scaling_for(category, function_name).mode == TABLE
    )
    return DEFAULT_TABLE_COST if is_table else DEFAULT_VOXEL_COST


def estimate_seconds(
    step: Any,
    *,
    voxels: int = 0,
    n_objects: int = 0,
    n_features: int = 0,
    tiles: int = 1,
    calibration: Calibration | None = None,
) -> float | None:
    """How long `step` should take, or None when there is no honest answer.

    None means one of two things, and a caller should treat them the same
    way - by showing a continuous progress bar rather than a fraction:
    either the step is `superlinear` and its runtime does not follow from
    its input size, or nobody has said how big the input is.

    `tiles` multiplies the estimate for a run that will process the image
    more than once over - a blocked run's per-tile cost is the same work
    done n times, plus a halo this deliberately ignores.
    """
    cost = cost_for(step.category, step.function_name)
    if cost.superlinear:
        return None

    seconds = (
        cost.per_voxel_ns * float(voxels)
        + cost.per_object_ns * float(n_objects)
        + cost.per_object_feature_ns * float(n_objects) * float(max(n_features, 1))
    ) / 1e9
    if seconds <= 0:
        return None
    seconds *= max(int(tiles), 1)
    if calibration is not None:
        seconds *= calibration.scale_for(step.category, step.function_name)
    return seconds


def format_duration(seconds: float | None) -> str:
    """A duration as a person would say it: "about 3 s", "about 4 min".

    None - no estimate - is rendered as an empty string rather than "0 s",
    because a bar labelled "0 s" claims the run is nearly over.
    """
    if seconds is None:
        return ""
    if seconds < NEGLIGIBLE_SECONDS:
        return "instant"
    if seconds < 60:
        return f"about {seconds:.0f} s" if seconds >= 1 else "under 1 s"
    minutes = seconds / 60
    if minutes < 60:
        return f"about {minutes:.0f} min"
    hours = minutes / 60
    return f"about {hours:.1f} h"


class Calibration:
    """What the steps on *this* machine actually cost, learned as they run.

    The table above is a guess made on somebody else's hardware. A run that
    took four times the estimate is not a reason to keep showing the same
    estimate next time: the ratio is recorded per step function and applied
    to later estimates, smoothed so that one unusually slow run (a cold
    cache, a busy machine) moves it a little rather than replacing it.

    Thread-safe, because the observations come from the worker thread a
    protocol runs on and the estimates are read from the GUI thread.
    """

    # How much of a new observation to believe. Low enough that a single
    # outlier barely moves the estimate, high enough that a machine
    # genuinely twice as slow is tracked within a handful of steps.
    WEIGHT = 0.35
    # Ratios outside this are not calibration, they are a different step:
    # a cellpose run that fell back to CPU, a filter whose radius was
    # changed by two orders of magnitude. Clamped so one of those cannot
    # make every later estimate absurd.
    MIN_SCALE = 0.05
    MAX_SCALE = 20.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scales: dict[tuple[str, str], float] = {}

    def scale_for(self, category: str, function_name: str) -> float:
        with self._lock:
            return self._scales.get((category, function_name), 1.0)

    def observe(self, step: Any, *, seconds: float, predicted: float | None) -> None:
        """Record that `step` took `seconds` when `predicted` was expected.

        Ignored when there was no prediction to compare against, or when
        either number is too small to carry information - timing a 3 ms step
        measures the timer, not the step.
        """
        if predicted is None or predicted < NEGLIGIBLE_SECONDS or seconds < NEGLIGIBLE_SECONDS:
            return
        observed_ratio = seconds / predicted
        key = (step.category, step.function_name)
        with self._lock:
            current = self._scales.get(key, 1.0)
            blended = current * (1 - self.WEIGHT) + observed_ratio * self.WEIGHT
            self._scales[key] = min(max(blended, self.MIN_SCALE), self.MAX_SCALE)

    def as_dict(self) -> dict[tuple[str, str], float]:
        with self._lock:
            return dict(self._scales)
