"""What a step does to memory, declared rather than discovered.

A step's function signature says what it takes; it does not say that
`watershed_split` allocates a float64 distance transform four times the size
of the uint16 image it was handed, nor that `gaussian_blur` needs to see
4*sigma voxels beyond a tile's edge to give the same answer a whole-image
run would. Both facts decide whether a dataset can be processed at all, and
neither can be recovered from `inspect.signature`.

So they are declared, in the same place and the same style as the rest of a
step's non-obvious I/O - `vtea_core.workflow.wiring.StepIO`, which already
carries which parameters are data and what the result should be called.
`Scaling` is the fifth thing that table knows.

Nothing here executes anything. It is the vocabulary the tile planner
(`vtea_core.blocked.plan`) and, later, the blocked executor read; the
algorithm functions stay ordinary NumPy functions that know nothing about
tiles. See docs/LARGE_IMAGES.md.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# How a step's output depends on its input, which is what decides whether -
# and how - it can be run one tile at a time.
#
# ELEMENTWISE: an output voxel depends only on the input voxel at the same
#   position. No halo, exact, trivially blocked.
# NEIGHBORHOOD: an output voxel depends on inputs within some distance. Run
#   on a tile grown by that distance and trimmed back; exact when the halo
#   is genuinely large enough, which is a thing to verify rather than assume
#   (see plan.TilePlan.requires_verification).
# GLOBAL_STAT: needs a statistic over the whole image before it can touch a
#   single voxel - Otsu's threshold, a percentile, a min/max rescale. Splits
#   into a streaming statistics pass followed by an elementwise apply.
# OBJECT_LOCAL: works per object, inside its bounding box. Scheduled by
#   object rather than by grid, because a window that holds one object fits
#   in memory by definition.
# ACCUMULATE: reduces voxels to per-object rows. Blockable by accumulating
#   partial sums per object and merging them, with a second pass over the
#   few objects a tile boundary cut for the features that do not compose.
# TABLE: consumes the feature table, not voxels at all. Scales with object
#   count, which is a different problem with different answers.
ELEMENTWISE = "elementwise"
NEIGHBORHOOD = "neighborhood"
GLOBAL_STAT = "global_stat"
OBJECT_LOCAL = "object_local"
ACCUMULATE = "accumulate"
TABLE = "table"

BLOCK_MODES = (ELEMENTWISE, NEIGHBORHOOD, GLOBAL_STAT, OBJECT_LOCAL, ACCUMULATE, TABLE)

# Whether the blocked answer is the answer.
#
# EXACT: identical to a whole-image run, unconditionally.
# EXACT_WITH_HALO: identical provided nothing relevant reached beyond the
#   halo. That precondition is checkable after the fact, and is checked.
# APPROXIMATE: not identical, by the nature of the algorithm. Says so here
#   so that a result can say so too, rather than a user discovering it from
#   a seam in a figure.
EXACT = "exact"
EXACT_WITH_HALO = "exact_with_halo"
APPROXIMATE = "approximate"

EXACTNESS = (EXACT, EXACT_WITH_HALO, APPROXIMATE)

# Modes whose cost is per voxel, and so participate in sizing a tile. The
# rest are sized by object or row count and are planned separately.
VOXEL_MODES = (ELEMENTWISE, NEIGHBORHOOD, GLOBAL_STAT, OBJECT_LOCAL, ACCUMULATE)

# What to assume an object's largest extent is, in voxels, when a step's
# halo is bounded by object size rather than by a parameter and nobody has
# said how big the objects are. Deliberately generous: an under-sized halo
# silently truncates objects, an over-sized one only costs time. Callers
# that know better should say so, and the plan records which happened.
DEFAULT_OBJECT_EXTENT_VOXELS = 64


class HaloTooLarge(ValueError):
    """A step's halo is so large that no useful tile fits inside the budget."""


@dataclass(frozen=True)
class HaloSpec:
    """How far beyond a tile a step reaches, in voxels.

    A halo is rarely a constant: it is 4*sigma for a Gaussian, the radius
    for a median filter, the distance for `expand_labels`. `param` names the
    step parameter it comes from and `scale` multiplies it, so the halo
    follows the setting the user actually chose instead of a guess made when
    the table was written.

    `physical` matters more than it looks. A 5 um dilation is 2.5 voxels
    along z at a 2 um z-step and 25 voxels in x at 0.2 um pixels; a scalar
    halo would be wrong on both axes at once. A physical halo therefore
    resolves to a *per-axis* tuple against the `Spacing`, the same way
    `vtea_core.segmentation.derived` already does its distance transforms.

    `object_extent` is for the steps whose reach is bounded by how big the
    objects are rather than by any parameter - the distance transform in
    `watershed_split`, the inward distance in `label_shell`. There is no
    honest way to derive that from the parameters, so it comes from the
    caller, and a plan built without one records that its halo is an
    assumption (see `TilePlan.requires_verification`).
    """

    voxels: int = 0
    param: str | None = None
    scale: float = 1.0
    physical: bool = False
    object_extent: bool = False
    minimum: int = 0

    @property
    def is_none(self) -> bool:
        return not self.voxels and self.param is None and not self.object_extent

    def resolve(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        spacing: Any = None,
        ndim: int = 3,
        object_extent: float | None = None,
    ) -> tuple[int, ...]:
        """This halo in voxels, one entry per spatial axis.

        The voxel-valued terms (a fixed `voxels`, an object-extent estimate)
        and the parameter-valued one are converted separately and combined
        by a per-axis maximum. Keeping them apart is not fussiness: mixing a
        voxel count into a physical parameter and then dividing the total by
        the voxel size converts a number that was already in voxels, which
        on a 2 um z-step would halve it.

        `spacing` is a `vtea_core.data.Spacing` (or None) and is consulted
        only for a `physical` parameter, and only when it is actually known.
        An unknown spacing means the parameter is already in voxels, which
        is what the rest of the codebase assumes when nobody has said
        otherwise.
        """
        params = params or {}
        floor = float(self.voxels)
        if self.object_extent:
            extent = DEFAULT_OBJECT_EXTENT_VOXELS if object_extent is None else object_extent
            floor = max(floor, float(extent))

        per_axis = [floor] * ndim
        given = params.get(self.param) if self.param is not None else None
        if given is not None:
            per_axis = [
                max(floor, value)
                for value in self._from_parameter(given, spacing=spacing, ndim=ndim)
            ]

        return tuple(max(self.minimum, math.ceil(value)) for value in per_axis)

    def _from_parameter(self, given: Any, *, spacing: Any, ndim: int) -> list[float]:
        """The parameter's contribution, per axis, in voxels."""
        if isinstance(given, (tuple, list, np.ndarray)):
            # A per-axis parameter - a tuple of sigmas, a tuple of radii -
            # already answers the question for each axis.
            values = _broadcast([self.scale * float(item) for item in given], ndim)
        else:
            values = [self.scale * float(given)] * ndim
        if not self.physical:
            return values
        sizes = _voxel_sizes(spacing, ndim)
        if sizes is None:
            return values
        return [value / size for value, size in zip(values, sizes)]


def _broadcast(values: list[float], ndim: int) -> list[float]:
    if len(values) == ndim:
        return values
    if len(values) == 1:
        return values * ndim
    # A 2-tuple of radii on a 3D array is a user error worth naming rather
    # than silently padding with zeros.
    raise ValueError(f"a per-axis halo parameter of length {len(values)} does not fit {ndim} axes")


def _voxel_sizes(spacing: Any, ndim: int) -> tuple[float, ...] | None:
    if spacing is None or not getattr(spacing, "is_known", False):
        return None
    return tuple(float(size) for size in spacing.for_ndim(ndim))


@dataclass(frozen=True)
class Scaling:
    """A step's behaviour on data too large to hold at once.

    `bytes_per_voxel` is the peak *live* bytes per voxel of the tile, over
    every array the step keeps alive at once - its input, its output, and
    the intermediates the library allocates inside. It assumes a `uint16`
    input, which is what fluorescence data almost always is, and it is
    deliberately an upper bound: overestimating costs smaller tiles, and
    underestimating costs an out-of-memory kill three hours into a run.

    `variants` is for the two steps whose scaling genuinely depends on a
    parameter rather than on the function: `threshold_mask` is elementwise
    with `method="fixed"` and needs a whole-image histogram with
    `method="otsu"`, and those are not the same step to a planner. Rather
    than hedge in a comment, the entry keyed on that parameter's value wins.
    """

    mode: str = ELEMENTWISE
    halo: HaloSpec = field(default_factory=HaloSpec)
    bytes_per_voxel: int = 8
    exactness: str = EXACT
    notes: str = ""
    variant_param: str | None = None
    variants: Mapping[str, Scaling] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in BLOCK_MODES:
            raise ValueError(f"unknown block mode {self.mode!r}, expected one of {BLOCK_MODES}")
        if self.exactness not in EXACTNESS:
            raise ValueError(f"unknown exactness {self.exactness!r}, expected one of {EXACTNESS}")
        if self.variants and self.variant_param is None:
            raise ValueError("variants need a variant_param naming the parameter they key on")

    @property
    def is_voxel_scaled(self) -> bool:
        """Whether this step's cost is per voxel, and so sizes a tile."""
        return self.mode in VOXEL_MODES

    def resolve(self, params: Mapping[str, Any] | None = None) -> Scaling:
        """This scaling with any parameter-dependent variant applied."""
        if not self.variants:
            return self
        params = params or {}
        chosen = params.get(self.variant_param)
        if chosen is None:
            # The parameter was left at the function's own default,
            # which is not visible from here. Each entry's base scaling is
            # therefore written to mirror that default.
            return self
        return self.variants.get(str(chosen), self)


# The default for a step nobody has characterised yet: assume it reaches
# nowhere and costs little, and be wrong cheaply rather than confidently.
DEFAULT_SCALING = Scaling()
