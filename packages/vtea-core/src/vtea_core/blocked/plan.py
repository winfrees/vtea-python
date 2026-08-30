"""Turning a memory budget into a grid of tiles, and saying so out loud.

The plan is the thing a user needs to see. A run that reports "512x512x512
tiles, 64-voxel halo, 4,096 tiles, bounded by watershed_split_1" is one
whose progress bar means something and whose slowness has an explanation. A
run that just takes four hours does not.

It is also provenance. Change the tile size and objects that straddle a seam
can change with it, so a result that does not record the plan that produced
it cannot be reproduced - see docs/LARGE_IMAGES.md on the invariance tests
that hold this to account.

Nothing here reads or writes data. A `TilePlan` is a description; the
executor that acts on one arrives in Phase L2.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

from vtea_core.blocked.budget import BudgetTooSmall, MemoryBudget, format_bytes
from vtea_core.blocked.contract import (
    APPROXIMATE,
    EXACT_WITH_HALO,
    HaloTooLarge,
    Scaling,
)

# A tile whose core is smaller than its own halo costs more in overlap than
# it does in work, and the arithmetic gets worse from there. Below this the
# planner refuses rather than producing a plan that would technically run.
MIN_CORE_TO_HALO = 1.0

# Below this, the plan still runs but is wasteful enough to say so: a core
# only twice its halo re-reads and re-computes most voxels several times.
EFFICIENT_CORE_TO_HALO = 4.0


@dataclass(frozen=True)
class Tile:
    """One tile: where its core is, and what has to be read to compute it.

    `core` is the region this tile is responsible for. `padded` is what must
    actually be read - the core grown by the halo and clipped to the array.
    `pad_width` is the part of the halo that fell off the edge of the
    dataset and therefore has to be synthesized rather than read; passing it
    to `numpy.pad` with the step's own boundary mode is what makes a tiled
    result identical to a whole-image one at the volume's borders, rather
    than merely similar.

    `inner` locates the core inside the padded block, which is how a result
    is trimmed back after the step has run.
    """

    index: tuple[int, ...]
    core: tuple[slice, ...]
    padded: tuple[slice, ...]
    pad_width: tuple[tuple[int, int], ...]

    @property
    def ndim(self) -> int:
        return len(self.core)

    @property
    def core_shape(self) -> tuple[int, ...]:
        return tuple(s.stop - s.start for s in self.core)

    @property
    def padded_shape(self) -> tuple[int, ...]:
        """Including the part that has to be synthesized by padding."""
        return tuple(
            (s.stop - s.start) + before + after
            for s, (before, after) in zip(self.padded, self.pad_width)
        )

    @property
    def inner(self) -> tuple[slice, ...]:
        """Where the core sits inside the padded block."""
        result = []
        for core, padded, (before, _after) in zip(self.core, self.padded, self.pad_width):
            start = core.start - padded.start + before
            result.append(slice(start, start + (core.stop - core.start)))
        return tuple(result)

    @property
    def at_dataset_border(self) -> bool:
        """Whether any of this tile's halo fell off the edge of the data.

        Not the same question as whether an *object* touches the dataset
        border, which is what a cytometry "exclude edge objects" option
        means - but it is what tells the reconciler that a seam here is not
        a seam at all.
        """
        return any(before or after for before, after in self.pad_width)


@dataclass(frozen=True)
class TilePlan:
    """How a dataset of `shape` is divided up for a given budget.

    `tile` is the core shape; `halo` is added on every side of it and
    trimmed off afterwards. `tiled_axes` are the axes that may be split -
    everything else (a channel axis, a time axis of length one) is taken
    whole by every tile, and counts towards the tile's cost.
    """

    shape: tuple[int, ...]
    tile: tuple[int, ...]
    halo: tuple[int, ...]
    budget: MemoryBudget
    bytes_per_voxel: int
    tiled_axes: tuple[int, ...]
    chunks: tuple[int, ...] | None = None
    bound_by: str = ""
    requires_verification: bool = False
    approximate_steps: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def splits(self) -> tuple[int, ...]:
        return tuple(
            math.ceil(extent / size) for extent, size in zip(self.shape, self.tile)
        )

    @property
    def n_tiles(self) -> int:
        return math.prod(self.splits)

    @property
    def is_single_tile(self) -> bool:
        """Whether the data fits whole, in which case the executor should
        get out of the way and call the function directly - and the result
        must be identical to a blocked run of one tile, which is the
        cheapest and most valuable test in the suite."""
        return self.n_tiles == 1

    @property
    def tile_bytes(self) -> int:
        return _cost(self.padded_tile, self.bytes_per_voxel)

    @property
    def padded_tile(self) -> tuple[int, ...]:
        return tuple(
            min(extent, size + 2 * pad)
            for extent, size, pad in zip(self.shape, self.tile, self.halo)
        )

    @property
    def overlap_ratio(self) -> float:
        """Voxels read divided by voxels in the dataset. 1.0 is a perfect
        partition; 2.0 means the halo doubles the work."""
        return (math.prod(self.padded_tile) * self.n_tiles) / max(math.prod(self.shape), 1)

    @property
    def is_efficient(self) -> bool:
        return all(
            size >= EFFICIENT_CORE_TO_HALO * pad
            for axis, (size, pad) in enumerate(zip(self.tile, self.halo))
            if axis in self.tiled_axes and pad
        )

    def tiles(self) -> Iterator[Tile]:
        """Every tile, in C order."""
        ranges = [range(count) for count in self.splits]
        for index in product(*ranges):
            yield self._tile_at(index)

    def tile_at(self, index: Sequence[int]) -> Tile:
        index = tuple(index)
        if len(index) != self.ndim:
            raise ValueError(f"tile index {index} does not have {self.ndim} axes")
        for value, count in zip(index, self.splits):
            if not 0 <= value < count:
                raise IndexError(f"tile index {index} is outside the {self.splits} grid")
        return self._tile_at(index)

    def _tile_at(self, index: tuple[int, ...]) -> Tile:
        core, padded, pad_width = [], [], []
        for value, size, extent, pad in zip(index, self.tile, self.shape, self.halo):
            start = value * size
            stop = min(start + size, extent)
            core.append(slice(start, stop))
            # The halo is read where the data exists and synthesized where
            # it does not, which is only ever at the dataset's own border.
            padded.append(slice(max(start - pad, 0), min(stop + pad, extent)))
            pad_width.append((max(pad - start, 0), max((stop + pad) - extent, 0)))
        return Tile(index, tuple(core), tuple(padded), tuple(pad_width))

    def describe(self) -> str:
        """The one line worth putting in front of a user before a long run."""
        tile = "x".join(str(size) for size in self.tile)
        if self.is_single_tile:
            summary = f"1 tile of {tile} - the whole dataset fits in the budget"
        else:
            halo = (
                "no halo"
                if not any(self.halo)
                else f"{'x'.join(str(pad) for pad in self.halo)} halo"
            )
            summary = f"{self.n_tiles:,} tiles of {tile}, {halo}"
        parts = [summary, f"{format_bytes(self.tile_bytes)}/tile"]
        if self.bound_by:
            parts.append(f"bounded by {self.bound_by}")
        parts.append(self.budget.describe())
        line = "; ".join(parts)
        for note in self.warnings():
            line += f"\n  ! {note}"
        return line

    def warnings(self) -> list[str]:
        """Everything about this plan a user should know before it runs."""
        messages = list(self.notes)
        if not self.budget.is_measured:
            messages.append(
                "the memory budget is a fallback constant - nothing could be detected. "
                "Set one explicitly if this machine has more or less than 4 GiB to spare."
            )
        if self.requires_verification:
            messages.append(
                "at least one step's halo is bounded by object size rather than by a "
                "parameter, so it is an assumption. Objects reaching the halo's edge will "
                "be flagged after segmentation rather than silently truncated."
            )
        if self.approximate_steps:
            messages.append(
                "these steps do not give the same answer tiled as whole: "
                + ", ".join(self.approximate_steps)
            )
        if not self.is_single_tile and not self.is_efficient:
            messages.append(
                f"the halo is large relative to the tile ({self.overlap_ratio:.1f}x the "
                "data will be read). A larger memory budget would help more than a faster disk."
            )
        return messages


def _cost(shape: Sequence[int], bytes_per_voxel: int) -> int:
    return math.prod(shape) * bytes_per_voxel


def plan_tiles(
    shape: Sequence[int],
    *,
    budget: MemoryBudget,
    bytes_per_voxel: int,
    halo: int | Sequence[int] = 0,
    chunks: Sequence[int] | None = None,
    tiled_axes: Sequence[int] | None = None,
    bound_by: str = "",
    requires_verification: bool = False,
    approximate_steps: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> TilePlan:
    """The largest tile of `shape` whose padded cost fits the budget.

    Axes are halved - the longest tiled axis first - until the padded tile
    fits. Halving rather than solving for a cube keeps a slab shaped like a
    slab: a 2048x2048x24 stack tiles in XY and stays whole in Z, which is
    both what the data wants and what the disk wants.

    `chunks` snaps the result *down* to a multiple of the store's chunk
    shape, so a tile boundary is also a chunk boundary and no chunk is read
    twice for two neighbouring tiles.
    """
    shape = tuple(int(extent) for extent in shape)
    if not shape or any(extent <= 0 for extent in shape):
        raise ValueError(f"cannot plan tiles for shape {shape}")
    if bytes_per_voxel <= 0:
        raise ValueError(f"bytes_per_voxel must be positive, got {bytes_per_voxel}")

    ndim = len(shape)
    halo = _per_axis(halo, ndim, "halo")
    tiled = tuple(sorted(range(ndim) if tiled_axes is None else {int(a) for a in tiled_axes}))
    for axis in tiled:
        if not 0 <= axis < ndim:
            raise ValueError(f"tiled axis {axis} is out of range for a {ndim}D shape")
    # An axis nobody may split is taken whole, and its halo is meaningless.
    halo = tuple(pad if axis in tiled else 0 for axis, pad in enumerate(halo))
    chunk_shape = _per_axis(chunks, ndim, "chunks") if chunks is not None else None

    usable = budget.usable_bytes
    splits = [1] * ndim
    tile = list(shape)

    while _cost(_padded(tile, shape, halo), bytes_per_voxel) > usable:
        axis = _axis_to_split(tile, halo, tiled, chunk_shape)
        if axis is None:
            _refuse(shape, tile, halo, bytes_per_voxel, budget)
        splits[axis] += 1
        tile[axis] = _extent(shape[axis], splits[axis], chunk_shape, axis)

    return TilePlan(
        shape=shape,
        tile=tuple(tile),
        halo=halo,
        budget=budget,
        bytes_per_voxel=bytes_per_voxel,
        tiled_axes=tiled,
        chunks=chunk_shape,
        bound_by=bound_by,
        requires_verification=requires_verification,
        approximate_steps=tuple(approximate_steps),
        notes=tuple(notes),
    )


def _padded(tile: Sequence[int], shape: Sequence[int], halo: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        min(extent, size + 2 * pad) for extent, size, pad in zip(shape, tile, halo)
    )


def _axis_to_split(
    tile: Sequence[int],
    halo: Sequence[int],
    tiled: Sequence[int],
    chunk_shape: Sequence[int] | None,
) -> int | None:
    """The longest axis that can still usefully be halved, or None.

    An axis stops being splittable once its core would fall below its own
    halo (the overlap would exceed the work) or below one storage chunk
    (every read would straddle two chunks).
    """
    candidates = []
    for axis in tiled:
        floor = max(halo[axis] * MIN_CORE_TO_HALO, 1)
        if chunk_shape is not None:
            floor = max(floor, chunk_shape[axis])
        if tile[axis] > floor:
            candidates.append(axis)
    if not candidates:
        return None
    return max(candidates, key=lambda axis: (tile[axis], -axis))


def _refuse(shape, tile, halo, bytes_per_voxel, budget) -> None:
    needed = _cost(_padded(tile, shape, halo), bytes_per_voxel)
    if any(halo):
        raise HaloTooLarge(
            f"a halo of {tuple(halo)} leaves no useful tile inside "
            f"{format_bytes(budget.usable_bytes)}: the smallest tile that keeps a core at "
            f"least as large as its halo is {tuple(tile)}, needing "
            f"{format_bytes(needed)}. Either raise the memory budget "
            f"({budget.describe()}), or reduce the parameter the halo comes from "
            f"(a smaller filter radius, dilation distance, or expected object size)."
        )
    raise BudgetTooSmall(
        f"{format_bytes(needed)} is needed for the smallest tile of a {tuple(shape)} "
        f"array at {bytes_per_voxel} bytes/voxel, and the budget is "
        f"{format_bytes(budget.usable_bytes)} ({budget.describe()})."
    )


def _extent(extent: int, splits: int, chunk_shape: Sequence[int] | None, axis: int) -> int:
    """One axis of the tile, given how many pieces it is cut into.

    Rounded up to a whole number of storage chunks, so a tile boundary is
    also a chunk boundary and no chunk has to be read twice for two
    neighbouring tiles. Up rather than down because the search continues
    until the result fits anyway, and rounding down wastes up to a whole
    chunk per axis - a 342-voxel tile snapped down to 256 throws away a
    quarter of the tile and a quarter of the budget with it.
    """
    size = math.ceil(extent / splits)
    if chunk_shape is not None and chunk_shape[axis] > 0:
        chunk = chunk_shape[axis]
        size = min(extent, math.ceil(size / chunk) * chunk)
    return size


def _per_axis(value: int | Sequence[int] | None, ndim: int, name: str) -> tuple[int, ...]:
    if value is None:
        return (0,) * ndim
    if isinstance(value, (int, float)):
        return (int(value),) * ndim
    values = tuple(int(item) for item in value)
    if len(values) == 1:
        return values * ndim
    if len(values) != ndim:
        raise ValueError(f"{name} has {len(values)} entries for a {ndim}D shape")
    return values


@dataclass(frozen=True)
class StepCost:
    """What one step contributes to a pipeline's plan.

    Kept per step rather than only as the maximum, so the answer to "why are
    the tiles so small?" is a table rather than a guess.
    """

    name: str
    scaling: Scaling
    halo: tuple[int, ...]
    bytes_per_voxel: int

    @property
    def is_voxel_scaled(self) -> bool:
        return self.scaling.is_voxel_scaled


def step_costs(
    steps: Iterable[Any],
    *,
    ndim: int = 3,
    spacing: Any = None,
    object_extent: float | None = None,
) -> list[StepCost]:
    """Each step's resolved scaling contract, halo and per-voxel cost.

    `steps` are `vtea_core.workflow.Step`-shaped: anything with `category`,
    `function_name`, `params` and `name`. Duck-typed on purpose, so the
    planner does not drag the workflow engine into a module the workflow
    engine's own wiring table imports from.
    """
    from vtea_core.workflow.wiring import scaling_for

    costs = []
    for step in steps:
        params = dict(getattr(step, "params", {}) or {})
        scaling = scaling_for(
            getattr(step, "category", ""), getattr(step, "function_name", "")
        ).resolve(params)
        halo = scaling.halo.resolve(
            params, spacing=spacing, ndim=ndim, object_extent=object_extent
        )
        costs.append(
            StepCost(
                name=getattr(step, "name", "") or getattr(step, "function_name", "?"),
                scaling=scaling,
                halo=halo,
                bytes_per_voxel=scaling.bytes_per_voxel,
            )
        )
    return costs


def plan_for_steps(
    steps: Iterable[Any],
    shape: Sequence[int],
    *,
    budget: MemoryBudget,
    spacing: Any = None,
    object_extent: float | None = None,
    chunks: Sequence[int] | None = None,
    tiled_axes: Sequence[int] | None = None,
) -> TilePlan:
    """One plan the whole pipeline can run on.

    A pipeline needs a single grid: intermediates are handed from step to
    step, and re-tiling between them would mean writing and re-reading the
    whole volume at every boundary. So the grid is the most demanding step's
    - the largest per-voxel cost, and the per-axis maximum of every halo -
    which is conservative for all the others and is the price of one grid.

    Steps whose cost is per object or per row (`TABLE`) are ignored here.
    They are real work and can be the slow part of a run, but they are not
    what decides how the *voxels* are divided, and pretending otherwise
    would shrink every tile for a clustering step that never touches an
    image.
    """
    shape = tuple(int(extent) for extent in shape)
    ndim = len(shape)
    spatial = ndim if tiled_axes is None else len(tuple(tiled_axes))
    costs = step_costs(steps, ndim=spatial, spacing=spacing, object_extent=object_extent)
    voxel_costs = [cost for cost in costs if cost.is_voxel_scaled]

    if not voxel_costs:
        # Nothing here reads voxels: a pipeline of clustering and gating
        # steps. One tile is the honest plan, and it costs nothing.
        return plan_tiles(
            shape,
            budget=budget,
            bytes_per_voxel=1,
            chunks=chunks,
            tiled_axes=tiled_axes,
            notes=("no step in this pipeline processes voxels",),
        )

    heaviest = max(voxel_costs, key=lambda cost: cost.bytes_per_voxel)
    halo = _combine_halos([cost.halo for cost in voxel_costs], ndim, tiled_axes)
    requires_verification = any(
        cost.scaling.halo.object_extent and cost.scaling.exactness == EXACT_WITH_HALO
        for cost in voxel_costs
    )
    approximate = tuple(
        cost.name for cost in voxel_costs if cost.scaling.exactness == APPROXIMATE
    )

    return plan_tiles(
        shape,
        budget=budget,
        bytes_per_voxel=heaviest.bytes_per_voxel,
        halo=halo,
        chunks=chunks,
        tiled_axes=tiled_axes,
        bound_by=heaviest.name,
        requires_verification=requires_verification and object_extent is None,
        approximate_steps=approximate,
    )


def _combine_halos(
    halos: Sequence[Sequence[int]], ndim: int, tiled_axes: Sequence[int] | None
) -> tuple[int, ...]:
    """The per-axis maximum, mapped onto the array's own axes.

    A step's halo is computed over the *spatial* axes it works on, which is
    not the same list as the array's axes when the array carries a channel
    or time axis. The spatial halo is placed on the tiled axes, in order,
    and everything else gets zero.
    """
    axes = tuple(range(ndim)) if tiled_axes is None else tuple(sorted(tiled_axes))
    combined = [0] * ndim
    for halo in halos:
        for position, axis in enumerate(axes):
            if position < len(halo):
                combined[axis] = max(combined[axis], int(halo[position]))
    return tuple(combined)
