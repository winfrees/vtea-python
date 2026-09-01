"""Associating two segmentations when neither fits in memory.

Three scoring methods with three different shapes, and the useful thing is
that none of them needs the whole image at once:

- **`containment` is additive.** It is a `bincount` over paired label
  arrays: the per-(child, parent) overlaps and the per-child totals are both
  sums, so a tiled run adds up to the whole-image answer exactly. One
  streaming pass, no halo, no approximation - the `ACCUMULATE` pattern that
  `vtea_core.blocked.measure` already uses for measurements.
- **`centroid_distance` needs no voxels at all.** The centroids are in the
  measurement table the run has already produced. Two sets of coordinates
  and a radius is a kd-tree query, and reading images for it was habit
  rather than necessity.
- **`boundary_distance` is object-local.** Each parent needs its own
  bounding box grown by `max_distance`, and nothing else, so a blocked run
  reads one window per parent from the store rather than the volume.

What does *not* fit in memory at ten million objects is the candidate
structure itself, and that is dealt with in
`vtea_core.objects.scoring.CandidateScores` rather than here - these
functions build its arrays directly.

The bounding boxes and the id lists both come from the run when it has
them: a blocked segmentation's `LabelLedger` already knows every object and
the box it occupies, so `boxes_from_ledger` is a lookup rather than a scan.
Without a ledger - labels loaded from a store somebody else wrote - the
same facts are recovered by one pass over the tiles.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from vtea_core.blocked.plan import TilePlan
from vtea_core.data.spacing import Spacing
from vtea_core.objects.assignment import MANY_TO_ONE, assign, posterior
from vtea_core.objects.association import (
    ASSIGNED,
    CONTAINED,
    Association,
    AssociationSet,
    ObjectRef,
)
from vtea_core.objects.identity import associate_ids
from vtea_core.objects.scoring import (
    BOUNDARY_DISTANCE,
    CENTROID_DISTANCE,
    CONTAINMENT,
    SCORING_METHODS,
    CandidateScores,
    boundary_pairs,
    centroid_pairs,
    child_sizes,
    grow_box,
    pair_counts,
)

# How many pending pair rows to let build up before adding them together.
# Coalescing on every tile costs a sort per tile; never coalescing lets the
# pending rows grow with the number of tiles rather than with the number of
# distinct pairs, which is the thing being avoided.
COALESCE_AFTER = 1_000_000


def object_ids_blocked(
    labels: Any,
    *,
    plan: TilePlan,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Every object id in a stored label array, sorted.

    One pass over the tile cores. Prefer `ledger.object_ids` when the labels
    came from a blocked segmentation in the same run - the ledger already
    knows, and knowing without reading is the point of keeping it.
    """
    seen: list[np.ndarray] = []
    for index, tile in enumerate(plan.tiles()):
        block = np.asarray(labels[tile.core])
        values = np.unique(block)
        seen.append(values[values != 0])
        if progress is not None:
            progress(index + 1, plan.n_tiles)
    if not seen:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(seen)).astype(np.int64)


class _PairAccumulator:
    """Per-(child, parent) counts summed across tiles.

    Pairs are held as one integer each - `child_index * n_parents +
    parent_index` - so adding two tiles' worth is a sort and a segmented
    sum rather than a dictionary merge. This is the whole reason containment
    tiles for free; the arithmetic is the same arithmetic either way, and
    the representation is what decides whether it finishes.
    """

    def __init__(self, n_parents: int):
        self.n_parents = int(n_parents)
        self._codes: list[np.ndarray] = []
        self._counts: list[np.ndarray] = []
        self._pending = 0

    def add(self, child_index: np.ndarray, parent_index: np.ndarray, counts: np.ndarray) -> None:
        if not len(counts):
            return
        self._codes.append(child_index.astype(np.int64) * self.n_parents + parent_index)
        self._counts.append(counts.astype(np.float64))
        self._pending += len(counts)
        if self._pending > COALESCE_AFTER:
            self._coalesce()

    def _coalesce(self) -> None:
        if len(self._codes) < 2:
            return
        codes = np.concatenate(self._codes)
        counts = np.concatenate(self._counts)
        unique, inverse = np.unique(codes, return_inverse=True)
        self._codes = [unique]
        self._counts = [np.bincount(inverse, weights=counts, minlength=len(unique))]
        self._pending = len(unique)

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._codes:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
            )
        self._coalesce()
        codes, counts = self._codes[0], self._counts[0]
        if len(self._codes) == 1 and len(codes) and not np.all(np.diff(codes) > 0):
            unique, inverse = np.unique(codes, return_inverse=True)
            codes = unique
            counts = np.bincount(inverse, weights=counts, minlength=len(unique))
        return (
            (codes // self.n_parents).astype(np.int32),
            (codes % self.n_parents).astype(np.int32),
            counts,
        )


def containment_blocked(
    child_labels: Any,
    parent_labels: Any,
    *,
    plan: TilePlan,
    child_ids: Sequence[int] | np.ndarray | None = None,
    parent_ids: Sequence[int] | np.ndarray | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> CandidateScores:
    """The fraction of each child's voxels lying inside each parent, tiled.

    Exact, not approximate, and not exact-with-a-halo either: an overlap
    count is a sum over voxels, every voxel is in exactly one tile core, and
    a sum does not care how it was grouped. A tiled run and a whole-image
    run return the same fractions to the last bit.
    """
    child_ids = _ids_or_scan(child_labels, child_ids, plan=plan)
    parent_ids = _ids_or_scan(parent_labels, parent_ids, plan=plan)

    overlaps = _PairAccumulator(max(len(parent_ids), 1))
    sizes = np.zeros(len(child_ids), dtype=np.float64)
    for index, tile in enumerate(plan.tiles()):
        child_block = np.asarray(child_labels[tile.core])
        foreground = child_block != 0
        if foreground.any():
            child_flat = child_block[foreground].ravel()
            parent_flat = np.asarray(parent_labels[tile.core])[foreground].ravel()
            _positions(child_ids, child_flat, "child")
            _positions(parent_ids, parent_flat[parent_flat != 0], "parent")
            overlaps.add(*pair_counts(child_flat, parent_flat, child_ids, parent_ids))
            sizes += child_sizes(child_flat, child_ids)
        if progress is not None:
            progress(index + 1, plan.n_tiles)

    child_index, parent_index, counts = overlaps.result()
    with np.errstate(divide="ignore", invalid="ignore"):
        affinity = counts / sizes[child_index]

    return CandidateScores(
        child_ids=child_ids,
        parent_ids=parent_ids,
        child_index=child_index,
        parent_index=parent_index,
        affinity=affinity,
        method=CONTAINMENT,
        params={},
    )


def centroid_distance_table(
    child_frame,
    parent_frame,
    *,
    spacing: Spacing | None = None,
    max_distance: float = 10.0,
    id_column: str = "object_id",
    child_prefix: str = "",
    parent_prefix: str = "",
) -> CandidateScores:
    """Centroid-distance affinity from two measurement tables.

    No image is read at all. The centroids are already in the tables the
    measurement step produced, in voxels, so the only thing this adds is the
    physical scaling and the kd-tree - and the answer is identical to
    scoring the same objects from the label arrays, because it is the same
    centroids and the same arithmetic.

    That makes this the cheapest method on a large dataset by a wide margin:
    a 200 GB pair of segmentations costs two table reads.
    """
    reach = float(max_distance)
    if reach <= 0:
        raise ValueError(f"max_distance must be positive, got {max_distance}")

    child_ids, child_points = _points(child_frame, id_column, child_prefix)
    parent_ids, parent_points = _points(parent_frame, id_column, parent_prefix)
    ndim = child_points.shape[1] if len(child_points) else 0
    if ndim and parent_points.shape[1] != ndim:
        raise ValueError(
            f"the two tables have different centroid dimensions: "
            f"{ndim} and {parent_points.shape[1]}"
        )
    sampling = _sampling_for(ndim, spacing)
    child_index, parent_index, affinity = centroid_pairs(
        child_points * sampling, parent_points * sampling, reach
    )

    return CandidateScores(
        child_ids=child_ids,
        parent_ids=parent_ids,
        child_index=child_index,
        parent_index=parent_index,
        affinity=affinity,
        method=CENTROID_DISTANCE,
        params={"max_distance": reach},
    )


def boxes_blocked(
    labels: Any,
    *,
    plan: TilePlan,
    ids: Sequence[int] | np.ndarray,
    progress: Callable[[int, int], None] | None = None,
) -> dict[int, tuple[slice, ...]]:
    """Each object's bounding box, from one pass over the tiles.

    The per-tile boxes of one object are combined by taking the smallest
    start and the largest stop on each axis, which is exactly right for an
    object a tile boundary ran through: the box is the union, and the union
    is what a window has to cover.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if not len(ids):
        return {}
    ndim = plan.ndim
    starts = np.full((len(ids), ndim), np.iinfo(np.int64).max, dtype=np.int64)
    stops = np.full((len(ids), ndim), -1, dtype=np.int64)

    for index, tile in enumerate(plan.tiles()):
        block = np.asarray(labels[tile.core])
        foreground = block != 0
        if foreground.any():
            positions = _positions(ids, block[foreground].ravel(), "label")
            for axis, coordinate in enumerate(np.nonzero(foreground)):
                # Global coordinates, so the box of an object a tile
                # boundary ran through is the union of what each tile saw
                # rather than two boxes in two coordinate systems.
                global_coordinate = coordinate.astype(np.int64) + tile.core[axis].start
                np.minimum.at(starts[:, axis], positions, global_coordinate)
                np.maximum.at(stops[:, axis], positions, global_coordinate + 1)
        if progress is not None:
            progress(index + 1, plan.n_tiles)

    return {
        int(object_id): tuple(
            slice(int(starts[slot, axis]), int(stops[slot, axis])) for axis in range(ndim)
        )
        for slot, object_id in enumerate(ids)
        if stops[slot, 0] >= 0
    }


def boxes_from_ledger(ledger) -> dict[int, tuple[slice, ...]]:
    """The same boxes, straight off a blocked segmentation's ledger.

    A `Fragment` already carries the box its tile saw, in global
    coordinates, so an object's box is the union of its fragments' - which
    the ledger knows without anything being read at all.
    """
    boxes = {}
    for object_id, fragments in ledger.fragments.items():
        if not fragments:
            continue
        bounds = np.array([fragment.bbox for fragment in fragments], dtype=np.int64)
        boxes[int(object_id)] = tuple(
            slice(int(bounds[:, axis, 0].min()), int(bounds[:, axis, 1].max()))
            for axis in range(bounds.shape[1])
        )
    return boxes


def boundary_distance_blocked(
    child_labels: Any,
    parent_labels: Any,
    *,
    plan: TilePlan,
    spacing: Spacing | None = None,
    max_distance: float = 10.0,
    child_ids: Sequence[int] | np.ndarray | None = None,
    parent_ids: Sequence[int] | np.ndarray | None = None,
    boxes: dict[int, tuple[slice, ...]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> CandidateScores:
    """The gap between each child's surface and each parent's, tiled.

    One window per parent - its own bounding box grown by `max_distance` -
    read from the store rather than held. Exact given that window, which is
    the same statement the whole-image version makes: everything within
    reach of the parent is inside it by construction, so nothing outside it
    could have scored.

    The cost is proportional to the objects rather than to the volume, and
    that holds however large the volume gets. What it is *not* proportional
    to is the tiling, which is why this reads boxes rather than tiles.
    """
    reach = float(max_distance)
    if reach <= 0:
        raise ValueError(f"max_distance must be positive, got {max_distance}")

    child_ids = _ids_or_scan(child_labels, child_ids, plan=plan)
    parent_ids = _ids_or_scan(parent_labels, parent_ids, plan=plan)
    sampling = _sampling_for(plan.ndim, spacing)
    if boxes is None:
        boxes = boxes_blocked(parent_labels, plan=plan, ids=parent_ids, progress=progress)

    children, parents, affinities = [], [], []
    for position, parent_id in enumerate(parent_ids):
        box = boxes.get(int(parent_id))
        if box is None:
            continue
        window = grow_box(box, plan.shape, reach, sampling)
        pairs = boundary_pairs(
            np.asarray(child_labels[window]),
            np.asarray(parent_labels[window]),
            int(parent_id),
            reach=reach,
            sampling=sampling,
        )
        if pairs is None:
            continue
        local_child_ids, local_affinity = pairs
        children.append(np.searchsorted(child_ids, local_child_ids))
        parents.append(np.full(len(local_child_ids), position, dtype=np.int32))
        affinities.append(local_affinity)
        if progress is not None:
            progress(position + 1, len(parent_ids))

    return CandidateScores(
        child_ids=child_ids,
        parent_ids=parent_ids,
        child_index=(
            np.concatenate(children).astype(np.int32) if children else np.empty(0, np.int32)
        ),
        parent_index=(
            np.concatenate(parents).astype(np.int32) if parents else np.empty(0, np.int32)
        ),
        affinity=np.concatenate(affinities) if affinities else np.empty(0, float),
        method=BOUNDARY_DISTANCE,
        params={"max_distance": reach},
    )


def associate_by_identity_blocked(
    child_labels: Any,
    parent_labels: Any,
    *,
    plan: TilePlan,
    child_name: str = "child",
    parent_name: str = "parent",
    require_parent: bool = True,
    child_ids: Sequence[int] | np.ndarray | None = None,
    parent_ids: Sequence[int] | np.ndarray | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> AssociationSet:
    """Identity association without materializing either label array.

    The whole computation is a comparison of two id sets, and a blocked run
    knows both - by scan here, or from a ledger when one exists. The links
    it returns are the same links, certain for the same reason.
    """
    return associate_ids(
        _ids_or_scan(child_labels, child_ids, plan=plan),
        _ids_or_scan(parent_labels, parent_ids, plan=plan),
        child_name=child_name,
        parent_name=parent_name,
        require_parent=require_parent,
    )


def score_candidates_blocked(
    child_labels: Any,
    parent_labels: Any,
    *,
    plan: TilePlan,
    method: str = CONTAINMENT,
    spacing: Spacing | None = None,
    max_distance: float = 10.0,
    child_table=None,
    parent_table=None,
    progress: Callable[[int, int], None] | None = None,
    **kwargs,
) -> CandidateScores:
    """Dispatch by method name, the blocked counterpart of
    `vtea_core.objects.scoring.score_candidates`.

    `centroid_distance` takes the measurement tables when they are offered
    and falls back to the label arrays when they are not - the answer is the
    same either way, and the tables are free.
    """
    if method == CONTAINMENT:
        return containment_blocked(
            child_labels, parent_labels, plan=plan, progress=progress, **kwargs
        )
    if method == CENTROID_DISTANCE:
        if child_table is not None and parent_table is not None:
            return centroid_distance_table(
                child_table, parent_table, spacing=spacing, max_distance=max_distance
            )
        return _centroid_distance_from_labels(
            child_labels,
            parent_labels,
            plan=plan,
            spacing=spacing,
            max_distance=max_distance,
            progress=progress,
            **kwargs,
        )
    if method == BOUNDARY_DISTANCE:
        return boundary_distance_blocked(
            child_labels,
            parent_labels,
            plan=plan,
            spacing=spacing,
            max_distance=max_distance,
            progress=progress,
            **kwargs,
        )
    raise ValueError(f"unknown scoring method {method!r}, expected one of {list(SCORING_METHODS)}")


def associate_objects_blocked(
    child_labels: Any,
    parent_labels: Any,
    *,
    plan: TilePlan,
    spacing: Spacing | None = None,
    child_name: str = "child",
    parent_name: str = "parent",
    method: str = CONTAINMENT,
    mode: str = MANY_TO_ONE,
    max_distance: float = 10.0,
    orphan_score: float = 0.05,
    min_probability: float = 0.0,
    child_table=None,
    parent_table=None,
    progress: Callable[[int, int], None] | None = None,
    **kwargs,
) -> AssociationSet:
    """`associate_objects`, out of core.

    Scoring is where the volume matters and it is done a tile or a window at
    a time above; everything after it - the posterior, the assignment, the
    record - is per object and per candidate rather than per voxel, and is
    the same code the in-memory path runs.
    """
    candidates = score_candidates_blocked(
        child_labels,
        parent_labels,
        plan=plan,
        method=method,
        spacing=spacing,
        max_distance=max_distance,
        child_table=child_table,
        parent_table=parent_table,
        progress=progress,
        **kwargs,
    )
    matches = assign(
        posterior(candidates, orphan_score=orphan_score),
        mode=mode,
        min_probability=min_probability,
    )

    params = {
        "method": method,
        "mode": mode,
        "orphan_score": float(orphan_score),
        "min_probability": float(min_probability),
        "blocked": True,
    }
    if method != CONTAINMENT:
        params["max_distance"] = float(max_distance)
    relationship = CONTAINED if method == CONTAINMENT else ASSIGNED

    associations = AssociationSet()
    for match in matches:
        child = ObjectRef(child_name, match.child_id)
        if match.parent_id is None:
            associations.add_unassigned(child)
            continue
        associations.add(
            Association(
                child=child,
                parent=ObjectRef(parent_name, match.parent_id),
                relationship=relationship,
                probability=match.probability,
                method=method,
                params=params,
                alternatives=[
                    (ObjectRef(parent_name, parent_id), probability)
                    for parent_id, probability in match.alternatives
                ],
            )
        )
    return associations


# -- shared -----------------------------------------------------------------


def _ids_or_scan(labels, ids, *, plan: TilePlan) -> np.ndarray:
    if ids is None:
        return object_ids_blocked(labels, plan=plan)
    return np.sort(np.asarray(ids, dtype=np.int64))


def _positions(ids: np.ndarray, values: np.ndarray, what: str) -> np.ndarray:
    """Where each label sits in `ids`, refusing one that is not there.

    `searchsorted` on a label the id list does not contain returns the slot
    it *would* occupy, which is some other object's - so an id list that
    does not cover the labels produces a plausible wrong answer rather than
    an error. Checked here rather than trusted, because the two come from
    different places (a ledger, a table, a scan) and only have to agree.
    """
    if not len(ids):
        if len(values):
            raise ValueError(f"there are no ids to score {what} labels against")
        return np.empty(0, dtype=np.int64)
    positions = np.searchsorted(ids, values)
    np.clip(positions, 0, len(ids) - 1, out=positions)
    wrong = ids[positions] != values
    if wrong.any():
        missing = np.unique(values[wrong])[:5]
        raise ValueError(
            f"{what} {missing.tolist()} is not in the id list this is being scored against - "
            f"the labels and the ids come from different runs"
        )
    return positions


def _sampling_for(ndim: int, spacing: Spacing | None) -> np.ndarray:
    """Physical voxel size, or ones when it is not known - stated rather
    than omitted, because a distance in voxels and a distance in microns
    differ by a factor nobody notices until the answer is wrong."""
    if spacing is None or not spacing.is_known:
        return np.ones(ndim, dtype=float)
    return np.asarray(spacing.for_ndim(ndim), dtype=float)


def _points(frame, id_column: str, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """A measurement table's ids and centroids, in voxels."""
    if id_column not in frame.columns:
        raise KeyError(f"the table has no {id_column!r} column; found {list(frame.columns)}")
    columns = sorted(
        column for column in frame.columns if column.startswith(f"{prefix}centroid-")
    )
    if not columns:
        raise KeyError(
            f"the table has no {prefix}centroid-* columns, so there is nothing to measure "
            f"a distance between; found {list(frame.columns)}"
        )
    ids = np.asarray(frame[id_column], dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    points = np.column_stack([np.asarray(frame[column], dtype=float) for column in columns])
    return ids[order], points[order]


def _centroid_distance_from_labels(
    child_labels,
    parent_labels,
    *,
    plan: TilePlan,
    spacing: Spacing | None,
    max_distance: float,
    progress=None,
    child_ids=None,
    parent_ids=None,
) -> CandidateScores:
    """Centroids accumulated from the tiles, for the case where no
    measurement table exists yet. Sums of coordinates and of counts, so it
    is the same centroid the whole-image version computes."""
    child_ids = _ids_or_scan(child_labels, child_ids, plan=plan)
    parent_ids = _ids_or_scan(parent_labels, parent_ids, plan=plan)
    sampling = _sampling_for(plan.ndim, spacing)
    child_points = centroids_blocked(child_labels, plan=plan, ids=child_ids, progress=progress)
    parent_points = centroids_blocked(parent_labels, plan=plan, ids=parent_ids)
    child_index, parent_index, affinity = centroid_pairs(
        child_points * sampling, parent_points * sampling, float(max_distance)
    )
    return CandidateScores(
        child_ids=child_ids,
        parent_ids=parent_ids,
        child_index=child_index,
        parent_index=parent_index,
        affinity=affinity,
        method=CENTROID_DISTANCE,
        params={"max_distance": float(max_distance)},
    )


def centroids_blocked(
    labels: Any,
    *,
    plan: TilePlan,
    ids: Sequence[int] | np.ndarray,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Each object's centre of mass, in voxels, accumulated over the tiles.

    A centroid is a sum of coordinates over a count, and both add across
    tiles, so an object a boundary ran through gets the same centre it would
    have got whole. Coordinates are global - each tile adds its own origin -
    which is what makes that true.
    """
    ids = np.asarray(ids, dtype=np.int64)
    totals = np.zeros((len(ids), plan.ndim), dtype=np.float64)
    counts = np.zeros(len(ids), dtype=np.float64)
    for index, tile in enumerate(plan.tiles()):
        block = np.asarray(labels[tile.core])
        foreground = block != 0
        if foreground.any():
            positions = _positions(ids, block[foreground].ravel(), "label")
            np.add.at(counts, positions, 1.0)
            coordinates = np.nonzero(foreground)
            for axis, coordinate in enumerate(coordinates):
                np.add.at(
                    totals[:, axis],
                    positions,
                    coordinate.astype(np.float64) + tile.core[axis].start,
                )
        if progress is not None:
            progress(index + 1, plan.n_tiles)
    with np.errstate(invalid="ignore", divide="ignore"):
        return totals / counts[:, None]
