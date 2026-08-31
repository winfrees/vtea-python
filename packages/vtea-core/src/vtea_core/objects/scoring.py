"""How well each parent explains a child object.

Two segmentations made independently in different channels share nothing: a
cytoplasm labelled 12 in a cytoskeleton channel and a nucleus labelled 7 in
DAPI have unrelated ids, and the only evidence that one belongs to the other
is geometric. These functions turn that evidence into a number.

Each returns `CandidateScores` - a *sparse* child x parent affinity, higher
being better. Sparsity is not an optimisation detail here: restricting a
child's candidates to the parents actually near it is what keeps the global
one-to-one assignment in `vtea_core.objects.assignment` tractable, because
it splits the cost matrix into independent blocks instead of one
n_children x n_parents problem. A field of 50,000 objects is unsolvable as
one matrix and trivial as 50,000 small ones.

Affinities are on a common [0, 1] scale so they can be compared against the
"no parent at all" option, and so the posterior that follows means the same
thing whichever method produced it.

Distances are physical wherever a `Spacing` is known and in voxels
otherwise - see `vtea_core.data.Spacing`. A `max_distance` of 10 means ten
microns in a stack whose z-step is four times its pixel size, not ten
z-slices.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from vtea_core.data.spacing import Spacing

# The scoring methods, by name, as a protocol records them.
CONTAINMENT = "containment"
CENTROID_DISTANCE = "centroid_distance"
BOUNDARY_DISTANCE = "boundary_distance"

SCORING_METHODS = (CONTAINMENT, CENTROID_DISTANCE, BOUNDARY_DISTANCE)


class CandidateScores:
    """Which parents each child might belong to, and how well each fits.

    `scores[child_id][parent_id]` is an affinity in (0, 1]; a pair that is
    absent is not a candidate at all, which is a stronger statement than a
    score of zero and is what makes the structure sparse.

    `child_ids` lists *every* child object, including those with no
    candidate: a cytoplasm with no nucleus anywhere near it is a result, not
    an absence of one, and the assignment step needs to be able to report it.

    **Held as three arrays rather than a dict of dicts**, which is what makes
    this survive ten million objects. A `dict[int, dict[int, float]]` costs
    roughly 200 bytes per candidate pair once the boxed ints and the inner
    dicts are counted; the same pair here is a 4-byte child index, a 4-byte
    parent index and an 8-byte affinity. At 10^7 children with a few
    candidates each that is the difference between a structure that fits in
    memory and one that is the reason the run does not - and the images this
    is scored from are already being read a tile at a time by then.

    The rows are kept sorted by child, so one child's candidates are a
    contiguous slice found by binary search rather than a hash lookup, and
    `for_child` costs what it always did. The nested dict is still available
    as `.scores` for the code and the tests that read it that way; it is
    built on demand and costs exactly what it used to.
    """

    __slots__ = (
        "_row_starts",
        "affinity",
        "child_ids",
        "child_index",
        "method",
        "params",
        "parent_ids",
        "parent_index",
    )

    def __init__(
        self,
        scores: Mapping[int, Mapping[int, float]] | None = None,
        child_ids: Any = (),
        parent_ids: Any = (),
        method: str = "",
        params: dict[str, Any] | None = None,
        *,
        child_index: Any = None,
        parent_index: Any = None,
        affinity: Any = None,
    ):
        """Either from a `{child: {parent: affinity}}` mapping, or from the
        three arrays directly - the second is how the scoring functions build
        one, and the first is how a caller with a handful of pairs does."""
        self.child_ids, child_remap = _ids_and_remap(child_ids)
        self.parent_ids, parent_remap = _ids_and_remap(parent_ids)
        self.method = method
        self.params = dict(params or {})

        if scores:
            # Built against the sorted ids, so no remapping applies.
            child_index, parent_index, affinity = _coo_from_mapping(
                scores, self.child_ids, self.parent_ids
            )
        elif child_index is None:
            child_index = parent_index = np.empty(0, dtype=np.int32)
            affinity = np.empty(0, dtype=float)
        else:
            child_index, parent_index = _remapped(
                child_index, parent_index, child_remap, parent_remap
            )

        self.child_index = np.asarray(child_index, dtype=np.int32)
        self.parent_index = np.asarray(parent_index, dtype=np.int32)
        self.affinity = np.asarray(affinity, dtype=float)
        if not (len(self.child_index) == len(self.parent_index) == len(self.affinity)):
            raise ValueError(
                f"the three candidate arrays must be the same length, got "
                f"{len(self.child_index)}, {len(self.parent_index)}, {len(self.affinity)}"
            )
        self.child_index, self.parent_index, self.affinity = _sorted_by_child(
            self.child_index, self.parent_index, self.affinity
        )
        self._row_starts = _row_starts(self.child_index, len(self.child_ids))

    # -- reading ----------------------------------------------------------

    def row(self, child_id: int) -> slice:
        """Where this child's candidates live in the three arrays."""
        return _row_slice(self.child_ids, self._row_starts, child_id)

    def candidates_for(self, child_id: int) -> tuple[np.ndarray, np.ndarray]:
        """One child's candidate parents and affinities, as arrays.

        The allocation-free way to read a row - what the assignment step
        walks. `for_child` is the same thing as a dict.
        """
        rows = self.row(child_id)
        return self.parent_ids[self.parent_index[rows]], self.affinity[rows]

    def for_child(self, child_id: int) -> dict[int, float]:
        parents, values = self.candidates_for(child_id)
        return {int(parent): float(value) for parent, value in zip(parents, values)}

    @property
    def scores(self) -> dict[int, dict[int, float]]:
        """The nested dict this used to hold, rebuilt on demand.

        Convenient, and it costs what it always cost - so it is for reading
        a small result, not for walking a large one.
        """
        built: dict[int, dict[int, float]] = {}
        for position, parent, value in zip(self.child_index, self.parent_index, self.affinity):
            child = int(self.child_ids[position])
            built.setdefault(child, {})[int(self.parent_ids[parent])] = float(value)
        return built

    @property
    def n_candidates(self) -> int:
        return len(self.affinity)

    @property
    def nbytes(self) -> int:
        """What the candidates actually weigh - the number the dict form
        could not answer without walking itself."""
        return int(
            self.child_index.nbytes
            + self.parent_index.nbytes
            + self.affinity.nbytes
            + self.child_ids.nbytes
            + self.parent_ids.nbytes
        )

    def __len__(self) -> int:
        return len(self.child_ids)

    def __repr__(self) -> str:
        return (
            f"CandidateScores(method={self.method!r}, {len(self.child_ids)} children, "
            f"{len(self.parent_ids)} parents, {self.n_candidates} candidates)"
        )


def _id_array(ids: Any) -> np.ndarray:
    """Object ids as a sorted int64 array."""
    return _ids_and_remap(ids)[0]


def _ids_and_remap(ids: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Object ids sorted, and where each one moved to.

    Sorted because every lookup here is a binary search into it. The second
    return value is the trap: an index array the caller built against the
    order *they* passed points at the wrong object once the ids are sorted,
    so the caller's indices have to be remapped through it. `None` when the
    ids were already in order, which is every call from the scoring
    functions - `np.unique` sorts - so the common path costs nothing.
    """
    array = np.asarray(list(ids) if isinstance(ids, (tuple, list)) else ids, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"object ids must be one-dimensional, got shape {array.shape}")
    if len(array) < 2 or bool(np.all(np.diff(array) > 0)):
        return array, None
    order = np.argsort(array, kind="stable")
    remap = np.empty(len(array), dtype=np.int64)
    remap[order] = np.arange(len(array), dtype=np.int64)
    return array[order], remap


def _coo_from_mapping(
    scores: Mapping[int, Mapping[int, float]],
    child_ids: np.ndarray,
    parent_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    children, parents, values = [], [], []
    for child_id, row in scores.items():
        position = _position(child_ids, child_id, "child")
        for parent_id, value in row.items():
            children.append(position)
            parents.append(_position(parent_ids, parent_id, "parent"))
            values.append(value)
    return (
        np.asarray(children, dtype=np.int32),
        np.asarray(parents, dtype=np.int32),
        np.asarray(values, dtype=float),
    )


def _position(ids: np.ndarray, object_id: int, what: str) -> int:
    position = int(np.searchsorted(ids, object_id))
    if position >= len(ids) or int(ids[position]) != int(object_id):
        raise KeyError(f"{what} {object_id} is not in this scoring's {what} ids")
    return position


def _remapped(
    child_index: Any, parent_index: Any, child_remap, parent_remap
) -> tuple[np.ndarray, np.ndarray]:
    """Caller-supplied indices moved onto the sorted ids."""
    child_index = np.asarray(child_index)
    parent_index = np.asarray(parent_index)
    if child_remap is not None:
        child_index = child_remap[child_index]
    if parent_remap is not None:
        parent_index = parent_remap[parent_index]
    return child_index, parent_index


def _sorted_by_child(
    child_index: np.ndarray, parent_index: np.ndarray, affinity: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(child_index) and not np.all(np.diff(child_index) >= 0):
        order = np.argsort(child_index, kind="stable")
        return child_index[order], parent_index[order], affinity[order]
    return child_index, parent_index, affinity


def _row_starts(child_index: np.ndarray, n_children: int) -> np.ndarray:
    """Where each child's run begins, CSR-style, with a closing bound."""
    return np.searchsorted(child_index, np.arange(n_children + 1, dtype=np.int64))


def _row_slice(ids: np.ndarray, row_starts: np.ndarray, object_id: int) -> slice:
    position = int(np.searchsorted(ids, object_id))
    if position >= len(ids) or int(ids[position]) != int(object_id):
        return slice(0, 0)
    return slice(int(row_starts[position]), int(row_starts[position + 1]))


def _as_labels(labels: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(labels)
    if not np.issubdtype(array.dtype, np.integer) and array.dtype != bool:
        raise TypeError(f"{name} must be a label image (integer or boolean), got {array.dtype}")
    return array.astype(np.int32, copy=False) if array.dtype == bool else array


def _pair(child_labels, parent_labels) -> tuple[np.ndarray, np.ndarray]:
    child = _as_labels(child_labels, "child_labels")
    parent = _as_labels(parent_labels, "parent_labels")
    if child.shape != parent.shape:
        raise ValueError(f"shapes differ: {child.shape} != {parent.shape}")
    return child, parent


def _ids(labels: np.ndarray) -> np.ndarray:
    """Every object id in a label image, sorted, background excluded."""
    values = np.unique(labels)
    return values[values != 0].astype(np.int64, copy=False)


def _sampling(labels: np.ndarray, spacing: Spacing | None) -> tuple[float, ...]:
    """Physical voxel size, or ones when it isn't known. Unlike the derived
    segmentations, distance scoring always needs *some* sampling, so an
    unknown spacing means voxels explicitly rather than by omission."""
    if spacing is None or not spacing.is_known:
        return (1.0,) * labels.ndim
    return spacing.for_ndim(labels.ndim)


def _check_max_distance(max_distance: float) -> float:
    if max_distance <= 0:
        raise ValueError(f"max_distance must be positive, got {max_distance}")
    return float(max_distance)


def containment(child_labels: np.ndarray, parent_labels: np.ndarray) -> CandidateScores:
    """The fraction of each child's voxels lying inside each parent.

    The right evidence when the child is genuinely *within* the parent - a
    nucleus inside a whole-cell mask, a lysosome inside a cytoplasm - and it
    needs no distance parameter at all, because overlap either exists or it
    does not. A child straddling two parents scores against both, in
    proportion, which is exactly the ambiguity the posterior should carry.

    Counted as a `bincount` over paired label arrays, which is why this one
    tiles for free: the per-pair overlaps and the per-child totals are both
    sums, so a tiled run adds up to the whole-image answer exactly. See
    `vtea_core.blocked.associate.containment_blocked`.
    """
    child, parent = _pair(child_labels, parent_labels)
    child_ids, parent_ids = _ids(child), _ids(parent)

    foreground = child != 0
    child_flat = child[foreground].ravel()
    parent_flat = parent[foreground].ravel()

    overlaps = pair_counts(child_flat, parent_flat, child_ids, parent_ids)
    sizes = child_sizes(child_flat, child_ids)
    child_index, parent_index, counts = overlaps
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


def pair_counts(
    child_flat: np.ndarray,
    parent_flat: np.ndarray,
    child_ids: np.ndarray,
    parent_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxels shared by each (child, parent) pair, as three arrays.

    Positions into `child_ids`/`parent_ids` rather than the ids themselves,
    so that counts from separate tiles are added by adding arrays. Pairs
    with background on either side are not pairs and are dropped.
    """
    if not child_flat.size or not len(parent_ids):
        return _empty_pairs()
    keep = parent_flat != 0
    if not keep.any():
        return _empty_pairs()
    child_index = np.searchsorted(child_ids, child_flat[keep])
    parent_index = np.searchsorted(parent_ids, parent_flat[keep])
    # One integer per pair, so the grouping is a sort rather than a hash of
    # tuples - the difference between seconds and minutes at 10^8 voxels.
    codes = child_index.astype(np.int64) * len(parent_ids) + parent_index
    unique, counts = np.unique(codes, return_counts=True)
    return (
        (unique // len(parent_ids)).astype(np.int32),
        (unique % len(parent_ids)).astype(np.int32),
        counts.astype(np.float64),
    )


def child_sizes(child_flat: np.ndarray, child_ids: np.ndarray) -> np.ndarray:
    """Voxels per child, aligned with `child_ids` - the denominator of the
    containment fraction, and additive across tiles for the same reason."""
    sizes = np.zeros(len(child_ids), dtype=np.float64)
    if child_flat.size and len(child_ids):
        positions = np.searchsorted(child_ids, child_flat)
        np.add.at(sizes, positions, 1.0)
    return sizes


def _empty_pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )


def _centroids(labels: np.ndarray, sampling: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    ids = _ids(labels)
    if not len(ids):
        return ids, np.empty((0, labels.ndim))
    centroids = ndi.center_of_mass(labels != 0, labels, list(ids))
    return ids, np.asarray(centroids, dtype=float) * np.asarray(sampling, dtype=float)


def centroid_distance(
    child_labels: np.ndarray,
    parent_labels: np.ndarray,
    *,
    spacing: Spacing | None = None,
    max_distance: float = 10.0,
) -> CandidateScores:
    """Affinity falling off linearly with the distance between centres.

    The cheap, robust choice when child and parent do not overlap - puncta
    scattered around a nucleus, a cytoplasm whose mask stops short of the
    nucleus it belongs to. `max_distance` does double duty: it is the reach
    beyond which a parent is not a candidate at all, and the scale over
    which the affinity decays, so there is one number to set rather than two
    that interact.

    Centroid distance is blind to shape - a large parent is no nearer than a
    small one at the same centre - which is why `boundary_distance` exists
    for the case where object size varies a lot.
    """
    child, parent = _pair(child_labels, parent_labels)
    reach = _check_max_distance(max_distance)
    sampling = _sampling(child, spacing)

    child_ids, child_points = _centroids(child, sampling)
    parent_ids, parent_points = _centroids(parent, sampling)
    child_index, parent_index, affinity = centroid_pairs(child_points, parent_points, reach)

    return CandidateScores(
        child_ids=child_ids,
        parent_ids=parent_ids,
        child_index=child_index,
        parent_index=parent_index,
        affinity=affinity,
        method=CENTROID_DISTANCE,
        params={"max_distance": reach},
    )


def centroid_pairs(
    child_points: np.ndarray, parent_points: np.ndarray, reach: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every child-parent pair within `reach`, and its linear affinity.

    Split out because it needs no image at all - two sets of coordinates and
    a radius - which is what lets a blocked run score centroid distance from
    the measurement table it already has rather than by reading voxels back.
    """
    if not len(child_points) or not len(parent_points):
        return _empty_pairs()
    tree = cKDTree(parent_points)
    neighbours = tree.query_ball_point(child_points, reach)
    counts = np.fromiter((len(row) for row in neighbours), dtype=np.int64, count=len(neighbours))
    if not counts.sum():
        return _empty_pairs()
    child_index = np.repeat(np.arange(len(child_points), dtype=np.int64), counts)
    parent_index = np.concatenate([np.asarray(row, dtype=np.int64) for row in neighbours if row])
    distance = np.linalg.norm(child_points[child_index] - parent_points[parent_index], axis=1)
    affinity = 1.0 - distance / reach
    keep = affinity > 0
    return (
        child_index[keep].astype(np.int32),
        parent_index[keep].astype(np.int32),
        affinity[keep],
    )


def boundary_distance(
    child_labels: np.ndarray,
    parent_labels: np.ndarray,
    *,
    spacing: Spacing | None = None,
    max_distance: float = 10.0,
) -> CandidateScores:
    """Affinity falling off with the gap between the two objects' surfaces.

    The gap is measured between nearest voxels: zero where the two overlap,
    one voxel step where they merely touch. This is the measure that matches
    how a person decides these by eye ("that punctum is right up against that
    nucleus"), and unlike centroid distance it is not fooled by a parent much
    larger than its children.

    Computed per parent inside its own bounding box, expanded by
    `max_distance`, so the cost is proportional to the objects rather than to
    the volume times the number of parents.
    """
    child, parent = _pair(child_labels, parent_labels)
    reach = _check_max_distance(max_distance)
    sampling = _sampling(child, spacing)
    child_ids, parent_ids = _ids(child), _ids(parent)

    children, parents, affinities = [], [], []
    for position, box in enumerate(object_boxes(parent, parent_ids)):
        if box is None:
            continue
        parent_id = int(parent_ids[position])
        window = grow_box(box, parent.shape, reach, sampling)
        pairs = boundary_pairs(
            child[window], parent[window], parent_id, reach=reach, sampling=sampling
        )
        if pairs is None:
            continue
        local_child_ids, local_affinity = pairs
        children.append(np.searchsorted(child_ids, local_child_ids))
        parents.append(np.full(len(local_child_ids), position, dtype=np.int32))
        affinities.append(local_affinity)

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


def object_boxes(labels: np.ndarray, ids: np.ndarray) -> list:
    """One bounding box per id in `ids`, `None` where the object is absent.

    `find_objects` indexes by label value, which is only the same as
    indexing by position when the ids happen to run 1..n. After a blocked
    segmentation they do not, so the mapping is made explicit here rather
    than assumed at each call site.
    """
    found = ndi.find_objects(labels)
    return [found[object_id - 1] if object_id <= len(found) else None for object_id in ids]


def grow_box(box, shape, reach: float, sampling) -> tuple[slice, ...]:
    """A bounding box grown by `reach` physical units, clipped to the array.

    Anisotropically: eight microns is four voxels along a 2 um z-step and
    sixteen in x at 0.5, and a scalar margin would be right on one axis and
    wrong on the others.
    """
    margin = [int(np.ceil(reach / size)) for size in sampling]
    return tuple(
        slice(max(0, axis.start - pad), min(extent, axis.stop + pad))
        for axis, pad, extent in zip(box, margin, shape)
    )


def boundary_pairs(
    local_child: np.ndarray,
    local_parent: np.ndarray,
    parent_id: int,
    *,
    reach: float,
    sampling,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Which children lie within `reach` of one parent's surface, and how near.

    The whole computation for one parent, inside a window that already holds
    everything within reach of it - which is what makes this object-local,
    and so the same function whether the window came from an array in memory
    or from a store read block by block.
    """
    if not local_child.any():
        return None
    distance = ndi.distance_transform_edt(local_parent != parent_id, sampling=sampling)
    within = (local_child != 0) & (distance <= reach)
    if not within.any():
        return None

    ids = local_child[within]
    gaps = distance[within]
    # The nearest voxel of each child to this parent: sort by (child, gap)
    # and keep the first row of each child's run.
    order = np.lexsort((gaps, ids))
    ids, gaps = ids[order], gaps[order]
    first = np.concatenate(([True], ids[1:] != ids[:-1]))
    ids, gaps = ids[first], gaps[first]
    affinity = 1.0 - gaps / reach
    keep = affinity > 0
    if not keep.any():
        return None
    return ids[keep].astype(np.int64), affinity[keep]


def score_candidates(
    child_labels: np.ndarray,
    parent_labels: np.ndarray,
    *,
    method: str = CONTAINMENT,
    spacing: Spacing | None = None,
    max_distance: float = 10.0,
) -> CandidateScores:
    """Dispatch by method name, so a protocol can record the choice as a
    string and a caller doesn't have to import three functions to offer
    three options."""
    if method == CONTAINMENT:
        return containment(child_labels, parent_labels)
    if method == CENTROID_DISTANCE:
        return centroid_distance(
            child_labels, parent_labels, spacing=spacing, max_distance=max_distance
        )
    if method == BOUNDARY_DISTANCE:
        return boundary_distance(
            child_labels, parent_labels, spacing=spacing, max_distance=max_distance
        )
    raise ValueError(f"unknown scoring method {method!r}, expected one of {list(SCORING_METHODS)}")
