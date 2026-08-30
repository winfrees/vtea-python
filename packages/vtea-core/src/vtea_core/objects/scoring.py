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

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class CandidateScores:
    """Which parents each child might belong to, and how well each fits.

    `scores[child_id][parent_id]` is an affinity in (0, 1]; a pair that is
    absent is not a candidate at all, which is a stronger statement than a
    score of zero and is what makes the structure sparse.

    `child_ids` lists *every* child object, including those with no
    candidate: a cytoplasm with no nucleus anywhere near it is a result, not
    an absence of one, and the assignment step needs to be able to report it.
    """

    scores: dict[int, dict[int, float]]
    child_ids: tuple[int, ...]
    parent_ids: tuple[int, ...]
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def for_child(self, child_id: int) -> dict[int, float]:
        return self.scores.get(int(child_id), {})

    @property
    def n_candidates(self) -> int:
        return sum(len(candidates) for candidates in self.scores.values())

    def __len__(self) -> int:
        return len(self.child_ids)


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


def _ids(labels: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.unique(labels) if value != 0)


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
    """
    child, parent = _pair(child_labels, parent_labels)

    foreground = child != 0
    child_flat = child[foreground].ravel()
    parent_flat = parent[foreground].ravel()
    sizes = np.bincount(child_flat) if child_flat.size else np.zeros(1, dtype=np.int64)

    scores: dict[int, dict[int, float]] = {}
    if child_flat.size:
        pairs, counts = np.unique(np.stack([child_flat, parent_flat]), axis=1, return_counts=True)
        for (child_id, parent_id), overlap in zip(pairs.T, counts):
            if parent_id == 0:
                continue
            fraction = float(overlap) / float(sizes[child_id])
            scores.setdefault(int(child_id), {})[int(parent_id)] = fraction

    return CandidateScores(
        scores=scores,
        child_ids=_ids(child),
        parent_ids=_ids(parent),
        method=CONTAINMENT,
        params={},
    )


def _centroids(labels: np.ndarray, sampling: tuple[float, ...]) -> tuple[tuple[int, ...], np.ndarray]:
    ids = _ids(labels)
    if not ids:
        return (), np.empty((0, labels.ndim))
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

    scores: dict[int, dict[int, float]] = {}
    if len(child_ids) and len(parent_ids):
        tree = cKDTree(parent_points)
        for index, point in enumerate(child_points):
            near = tree.query_ball_point(point, reach)
            candidates = {}
            for other in near:
                distance = float(np.linalg.norm(point - parent_points[other]))
                affinity = 1.0 - distance / reach
                if affinity > 0:
                    candidates[int(parent_ids[other])] = affinity
            if candidates:
                scores[int(child_ids[index])] = candidates

    return CandidateScores(
        scores=scores,
        child_ids=child_ids,
        parent_ids=parent_ids,
        method=CENTROID_DISTANCE,
        params={"max_distance": reach},
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
    margin = [int(np.ceil(reach / size)) for size in sampling]

    scores: dict[int, dict[int, float]] = {}
    for index, box in enumerate(ndi.find_objects(parent)):
        if box is None:
            continue
        parent_id = index + 1
        window = tuple(
            slice(max(0, axis.start - pad), min(extent, axis.stop + pad))
            for axis, pad, extent in zip(box, margin, parent.shape)
        )
        local_child = child[window]
        if not local_child.any():
            continue

        distance = ndi.distance_transform_edt(parent[window] != parent_id, sampling=sampling)
        within = (local_child != 0) & (distance <= reach)
        if not within.any():
            continue

        ids = local_child[within]
        gaps = distance[within]
        # The nearest voxel of each child to this parent: sort by (child,
        # gap) and keep the first row of each child's run.
        order = np.lexsort((gaps, ids))
        ids, gaps = ids[order], gaps[order]
        first = np.concatenate(([True], ids[1:] != ids[:-1]))
        for child_id, gap in zip(ids[first], gaps[first]):
            affinity = 1.0 - float(gap) / reach
            if affinity > 0:
                scores.setdefault(int(child_id), {})[parent_id] = affinity

    return CandidateScores(
        scores=scores,
        child_ids=_ids(child),
        parent_ids=_ids(parent),
        method=BOUNDARY_DISTANCE,
        params={"max_distance": reach},
    )


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
