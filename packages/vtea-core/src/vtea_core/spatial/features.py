"""Per-object spatial features: distances, neighbour counts and contacts by class.

See the package docstring for what each measures. All functions return a
DataFrame with one row per object, in the order given, so it can be joined
column-wise onto the object table.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from vtea_core.data.spacing import Spacing


def class_targets(classes, targets: Sequence | None = None) -> list:
    """The classes to measure: `targets` (each must be present or is still
    reported, as all-NaN / zero columns), else every class present, sorted."""
    if targets is not None:
        return list(targets)
    values = pd.Series(classes, dtype=object)
    present = values[values.notna()].unique().tolist()
    return sorted(present, key=lambda v: (not isinstance(v, (int, np.integer)), str(v)))


def _classes(classes, n: int) -> pd.Series:
    values = pd.Series(np.asarray(classes, dtype=object))
    if len(values) != n:
        raise ValueError(f"{len(values)} classes for {n} objects")
    return values


def _members(values: pd.Series, target) -> np.ndarray:
    return (values == target).to_numpy(dtype=bool) & values.notna().to_numpy()


def _positions(positions) -> np.ndarray:
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2:
        raise ValueError("positions must be an (objects x axes) array")
    return points


def nearest_distance(positions, classes, *, targets: Sequence | None = None) -> pd.DataFrame:
    """Distance from each object's centroid to the nearest centroid of another
    object of each class (`distance to <class>`; NaN when there is none)."""
    points = _positions(positions)
    values = _classes(classes, len(points))
    out = pd.DataFrame(index=range(len(points)))
    for target in class_targets(values, targets):
        member = _members(values, target)
        column = np.full(len(points), np.nan)
        rows = np.flatnonzero(member)
        if len(rows) and len(points):
            tree = cKDTree(points[rows])
            k = min(2, len(rows))
            dist, hit = tree.query(points, k=k)
            dist, hit = dist.reshape(len(points), k), hit.reshape(len(points), k)
            first_is_self = rows[hit[:, 0]] == np.arange(len(points))
            column = dist[:, 0].copy()
            if k == 2:
                column[first_is_self] = dist[first_is_self, 1]
            else:
                column[first_is_self] = np.nan  # the only member is itself
        out[f"distance to {target}"] = column
    return out


def neighbors_within(
    positions,
    classes,
    radius: float,
    *,
    targets: Sequence | None = None,
    fraction: bool = True,
) -> pd.DataFrame:
    """How many other objects of each class have their centroid within `radius`
    of each object's centroid (`n <class> within <r>`), and, with `fraction`,
    what fraction of all the other objects within `radius` they are
    (`fraction <class> within <r>`, NaN when there are none)."""
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    points = _positions(positions)
    values = _classes(classes, len(points))
    out = pd.DataFrame(index=range(len(points)))
    if not len(points):
        return out
    r = f"{radius:g}"
    total = cKDTree(points).query_ball_point(points, r=radius, return_length=True) - 1
    for target in class_targets(values, targets):
        member = _members(values, target)
        count = np.zeros(len(points), dtype=np.int64)
        if member.any():
            count = cKDTree(points[member]).query_ball_point(points, r=radius, return_length=True)
            count = count - member  # not itself
        out[f"n {target} within {r}"] = count
        if fraction:
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"fraction {target} within {r}"] = np.where(
                    total > 0, count / np.maximum(total, 1), np.nan
                )
    return out


def _sampling(spacing: Spacing | None, ndim: int) -> np.ndarray:
    if spacing is None or not spacing.is_known:
        return np.ones(ndim)
    return np.asarray(spacing.for_ndim(ndim), dtype=float)


def surface_distance(
    labels: np.ndarray,
    ids,
    classes,
    spacing: Spacing | None = None,
    *,
    targets: Sequence | None = None,
) -> pd.DataFrame:
    """Exact distance between masks: for each object (`ids`, its label values
    in `labels`, a 2D or 3D label image) the distance from its nearest voxel
    to the nearest voxel of another object of each class (`surface distance
    to <class>`; NaN when there is none, or the object is not in `labels`).

    Measured between voxel centres with the voxel size of `spacing`, so
    touching objects are one voxel apart and overlapping (impossible for a
    label image) would be 0. One Euclidean distance transform per class gives
    every non-member its distance; a member's distance to the *other* members
    is computed in a window around it, bounded by the distance to the nearest
    one (an upper bound found from one voxel of each), so it stays exact
    without a transform of the whole image per object.
    """
    labels = np.asarray(labels)
    if labels.ndim not in (2, 3):
        raise ValueError("labels must be a 2D or 3D label image")
    ids = np.asarray(ids, dtype=np.int64)
    values = _classes(classes, len(ids))
    sampling = _sampling(spacing, labels.ndim)
    out = pd.DataFrame(index=range(len(ids)))
    top = int(labels.max()) if labels.size else 0
    boxes = ndi.find_objects(labels.astype(np.int64, copy=False)) if top > 0 else []
    present = np.array([0 < i <= len(boxes) and boxes[i - 1] is not None for i in ids], dtype=bool)
    surface = _surface_voxels(labels) if top > 0 else None
    for target in class_targets(values, targets):
        member = _members(values, target) & present
        column = np.full(len(ids), np.nan)
        member_ids = ids[member]
        if len(member_ids):
            lookup = np.zeros(top + 1, dtype=bool)  # label -> member, faster than np.isin
            lookup[member_ids] = True
            inside = lookup[labels]
            dist = ndi.distance_transform_edt(~inside, sampling=sampling)
            others = present & ~member
            if others.any():
                column[others] = _label_minimum(dist, labels, top)[ids[others]]
            if len(member_ids) > 1:
                column[member] = _within_class(labels, surface, member_ids, boxes, sampling, lookup)
        out[f"surface distance to {target}"] = column
    return out


def _surface_voxels(labels) -> np.ndarray:
    """Object voxels with a face neighbour of another label (or background).

    The closest pair of voxels of two disjoint masks is always made of two such
    voxels: from an inner voxel, the face neighbour towards the other mask is in
    the same mask and closer. So distances between masks need only these."""
    surface = np.zeros(labels.shape, dtype=bool)
    for axis in range(labels.ndim):
        a = [slice(None)] * labels.ndim
        b = [slice(None)] * labels.ndim
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        differ = labels[tuple(a)] != labels[tuple(b)]
        surface[tuple(a)] |= differ
        surface[tuple(b)] |= differ
    return surface & (labels > 0)


def _within_class(labels, surface, member_ids, boxes, sampling, lookup) -> np.ndarray:
    """Distance from each member to the nearest other member, mask to mask: in a
    window around the member, nearest surface voxel to surface voxel."""
    # One voxel of each member, the most central one: the distance between two of
    # them bounds the distance between the two masks from above (the more central,
    # the tighter the bound, and the smaller the window searched below).
    anchors = np.empty((len(member_ids), labels.ndim))
    for k, i in enumerate(member_ids):
        box = boxes[i - 1]
        voxels = np.argwhere(labels[box] == i)
        middle = (np.asarray([s.stop - s.start for s in box]) - 1) / 2
        local = voxels[np.argmin((((voxels - middle) * sampling) ** 2).sum(axis=1))]
        anchors[k] = [s.start + v for s, v in zip(box, local)]
    physical = anchors * sampling
    bound, _ = cKDTree(physical).query(physical, k=2)
    bound = bound[:, 1]
    result = np.empty(len(member_ids))
    for k, i in enumerate(member_ids):
        box = boxes[i - 1]
        pad = np.ceil(bound[k] / sampling).astype(int) + 1
        window = tuple(
            slice(max(s.start - p, 0), min(s.stop + p, n))
            for s, p, n in zip(box, pad, labels.shape)
        )
        points = np.argwhere(surface[window])  # few voxels: filter those, not the window
        which = labels[window][tuple(points.T)]
        other = points[lookup[which] & (which != i)]
        if not len(other):  # cannot happen (the bound reaches one), kept safe
            result[k] = bound[k]
            continue
        own = points[which == i]
        dist, _ = cKDTree(other * sampling).query(own * sampling, k=1)
        result[k] = dist.min()
    return result


def _label_minimum(values, labels, top) -> np.ndarray:
    """Minimum of `values` over each label 0..top (inf where a label is absent),
    without the sort ndimage.minimum makes."""
    out = np.full(top + 1, np.inf)
    np.minimum.at(out, labels.ravel(), values.ravel())
    return out


def touching_pairs(labels: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """(n, 2) array of label pairs whose masks touch (each pair both ways).
    `connectivity` 1: sharing a face; `labels.ndim`: also an edge or corner."""
    labels = np.asarray(labels)
    if labels.ndim < 1 or not labels.size:
        return np.zeros((0, 2), dtype=np.int64)
    if not 1 <= connectivity <= labels.ndim:
        raise ValueError(f"connectivity must be 1..{labels.ndim}, got {connectivity}")
    offsets = [
        o
        for o in np.ndindex(*(3,) * labels.ndim)
        if 0 < sum(abs(v - 1) for v in o) <= connectivity and next(v for v in o if v != 1) == 2
    ]  # half of the neighbourhood: each pair is found once per direction, then mirrored
    found = []
    for offset in offsets:
        shift = [v - 1 for v in offset]
        a = labels[
            tuple(slice(max(-s, 0), labels.shape[d] - max(s, 0)) for d, s in enumerate(shift))
        ]
        b = labels[
            tuple(slice(max(s, 0), labels.shape[d] - max(-s, 0)) for d, s in enumerate(shift))
        ]
        hit = (a != b) & (a > 0) & (b > 0)
        if hit.any():
            found.append(np.column_stack([a[hit], b[hit]]).astype(np.int64))
    if not found:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.unique(np.vstack(found), axis=0)
    return np.unique(np.vstack([pairs, pairs[:, ::-1]]), axis=0)


def contacts(
    labels: np.ndarray,
    ids,
    classes,
    *,
    targets: Sequence | None = None,
    connectivity: int = 1,
) -> pd.DataFrame:
    """How many other objects of each class touch each object's mask
    (`contacts <class>`), and how many objects touch it in all (`contacts`)."""
    ids = np.asarray(ids, dtype=np.int64)
    values = _classes(classes, len(ids))
    pairs = touching_pairs(labels, connectivity)
    class_of = pd.Series(values.to_numpy(), index=ids)
    class_of = class_of[~class_of.index.duplicated()]
    frame = pd.DataFrame(pairs, columns=["a", "b"])
    frame["class"] = class_of.reindex(frame["b"]).to_numpy()
    out = pd.DataFrame(index=range(len(ids)))
    out["contacts"] = frame.groupby("a").size().reindex(ids, fill_value=0).to_numpy(dtype=np.int64)
    for target in class_targets(values, targets):
        hits = frame[frame["class"] == target].groupby("a").size()
        out[f"contacts {target}"] = hits.reindex(ids, fill_value=0).to_numpy(dtype=np.int64)
    return out
