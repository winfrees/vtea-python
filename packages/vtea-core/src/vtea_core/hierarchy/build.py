"""Building a level on its parent: windows, and regions.

**Windows** (`build_windows`) are the neighbourhoods of `vtea_core.neighborhoods`
at any depth: a window is drawn around each seed object of the parent - its
`k` nearest, or everything within `radius` - so they overlap, and a parent
object can be in any number of them. Seeds and members can each be a subset
of the parent (a population): windows around tubular neighbourhoods only,
counting every object around them, say. A seed is always a member of its own
window.

**Regions** (`build_regions`) are functional units: contiguous parent
objects of the same type, merged. They do not overlap. Two parent objects of
the same type are joined in two stages:

1. *Candidates*: the two touch (see `Hierarchy.adjacency`: touching masks at
   level 0; above it, windows whose seeds touch, regions whose members
   touch), or their centres are within a distance `d` (k-d tree pairs). With
   `d = 0` only touching counts.
2. *Weight* (only when the parent objects have members of their own, i.e.
   the parent is not the segmentation): how close each one's members sit to
   the other's,

       w(A, B) = mean over m in A u B of exp(-delta(m)^2 / 2 sigma^2)

   where delta(m) is the distance from member m to the nearest member of the
   other (0 for a member both contain). The pair is joined when w >= `tau`.
   Two windows whose seeds are near but whose members extend in different
   directions - two tubules side by side - score low and stay apart; sharing
   members alone never makes a candidate, since overlapping windows share
   members almost everywhere.

The joined pairs' connected components with at least `min_size` parent
objects are the regions.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from vtea_core.hierarchy.level import LEVEL_ID, REGIONS, WINDOWS, ObjectLevel
from vtea_core.hierarchy.measure import aggregate, composition, shape
from vtea_core.neighborhoods.measure import _names
from vtea_core.neighborhoods.model import centroid_columns

WINDOW_METHODS = ("radius", "nearest")


def voxel_positions(level: ObjectLevel) -> np.ndarray:
    columns = centroid_columns(level.table)
    if not columns:
        raise ValueError(f"level {level.name!r} has no centroid-* columns")
    return level.table[columns].to_numpy(dtype=float)


def _member_columns(
    level: ObjectLevel,
    membership: sparse.csr_matrix,
    parent: ObjectLevel,
    classes: str | None,
    features: str,
    aggregations: str,
) -> list[pd.DataFrame]:
    blocks = []
    if classes:
        if classes not in parent.table.columns:
            raise ValueError(f"level {parent.name!r} has no class column {classes!r}")
        blocks.append(composition(membership, parent.table[classes]))
    wanted = _names(features)
    if wanted:
        missing = [name for name in wanted if name not in parent.table.columns]
        if missing:
            raise ValueError(f"level {parent.name!r} has no feature(s) {missing}")
        blocks.append(aggregate(membership, parent.table[wanted], aggregations))
    return blocks


def build_windows(
    parent: ObjectLevel,
    positions: np.ndarray,
    *,
    name: str,
    method: Literal["radius", "nearest"] = "radius",
    radius: float = 50.0,
    k: int = 10,
    classes: str | None = None,
    seeds=None,
    members=None,
    features: str = "",
    aggregations: str = "mean",
    unit: str = "voxel",
) -> ObjectLevel:
    """A windows level on `parent`. `positions` are the parent's physical
    positions (what distances are measured in). `classes`: a column of the
    parent to measure composition over. `seeds` / `members`: which parent
    objects get a window, and which can be counted in one (see
    `ObjectLevel.selection`); every object by default."""
    if method not in WINDOW_METHODS:
        raise ValueError(f"unknown window method {method!r}, expected one of {WINDOW_METHODS}")
    if method == "nearest" and k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if method == "radius" and radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    seed_rows = np.flatnonzero(parent.selection(seeds))
    member_rows = np.flatnonzero(parent.selection(members))
    n_parent = len(parent)

    rows: list[np.ndarray] = []
    if len(seed_rows) and len(member_rows):
        tree = cKDTree(positions[member_rows])
        if method == "radius":
            hits = tree.query_ball_point(positions[seed_rows], r=radius)
            for seed, found in zip(seed_rows, hits):
                rows.append(np.union1d(member_rows[np.asarray(found, dtype=np.int64)], [seed]))
        else:
            # One more than k, as the seed is usually among them - but not
            # necessarily first (ties at distance 0), or at all (a seed that
            # is not a member): drop it wherever it is, keep k others.
            wanted = min(int(k) + 1, len(member_rows))
            _d, found = tree.query(positions[seed_rows], k=wanted)
            found = np.asarray(found).reshape(len(seed_rows), wanted)
            for seed, near in zip(seed_rows, found):
                others = [r for r in member_rows[near] if r != seed][: int(k)]
                rows.append(np.union1d(np.asarray(others, dtype=np.int64), [seed]))
    lengths = np.array([len(r) for r in rows], dtype=np.int64)
    membership = sparse.csr_matrix(
        (
            np.ones(int(lengths.sum()), dtype=np.int8),
            np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64),
            np.concatenate([[0], np.cumsum(lengths)]),
        ),
        shape=(len(seed_rows), n_parent),
    )
    voxels = voxel_positions(parent)
    spread = np.array(
        [
            np.linalg.norm(positions[r] - positions[s], axis=1).mean()
            for s, r in zip(seed_rows, rows)
        ]
    )
    table = pd.DataFrame({LEVEL_ID: np.arange(1, len(seed_rows) + 1, dtype=np.int64)})
    table["seed"] = parent.ids[seed_rows]
    table["n_objects"] = lengths
    for axis in range(voxels.shape[1]):
        table[f"centroid-{axis}"] = voxels[seed_rows, axis]
    table["mean_distance"] = spread if len(spread) else np.zeros(0)
    level = ObjectLevel(
        name=name,
        kind=WINDOWS,
        table=table,
        parent=parent.name,
        depth=parent.depth + 1,
        membership=membership,
        seeds=seed_rows,
        provenance={
            "builder": WINDOWS,
            "method": method,
            "radius": float(radius),
            "k": int(k),
            "classes": classes,
            "features": features,
            "aggregations": aggregations,
            "seeds": None if seeds is None else len(seed_rows),
            "members": None if members is None else len(member_rows),
            "unit": unit,
        },
    )
    for block in _member_columns(level, membership, parent, classes, features, aggregations):
        level.table = pd.concat([level.table, block], axis=1)
    return level


def member_weights(
    pairs: np.ndarray, membership: sparse.csr_matrix, member_positions: np.ndarray, sigma: float
) -> np.ndarray:
    """w(A, B) for each row (A, B) of `pairs` - see the module docstring."""
    m = membership.tocsr()
    weights = np.empty(len(pairs))
    trees: dict[int, cKDTree] = {}

    def members(row):
        return m.indices[m.indptr[row] : m.indptr[row + 1]]

    def tree(row):
        if row not in trees:
            trees[row] = cKDTree(member_positions[members(row)])
        return trees[row]

    scale = 2.0 * sigma * sigma
    for index, (a, b) in enumerate(pairs):
        ma, mb = members(a), members(b)
        if len(ma) == 0 or len(mb) == 0:
            weights[index] = 0.0
            continue
        da, _ = tree(b).query(member_positions[ma])
        db, _ = tree(a).query(member_positions[mb])
        weights[index] = float(np.exp(-(np.concatenate([da, db]) ** 2) / scale).mean())
    return weights


def build_regions(
    parent: ObjectLevel,
    positions: np.ndarray,
    *,
    name: str,
    types: str,
    values=None,
    adjacency: sparse.csr_matrix | None = None,
    d: float = 0.0,
    member_positions: np.ndarray | None = None,
    sigma: float | None = None,
    tau: float = 0.5,
    min_size: int = 1,
    classes: str | None = None,
    features: str = "",
    aggregations: str = "mean",
    unit: str = "voxel",
) -> ObjectLevel:
    """A regions level on `parent`: contiguous parent objects with the same
    value of `types` merged. `values`: which types to make regions of (all
    by default; objects with no type never are). `adjacency`: which parent
    objects touch (None: only the distance `d` makes candidates).
    `member_positions`: physical positions of the parent's own members (the
    grandparent's objects), for the member weighting; None when the parent
    is the segmentation. `sigma` defaults to the median nearest-neighbour
    distance among those members."""
    if types not in parent.table.columns:
        raise ValueError(f"level {parent.name!r} has no type column {types!r}")
    if d < 0:
        raise ValueError(f"d must be 0 or more, got {d}")
    if adjacency is None and d <= 0:
        raise ValueError(
            "nothing joins objects into regions: give touching objects or a distance d > 0"
        )
    if min_size < 1:
        raise ValueError(f"min_size must be at least 1, got {min_size}")
    kind = parent.table[types]
    eligible = kind.notna().to_numpy()
    if values is not None:
        eligible = eligible & np.array(kind.isin(list(np.atleast_1d(values))), dtype=bool)
    codes = pd.factorize(kind)[0]
    n = len(parent)

    found = []
    if adjacency is not None:
        coo = sparse.triu(adjacency, k=1).tocoo()
        found.append(np.column_stack([coo.row, coo.col]))
    if d > 0 and n > 1:
        near = cKDTree(positions).query_pairs(d, output_type="ndarray")
        found.append(np.asarray(near, dtype=np.int64).reshape(-1, 2))
    pairs = np.unique(np.vstack(found), axis=0) if found else np.zeros((0, 2), dtype=np.int64)
    if len(pairs):
        keep = (
            eligible[pairs[:, 0]]
            & eligible[pairs[:, 1]]
            & (codes[pairs[:, 0]] == codes[pairs[:, 1]])
        )
        pairs = pairs[keep]

    weighted = parent.membership is not None and member_positions is not None
    weights = None
    if weighted:
        if sigma is None:
            if len(member_positions) > 1:
                gaps, _ = cKDTree(member_positions).query(member_positions, k=2)
                sigma = float(np.median(gaps[:, 1]))
            else:
                sigma = 1.0
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        weights = member_weights(pairs, parent.membership, member_positions, sigma)
        joined = pairs[weights >= tau]
    else:
        joined = pairs

    graph = sparse.coo_matrix(
        (np.ones(len(joined)), (joined[:, 0], joined[:, 1])), shape=(n, n)
    ).tocsr()
    _count, component = connected_components(graph, directed=False)
    component = np.where(eligible, component, -1)
    labels, sizes = np.unique(component[component >= 0], return_counts=True)
    keep = sizes >= min_size
    # Regions numbered from the largest down, so region 1 is the biggest.
    order = labels[keep][np.argsort(-sizes[keep], kind="stable")]
    rank = np.full(int(component.max()) + 1 if n else 0, -1, dtype=np.int64)
    rank[order] = np.arange(len(order))
    region_of = np.where(component >= 0, rank[np.maximum(component, 0)], -1) if n else component
    in_region = np.flatnonzero(region_of >= 0)
    in_region = in_region[np.argsort(region_of[in_region], kind="stable")]
    lengths = np.bincount(region_of[in_region], minlength=len(order)).astype(np.int64)
    membership = sparse.csr_matrix(
        (
            np.ones(len(in_region), dtype=np.int8),
            in_region,
            np.concatenate([[0], np.cumsum(lengths)]),
        ),
        shape=(len(order), n),
    )
    first = in_region[np.concatenate([[0], np.cumsum(lengths)[:-1]])] if len(order) else in_region
    voxels = voxel_positions(parent)
    table = pd.DataFrame({LEVEL_ID: np.arange(1, len(order) + 1, dtype=np.int64)})
    table[types] = kind.to_numpy()[first]
    table["n_objects"] = lengths
    for axis in range(voxels.shape[1]):
        table[f"centroid-{axis}"] = (membership @ voxels[:, axis]) / np.maximum(lengths, 1)
    table = pd.concat([table, shape(membership, positions)], axis=1)
    level = ObjectLevel(
        name=name,
        kind=REGIONS,
        table=table,
        parent=parent.name,
        depth=parent.depth + 1,
        membership=membership,
        provenance={
            "builder": REGIONS,
            "types": types,
            "values": None if values is None else [_plain(v) for v in np.atleast_1d(values)],
            "touching": adjacency is not None,
            "d": float(d),
            "sigma": None if sigma is None else float(sigma),
            "tau": float(tau) if weighted else None,
            "min_size": int(min_size),
            "classes": classes,
            "features": features,
            "aggregations": aggregations,
            "candidate_pairs": len(pairs),
            "joined_pairs": len(joined),
            "unit": unit,
        },
    )
    for block in _member_columns(level, membership, parent, classes, features, aggregations):
        level.table = pd.concat([level.table, block], axis=1)
    return level


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value
