"""Feature families that describe a window or region beyond its composition.

A neighbourhood's class composition (`measure.composition`) is the CODEX
recipe and a good default, but it says nothing about how mixed the window
is, how its members are arranged, whether its classes sit side by side or
intermingle, or whether it is on a border. Each family here is one function
returning named columns, one row per object of the level, computed from the
level's membership matrix (objects x parent objects, row order) and the
parent's positions and classes - so the same features describe windows and
regions at any depth. `Hierarchy.describe` adds them to a level by name.

| family | columns | what it separates |
| --- | --- | --- |
| `weighted` | `Class_<c>_WeightedFraction` | composition with near members counting more (Gaussian in distance from the centre): less sensitive to the hard radius edge |
| `multiscale` | `Class_<c>_ClassFraction_r<r>` | composition at several radii |
| `diversity` | `shannon`, `simpson`, `richness`, `dominant_class`, `dominant_fraction` | mixed interfaces from pure compartments |
| `arrangement` | `radius_of_gyration`, `elongation`, `seed_offset`, `member_nn_distance`, `density` | the shape of the local tissue: strands (tubules), edges |
| `contacts` | `internal_edges`, `homophily` | whether classes intermingle or sit side by side within the window |
| `asymmetry` | `asymmetry` | a gradient across the window: one side differs from the other |
| `interface` | `distance_to_other_type` | how far an object is from one of another type: borders |

`transform_composition` turns composition columns into centred log-ratios
(or square roots) before clustering, so rare classes are not swamped.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from vtea_core.hierarchy.measure import composition, shape
from vtea_core.neighborhoods.measure import class_column_name, class_values

FAMILIES = (
    "weighted",
    "multiscale",
    "diversity",
    "arrangement",
    "contacts",
    "asymmetry",
    "interface",
)


def _class_codes(classes) -> tuple[list, np.ndarray]:
    """(class values, code per object: index into values, -1 for none)."""
    series = pd.Series(np.asarray(classes, dtype=object))
    values = class_values(series)
    as_text = bool(values) and isinstance(values[0], str)
    keys = series.astype(str) if as_text else pd.to_numeric(series, errors="coerce")
    lookup = {value: code for code, value in enumerate(values)}
    codes = np.array(
        [lookup.get(k, -1) if pd.notna(s) else -1 for k, s in zip(keys, series)], dtype=np.int64
    )
    return values, codes


def _counts(membership: sparse.csr_matrix, codes: np.ndarray, n_classes: int) -> np.ndarray:
    """(objects, classes) member counts per class."""
    known = codes >= 0
    onehot = sparse.csr_matrix(
        (np.ones(int(known.sum())), (np.flatnonzero(known), codes[known])),
        shape=(len(codes), n_classes),
    )
    return np.asarray((membership.astype(np.float64) @ onehot).todense())


def weighted_composition(
    membership: sparse.csr_matrix,
    positions: np.ndarray,
    centers: np.ndarray,
    classes,
    sigma: float,
) -> pd.DataFrame:
    """Composition (percent) with each member weighted by
    exp(-d^2 / 2 sigma^2), d its distance from the object's centre."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    values, codes = _class_codes(classes)
    coo = membership.tocoo()
    d2 = ((positions[coo.col] - centers[coo.row]) ** 2).sum(axis=1)
    weight = np.exp(-d2 / (2.0 * sigma * sigma))
    weights = sparse.csr_matrix((weight, (coo.row, coo.col)), shape=membership.shape)
    total = np.asarray(weights.sum(axis=1)).ravel()
    counts = _counts(weights, codes, len(values))
    out = {}
    for index, value in enumerate(values):
        out[class_column_name(value, "WeightedFraction")] = (
            100.0 * counts[:, index] / np.where(total > 0, total, 1)
        )
    return pd.DataFrame(out, index=range(membership.shape[0]))


def multiscale_composition(
    positions: np.ndarray, centers: np.ndarray, classes, radii, *, include=None
) -> pd.DataFrame:
    """Composition around each centre at several `radii`, as
    `Class_<c>_ClassFraction_r<r>` (percent). `include`: which positions can
    be counted (all by default)."""
    radii = [float(r) for r in np.atleast_1d(radii)]
    if not radii or min(radii) <= 0:
        raise ValueError("give one or more positive radii")
    rows = np.arange(len(positions)) if include is None else np.flatnonzero(include)
    tree = cKDTree(positions[rows])
    frames = []
    for radius in radii:
        hits = tree.query_ball_point(centers, r=radius)
        lengths = np.array([len(h) for h in hits], dtype=np.int64)
        cols = rows[np.concatenate([np.asarray(h, dtype=np.int64) for h in hits])] if len(hits) else []
        membership = sparse.csr_matrix(
            (np.ones(int(lengths.sum())), cols, np.concatenate([[0], np.cumsum(lengths)])),
            shape=(len(centers), len(positions)),
        )
        block = composition(membership, classes)
        block = block[[c for c in block.columns if c.endswith("_ClassFraction")]]
        frames.append(block.rename(columns=lambda c, r=radius: f"{c}_r{r:g}"))
    return pd.concat(frames, axis=1)


def diversity(membership: sparse.csr_matrix, classes) -> pd.DataFrame:
    """How mixed each object's members are, over the classed members:
    Shannon entropy (natural log), Simpson's index (1 - sum p^2), richness
    (classes present), the dominant class and its share (0-1)."""
    values, codes = _class_codes(classes)
    counts = _counts(membership, codes, len(values))
    total = counts.sum(axis=1)
    p = counts / np.where(total > 0, total, 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(p > 0, np.log(p), 0.0)
    shannon = -(p * logs).sum(axis=1)
    simpson = 1.0 - (p**2).sum(axis=1)
    dominant = p.argmax(axis=1) if len(values) else np.zeros(len(p), dtype=np.int64)
    empty = total == 0
    out = pd.DataFrame(
        {
            "shannon": np.where(empty, np.nan, shannon),
            "simpson": np.where(empty, np.nan, simpson),
            "richness": (counts > 0).sum(axis=1).astype(np.int64),
            "dominant_class": pd.Series(
                [None if e else values[d] for d, e in zip(dominant, empty)], dtype=object
            ),
            "dominant_fraction": np.where(empty, np.nan, p.max(axis=1) if len(values) else 0.0),
        }
    )
    return out


def arrangement(
    membership: sparse.csr_matrix,
    positions: np.ndarray,
    centers: np.ndarray,
    *,
    radius: float | None = None,
) -> pd.DataFrame:
    """How each object's members are arranged (physical units):
    `radius_of_gyration` and `elongation` (see `measure.shape`);
    `seed_offset`, the distance from the centre to the members' mean over the
    radius of gyration (0 in the middle of the tissue, ~1 on an edge);
    `member_nn_distance`, the mean over members of their nearest-neighbour
    distance in the whole level (how crowded); and `density`, members per
    unit area (2D) or volume (3D) of a window of `radius` (NaN without one)."""
    m = sparse.csr_matrix(membership, dtype=np.float64)
    counts = np.asarray(m.sum(axis=1)).ravel()
    out = shape(membership, positions)
    mean = (m @ positions) / np.maximum(counts, 1)[:, None]
    offset = np.linalg.norm(mean - centers, axis=1)
    gyration = out["radius_of_gyration"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["seed_offset"] = np.where(gyration > 0, offset / gyration, np.where(counts > 0, 0.0, np.nan))
    if len(positions) > 1:
        nn, _ = cKDTree(positions).query(positions, k=2)
        nearest = nn[:, 1]
    else:
        nearest = np.full(len(positions), np.nan)
    out["member_nn_distance"] = (m @ nearest) / np.where(counts > 0, counts, np.nan)
    if radius and radius > 0:
        ndim = positions.shape[1]
        size = math.pi * radius**2 if ndim == 2 else 4.0 / 3.0 * math.pi * radius**3
        out["density"] = counts / size
    else:
        out["density"] = np.nan
    return out


def internal_contacts(
    membership: sparse.csr_matrix, adjacency: sparse.csr_matrix, classes
) -> pd.DataFrame:
    """Neighbour pairs among each object's members: how many
    (`internal_edges`), and the fraction joining two members of the same
    class (`homophily`; NaN with no classed edge). High homophily: the
    classes sit side by side in compartments; low: they intermingle."""
    _values, codes = _class_codes(classes)
    a = sparse.triu(sparse.csr_matrix(adjacency, dtype=np.float64), k=1).tocsr()
    m = sparse.csr_matrix(membership, dtype=np.float64)
    # Edges (u, v) with both ends in the object: rowsum((M A) .* M).
    edges = np.asarray((m @ a).multiply(m).sum(axis=1)).ravel()
    coo = a.tocoo()
    same = (codes[coo.row] == codes[coo.col]) & (codes[coo.row] >= 0)
    classed = (codes[coo.row] >= 0) & (codes[coo.col] >= 0)
    shape_ = a.shape

    def within(mask):
        sub = sparse.csr_matrix((np.ones(int(mask.sum())), (coo.row[mask], coo.col[mask])), shape=shape_)
        return np.asarray((m @ sub).multiply(m).sum(axis=1)).ravel()

    same_edges, classed_edges = within(same), within(classed)
    with np.errstate(divide="ignore", invalid="ignore"):
        homophily = np.where(classed_edges > 0, same_edges / classed_edges, np.nan)
    return pd.DataFrame({"internal_edges": edges.astype(np.int64), "homophily": homophily})


def asymmetry(
    membership: sparse.csr_matrix, positions: np.ndarray, centers: np.ndarray, classes
) -> pd.DataFrame:
    """How different the two halves of each object are: for each axis, the
    members on either side of the centre are compared by composition (total
    variation distance, 0 the same to 1 nothing in common), and the largest
    over the axes is kept. A window on a border or a gradient is asymmetric;
    one inside a uniform compartment is not."""
    values, codes = _class_codes(classes)
    coo = membership.tocoo()
    n = membership.shape[0]
    best = np.zeros(n)
    seen = np.zeros(n, dtype=bool)
    for axis in range(positions.shape[1]):
        delta = positions[coo.col, axis] - centers[coo.row, axis]
        sides = []
        for side in (delta < 0, delta > 0):
            sub = sparse.csr_matrix(
                (np.ones(int(side.sum())), (coo.row[side], coo.col[side])), shape=membership.shape
            )
            counts = _counts(sub, codes, len(values))
            total = counts.sum(axis=1)
            sides.append((counts / np.where(total > 0, total, 1)[:, None], total > 0))
        (pa, a_ok), (pb, b_ok) = sides
        both = a_ok & b_ok
        tv = 0.5 * np.abs(pa - pb).sum(axis=1)
        best = np.where(both, np.maximum(best, tv), best)
        seen |= both
    return pd.DataFrame({"asymmetry": np.where(seen, best, np.nan)})


def interface_distance(positions: np.ndarray, types) -> pd.DataFrame:
    """Each object's distance to the nearest object of another type
    (`distance_to_other_type`; NaN with no type or no other type)."""
    series = pd.Series(np.asarray(types, dtype=object))
    out = np.full(len(series), np.nan)
    known = series.notna().to_numpy()
    kinds = pd.unique(series[known])
    for kind in kinds:
        mine = known & (series == kind).to_numpy()
        others = known & ~mine
        if not others.any():
            continue
        distance, _ = cKDTree(positions[others]).query(positions[mine])
        out[mine] = distance
    return pd.DataFrame({"distance_to_other_type": out})


def transform_composition(
    frame: pd.DataFrame, how: str = "clr", *, columns=None, pseudocount: float = 0.5
) -> pd.DataFrame:
    """Composition columns (percent) transformed for clustering. `"clr"`:
    centred log-ratio, log(x + pseudo) minus the row's mean log, over the
    columns given (by default every `*_ClassFraction` column); `"sqrt"`:
    square root of the fraction; `"none"`: unchanged. Returns new columns
    named `<column>_<how>`."""
    if how not in ("clr", "sqrt", "none"):
        raise ValueError(f"unknown composition transform {how!r}, expected clr, sqrt or none")
    columns = list(columns) if columns is not None else [
        c for c in frame.columns if str(c).endswith("_ClassFraction")
    ]
    if not columns:
        raise ValueError("no composition columns to transform")
    values = frame[columns].to_numpy(dtype=float)
    if how == "clr":
        logs = np.log(values + pseudocount)
        values = logs - logs.mean(axis=1, keepdims=True)
    elif how == "sqrt":
        values = np.sqrt(np.clip(values, 0, None) / 100.0)
    return pd.DataFrame(values, columns=[f"{c}_{how}" for c in columns], index=frame.index)
