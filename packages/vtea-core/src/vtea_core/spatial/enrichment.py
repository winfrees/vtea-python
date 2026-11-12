"""Enrichment: which classes sit together more (or less) than chance.

Two questions, the two usual ones of a neighbourhood analysis (Schürch et
al., Cell 2020; Goltsev et al., Cell 2018):

- `type_enrichment`: is a cell type over-represented in a neighbourhood
  type? A contingency table of class x type with log2(observed / expected),
  expected from the margins (independence).
- `neighbor_enrichment`: are cells of class B neighbours of cells of class A
  more often than chance? The neighbour graph (within a radius, or k nearest)
  is fixed - the tissue keeps its shape and density - and the class labels
  are permuted among the objects `n_permutations` times. Each ordered pair
  (A, B) gets the observed mean number of B neighbours per A cell, the
  permutation mean and standard deviation, a z-score and a two-sided
  empirical p-value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from vtea_core.spatial.features import class_targets


def type_enrichment(classes, types) -> pd.DataFrame:
    """Long table, one row per (class, type) pair: `class`, `type`, `observed`
    (objects), `expected` (from the row and column totals), `log2 enrichment`
    (log2 of observed / expected, with 0.5 added to both so empty cells stay
    finite) and `fraction of type` (share of the type's objects in the class).
    Objects with no class or no type are left out."""
    frame = pd.DataFrame(
        {"class": np.asarray(classes, dtype=object), "type": np.asarray(types, dtype=object)}
    )
    frame = frame[frame["class"].notna() & frame["type"].notna()]
    table = pd.crosstab(frame["class"], frame["type"])
    total = table.to_numpy().sum()
    if total == 0:
        return pd.DataFrame(
            columns=["class", "type", "observed", "expected", "log2 enrichment", "fraction of type"]
        )
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    observed = table.to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "class": np.repeat(table.index.to_numpy(), table.shape[1]),
            "type": np.tile(table.columns.to_numpy(), table.shape[0]),
            "observed": observed.ravel().astype(np.int64),
            "expected": expected.ravel(),
            "log2 enrichment": np.log2((observed + 0.5) / (expected + 0.5)).ravel(),
            "fraction of type": (observed / np.maximum(observed.sum(axis=0), 1)).ravel(),
        }
    )
    return out


def neighbor_graph(positions, *, radius: float | None = None, k: int | None = None) -> np.ndarray:
    """(edges, 2) directed neighbour pairs (i, j), i != j: every j within
    `radius` of i, or the `k` nearest j of each i."""
    points = np.asarray(positions, dtype=float)
    if (radius is None) == (k is None):
        raise ValueError("give either a radius or k")
    if len(points) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    tree = cKDTree(points)
    if radius is not None:
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        pairs = tree.query_pairs(r=radius, output_type="ndarray").astype(np.int64)
        return (
            np.vstack([pairs, pairs[:, ::-1]]) if len(pairs) else np.zeros((0, 2), dtype=np.int64)
        )
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    kk = min(int(k) + 1, len(points))
    _, hit = tree.query(points, k=kk)
    hit = np.asarray(hit).reshape(len(points), kk)
    rows = np.repeat(np.arange(len(points)), kk)
    cols = hit.ravel()
    keep = rows != cols
    edges = np.column_stack([rows[keep], cols[keep]])
    # Each object keeps exactly k (the object itself may not come first on ties).
    order = np.argsort(edges[:, 0], kind="stable")
    edges = edges[order]
    first = np.searchsorted(edges[:, 0], np.arange(len(points)))
    rank = np.arange(len(edges)) - first[edges[:, 0]]
    return edges[rank < k]


def neighbor_enrichment(
    positions,
    classes,
    *,
    radius: float | None = None,
    k: int | None = None,
    n_permutations: int = 200,
    random_state: int | None = 0,
    targets=None,
) -> pd.DataFrame:
    """Pairwise neighbour enrichment against a permutation null (see the module
    docstring). Long table, one row per ordered pair: `class` (the centre),
    `neighbor`, `observed` (mean number of `neighbor` objects among a `class`
    object's neighbours), `expected` and `sd` (over the permutations), `z`
    and `p` (two-sided, (hits + 1) / (permutations + 1)). Objects with no
    class stay in the graph, as neighbours of no class."""
    points = np.asarray(positions, dtype=float)
    values = pd.Series(np.asarray(classes, dtype=object))
    if len(values) != len(points):
        raise ValueError(f"{len(values)} classes for {len(points)} objects")
    names = class_targets(values, targets)
    lookup = {name: code for code, name in enumerate(names)}
    none = len(names)  # objects of no (wanted) class
    codes = np.array([lookup.get(v, none) if pd.notna(v) else none for v in values], dtype=np.int64)
    edges = neighbor_graph(points, radius=radius, k=k)
    n_codes = none + 1

    def counts(labels):
        flat = labels[edges[:, 0]] * n_codes + labels[edges[:, 1]]
        return np.bincount(flat, minlength=n_codes * n_codes).reshape(n_codes, n_codes)[
            :none, :none
        ]

    sizes = np.bincount(codes, minlength=n_codes)[:none].astype(float)
    per = np.maximum(sizes, 1)[:, None]
    observed = counts(codes) / per
    rng = np.random.default_rng(random_state)
    null = np.empty((max(int(n_permutations), 0), none, none))
    for p in range(len(null)):
        null[p] = counts(rng.permutation(codes)) / per
    if len(null):
        mean, sd = (
            null.mean(axis=0),
            null.std(axis=0, ddof=1) if len(null) > 1 else np.zeros_like(observed),
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(sd > 0, (observed - mean) / sd, 0.0)
        extreme = (np.abs(null - mean) >= np.abs(observed - mean)[None] - 1e-12).sum(axis=0)
        pvalue = (extreme + 1) / (len(null) + 1)
    else:
        mean = sd = z = pvalue = np.full_like(observed, np.nan)
    out = pd.DataFrame(
        {
            "class": np.repeat(np.array(names, dtype=object), none),
            "neighbor": np.tile(np.array(names, dtype=object), none),
            "observed": observed.ravel(),
            "expected": mean.ravel(),
            "sd": sd.ravel(),
            "z": z.ravel(),
            "p": pvalue.ravel(),
        }
    )
    return out
