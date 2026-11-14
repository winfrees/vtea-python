"""Measuring a level from its members: composition, aggregates, shape.

Every function here takes a membership matrix (objects x parent objects, in
row order) and the parent's table, so the same measurements describe
windows and regions, at any depth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from vtea_core.neighborhoods.measure import AGGREGATIONS, _names, class_column_name, class_values


def composition(membership: sparse.csr_matrix, classes: pd.Series) -> pd.DataFrame:
    """Per object, the fraction (percent, `Class_<c>_ClassFraction`) and the
    count (`Class_<c>_ClassSums`) of its members in each class - the column
    names of `vtea_core.neighborhoods`, so a window of any level reads like
    a neighbourhood. Members with no class (NaN / None) count towards the
    total but in no class."""
    classes = pd.Series(np.asarray(classes, dtype=object))
    values = class_values(classes)
    totals = np.asarray(membership.sum(axis=1)).ravel().astype(float)
    out = {}
    counts = {}
    as_text = bool(values) and isinstance(values[0], str)
    keys = classes.astype(str) if as_text else pd.to_numeric(classes, errors="coerce")
    for value in values:
        hit = (keys == value).to_numpy() & classes.notna().to_numpy()
        counts[value] = np.asarray(membership @ hit.astype(np.int64)).ravel()
    for value in values:
        out[class_column_name(value, "ClassFraction")] = (
            100.0 * counts[value] / np.where(totals > 0, totals, 1)
        )
    for value in values:
        out[class_column_name(value, "ClassSums")] = counts[value].astype(np.int64)
    return pd.DataFrame(out, index=range(membership.shape[0]))


def aggregate(
    membership: sparse.csr_matrix, values: pd.DataFrame, aggregations: str = "mean"
) -> pd.DataFrame:
    """Each column of `values` (one row per parent object) reduced over every
    object's members by each of `aggregations` (comma-separated, from
    `AGGREGATIONS`), as `<aggregation>_<column>`. An object with no members
    gets NaN."""
    reductions = _names(aggregations)
    unknown = set(reductions) - set(AGGREGATIONS)
    if unknown:
        raise ValueError(f"unknown aggregation(s) {sorted(unknown)}, expected {list(AGGREGATIONS)}")
    n = membership.shape[0]
    coo = membership.tocoo()
    pairs = values.iloc[coo.col].reset_index(drop=True)
    pairs.insert(0, "_object", coo.row)
    grouped = pairs.groupby("_object")
    out = {}
    for reduction in reductions:
        block = grouped[list(values.columns)].agg(reduction).reindex(range(n))
        for column in values.columns:
            out[f"{reduction}_{column}"] = block[column].to_numpy()
    return pd.DataFrame(out, index=range(n))


def shape(membership: sparse.csr_matrix, positions: np.ndarray) -> pd.DataFrame:
    """How each object's members are arranged, from their physical positions:
    `radius_of_gyration` (root mean square distance from their mean) and
    `elongation` (square root of the ratio of the two largest eigenvalues of
    their gyration tensor: 1 for a round cluster, large for a strand; NaN
    with fewer than two members)."""
    m = sparse.csr_matrix(membership, dtype=np.float64)
    n, ndim = m.shape[0], positions.shape[1]
    counts = np.asarray(m.sum(axis=1)).ravel()
    safe = np.maximum(counts, 1)[:, None]
    mean = (m @ positions) / safe
    # Gyration tensor per object: E[x x^T] - mean mean^T, from sparse sums.
    second = np.empty((n, ndim, ndim))
    for i in range(ndim):
        for j in range(i, ndim):
            second[:, i, j] = second[:, j, i] = (m @ (positions[:, i] * positions[:, j])) / safe[
                :, 0
            ]
    tensor = second - mean[:, :, None] * mean[:, None, :]
    gyration = np.sqrt(np.clip(np.trace(tensor, axis1=1, axis2=2), 0, None))
    eig = np.clip(np.sort(np.linalg.eigvalsh(tensor), axis=1)[:, ::-1], 0, None)
    # E[x x^T] - mean mean^T cancels: below this the second eigenvalue is
    # rounding, not spread, and the members lie on a line.
    noise = 1e-9 * ((mean**2).sum(axis=1) + eig[:, 0]) + 1e-12
    if ndim > 1:
        flat = eig[:, 1] <= noise
        with np.errstate(divide="ignore", invalid="ignore"):
            elongation = np.where(flat, np.inf, np.sqrt(eig[:, 0] / np.where(flat, 1, eig[:, 1])))
    else:
        elongation = np.full(n, np.nan)
    gyration = np.where(counts > 0, gyration, np.nan)
    elongation = np.where(counts >= 2, elongation, np.nan)
    return pd.DataFrame({"radius_of_gyration": gyration, "elongation": elongation})
