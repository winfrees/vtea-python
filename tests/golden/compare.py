"""Comparison utilities for golden-dataset parity tests.

These compare a Python (vtea-core) output against a Java-VTEA reference
fixture. Exact bit-for-bit equality isn't the goal - algorithmic
equivalence within a documented tolerance is. See README.md for the
fixture layout these expect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def segmentation_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union of two binary (or boolean-cast label) masks.

    For multi-label masks, pass ``mask == label_id`` for the label of
    interest - this compares foreground/background agreement, not
    label-id agreement (label IDs are not expected to match across
    independently-run segmentations).
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def feature_table_diff(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    key: str,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> pd.DataFrame:
    """Row-aligned (on `key`) relative/absolute diff between two feature tables.

    Returns a DataFrame of per-column max absolute and relative errors for
    numeric columns; empty if every numeric column is within tolerance.
    Raises if `key` values don't match between the two tables (that's a
    correctness bug, not a tolerance issue).
    """
    a = actual.set_index(key).sort_index()
    e = expected.set_index(key).sort_index()
    if not a.index.equals(e.index):
        missing = e.index.difference(a.index)
        extra = a.index.difference(e.index)
        raise ValueError(
            f"Row key mismatch for '{key}': missing={list(missing)[:10]} extra={list(extra)[:10]}"
        )

    numeric_cols = [c for c in e.columns if pd.api.types.is_numeric_dtype(e[c])]
    rows = []
    for col in numeric_cols:
        if col not in a.columns:
            rows.append({"column": col, "status": "missing_in_actual"})
            continue
        close = np.isclose(a[col].to_numpy(), e[col].to_numpy(), rtol=rtol, atol=atol)
        if not close.all():
            diff = (a[col] - e[col]).abs()
            rows.append(
                {
                    "column": col,
                    "status": "out_of_tolerance",
                    "max_abs_error": float(diff.max()),
                    "max_rel_error": float((diff / e[col].abs().clip(lower=atol)).max()),
                    "n_failing": int((~close).sum()),
                }
            )
    return pd.DataFrame(rows)


def cluster_assignment_ari(actual_labels: np.ndarray, expected_labels: np.ndarray) -> float:
    """Adjusted Rand Index between two cluster assignments.

    Cluster-*id* equality isn't meaningful across independent runs
    (labels are arbitrary/permutable); ARI measures whether the same
    points end up grouped together, which is the actual invariant.
    1.0 is a perfect match, ~0.0 is random.
    """
    return float(adjusted_rand_score(expected_labels, actual_labels))
