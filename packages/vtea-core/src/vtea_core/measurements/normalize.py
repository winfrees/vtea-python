"""Putting features on a common scale before they are compared.

A feature table mixes quantities in unrelated units: a volume in thousands
of voxels, a mean intensity in the hundreds, a fraction between 0 and 1.
Every clustering and reduction here measures distance between rows, so
without rescaling the largest-valued feature decides the answer on its own
and the others might as well not have been measured.

The Java VTEA offered one remedy, as a "Z-scale all data" checkbox that is
the first entry of every clustering and reduction protocol
(`AbstractFeatureProcessing.normalizeColumns`). It is ported here as the
`"zscore"` option of a `normalize` parameter every clustering and reduction
step carries, off by default as the checkbox was. Two more are offered
because a z-score is the wrong tool for the data a tissue produces often
enough to matter:

- `"robust"`: centre on the median and divide by the interquartile range.
  A few very bright or very large objects - debris, a clump the
  segmentation did not split - inflate a standard deviation and flatten
  every ordinary object towards zero; they barely move a quartile.
- `"minmax"`: rescale to [0, 1]. For features already bounded (fractions,
  scores) whose range is meaningful.

A feature with no spread at all carries no information to compare on, and
becomes a column of zeros under every option. (The Java writes 1 there
instead; a constant column is a constant column, and neither value changes
a distance.) The z-score divides by the population standard deviation (n),
as the Java does, not the sample one.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

NORMALIZATIONS = ("none", "zscore", "robust", "minmax")

Normalization = Literal["none", "zscore", "robust", "minmax"]


def normalize_features(data: np.ndarray, method: Normalization = "zscore") -> np.ndarray:
    """`data` (n_objects, n_features) with each column rescaled by `method`.

    Returns a new float array; `data` is left alone. `"none"` returns the
    data as floats, unscaled, so a caller can pass the choice straight
    through without special-casing it.
    """
    if method not in NORMALIZATIONS:
        raise ValueError(f"unknown normalization {method!r}, expected one of {list(NORMALIZATIONS)}")
    matrix = np.array(data, dtype=float, copy=True)
    if method == "none" or matrix.size == 0:
        return matrix
    if matrix.ndim == 1:
        return normalize_features(matrix[:, np.newaxis], method)[:, 0]

    if method == "zscore":
        centre = matrix.mean(axis=0)
        scale = matrix.std(axis=0)  # ddof=0: population, as the Java divides by n
    elif method == "robust":
        centre = np.median(matrix, axis=0)
        upper, lower = np.percentile(matrix, [75, 25], axis=0)
        scale = upper - lower
        # A feature whose middle half is one value (a count that is mostly
        # zero, say) still has a spread worth keeping; fall back to the
        # standard deviation rather than discarding it as constant.
        flat = scale == 0
        if flat.any():
            scale[flat] = matrix[:, flat].std(axis=0)
    else:  # minmax
        centre = matrix.min(axis=0)
        scale = matrix.max(axis=0) - centre

    constant = scale == 0
    scale[constant] = 1.0
    result = (matrix - centre) / scale
    result[:, constant] = 0.0
    return result
