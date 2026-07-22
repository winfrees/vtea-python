"""Per-object measurement extraction via skimage.measure.regionprops_table.

Replaces vtea.objects.measurements' Count/Mean/Sum/Minimum/Maximum/
StandardDeviation/ThresholdMean (seven classes) - regionprops_table already
computes count/mean/min/max directly, and accepts extra per-region reduction
functions for the rest via its extra_properties parameter, so one call
replaces six of the seven. ThresholdMean's "mean of the top 25% of values by
intensity range" has no library equivalent and is ported directly.
vtea.objects.measurements.TheAnswer is a joke class (unregistered, returns
the constant 42) and isn't ported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

_COLUMN_RENAME = {
    "label": "object_id",
    "area": "count",
    "intensity_mean": "mean",
    "intensity_min": "min",
    "intensity_max": "max",
}


def threshold_mean(region_mask: np.ndarray, region_intensity: np.ndarray) -> float:
    """Mean of the values in the top quartile of the region's intensity range.

    Ports vtea.objects.measurements.ThresholdMean.getMean() directly - no
    library equivalent exists for this one. Matches skimage's extra_properties
    signature: (bbox-cropped boolean mask, bbox-cropped intensity image).
    """
    values = region_intensity[region_mask]
    if values.size == 0:
        return float("nan")
    cutoff = values.max() - (values.max() - values.min()) / 4
    selected = values[values >= cutoff]
    return float(selected.mean()) if selected.size else float("nan")


def _region_sum(region_mask: np.ndarray, region_intensity: np.ndarray) -> float:
    return float(np.sum(region_intensity[region_mask]))


def _region_stddev(region_mask: np.ndarray, region_intensity: np.ndarray) -> float:
    return float(np.std(region_intensity[region_mask]))


def extract_measurements(labels: np.ndarray, intensity: np.ndarray) -> pd.DataFrame:
    """Per-object measurement table: object_id, count, mean, sum, stddev, min, max, threshold_mean.

    `labels` and `intensity` must be the same shape (any dimensionality).
    """
    if labels.shape != intensity.shape:
        raise ValueError(f"labels shape {labels.shape} != intensity shape {intensity.shape}")

    properties = ["label", "area", "intensity_mean", "intensity_min", "intensity_max"]
    extra_properties = [_region_sum, _region_stddev, threshold_mean]

    table = regionprops_table(
        labels, intensity_image=intensity, properties=properties, extra_properties=extra_properties
    )
    frame = pd.DataFrame(table).rename(columns=_COLUMN_RENAME)
    return frame.rename(columns={"_region_sum": "sum", "_region_stddev": "stddev"})
