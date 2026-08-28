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
    """Per-object measurement table: object_id, centroid-0..N, count, mean, sum,
    stddev, min, max, threshold_mean.

    `labels` and `intensity` must be the same shape (any dimensionality). Centroid
    columns follow the array's own axis order (e.g. centroid-0/1/2 = Z/Y/X for a
    3D label array) - used for plot axes and to locate objects for gallery crops.
    """
    if labels.shape != intensity.shape:
        raise ValueError(f"labels shape {labels.shape} != intensity shape {intensity.shape}")

    properties = ["label", "centroid", "area", "intensity_mean", "intensity_min", "intensity_max"]
    extra_properties = [_region_sum, _region_stddev, threshold_mean]

    table = regionprops_table(
        labels, intensity_image=intensity, properties=properties, extra_properties=extra_properties
    )
    frame = pd.DataFrame(table).rename(columns=_COLUMN_RENAME)
    return frame.rename(columns={"_region_sum": "sum", "_region_stddev": "stddev"})


# Columns describing an object's geometry rather than its brightness. They
# are identical for every channel, so a multi-channel table carries them
# once instead of repeating them per channel.
GEOMETRY_COLUMNS = ("object_id", "count")

# Columns that identify or locate an object rather than describe it. Feeding
# a centroid into PCA or k-means clusters objects by where they sit in the
# field of view, which is nearly never what's wanted, and an object_id is
# just a row number - so feature_matrix() leaves both out by default.
NON_FEATURE_COLUMNS = ("object_id",)
NON_FEATURE_PREFIXES = ("centroid-",)


def is_feature_column(name: str) -> bool:
    return name not in NON_FEATURE_COLUMNS and not name.startswith(NON_FEATURE_PREFIXES)


def feature_matrix(
    frame: pd.DataFrame, columns: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """The measurement table as the (n_objects, n_features) float array that
    clustering and dimensionality reduction take as `data`, plus the names of
    the columns it was built from.

    Without this the analysis steps have no way to be run from the GUI at
    all: they declare a `data` input and nothing in the protocol produces
    one. Non-numeric columns, identifiers and centroids are dropped (see
    NON_FEATURE_COLUMNS); NaNs - a measurement can legitimately produce one
    for an empty region - become 0.0, since scikit-learn refuses to fit on
    them.
    """
    if columns is None:
        columns = [
            name
            for name in frame.columns
            if is_feature_column(name) and pd.api.types.is_numeric_dtype(frame[name])
        ]
    if not columns:
        return np.empty((len(frame), 0), dtype=float), []
    matrix = frame.loc[:, columns].to_numpy(dtype=float)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), list(columns)


def extract_measurements_by_channel(
    labels: np.ndarray,
    intensity: np.ndarray,
    *,
    channel_axis: int | None = None,
    channel: int | None = None,
) -> pd.DataFrame:
    """Measure one segmentation against every channel, as one flat table.

    Objects come from `labels` (a single segmentation), intensities from
    `intensity` (which may still carry a channel axis). Every intensity column
    is suffixed with the channel it was measured on - `mean_ch0`, `mean_ch2`
    - so features from different channels coexist in one table and can be
    told apart when picking plot axes. Geometry columns (object_id, count,
    centroid-*) describe the object itself and appear once, unsuffixed.

    `channel_axis=None`, or an `intensity` already matching `labels`, measures a
    single channel and produces unsuffixed names. `channel` restricts the
    measurement to one channel instead of all of them.
    """
    if channel_axis is None or intensity.ndim == labels.ndim:
        return extract_measurements(labels, intensity)

    if not -intensity.ndim <= channel_axis < intensity.ndim:
        raise ValueError(
            f"channel axis {channel_axis} is out of range for a volume of shape {intensity.shape}"
        )

    n_channels = intensity.shape[channel_axis]
    wanted = range(n_channels) if channel is None else [channel]
    for index in wanted:
        if not 0 <= index < n_channels:
            raise ValueError(
                f"channel {index} is out of range - axis {channel_axis} has {n_channels} channel(s)"
            )

    merged: pd.DataFrame | None = None
    for index in wanted:
        single = np.take(intensity, index, axis=channel_axis)
        table = extract_measurements(labels, single)
        if merged is None:
            merged = table.rename(
                columns={
                    name: f"{name}_ch{index}"
                    for name in table.columns
                    if name not in GEOMETRY_COLUMNS and not name.startswith("centroid-")
                }
            )
            continue
        intensity_columns = [
            name
            for name in table.columns
            if name not in GEOMETRY_COLUMNS and not name.startswith("centroid-")
        ]
        renamed = table[intensity_columns].rename(
            columns={name: f"{name}_ch{index}" for name in intensity_columns}
        )
        merged = pd.concat([merged, renamed], axis=1)
    return merged
