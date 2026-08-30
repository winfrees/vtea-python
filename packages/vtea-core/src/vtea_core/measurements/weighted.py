"""Measuring objects whose edges are probabilities rather than facts.

With a soft ownership (see `vtea_core.objects.ownership`), a voxel belongs
to a cell with probability p, and every summary statistic has to change to
match. A count becomes an expected volume, the sum of those probabilities; a
mean becomes a probability-weighted mean, so a voxel a cell only half owns
contributes half as much to its brightness.

`regionprops_table` cannot express this - it takes a hard label image - so
this is its own reducer rather than a parameter. The column names are the
same as the unweighted table's on purpose: a protocol can swap a hard
measurement for a weighted one without every plot axis and gate changing its
name underneath.

What that buys is the thing soft ownership is for. A cell whose boundary is
genuinely ambiguous gets a brightness that reflects the ambiguity instead of
one that depends on where a watershed happened to draw the line, and the
same cell measured with a slightly different falloff moves a little rather
than jumping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vtea_core.data.spacing import Spacing
from vtea_core.measurements.regionprops import GEOMETRY_COLUMNS, VOLUME_COLUMN
from vtea_core.objects.ownership import Ownership


def _accumulate(ownership: Ownership, intensity: np.ndarray):
    """Per-owner sums over the voxels each owner has any claim on.

    Restricted to owned voxels rather than swept over the whole array, so
    the cost follows the objects rather than the volume - which for a field
    that is mostly background is the difference between usable and not.
    """
    ids = ownership.object_ids()
    if not ids:
        return ids, {}

    size = max(ids) + 1
    weight = np.zeros(size)
    value = np.zeros(size)
    square = np.zeros(size)
    lowest = np.full(size, np.inf)
    highest = np.full(size, -np.inf)
    centroids = [np.zeros(size) for _ in ownership.shape]

    flat_intensity = intensity.ravel()
    for slot in range(ownership.top_k):
        owners = ownership.owners[slot].ravel()
        probabilities = ownership.probabilities[slot].ravel()
        where = np.flatnonzero((owners != 0) & (probabilities > 0))
        if where.size == 0:
            continue
        claimed = owners[where]
        p = probabilities[where]
        values = flat_intensity[where]

        weight += np.bincount(claimed, weights=p, minlength=size)
        value += np.bincount(claimed, weights=p * values, minlength=size)
        square += np.bincount(claimed, weights=p * values * values, minlength=size)
        np.minimum.at(lowest, claimed, values)
        np.maximum.at(highest, claimed, values)
        for axis, coordinate in enumerate(np.unravel_index(where, ownership.shape)):
            centroids[axis] += np.bincount(claimed, weights=p * coordinate, minlength=size)

    return ids, {
        "weight": weight,
        "value": value,
        "square": square,
        "min": lowest,
        "max": highest,
        "centroids": centroids,
    }


def weighted_measurements(
    ownership: Ownership,
    intensity: np.ndarray,
    *,
    spacing: Spacing | None = None,
) -> pd.DataFrame:
    """Per-owner measurements over a probabilistic ownership.

    `count` is the expected voxel count - the sum of this owner's
    probabilities - so a cell that half-owns twenty contested voxels counts
    ten of them, and `volume` scales it into physical units where the voxel
    size is known. `mean`, `sum` and `stddev` are probability-weighted, and
    the centroid is the weighted centre of mass.

    `min` and `max` are taken over every voxel the owner has any claim on,
    weighted by nothing: an extreme is an extreme, and scaling it by a
    probability would report a value that occurs nowhere in the image.
    """
    if ownership.shape != intensity.shape:
        raise ValueError(f"shapes differ: {ownership.shape} != {intensity.shape}")

    ids, totals = _accumulate(ownership, intensity)
    if not ids:
        columns = ["object_id", "count", "mean", "sum", "stddev", "min", "max"]
        return pd.DataFrame({name: pd.Series(dtype=float) for name in columns})

    index = np.array(ids)
    weight = totals["weight"][index]
    safe = np.where(weight > 0, weight, np.nan)
    mean = totals["value"][index] / safe
    variance = np.maximum(totals["square"][index] / safe - mean * mean, 0.0)

    frame = pd.DataFrame({"object_id": index})
    for axis, centroid in enumerate(totals["centroids"]):
        frame[f"centroid-{axis}"] = centroid[index] / safe
    frame["count"] = weight
    if spacing is not None and spacing.is_known:
        voxel_volume = float(np.prod(spacing.for_ndim(len(ownership.shape))))
        frame[VOLUME_COLUMN] = weight * voxel_volume
    frame["mean"] = mean
    frame["sum"] = totals["value"][index]
    frame["stddev"] = np.sqrt(variance)
    frame["min"] = totals["min"][index]
    frame["max"] = totals["max"][index]
    return frame


def weighted_measurements_by_channel(
    ownership: Ownership,
    intensity: np.ndarray,
    *,
    channel_axis: int | None = None,
    channel: int | None = None,
    spacing: Spacing | None = None,
) -> pd.DataFrame:
    """The weighted measurements against every channel, as one flat table.

    The same shape of answer `extract_measurements_by_channel` gives, with
    intensity columns suffixed by the channel they were measured on, so a
    weighted table drops into the same plots, gates and clustering steps as
    a hard one.
    """
    if channel_axis is None or intensity.ndim == len(ownership.shape):
        return weighted_measurements(ownership, intensity, spacing=spacing)

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
        table = weighted_measurements(ownership, single, spacing=spacing)
        per_channel = [
            name
            for name in table.columns
            if name not in GEOMETRY_COLUMNS and not name.startswith("centroid-")
        ]
        renamed = {name: f"{name}_ch{index}" for name in per_channel}
        if merged is None:
            merged = table.rename(columns=renamed)
            continue
        merged = pd.concat([merged, table[per_channel].rename(columns=renamed)], axis=1)
    return merged
