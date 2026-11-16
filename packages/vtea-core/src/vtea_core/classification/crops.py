"""Fixed-size crops centred on each object - what an image model is fed.

A per-object model (the VAE, the CNN) sees an object as a small cube of
image around it rather than as a row of measurements. This cuts those
cubes out, one per object of a label image, in the same order the
measurement table lists the objects (ascending id), so whatever the model
returns lines up with the table row for row.

Follows the Java `CellRegionExtractor` the VAE plugins used: a cube of
`size` voxels a side centred on the object's centroid, edge voxels repeated
where the cube runs off the image ("REPLICATE" padding), and each crop
z-scored on its own so that a model learns shape and texture rather than
how bright the field happened to be. No torch needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy import ndimage as ndi

CROP_NORMALIZATIONS = ("zscore", "minmax", "none")


def object_ids(labels: np.ndarray) -> np.ndarray:
    """The ids of the objects in `labels`, ascending - measurement-table order."""
    ids = np.unique(labels)
    return ids[ids != 0].astype(np.int64)


def _channels_first(
    intensity: np.ndarray, labels: np.ndarray, channel_axis: int | None, channel: int | None
) -> np.ndarray:
    """The intensity image as (C, *spatial), matching `labels`' spatial shape."""
    intensity = np.asarray(intensity)
    if intensity.ndim == labels.ndim:
        if intensity.shape != labels.shape:
            raise ValueError(f"intensity shape {intensity.shape} != labels shape {labels.shape}")
        return intensity[np.newaxis]
    if channel_axis is None:
        raise ValueError(
            f"the intensity image has {intensity.ndim} axes and the labels {labels.ndim}; "
            f"say which axis is the channel axis"
        )
    stacked = np.moveaxis(intensity, channel_axis, 0)
    if stacked.shape[1:] != labels.shape:
        raise ValueError(
            f"intensity spatial shape {stacked.shape[1:]} != labels shape {labels.shape}"
        )
    if channel is not None:
        if not 0 <= channel < stacked.shape[0]:
            raise ValueError(f"channel {channel} is out of range - there are {stacked.shape[0]}")
        return stacked[channel : channel + 1]
    return stacked


def _normalize_crop(crop: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return crop
    if method == "minmax":
        low, high = crop.min(), crop.max()
        return (crop - low) / (high - low) if high > low else np.zeros_like(crop)
    mean, std = crop.mean(), crop.std()
    return (crop - mean) / std if std > 0 else np.zeros_like(crop)


def _crop_shape(size, ndim: int) -> tuple[int, ...]:
    shape = (int(size),) * ndim if np.isscalar(size) else tuple(int(v) for v in size)
    if len(shape) != ndim:
        raise ValueError(f"crop size {size} has {len(shape)} axes; the image has {ndim}")
    if min(shape) < 1:
        raise ValueError(f"crop size must be at least 1, got {size}")
    return shape


def extract_crops(
    labels: np.ndarray,
    intensity: np.ndarray,
    *,
    size: int | Sequence[int] = 32,
    channel_axis: int | None = None,
    channel: int | None = None,
    ids: np.ndarray | None = None,
    normalize: Literal["zscore", "minmax", "none"] = "zscore",
    mask_outside: bool = False,
    spacing: Sequence[float] | None = None,
    target_spacing: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(crops, ids): one crop per object, float32, shaped (n_objects,
    n_channels, *size).

    `size` is one number (a cube) or one per axis, e.g. (7, 32, 32) for
    NephNet3D's Z x Y x X crops. `ids` restricts and orders the objects
    (default: every object, ascending). `channel` picks one channel of a
    multi-channel image; None keeps them all as the crop's channels.
    `normalize` is applied per crop and per channel. `mask_outside` zeroes
    whatever is not the object itself - for a model that should see the
    object's shape and not its neighbours'.

    **Voxel size**: with `spacing` (this image's voxel size, per axis) and
    `target_spacing` (the voxel size a model was trained at), each crop
    covers the same physical extent as the model's training crops - a window
    of size x target / spacing voxels around the object - and is resampled
    (linearly) to `size`. A model trained at one Z step then sees nuclei the
    same size at another.
    """
    labels = np.asarray(labels)
    if normalize not in CROP_NORMALIZATIONS:
        raise ValueError(
            f"unknown crop normalization {normalize!r}, expected {CROP_NORMALIZATIONS}"
        )
    ndim = labels.ndim
    shape = _crop_shape(size, ndim)
    window = shape
    if spacing is not None and target_spacing is not None:
        ratio = np.asarray(target_spacing, float)[-ndim:] / np.asarray(spacing, float)[-ndim:]
        window = tuple(max(1, round(n * r)) for n, r in zip(shape, ratio))
    image = _channels_first(intensity, labels, channel_axis, channel).astype(np.float32, copy=False)
    wanted = object_ids(labels) if ids is None else np.asarray(ids, dtype=np.int64)
    crops = np.zeros((len(wanted), image.shape[0], *shape), dtype=np.float32)
    if len(wanted) == 0:
        return crops, wanted

    centres = ndi.center_of_mass(np.ones(labels.shape), labels, wanted)
    for row, (object_id, centre) in enumerate(zip(wanted, centres)):
        if np.any(np.isnan(centre)):
            continue  # not in the image; left as zeros rather than guessed
        start = [int(np.round(value)) - n // 2 for value, n in zip(centre, window)]
        stop = [begin + n for begin, n in zip(start, window)]
        clipped = tuple(
            slice(max(begin, 0), min(end, extent))
            for begin, end, extent in zip(start, stop, labels.shape)
        )
        pads = [
            (max(0, -begin), max(0, end - extent))
            for begin, end, extent in zip(start, stop, labels.shape)
        ]
        region = image[(slice(None),) + clipped]
        if mask_outside:
            region = region * (labels[clipped] == object_id)
        region = np.pad(region, [(0, 0)] + pads, mode="edge")
        if window != shape:
            factors = [n / w for n, w in zip(shape, window)]
            region = np.stack([_zoom_to(region[c], factors, shape) for c in range(region.shape[0])])
        for c in range(region.shape[0]):
            crops[row, c] = _normalize_crop(region[c], normalize)
    return crops, wanted


def _zoom_to(array: np.ndarray, factors, shape) -> np.ndarray:
    """`array` zoomed by `factors` (linear), trimmed or edge-padded to exactly `shape`."""
    out = ndi.zoom(array, factors, order=1, mode="nearest", grid_mode=True)
    out = out[tuple(slice(0, n) for n in shape)]
    pads = [(0, n - m) for n, m in zip(shape, out.shape)]
    return np.pad(out, pads, mode="edge") if any(p for _, p in pads) else out
