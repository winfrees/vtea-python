"""OME-NGFF 0.4: the format VTEA writes large data into.

Why this and not "a Zarr array of the pixels": a bare array records a shape
and a dtype, which is not enough to open it again. Which axis is z, how big
a voxel is, and which arrays are downsampled copies of which are exactly
the facts an analysis depends on and that get lost between a microscope and
a collaborator. NGFF has an agreed answer for all three, and other tools
read it.

Two decisions show up in every function here, both from
docs/LARGE_IMAGES.md:

- **Five axes, always.** A store is written `TCZYX` with a time axis of
  length one, even though VTEA analyses one timepoint. It costs an axis of
  length one; the alternative is converting every store written before time
  support lands. Reading squeezes it back out, and a store with more than
  one timepoint raises `TimeSeriesNotSupported` rather than quietly
  analysing the first frame.
- **A pyramid, always.** Coarse levels are what make a 33 GB volume
  openable in a viewer at all, and they are nearly free to write while the
  data is already streaming past.

The metadata is written here rather than through `ome-zarr-py` so that the
zarr dependency stays inside `io.store` and the spec version VTEA writes
stays a constant this module owns. `ome-zarr-py` reading what this writes
is a test (`test_io_ome_zarr.py`), which is the part that actually proves
interoperability.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np

from vtea_core.data.axes import CANONICAL, VOLUME, Axes, to_canonical
from vtea_core.data.spacing import FROM_METADATA, Spacing
from vtea_core.data.volume import ChunkedVolumeDataset, VolumeDataset
from vtea_core.io import store

# How VTEA's unit names map to the UDUNITS-2 names NGFF requires. Written
# out rather than guessed: "um" is not a UDUNITS unit, and a reader that
# does not recognise a unit is entitled to ignore the scale entirely.
_UNIT_TO_NGFF = {
    "µm": "micrometer",
    "um": "micrometer",
    "micron": "micrometer",
    "microns": "micrometer",
    "micrometer": "micrometer",
    "nm": "nanometer",
    "nanometer": "nanometer",
    "mm": "millimeter",
    "millimeter": "millimeter",
    "m": "meter",
    "meter": "meter",
}
_NGFF_TO_UNIT = {
    "micrometer": "µm",
    "nanometer": "nm",
    "millimeter": "mm",
    "meter": "m",
}

# Stop halving once the largest spatial axis is this small: below it a level
# is cheaper to render than to fetch, and the levels stop earning their
# storage.
MIN_PYRAMID_EXTENT = 256

# However few levels that implies, never write more than this - a very
# anisotropic volume would otherwise generate levels that only shrink one
# axis, forever.
MAX_PYRAMID_LEVELS = 6

# How a coarse level is made. "mean" for intensity; "nearest" for a label
# image, where averaging two ids would invent a third.
MEAN = "mean"
NEAREST = "nearest"


@dataclass(frozen=True)
class MultiscaleInfo:
    """What a store says about itself, before any pixels are read."""

    axes: Axes
    shape: tuple[int, ...]
    dtype: np.dtype
    levels: tuple[str, ...]
    scales: tuple[tuple[float, ...], ...]
    spacing: Spacing | None
    ngff_version: str
    name: str = ""

    @property
    def n_timepoints(self) -> int:
        index = self.axes.index_of("T")
        return 1 if index < 0 else int(self.shape[index])

    @property
    def n_channels(self) -> int:
        index = self.axes.index_of("C")
        return 1 if index < 0 else int(self.shape[index])

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def describe(self) -> str:
        parts = [self.axes.describe(self.shape), f"{self.dtype}", f"{self.n_levels} levels"]
        if self.spacing is not None and self.spacing.is_known:
            parts.append(f"voxel {self.spacing.describe()}")
        return ", ".join(parts)


def write_ome_zarr(
    dataset: VolumeDataset | np.ndarray | da.Array,
    path: str | os.PathLike,
    *,
    spacing: Spacing | None = None,
    levels: int | None = None,
    chunks: tuple[int, ...] | None = None,
    compressor_name: str | None = store.DEFAULT_COMPRESSOR,
    reduction: str = MEAN,
    name: str = "",
    overwrite: bool = True,
) -> Path:
    """Write a (C, Z, Y, X) volume as a multiscale OME-NGFF 0.4 store.

    Streams: a Dask-backed dataset is never materialized, and each pyramid
    level is written a chunk at a time. `levels=None` picks enough of them
    to bring the largest spatial axis under `MIN_PYRAMID_EXTENT`.

    `reduction` decides how a coarse level is made. `MEAN` for intensity;
    `NEAREST` for a label image, where the average of ids 4 and 6 would be
    a nonexistent object 5.
    """
    array = _as_dask(dataset)
    if array.ndim != 4:
        raise ValueError(f"expected a 4D (C, Z, Y, X) array, got shape {array.shape}")
    # Five axes on disk, four in memory - see this module's docstring.
    full = to_canonical(array, VOLUME, target=CANONICAL)
    axes = Axes(CANONICAL)

    path = Path(os.fspath(path))
    chunks = tuple(chunks) if chunks else store.default_chunks(full.shape, axes.order)
    pyramid = _pyramid(full, axes, levels=levels, reduction=reduction)

    group = store.create_group(path, overwrite=overwrite)
    scales = []
    for index, level in enumerate(pyramid):
        target = store.create_array(
            group,
            str(index),
            shape=level.shape,
            dtype=level.dtype,
            chunks=_fit_chunks(chunks, level.shape),
            compressor_name=compressor_name,
        )
        store.store_dask(level, target)
        scales.append(_scale_for(level.shape, full.shape, axes, spacing))

    store.set_attrs(
        group,
        {
            "multiscales": [
                {
                    "version": store.NGFF_VERSION,
                    "name": name or path.stem,
                    "axes": _axes_metadata(axes, spacing),
                    "datasets": [
                        {
                            "path": str(index),
                            "coordinateTransformations": [{"type": "scale", "scale": scale}],
                        }
                        for index, scale in enumerate(scales)
                    ],
                }
            ]
        },
    )
    return path


def read_ome_zarr(
    path: str | os.PathLike, *, level: int = 0, chunks: str | tuple = "auto"
) -> ChunkedVolumeDataset:
    """Open one level of a multiscale store as a (C, Z, Y, X) dataset.

    Lazy: nothing is read until something asks for voxels. `level=0` is
    full resolution; a higher level is the cheap way to draw a thumbnail or
    estimate a background.
    """
    info = read_info(path)
    if not 0 <= level < info.n_levels:
        raise IndexError(
            f"level {level} is out of range - this store has {info.n_levels} "
            f"(0 is full resolution)"
        )
    array = store.read_dask(path, component=info.levels[level], chunks=chunks)
    # Squeezes the time axis, and refuses a real time series rather than
    # silently analysing its first frame.
    return ChunkedVolumeDataset(to_canonical(array, info.axes, target=VOLUME))


def read_info(path: str | os.PathLike) -> MultiscaleInfo:
    """What the store says about itself, without reading any pixels."""
    try:
        group = store.open_group(path, mode="r")
    except Exception as exc:  # a bare array, or not a zarr store at all
        raise ValueError(
            f"{path} is not an OME-NGFF multiscale image - it is not even a Zarr group. "
            f"Read it with read_zarr() if it is a plain array ({exc})."
        ) from exc
    attrs = store.group_attrs(group)
    multiscales = attrs.get("multiscales")
    if not multiscales:
        raise ValueError(
            f"{path} is a Zarr group but not an OME-NGFF multiscale image - it has no "
            f"'multiscales' metadata. Read it with read_zarr() if it is a plain array."
        )
    entry = multiscales[0]
    version = store.check_ngff_version(entry.get("version"))
    axes = _axes_from_metadata(entry, group)
    datasets = entry["datasets"]
    paths = tuple(str(item["path"]) for item in datasets)
    scales = tuple(_scale_of(item, axes.ndim) for item in datasets)
    level0 = group[paths[0]]
    return MultiscaleInfo(
        axes=axes,
        shape=tuple(int(size) for size in level0.shape),
        dtype=np.dtype(level0.dtype),
        levels=paths,
        scales=scales,
        spacing=_spacing_from(entry, axes, scales[0]),
        ngff_version=version,
        name=entry.get("name", ""),
    )


def ingest(
    source: str | os.PathLike,
    destination: str | os.PathLike,
    *,
    spacing: Spacing | None = None,
    levels: int | None = None,
    chunks: tuple[int, ...] | None = None,
    compressor_name: str | None = store.DEFAULT_COMPRESSOR,
    reduction: str = MEAN,
    overwrite: bool = True,
) -> Path:
    """Convert any readable volume into a chunked, pyramidal OME-Zarr.

    The honest answer to "my data is a 40 GB TIFF". A TIFF's own chunks are
    its strips or tiles, which are laid out for reading the file in order;
    pulling a cube out of the middle of one means touching most of the
    file. Converting once, up front, is what makes every later step's
    access pattern cheap - and it is a streaming read-and-write, so it does
    not need the memory the analysis was short of in the first place.
    """
    from vtea_core.io import open_volume

    dataset = open_volume(source, lazy=True)
    return write_ome_zarr(
        dataset,
        destination,
        spacing=spacing,
        levels=levels,
        chunks=chunks,
        compressor_name=compressor_name,
        reduction=reduction,
        name=Path(os.fspath(source)).stem,
        overwrite=overwrite,
    )


# -- pyramid ------------------------------------------------------------


def pyramid_levels(shape: tuple[int, ...], axes: Axes) -> int:
    """How many levels a volume of this shape gets.

    Halved until the largest spatial axis is under `MIN_PYRAMID_EXTENT`, and
    an axis stops halving once it reaches 1 - so a 24-slice slab keeps its
    24 slices while its 2048-pixel axes come down, instead of losing z
    entirely by level three.
    """
    spatial = [shape[index] for index in axes.spatial_indices]
    count = 1
    extents = list(spatial)
    while max(extents) > MIN_PYRAMID_EXTENT and count < MAX_PYRAMID_LEVELS:
        extents = [max(1, extent // 2) for extent in extents]
        count += 1
    return count


def _pyramid(array: da.Array, axes: Axes, *, levels: int | None, reduction: str) -> list[da.Array]:
    if reduction not in (MEAN, NEAREST):
        raise ValueError(f"unknown reduction {reduction!r}, expected {MEAN!r} or {NEAREST!r}")
    count = pyramid_levels(array.shape, axes) if levels is None else int(levels)
    if count < 1:
        raise ValueError(f"a store needs at least one level, got {levels}")

    result = [array]
    for _ in range(count - 1):
        current = result[-1]
        factors = {
            index: 2
            for index in axes.spatial_indices
            if current.shape[index] > 1 and current.shape[index] > MIN_PYRAMID_EXTENT // 2
        }
        if not factors:
            break
        result.append(_downsample(current, factors, reduction))
    return result


def _downsample(array: da.Array, factors: dict[int, int], reduction: str) -> da.Array:
    if reduction == NEAREST:
        # Striding, not averaging: the mean of label 4 and label 6 is an
        # object that does not exist.
        picks = tuple(
            slice(None, None, factors.get(axis, 1)) for axis in range(array.ndim)
        )
        return array[picks]
    coarse = da.coarsen(np.mean, array, factors, trim_excess=True)
    return coarse.astype(array.dtype)


def _fit_chunks(chunks: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    """Chunks capped at the array's own extent - a coarse level can be
    smaller than one chunk of the level above it."""
    return tuple(max(1, min(chunk, extent)) for chunk, extent in zip(chunks, shape))


def _as_dask(dataset: VolumeDataset | np.ndarray | da.Array) -> da.Array:
    if isinstance(dataset, VolumeDataset):
        array = dataset.array
    else:
        array = dataset
    if isinstance(array, da.Array):
        return array
    return da.from_array(array, chunks="auto")


# -- metadata -----------------------------------------------------------


def _axes_metadata(axes: Axes, spacing: Spacing | None) -> list[dict[str, Any]]:
    unit = _ngff_unit(spacing)
    entries = []
    for axis, kind in zip(axes.order, axes.types()):
        entry: dict[str, Any] = {"name": axis.lower(), "type": kind}
        if kind == "space" and unit:
            entry["unit"] = unit
        elif kind == "time":
            entry["unit"] = "second"
        entries.append(entry)
    return entries


def _axes_from_metadata(entry: dict[str, Any], group: Any) -> Axes:
    axes = entry.get("axes")
    if axes:
        return Axes("".join(str(item["name"]).upper() for item in axes))
    # NGFF 0.3 and earlier left the axes implicit, positional TCZYX.
    from vtea_core.data.axes import canonical_for

    level0 = group[str(entry["datasets"][0]["path"])]
    return canonical_for(level0.ndim)


def _ngff_unit(spacing: Spacing | None) -> str | None:
    if spacing is None or not spacing.is_known:
        return None
    return _UNIT_TO_NGFF.get(spacing.unit.strip().lower() if spacing.unit else "", None) or (
        _UNIT_TO_NGFF.get(spacing.unit, None)
    )


def _scale_for(
    shape: tuple[int, ...],
    full_shape: tuple[int, ...],
    axes: Axes,
    spacing: Spacing | None,
) -> list[float]:
    """This level's voxel size, in the store's own units.

    A coarse level's voxels are physically larger, by exactly the factor it
    was downsampled by - which is how a viewer knows the levels describe the
    same specimen rather than five differently sized ones. Without a known
    spacing the scale is that factor alone, which is what "we were not told
    the voxel size" honestly looks like.
    """
    sizes = _voxel_sizes(axes, spacing)
    return [
        size * (full / max(current, 1))
        for size, full, current in zip(sizes, full_shape, shape)
    ]


def _voxel_sizes(axes: Axes, spacing: Spacing | None) -> list[float]:
    if spacing is None or not spacing.is_known:
        return [1.0] * axes.ndim
    spatial = spacing.for_ndim(len(axes.spatial))
    sizes, position = [], 0
    for axis in axes.order:
        if axis in "ZYX":
            sizes.append(float(spatial[position]))
            position += 1
        else:
            sizes.append(1.0)
    return sizes


def _scale_of(dataset: dict[str, Any], ndim: int) -> tuple[float, ...]:
    for transformation in dataset.get("coordinateTransformations", []):
        if transformation.get("type") == "scale":
            return tuple(float(value) for value in transformation["scale"])
    return (1.0,) * ndim


def _spacing_from(
    entry: dict[str, Any], axes: Axes, scale: tuple[float, ...]
) -> Spacing | None:
    """The voxel size a store records, or None when it records none.

    An all-ones scale is the NGFF spelling of "nobody said", exactly as an
    all-ones napari `layer.scale` is - and `Spacing` exists to keep those
    distinguishable from a genuine one-micron isotropic acquisition, so this
    returns None rather than a confident 1.0.
    """
    metadata = {str(item["name"]).upper(): item for item in entry.get("axes", [])}
    spatial = [scale[index] for index in axes.spatial_indices]
    if not spatial or all(value == 1.0 for value in spatial):
        return None
    units = {
        _NGFF_TO_UNIT.get(str(metadata.get(axis, {}).get("unit", "")).lower())
        for axis in axes.spatial
    }
    units.discard(None)
    unit = units.pop() if len(units) == 1 else Spacing((1.0,)).unit
    return Spacing(tuple(float(value) for value in spatial), unit=unit, source=FROM_METADATA)
