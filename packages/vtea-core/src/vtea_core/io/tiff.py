"""TIFF / OME-TIFF / ImageJ-hyperstack I/O, via tifffile.

Ports the TIFF side of vtea.utilities.conversion and the ImagePlus-backed
read paths from the Java codebase.

Two ways in. `read_tiff` reads the file, which is the right thing for the
sizes TIFF is good at. `open_tiff` maps it a plane at a time, so a file
larger than memory can be opened and sliced without being read - the honest
answer for a 40 GB acquisition nobody has converted yet. Converting it (see
io.ome_zarr.ingest) is better still: a TIFF's finest unit is a plane, so
pulling a small cube out of the middle of a stack still costs a full plane
per slice.
"""

from __future__ import annotations

import math
import os

import dask
import dask.array as da
import numpy as np
import tifffile

from vtea_core.data.axes import VOLUME, Axes, to_canonical
from vtea_core.data.volume import ChunkedVolumeDataset, InMemoryVolumeDataset


def read_tiff(path: str | os.PathLike) -> InMemoryVolumeDataset:
    """Reads a TIFF/OME-TIFF/ImageJ hyperstack into a (C, Z, Y, X) InMemoryVolumeDataset."""
    with tifffile.TiffFile(os.fspath(path)) as tif:
        series = tif.series[0]
        array = series.asarray()
        axes = series.axes
    array = _to_czyx(array, axes)
    return InMemoryVolumeDataset(array)


def open_tiff(path: str | os.PathLike, chunks: str | tuple = "auto") -> ChunkedVolumeDataset:
    """Maps a TIFF as a (C, Z, Y, X) ChunkedVolumeDataset without reading it.

    One plane per chunk, read on demand, so a file far larger than memory
    can be opened and sliced. The reordering to canonical axes is Dask's, so
    it stays a description rather than becoming a copy.

    Built directly on `tifffile`'s pages rather than on its `aszarr` store:
    that store needs zarr 3, and VTEA is on zarr 2 (see io.store). Reading
    pages has the happier side effect of keeping zarr out of the TIFF path
    altogether.

    A plane at a time is the finest granularity a TIFF offers here, so
    reading a small cube out of the middle of a large stack still costs a
    full plane per slice. That is a property of the format, not of this
    function, and it is the reason `io.ome_zarr.ingest` exists: convert
    once, and every later access is a chunk rather than a plane.
    """
    with tifffile.TiffFile(os.fspath(path)) as tif:
        series = tif.series[0]
        axes = _check_axes_string(series.axes)
        shape = tuple(int(size) for size in series.shape)
        dtype = np.dtype(series.dtype)
        n_pages = len(series.pages)

    if axes[-2:] != "YX":
        raise NotImplementedError(
            f"lazy TIFF reading needs the last two axes to be Y and X, got {axes!r}. "
            f"Use read_tiff() to read this file eagerly."
        )
    plane_shape, leading = shape[-2:], shape[:-2]
    if math.prod(leading) != n_pages:
        raise NotImplementedError(
            f"this TIFF's {n_pages} pages do not correspond one-to-one with its "
            f"{axes[:-2]!r} axes {leading}, so it cannot be mapped a plane at a time. "
            f"Use read_tiff() to read it eagerly."
        )

    read_plane = dask.delayed(_read_plane, pure=True)
    planes = [
        da.from_delayed(read_plane(os.fspath(path), index), shape=plane_shape, dtype=dtype)
        for index in range(n_pages)
    ]
    stacked = da.stack(planes) if planes else da.zeros((0, *plane_shape), dtype=dtype)
    array = stacked.reshape(shape)
    if chunks != "auto":
        array = array.rechunk(chunks)
    return ChunkedVolumeDataset(_to_czyx_lazy(array, axes))


def _read_plane(path: str, index: int) -> np.ndarray:
    """One page of a TIFF, opened per read.

    Reopening the file each time rather than holding a handle keeps this
    usable from any Dask scheduler - including processes, where an open
    `TiffFile` could not be sent - at the cost of an open per plane, which
    is small beside decoding the plane itself.
    """
    with tifffile.TiffFile(path) as tif:
        return np.asarray(tif.series[0].pages[index].asarray())


def write_tiff(dataset: InMemoryVolumeDataset, path: str | os.PathLike) -> None:
    """Writes a VolumeDataset as an ImageJ-compatible hyperstack TIFF (axes ZCYX)."""
    array = dataset.to_numpy()  # (C, Z, Y, X)
    zcyx = np.transpose(array, (1, 0, 2, 3))
    tifffile.imwrite(os.fspath(path), zcyx, imagej=True, metadata={"axes": "ZCYX"})


def _to_czyx(array: np.ndarray, axes: str) -> np.ndarray:
    """Reorders/pads an arbitrary-axes TIFF array to canonical (C, Z, Y, X).

    The axis bookkeeping itself lives in `vtea_core.data.axes`, which every
    reader shares; what stays here is the one thing that is TIFF's alone -
    sample-interleaved pixels, which are not an axis to be transposed but a
    pixel layout to be decoded, and which nothing downstream handles.
    """
    return np.ascontiguousarray(to_canonical(array, _check_axes(array, axes), target=VOLUME))


def _to_czyx_lazy(array, axes: str):
    """`_to_czyx` for a Dask array: the same reordering, still unevaluated."""
    return to_canonical(array, _check_axes(array, axes), target=VOLUME)


def _check_axes(array, axes: str) -> Axes:
    return Axes(_check_axes_string(axes))


def _check_axes_string(axes: str) -> str:
    axes = axes.upper()
    if "S" in axes:
        raise NotImplementedError(
            f"sample-interleaved (e.g. RGB) TIFF axes are not supported: {axes!r}"
        )
    if not {"Y", "X"} <= set(axes):
        raise ValueError(f"expected TIFF axes to include Y and X, got {axes!r}")
    return axes
