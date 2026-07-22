"""TIFF / OME-TIFF / ImageJ-hyperstack I/O, via tifffile.

Ports the TIFF side of vtea.utilities.conversion and the ImagePlus-backed
read paths from the Java codebase.
"""

from __future__ import annotations

import os

import numpy as np
import tifffile

from vtea_core.data.volume import InMemoryVolumeDataset


def read_tiff(path: str | os.PathLike) -> InMemoryVolumeDataset:
    """Reads a TIFF/OME-TIFF/ImageJ hyperstack into a (C, Z, Y, X) InMemoryVolumeDataset."""
    with tifffile.TiffFile(os.fspath(path)) as tif:
        series = tif.series[0]
        array = series.asarray()
        axes = series.axes
    array = _to_czyx(array, axes)
    return InMemoryVolumeDataset(array)


def write_tiff(dataset: InMemoryVolumeDataset, path: str | os.PathLike) -> None:
    """Writes a VolumeDataset as an ImageJ-compatible hyperstack TIFF (axes ZCYX)."""
    array = dataset.to_numpy()  # (C, Z, Y, X)
    zcyx = np.transpose(array, (1, 0, 2, 3))
    tifffile.imwrite(os.fspath(path), zcyx, imagej=True, metadata={"axes": "ZCYX"})


def _to_czyx(array: np.ndarray, axes: str) -> np.ndarray:
    """Reorders/pads an arbitrary-axes TIFF array to canonical (C, Z, Y, X)."""
    axes = axes.upper()

    if "T" in axes:
        t_index = axes.index("T")
        if array.shape[t_index] != 1:
            raise NotImplementedError(f"time series TIFFs are not supported yet (axes={axes!r})")
        array = np.take(array, 0, axis=t_index)
        axes = axes[:t_index] + axes[t_index + 1 :]

    if "S" in axes:
        raise NotImplementedError(f"sample-interleaved (e.g. RGB) TIFF axes are not supported: {axes!r}")

    if not {"Y", "X"} <= set(axes):
        raise ValueError(f"expected TIFF axes to include Y and X, got {axes!r}")

    if "C" not in axes:
        array = array[np.newaxis, ...]
        axes = "C" + axes

    if "Z" not in axes:
        c_index = axes.index("C")
        array = np.expand_dims(array, axis=c_index + 1)
        axes = axes[: c_index + 1] + "Z" + axes[c_index + 1 :]

    if set(axes) != {"C", "Z", "Y", "X"}:
        raise ValueError(f"unsupported TIFF axes {axes!r}, expected some combination of C, Z, Y, X, T")

    order = [axes.index(a) for a in "CZYX"]
    return np.ascontiguousarray(np.transpose(array, order))
