"""Image I/O: TIFF/OME-TIFF (tifffile), Zarr and OME-NGFF, proprietary formats (bioio).

Ports vtea.io.zarr and vtea.utilities.conversion from the Java codebase.
Proprietary vendor format support (bioio) is not implemented yet - see
PORT_PLAN.md's dependency mapping table.

Every reader returns a (C, Z, Y, X) `VolumeDataset`, eagerly or lazily.
`open_volume(path)` reads; `open_volume(path, lazy=True)` maps, which is the
only option once a file is larger than memory. `ingest` converts anything
readable into a chunked, pyramidal OME-Zarr, which is the format the rest of
the large-data work assumes - see docs/LARGE_IMAGES.md.
"""

from __future__ import annotations

import os
from pathlib import Path

from vtea_core.data.volume import VolumeDataset
from vtea_core.io.ome_zarr import (
    MultiscaleInfo,
    ingest,
    read_info,
    read_ome_zarr,
    write_ome_zarr,
)
from vtea_core.io.store import is_zarr
from vtea_core.io.tiff import open_tiff, read_tiff, write_tiff
from vtea_core.io.zarr_io import read_zarr, write_zarr

__all__ = [
    "MultiscaleInfo",
    "ingest",
    "is_ome_zarr",
    "open_tiff",
    "open_volume",
    "read_info",
    "read_ome_zarr",
    "read_tiff",
    "read_zarr",
    "write_ome_zarr",
    "write_tiff",
    "write_zarr",
]

_TIFF_SUFFIXES = (".tif", ".tiff")


def is_ome_zarr(path: str | os.PathLike) -> bool:
    """Whether `path` is a multiscale OME-NGFF image rather than a bare array."""
    if not is_zarr(path):
        return False
    try:
        read_info(path)
    except Exception:  # noqa: BLE001 - "not an NGFF image" is the answer, not an error
        return False
    return True


def open_volume(path: str | os.PathLike, *, lazy: bool = False) -> VolumeDataset:
    """Open a volume, dispatching on the path's format.

    `lazy=True` maps the file instead of reading it, giving a Dask-backed
    dataset that can be far larger than memory. It is not the default
    because the eager path is faster for data that fits and is what every
    existing caller expects; a caller that might be handed 40 GB should
    pass it.
    """
    path = Path(os.fspath(path))
    suffix = path.suffix.lower()
    if suffix in _TIFF_SUFFIXES:
        return open_tiff(path) if lazy else read_tiff(path)
    if is_zarr(path):
        if is_ome_zarr(path):
            return read_ome_zarr(path)
        return read_zarr(path)
    raise ValueError(f"unrecognized volume format: {path}")
