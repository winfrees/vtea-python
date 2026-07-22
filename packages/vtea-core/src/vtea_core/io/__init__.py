"""Image I/O: TIFF/OME-TIFF (tifffile), Zarr (zarr-python), proprietary formats (bioio).

Ports vtea.io.zarr and vtea.utilities.conversion from the Java codebase.
Proprietary vendor format support (bioio) is not implemented yet - see
PORT_PLAN.md's dependency mapping table.
"""

from __future__ import annotations

import os
from pathlib import Path

from vtea_core.data.volume import VolumeDataset
from vtea_core.io.tiff import read_tiff, write_tiff
from vtea_core.io.zarr_io import read_zarr, write_zarr

__all__ = ["read_tiff", "write_tiff", "read_zarr", "write_zarr", "open_volume"]

_ZARR_MARKERS = (".zarray", ".zgroup", "zarr.json")


def open_volume(path: str | os.PathLike) -> VolumeDataset:
    """Dispatches to read_tiff or read_zarr based on the path's format."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return read_tiff(path)
    if suffix == ".zarr" or (path.is_dir() and any((path / marker).exists() for marker in _ZARR_MARKERS)):
        return read_zarr(path)
    raise ValueError(f"unrecognized volume format: {path}")
