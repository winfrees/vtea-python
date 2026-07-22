"""Zarr I/O for chunked, out-of-core volumes, via Dask + zarr-python.

Ports vtea.io.zarr (ZarrReader/ZarrWriter) from the Java codebase. Dask's
chunking replaces the hand-written Chunk/VolumePartitioner/ChunkIterator
system for the read/write path (see PORT_PLAN.md, Phase 1).
"""

from __future__ import annotations

import os

import dask.array as da

from vtea_core.data.volume import ChunkedVolumeDataset, VolumeDataset


def read_zarr(path: str | os.PathLike, chunks: str | tuple = "auto") -> ChunkedVolumeDataset:
    """Reads a Zarr array into a (C, Z, Y, X) ChunkedVolumeDataset."""
    array = da.from_zarr(os.fspath(path), chunks=chunks)
    if array.ndim != 4:
        raise ValueError(f"expected a 4D (C, Z, Y, X) Zarr array, got shape {array.shape}")
    return ChunkedVolumeDataset(array)


def write_zarr(dataset: VolumeDataset, path: str | os.PathLike, chunks: str | tuple | None = None) -> None:
    """Writes any VolumeDataset (in-memory or chunked) out as a Zarr array."""
    array = dataset.array if dataset.is_chunked else da.from_array(dataset.array, chunks=chunks or "auto")
    array.to_zarr(os.fspath(path), overwrite=True)
