"""Plain-Zarr I/O for chunked, out-of-core volumes.

Ports vtea.io.zarr (ZarrReader/ZarrWriter) from the Java codebase. Dask's
chunking replaces the hand-written Chunk/VolumePartitioner/ChunkIterator
system for the read/write path (see PORT_PLAN.md, Phase 1).

This is the bare-array path: a 4D (C, Z, Y, X) Zarr array with no metadata
beyond its shape. `vtea_core.io.ome_zarr` is the one to prefer for anything
that will be kept or shared - it records the axes, the voxel size and a
pyramid, none of which a plain array can. The zarr calls themselves live in
`vtea_core.io.store`, which is the only module that imports zarr.
"""

from __future__ import annotations

import os

from vtea_core.data.volume import ChunkedVolumeDataset, VolumeDataset
from vtea_core.io import store


def read_zarr(path: str | os.PathLike, chunks: str | tuple = "auto") -> ChunkedVolumeDataset:
    """Reads a Zarr array into a (C, Z, Y, X) ChunkedVolumeDataset."""
    array = store.read_dask(path, chunks=chunks)
    if array.ndim != 4:
        raise ValueError(f"expected a 4D (C, Z, Y, X) Zarr array, got shape {array.shape}")
    return ChunkedVolumeDataset(array)


def write_zarr(
    dataset: VolumeDataset, path: str | os.PathLike, chunks: str | tuple | None = None
) -> None:
    """Writes any VolumeDataset (in-memory or chunked) out as a Zarr array."""
    store.write_dask(dataset.array, path, chunks=chunks, compressor_name=None)
