"""The only module in VTEA that imports zarr.

That is the whole point of it. VTEA writes **Zarr 2 and OME-NGFF 0.4**,
because that is what the tools a collaborator will open the data with
actually read - `ome-zarr-py`, `napari-ome-zarr`, Fiji's N5/Zarr reader,
`bioio` - and because NGFF 0.4 is itself a Zarr-2 spec. Zarr 3 brings
sharding, which will matter when a pyramid runs to hundreds of thousands of
chunk files, and NGFF 0.5 with it. The move is a question of when, not
whether.

So the readiness for it is mechanical rather than aspirational:

1. **Every zarr call is here**, `dask.array.from_zarr`/`to_zarr` included -
   those are zarr use under another name, and leaving them scattered would
   put the migration back where it started.
2. **Only API that exists in both versions.** No store internals, no
   reading `.zarray` by hand, and no `numcodecs` object crossing this
   module's boundary: a compressor is named by a string outside and
   translated inside, because that translation is exactly what changes.
3. **The version written is a constant; the reader accepts more than it
   writes.** A user hitting a store from a newer writer is the case that
   actually costs them something.
4. **Every store says what wrote it**, in `vtea` attrs on the root group,
   so a future reader knows rather than infers.

See docs/LARGE_IMAGES.md, "The store: Zarr 2 now, Zarr 3 ready".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import zarr

# What this version writes. Read as a pair: NGFF 0.4 metadata inside a
# Zarr 2 store. NGFF 0.5 is the Zarr 3 spelling and is not written yet.
ZARR_FORMAT = 2
NGFF_VERSION = "0.4"

# What this version reads. Wider than what it writes, on purpose.
SUPPORTED_NGFF_VERSIONS = ("0.4", "0.5")

# Bumped when VTEA's own attrs change shape, independently of the NGFF
# version - they answer different questions and move at different times.
VTEA_FORMAT_VERSION = 1

# The key VTEA's own metadata lives under, kept out of the NGFF namespace
# so that a strict NGFF reader ignores it rather than choking on it.
VTEA_ATTRS_KEY = "vtea"

# Chunk size per spatial axis. 128 is chosen to survive Zarr 3: a shard
# subdivides into inner chunks, and 128-voxel chunks are a sensible inner
# size where 1024-voxel blocks would be one shard's worth on their own.
# 128^3 of uint16 is 4 MB, which is also a reasonable single read.
DEFAULT_CHUNK = 128

# Compressors are named here and translated below, so nothing outside this
# module holds a numcodecs object.
DEFAULT_COMPRESSOR = "zstd"
_COMPRESSOR_NAMES = ("zstd", "lz4", "blosclz", "zlib", "none")

# How chunk keys are spelled on disk: "0/1/2" rather than zarr 2's own
# default of "0.1.2". OME-NGFF asks for the nested form, and other readers
# take it at its word - a store written with the flat separator opens, reads
# the right shape, and returns fill values, because the chunk files are not
# where the reader looks. Zarr 3 uses the nested form as standard, so this
# is also one less difference to cross later.
DIMENSION_SEPARATOR = "/"

_MARKERS = (".zarray", ".zgroup", "zarr.json")


class UnsupportedStoreVersion(ValueError):
    """A store was written by something newer than this reader."""


def is_zarr(path: str | os.PathLike) -> bool:
    """Whether `path` looks like a Zarr store or group."""
    path = Path(path)
    if path.suffix.lower() == ".zarr":
        return True
    return path.is_dir() and any((path / marker).exists() for marker in _MARKERS)


def compressor(name: str | None = DEFAULT_COMPRESSOR, *, level: int = 5) -> Any:
    """A codec object from a name.

    The one place numcodecs is touched. Zarr 3 takes a list of codecs
    rather than one compressor, and spells the same algorithms
    differently; a name is the part that survives that.
    """
    if name is None or name == "none":
        return None
    if name not in _COMPRESSOR_NAMES:
        raise ValueError(f"unknown compressor {name!r}, expected one of {_COMPRESSOR_NAMES}")
    from numcodecs import Blosc

    return Blosc(cname=name, clevel=level, shuffle=Blosc.SHUFFLE)


def create_group(path: str | os.PathLike, *, overwrite: bool = False) -> Any:
    """A new (or reopened) root group, stamped with what wrote it."""
    group = zarr.open_group(os.fspath(path), mode="w" if overwrite else "a")
    stamp = dict(group.attrs.get(VTEA_ATTRS_KEY, {}))
    stamp.update({"format_version": VTEA_FORMAT_VERSION, "zarr_format": ZARR_FORMAT})
    group.attrs[VTEA_ATTRS_KEY] = stamp
    return group


def open_group(path: str | os.PathLike, *, mode: str = "r") -> Any:
    return zarr.open_group(os.fspath(path), mode=mode)


def group_attrs(node: Any) -> dict[str, Any]:
    """A plain dict of a group's or array's attributes.

    A copy, not a live view: an attrs mapping that writes to disk when a
    caller happens to mutate it is a surprise nobody needs.
    """
    return dict(node.attrs)


def set_attrs(node: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        node.attrs[key] = value


def create_array(
    group: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    chunks: tuple[int, ...],
    compressor_name: str | None = DEFAULT_COMPRESSOR,
    overwrite: bool = True,
) -> Any:
    """An empty array inside `group`, ready to be written region by region.

    Region-at-a-time is the point: an out-of-core result is produced one
    tile at a time and each is written as it is finished, so nothing ever
    holds the whole output.
    """
    return group.create_dataset(
        name,
        shape=tuple(shape),
        chunks=tuple(chunks),
        dtype=np.dtype(dtype),
        compressor=compressor(compressor_name),
        dimension_separator=DIMENSION_SEPARATOR,
        overwrite=overwrite,
    )


def write_region(array: Any, region: tuple[slice, ...], values: np.ndarray) -> None:
    """Write one region of a stored array. The out-of-core write path."""
    array[region] = values


def read_region(array: Any, region: tuple[slice, ...]) -> np.ndarray:
    """Read one region of a stored array. The out-of-core read path -
    an object's bounding box, a tile, a gallery crop."""
    return np.asarray(array[region])


def as_dask(array: Any, chunks: str | tuple = "auto") -> da.Array:
    """A Dask view of a stored array, read only when something asks."""
    return da.from_zarr(array, chunks=chunks)


def read_dask(
    path: str | os.PathLike, *, component: str | None = None, chunks: str | tuple = "auto"
) -> da.Array:
    array = da.from_zarr(os.fspath(path), component=component, chunks=chunks)
    return array


def write_dask(
    array: da.Array | np.ndarray,
    path: str | os.PathLike,
    *,
    component: str | None = None,
    chunks: tuple[int, ...] | None = None,
    compressor_name: str | None = DEFAULT_COMPRESSOR,
    overwrite: bool = True,
) -> None:
    """Compute and store a whole array, streaming chunk by chunk.

    A NumPy array is wrapped first, so this is one path rather than two;
    a Dask array is never materialized.
    """
    if not isinstance(array, da.Array):
        array = da.from_array(array, chunks=chunks or "auto")
    elif chunks is not None:
        array = array.rechunk(chunks)
    array.to_zarr(
        os.fspath(path),
        component=component,
        overwrite=overwrite,
        compressor=compressor(compressor_name),
        dimension_separator=DIMENSION_SEPARATOR,
    )


def store_dask(array: da.Array, target: Any) -> None:
    """Stream a Dask array into an array that already exists.

    The write path for a pyramid level or a blocked result: the target's
    chunking, dtype and compression are decided when it is created, and
    this fills it a chunk at a time without ever holding the whole thing.

    The source is rechunked to the target's chunks first, and that is not
    an optimisation. Two source chunks that land in one stored chunk are
    two read-modify-write cycles on the same compressed block, and with
    `lock=False` one silently overwrites the other - the symptom is a store
    that reads back partly zeroed. Aligning them makes every write cover a
    whole chunk, so there is nothing to race over and no lock to pay for.
    """
    chunks = getattr(target, "chunks", None)
    if chunks:
        array = array.rechunk(tuple(chunks))
    da.store(array, target, lock=False)


def default_chunks(
    shape: tuple[int, ...],
    axes: str,
    *,
    spatial_chunk: int = DEFAULT_CHUNK,
) -> tuple[int, ...]:
    """A chunk shape for an array with these axes.

    Spatial axes get `spatial_chunk`, capped at the extent, so a 24-slice
    slab is not chunked into fifths of a slice. Channel and time get one:
    a step reads one channel at a time, and pulling a whole 4-channel chunk
    to look at one of them would quadruple every read.
    """
    chunks = []
    for axis, extent in zip(axes.upper(), shape):
        chunks.append(min(int(extent), spatial_chunk) if axis in "ZYX" else 1)
    return tuple(chunks)


def check_ngff_version(version: str | None) -> str:
    """Accept the NGFF versions this reader understands, and say so plainly
    when a store is newer than it."""
    if version is None:
        # NGFF before 0.4 did not stamp a version. Reading it as 0.4 is the
        # best available guess and is what other readers do.
        return NGFF_VERSION
    if str(version) not in SUPPORTED_NGFF_VERSIONS:
        raise UnsupportedStoreVersion(
            f"this store declares OME-NGFF {version}, and VTEA reads "
            f"{', '.join(SUPPORTED_NGFF_VERSIONS)}. Upgrade VTEA, or re-export the "
            f"store from whatever wrote it at a version VTEA reads."
        )
    return str(version)
