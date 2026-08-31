"""Somewhere to put an intermediate that does not fit in memory.

`Pipeline.run` threads results through a dict, which is the right thing
while they are NumPy arrays: four steps, four arrays, all alive at once and
all small. At scale the same dict is the problem - a blurred copy, a mask, a
label image and a distance transform of a 33 GB volume are not four things
that can be held, and the label image alone is as large as the image.

So a blocked run's intermediates go here instead: a Zarr group in a scratch
directory, written tile by tile, read back by region. Which turns "hold
every intermediate" into "hold one tile of one intermediate", and makes the
context a dict of *handles* rather than of arrays.

Deliberately a plain temporary directory rather than anything cleverer.
Scratch is scratch: it should be easy to point at a fast local disk, easy to
delete, and obvious in `du`. Keeping it (`keep=True`) is what turns a
crashed six-hour run into a resumable one, which is why it is an option
rather than always cleaning up.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np

from vtea_core.io import store as _io_store

# Where scratch goes when nobody says. Honours TMPDIR, which is how a
# cluster points jobs at node-local disk instead of a shared filesystem.
ENV_VAR = "VTEA_SCRATCH_DIR"


class ZarrScratch:
    """A directory of Zarr arrays that live as long as a run does.

    Use it as a context manager; on exit the directory goes away unless
    `keep` was set. `root` overrides where it is created, `$VTEA_SCRATCH_DIR`
    overrides that default, and both matter more than they look: scratch on
    a shared network filesystem can be slower than the computation it exists
    to serve.
    """

    def __init__(
        self,
        root: str | os.PathLike | None = None,
        *,
        prefix: str = "vtea-scratch-",
        keep: bool = False,
    ):
        base = root if root is not None else os.environ.get(ENV_VAR) or None
        if base is not None:
            Path(base).mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
        self.keep = keep
        self._group = _io_store.create_group(self.path, overwrite=True)
        self._closed = False

    @classmethod
    def reopen(cls, path: str | os.PathLike, *, keep: bool = True) -> ZarrScratch:
        """Open a scratch directory a previous run left behind.

        What makes a resume possible across processes: the manifest records
        what was segmented, and this is where the segmentation actually is.
        Defaults to `keep=True`, because a store being reopened is by
        definition one somebody wanted to survive - deleting it on the way
        out of the resumed run would be a surprise, and the second one.
        """
        store = cls.__new__(cls)
        store.path = Path(os.fspath(path))
        if not store.path.is_dir():
            raise FileNotFoundError(f"no scratch store at {store.path}")
        store.keep = keep
        store._group = _io_store.open_group(store.path, mode="a")
        store._closed = False
        return store

    def __enter__(self) -> ZarrScratch:  # noqa: PYI034 - typing.Self needs Python 3.11+, this package supports 3.10
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __contains__(self, name: str) -> bool:
        return name in self._group

    def __len__(self) -> int:
        return len(self.names())

    def names(self) -> list[str]:
        return sorted(self._group.array_keys())

    def create(
        self,
        name: str,
        *,
        shape: tuple[int, ...],
        dtype: Any,
        chunks: tuple[int, ...] | None = None,
        axes: str = "CZYX",
        compressor_name: str | None = _io_store.DEFAULT_COMPRESSOR,
    ) -> Any:
        """An empty array to write a blocked result into, tile by tile."""
        self._check_open()
        return _io_store.create_array(
            self._group,
            name,
            shape=tuple(shape),
            dtype=dtype,
            chunks=tuple(chunks) if chunks else _io_store.default_chunks(tuple(shape), axes),
            compressor_name=compressor_name,
            overwrite=True,
        )

    def create_like(self, name: str, other: Any, *, dtype: Any = None, **kwargs) -> Any:
        """An array shaped like another - the common case, since a step's
        output usually has its input's shape and a different dtype."""
        return self.create(
            name,
            shape=tuple(other.shape),
            dtype=np.dtype(dtype) if dtype is not None else np.dtype(other.dtype),
            chunks=tuple(getattr(other, "chunksize", None) or getattr(other, "chunks", None) or ())
            or None,
            **kwargs,
        )

    def put(self, name: str, array: da.Array | np.ndarray, **kwargs) -> Any:
        """Write a whole array in, streaming rather than materializing it."""
        target = self.create(name, shape=array.shape, dtype=array.dtype, **kwargs)
        if isinstance(array, da.Array):
            _io_store.store_dask(array, target)
        else:
            target[...] = array
        return target

    def open(self, name: str) -> Any:
        self._check_open()
        if name not in self._group:
            raise KeyError(f"no scratch array named {name!r} - have {self.names()}")
        return self._group[name]

    def as_dask(self, name: str, chunks: str | tuple = "auto") -> da.Array:
        return _io_store.as_dask(self.open(name), chunks=chunks)

    def drop(self, name: str) -> None:
        """Delete an intermediate nothing needs any more.

        Worth doing explicitly during a long run: a protocol's third step
        rarely needs its first step's output, and scratch is finite in a way
        that is easy to forget once it stops being memory.
        """
        if name in self._group:
            del self._group[name]

    @property
    def nbytes(self) -> int:
        """Bytes on disk. Compressed, so this is the real figure rather
        than the sum of the arrays' nominal sizes."""
        return sum(
            item.stat().st_size for item in self.path.rglob("*") if item.is_file()
        )

    def close(self) -> None:
        """Delete the scratch directory, unless `keep` was set.

        **Anything still holding one of these arrays reads zeros
        afterwards, silently.** A Zarr array whose chunk files have gone
        does not raise; it returns its fill value. So a result has to be
        read, or copied somewhere durable, before the store is closed -
        which is why the executor's results are documented as valid only
        inside its context.
        """
        if self._closed:
            return
        self._closed = True
        if not self.keep:
            shutil.rmtree(self.path, ignore_errors=True)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("this scratch store has been closed")

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"{len(self.names())} arrays"
        return f"<ZarrScratch {self.path} ({state})>"
