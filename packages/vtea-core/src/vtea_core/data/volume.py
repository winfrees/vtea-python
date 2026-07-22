"""VolumeDataset: in-memory (NumPy) and chunked (Dask) volumetric image data.

Ports vtea.dataset.volume (VolumeDataset / ImagePlusVolumeDataset /
ZarrVolumeDataset) from the Java codebase. Canonical axis order is always
(C, Z, Y, X) - channel, depth, height, width - regardless of backing store;
2D images have depth=1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import dask.array as da
import numpy as np

Array = Union[np.ndarray, da.Array]

# Placeholder budget for fits_in_memory() when the caller doesn't supply one.
# Real memory-aware chunking (matching vtea.partition's "Memory-Based"
# strategy) is a Phase 2+ concern; this just keeps the method usable now.
DEFAULT_MEMORY_BUDGET_BYTES = 4 * 1024**3


class VolumeDataset(ABC):
    """An n-channel 3D (Z, Y, X) image volume, in-memory or chunked."""

    @property
    @abstractmethod
    def array(self) -> Array:
        """The underlying (C, Z, Y, X) array - numpy or dask depending on backing."""

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.array.shape

    @property
    def n_channels(self) -> int:
        return self.shape[0]

    @property
    def depth(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[2]

    @property
    def width(self) -> int:
        return self.shape[3]

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    @abstractmethod
    def is_chunked(self) -> bool:
        """True for a Dask-backed (out-of-core) dataset, False for NumPy in-memory."""

    def channel(self, index: int) -> Array:
        """(Z, Y, X) array (numpy or dask, matching this dataset's backing) for one channel."""
        return self.array[index]

    def voxel(self, c: int, z: int, y: int, x: int) -> float:
        value = self.array[c, z, y, x]
        return float(value.compute()) if self.is_chunked else float(value)

    def subvolume(self, c: int, z0: int, y0: int, x0: int, depth: int, height: int, width: int) -> np.ndarray:
        """Materialized (depth, height, width) numpy array for one channel."""
        region = self.array[c, z0 : z0 + depth, y0 : y0 + height, x0 : x0 + width]
        return np.asarray(region.compute() if self.is_chunked else region)

    def fits_in_memory(self, max_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES) -> bool:
        return self.array.nbytes <= max_bytes

    def to_numpy(self) -> np.ndarray:
        """Materialize the full (C, Z, Y, X) array in memory."""
        return np.asarray(self.array.compute() if self.is_chunked else self.array)


class InMemoryVolumeDataset(VolumeDataset):
    """NumPy-backed VolumeDataset. Ports vtea.dataset.volume.ImagePlusVolumeDataset."""

    def __init__(self, array: np.ndarray):
        if array.ndim != 4:
            raise ValueError(f"expected a 4D (C, Z, Y, X) array, got shape {array.shape}")
        self._array = array

    @property
    def array(self) -> np.ndarray:
        return self._array

    @property
    def is_chunked(self) -> bool:
        return False


class ChunkedVolumeDataset(VolumeDataset):
    """Dask-backed VolumeDataset. Ports vtea.dataset.volume.ZarrVolumeDataset."""

    def __init__(self, array: da.Array):
        if array.ndim != 4:
            raise ValueError(f"expected a 4D (C, Z, Y, X) array, got shape {array.shape}")
        self._array = array

    @property
    def array(self) -> da.Array:
        return self._array

    @property
    def is_chunked(self) -> bool:
        return True
