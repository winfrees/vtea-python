"""Segmented-object access over integer label masks.

Replaces vteaobjects.MicroObject's per-object pixel-coordinate storage from
the Java codebase: rather than one Python object per segmented region,
objects are represented implicitly as regions of an integer label array
(0 = background), and these functions extract per-object pixel coordinates
and intensity values from it on demand.
"""

from __future__ import annotations

import numpy as np


def object_ids(label_mask: np.ndarray) -> np.ndarray:
    """Sorted nonzero object ids present in a label mask (0 is background)."""
    ids = np.unique(label_mask)
    return ids[ids != 0]


def object_pixel_indices(label_mask: np.ndarray, object_id: int) -> tuple[np.ndarray, ...]:
    """Voxel coordinates for one object id, as a tuple of per-axis index arrays.

    label_mask may be of any dimensionality; returns one index array per axis,
    matching np.nonzero's convention (e.g. (z_idx, y_idx, x_idx) for a 3D mask).
    Replaces MicroObject.getPixelsX()/getPixelsY()/getPixelsZ().
    """
    return np.nonzero(label_mask == object_id)


def object_intensity_values(label_mask: np.ndarray, intensity: np.ndarray, object_id: int) -> np.ndarray:
    """Intensity values at one object's voxels, from a co-registered intensity volume."""
    if label_mask.shape != intensity.shape:
        raise ValueError(f"label_mask shape {label_mask.shape} != intensity shape {intensity.shape}")
    return intensity[label_mask == object_id]
