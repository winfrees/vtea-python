"""Label arrays from existing coordinates or masks, rather than intensity thresholding.

Replaces vtea.objects.Segmentation.Points (point-seeded objects) and
PreLabelled (import an existing label image) from the Java codebase.
ImageJROIBased (importing ImageJ .roi/.zip ROI files) is deferred - it's an
I/O format concern, not a segmentation algorithm.
"""

from __future__ import annotations

import numpy as np


def labels_from_points(points: np.ndarray, shape: tuple[int, ...], *, radius: float) -> np.ndarray:
    """One labeled sphere/disk per point.

    points: (N, ndim) array of coordinates, in the same axis order as `shape`.
    Later points overwrite earlier ones where spheres overlap.
    """
    labels = np.zeros(shape, dtype=np.int32)
    grid = np.indices(shape)
    for i, point in enumerate(points, start=1):
        distance_sq = sum((grid[axis] - point[axis]) ** 2 for axis in range(len(shape)))
        labels[distance_sq <= radius**2] = i
    return labels


def import_labels(array: np.ndarray) -> np.ndarray:
    """Passes through an already-labeled array (e.g. loaded via vtea_core.io).

    Exists for API symmetry with the other segmentation functions and as the
    documented replacement for PreLabelled - there's no algorithm to run.
    """
    return np.asarray(array)
