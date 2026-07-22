"""Connected-component labeling, watershed splitting, and size filtering.

Consolidates vtea.objects.Segmentation's LayerCake3D*, FloodFill3DSingleThreshold,
MorphoLibJConnectedComponents, and Imglib2ConnectedComponents from the Java
codebase - six classes that all reduce to "threshold, then connected-component
label a mask, optionally split touching objects, then filter by size" using
different libraries/workarounds for want of a single fast native 3D
connected-components implementation. scikit-image/scipy provide that
directly, so one implementation replaces all of them. Large-volume/chunked
handling (the point of LayerCake3DSingleThresholdkDTree and
LayerCake3DLargeScaleSingleThreshold) is Dask's job via ChunkedVolumeDataset,
not a separate algorithm.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def label_components(mask: np.ndarray, *, connectivity: int = 1) -> np.ndarray:
    """Connected-component labeling of a boolean mask.

    connectivity=1: face-adjacency only (6-connected in 3D, 4-connected in 2D).
    connectivity=mask.ndim: full corner-adjacency (26-connected in 3D).
    """
    structure = ndi.generate_binary_structure(mask.ndim, connectivity)
    labels, _ = ndi.label(mask, structure=structure)
    return labels


def watershed_split(intensity: np.ndarray, mask: np.ndarray, *, min_distance: int = 5) -> np.ndarray:
    """Splits touching objects in `mask` using a distance-transform watershed.

    Replaces the "Watershed" option in LayerCake3DSingleThreshold/
    Region2DSingleThreshold - here a separate, optional step rather than a
    boolean flag baked into a segmentation class. `intensity` is accepted
    for API symmetry with future intensity-guided watershed variants; the
    split itself is purely shape-based (distance transform of `mask`).
    """
    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=min_distance, labels=mask)
    markers = np.zeros(distance.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    return watershed(-distance, markers, mask=mask)


def filter_by_size(labels: np.ndarray, *, min_size: int | None = None, max_size: int | None = None) -> np.ndarray:
    """Removes labeled objects outside [min_size, max_size] voxels, relabeling sequentially.

    Replaces the minObjectSize/maxObjectSize filtering duplicated across every
    Java segmentation class.
    """
    if min_size is None and max_size is None:
        return labels

    ids, counts = np.unique(labels, return_counts=True)
    keep = ids != 0
    if min_size is not None:
        keep &= counts >= min_size
    if max_size is not None:
        keep &= counts <= max_size
    keep_ids = ids[keep]

    remap = np.zeros(int(ids.max()) + 1, dtype=labels.dtype)
    remap[keep_ids] = np.arange(1, len(keep_ids) + 1)
    return remap[labels]
