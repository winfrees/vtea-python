"""Intensity thresholding -> boolean foreground mask.

Ports the thresholding step shared by every Java segmentation method
(SingleThreshold, LayerCake3D*, FloodFill3DSingleThreshold, ...) as a
standalone, composable function instead of duplicating it inside each
algorithm class.
"""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu


def threshold_mask(
    volume: np.ndarray,
    *,
    method: str = "fixed",
    value: float | None = None,
    percentile: float | None = None,
) -> np.ndarray:
    """Boolean foreground mask from an intensity volume.

    method="fixed": voxels >= `value`.
    method="otsu": Otsu's method (`value`/`percentile` ignored).
    method="percentile": voxels >= the given percentile of intensities in `volume`.
    """
    if method == "fixed":
        if value is None:
            raise ValueError("method='fixed' requires `value`")
        threshold = value
    elif method == "otsu":
        threshold = threshold_otsu(volume)
    elif method == "percentile":
        if percentile is None:
            raise ValueError("method='percentile' requires `percentile`")
        threshold = np.percentile(volume, percentile)
    else:
        raise ValueError(f"unknown method {method!r}, expected 'fixed', 'otsu', or 'percentile'")
    return volume >= threshold
