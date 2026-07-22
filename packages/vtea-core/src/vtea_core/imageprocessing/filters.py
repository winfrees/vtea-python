"""Basic intensity filters: Gaussian blur, median filter, contrast, background subtraction.

Ports vtea.imageprocessing.builtin from the Java codebase. Denoise (a fixed
same-radius-all-axes median filter) and Median3D (separate X/Y/Z radii) are
consolidated into one median_filter() - Denoise is just Median3D's special
case where all three radii are equal. Both Java classes are themselves
one-line wrappers around ImageJ macro commands (IJ.run(..., "Gaussian
Blur...", ...), "Median...", "Enhance Contrast...") or ImageJ's
BackgroundSubtracter class; scipy.ndimage/scikit-image provide the same
operations directly. LinearUnmixing and IJMacro (arbitrary embedded ImageJ
macros) are not ported - see PORT_PLAN.md's open question on ImageJ macro
compatibility.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.restoration import rolling_ball


def gaussian_blur(volume: np.ndarray, sigma: float | tuple[float, ...]) -> np.ndarray:
    """Gaussian blur. Replaces vtea.imageprocessing.builtin.Gaussian."""
    return ndi.gaussian_filter(volume, sigma=sigma)


def median_filter(volume: np.ndarray, radius: float | tuple[float, ...]) -> np.ndarray:
    """Median filter with a scalar radius (all axes) or a per-axis tuple.

    Replaces both vtea.imageprocessing.builtin.Denoise (same radius on every
    axis) and Median3D (separate X/Y/Z radii) - Denoise is Median3D's
    scalar-radius special case.
    """
    if isinstance(radius, (int, float)):
        size = int(2 * radius + 1)
    else:
        size = tuple(int(2 * r + 1) for r in radius)
    return ndi.median_filter(volume, size=size)


def enhance_contrast(volume: np.ndarray, *, method: str = "normalize") -> np.ndarray:
    """Contrast enhancement. Replaces vtea.imageprocessing.builtin.EnhanceContrast.

    method="normalize": rescale intensities to fill the image's dtype range.
    method="equalize": adaptive histogram equalization (CLAHE).
    """
    if method == "normalize":
        return rescale_intensity(volume)
    elif method == "equalize":
        return equalize_adapthist(volume)
    else:
        raise ValueError(f"unknown method {method!r}, expected 'normalize' or 'equalize'")


def subtract_background(volume: np.ndarray, *, radius: float = 50) -> np.ndarray:
    """Rolling-ball background subtraction.

    Replaces vtea.imageprocessing.builtin.BackgroundSubtraction (ImageJ's
    BackgroundSubtracter - the same rolling-ball algorithm). Returns a
    non-negative float array regardless of the input dtype, to avoid
    unsigned-integer underflow.
    """
    background = rolling_ball(volume, radius=radius)
    return np.clip(volume.astype(np.float64) - background, 0, None)
