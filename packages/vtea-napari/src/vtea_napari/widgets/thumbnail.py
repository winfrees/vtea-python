"""Shared array -> QPixmap rendering, used by both the gallery view and
protocol-builder step-card previews.

Replaces the per-object BufferedImage cropping vtea.exploration.gallery.
GalleryImageProcessor did with a SwingWorker batch - here it's one small
numpy -> QPixmap helper reused wherever a thumbnail is needed, rather than
a dedicated processor class.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap


def max_projection(array: np.ndarray) -> np.ndarray:
    """Collapses any leading axes via max-intensity projection down to 2D
    (matches vtea's gallery crops, which projected the full Z extent)."""
    array = np.asarray(array)
    while array.ndim > 2:
        array = array.max(axis=0)
    return array


def array_to_pixmap(array: np.ndarray, size: int = 64) -> QPixmap:
    """Renders a 2D array as a grayscale QPixmap scaled to fit within
    `size` x `size`, normalized to its own min/max."""
    if array.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {array.shape}")
    data = np.nan_to_num(array.astype(np.float64))
    minimum, maximum = data.min(), data.max()
    span = maximum - minimum
    normalized = ((data - minimum) / span * 255) if span > 0 else np.zeros_like(data)
    image_data = np.ascontiguousarray(normalized.astype(np.uint8))

    height, width = image_data.shape
    qimage = QImage(image_data.data, width, height, width, QImage.Format.Format_Grayscale8)
    pixmap = QPixmap.fromImage(qimage.copy())  # detach from the numpy buffer before it's freed
    return pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio)
