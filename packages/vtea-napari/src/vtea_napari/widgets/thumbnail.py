"""Shared array -> QPixmap rendering, used by both the gallery view and
protocol-builder step-card previews.

Replaces the per-object BufferedImage cropping vtea.exploration.gallery.
GalleryImageProcessor did with a SwingWorker batch - here it's one small
numpy -> QPixmap helper reused wherever a thumbnail is needed, rather than
a dedicated processor class.

The gallery needs three things a grayscale thumbnail cannot give it, and
they live here because they are all "array to picture":

- **A composite.** A cell is not one channel. Reading a nucleus against its
  membrane marker means seeing them at once, each in its own colour, which
  is what `composite_rgb` does - normalise each channel, map it through its
  own LUT, and add the results the way a fluorescence composite is made.
- **The segmentation on top.** A dot on a scatter plot is an object id; the
  only way to know *which cell* it is is to see the outline the analysis
  drew around it. `overlay_mask` tints those voxels in a chosen colour at a
  chosen opacity - opacity because a mask painted opaque hides the very
  intensities it is meant to identify.
- **Masks at the intensity's resolution.** A crop may be read from a coarse
  pyramid level while the segmentation only exists at level 0, so
  `resize_nearest` puts the mask on the crop's grid. Nearest-neighbour by
  construction: a label image interpolated is a label image with invented
  ids in it.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps as _matplotlib_colormaps
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap

# The LUTs a channel can be shown in. The first six are the single-hue maps
# fluorescence is read in - black to the pure colour - built here rather than
# taken from matplotlib, which has no "green"; the rest are matplotlib's own,
# for a channel being read as a quantity rather than as a stain.
PURE_HUES = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "gray": (1.0, 1.0, 1.0),
}
MATPLOTLIB_LUTS = ("viridis", "inferno", "magma", "turbo")
GALLERY_LUTS = (*PURE_HUES, *MATPLOTLIB_LUTS)

# What each of the three channel slots starts on. Grey first, so a
# single-channel acquisition looks like the greyscale thumbnail it has
# always been rather than an unexplained green one; then magenta and cyan,
# which are distinguishable from grey and from each other including to a
# colour-blind reader.
DEFAULT_LUTS = ("gray", "magenta", "cyan")


def max_projection(array: np.ndarray) -> np.ndarray:
    """Collapses any leading axes via max-intensity projection down to 2D
    (matches vtea's gallery crops, which projected the full Z extent)."""
    array = np.asarray(array)
    while array.ndim > 2:
        array = array.max(axis=0)
    return array


def normalize(array: np.ndarray, limits=None) -> np.ndarray:
    """A 2D array as 0..1 floats, against `limits` or against its own range.

    Given limits - the contrast the same image is being displayed at in the
    viewer - every crop is scaled the same way, which is what makes a grid
    of them comparable and what stops a channel with no signal in *this*
    cell from being stretched into a screenful of amplified noise.

    Without them it falls back to per-crop scaling, which is what a
    thumbnail of a dim cell needs when there is no better reference:
    normalising forty crops against the brightest one in the stack shows
    thirty-nine black squares.
    """
    data = np.nan_to_num(np.asarray(array, dtype=np.float64))
    if limits is not None:
        minimum, maximum = (float(value) for value in limits)
    else:
        minimum, maximum = data.min(), data.max()
    span = maximum - minimum
    if span <= 0:
        return np.zeros_like(data)
    return np.clip((data - minimum) / span, 0.0, 1.0)


def lut_rgb(values: np.ndarray, lut: str) -> np.ndarray:
    """Normalised intensities through one LUT, as (H, W, 3) floats."""
    if lut in PURE_HUES:
        colour = np.asarray(PURE_HUES[lut], dtype=float)
        return values[..., np.newaxis] * colour
    colormap = _matplotlib_colormaps.get(lut, None)
    if colormap is None:
        # An unknown LUT shows the channel in grey rather than refusing to
        # draw it: a thumbnail is not the place to fail over a name.
        return values[..., np.newaxis] * np.ones(3)
    return np.asarray(colormap(values))[..., :3]


def composite_rgb(planes, luts, limits=None) -> np.ndarray:
    """Several channels as one colour image, the way a composite is made.

    Each plane is normalised (against `limits[i]` where given, else against
    itself), mapped through its own LUT and added; the sum is clipped rather
    than averaged, because averaging two channels dims both and a composite
    is meant to show where they overlap as the sum of the two colours.

    Returns (H, W, 3) uint8. An empty list gives a black square rather than
    an error - "no channels selected" is a state the controls allow.
    """
    planes = [np.asarray(plane) for plane in planes]
    if not planes:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    shape = planes[0].shape
    accumulated = np.zeros((*shape, 3), dtype=float)
    limits = list(limits) if limits is not None else [None] * len(planes)
    for index, (plane, lut) in enumerate(zip(planes, luts)):
        if plane.shape != shape:
            raise ValueError(
                f"channel crops must be the same shape; got {plane.shape} and {shape}"
            )
        span = limits[index] if index < len(limits) else None
        accumulated += lut_rgb(normalize(plane, span), lut)
    return (np.clip(accumulated, 0.0, 1.0) * 255).astype(np.uint8)


def resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """`array` on a different grid, by nearest neighbour.

    For putting a level-0 segmentation onto a crop read from a coarser
    pyramid level. Nearest by construction: interpolating a label image
    invents ids that are not in it, and an object outlined in an id that
    does not exist is worse than no outline.
    """
    array = np.asarray(array)
    height, width = shape
    if array.shape == (height, width):
        return array
    if array.size == 0 or height <= 0 or width <= 0:
        return np.zeros(shape, dtype=array.dtype)
    rows = np.clip((np.arange(height) * array.shape[0] / height).astype(int), 0, array.shape[0] - 1)
    columns = np.clip(
        (np.arange(width) * array.shape[1] / width).astype(int), 0, array.shape[1] - 1
    )
    return array[np.ix_(rows, columns)]


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color, opacity: float) -> np.ndarray:
    """Tint the voxels `mask` selects, at `opacity`.

    Alpha-blended rather than painted over: at full opacity the mask hides
    the intensities it is there to point at, and what the gallery is being
    asked is "which cell is this dot?" - which needs the outline *and* the
    cell.
    """
    rgb = np.asarray(rgb)
    mask = np.asarray(mask, dtype=bool)
    if opacity <= 0 or not mask.any():
        return rgb
    if mask.shape != rgb.shape[:2]:
        mask = resize_nearest(mask, rgb.shape[:2]).astype(bool)
    tint = np.asarray(to_rgb(color), dtype=float) * 255.0
    blended = rgb.astype(float)
    weight = float(np.clip(opacity, 0.0, 1.0))
    blended[mask] = blended[mask] * (1 - weight) + tint * weight
    return blended.astype(np.uint8)


def to_rgb(color) -> tuple[float, float, float]:
    """A colour, however it was written, as three 0..1 floats."""
    if isinstance(color, str):
        text = color.lstrip("#")
        if len(text) == 6:
            return tuple(int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4))
        if text in PURE_HUES:
            return PURE_HUES[text]
        from matplotlib.colors import to_rgb as _to_rgb

        return tuple(float(value) for value in _to_rgb(color))
    values = tuple(float(value) for value in color)[:3]
    return values if max(values, default=0) <= 1 else tuple(value / 255.0 for value in values)


def rgb_to_pixmap(rgb: np.ndarray, size: int = 64) -> QPixmap:
    """An (H, W, 3) uint8 image as a QPixmap of `size` x `size`.

    Scaled to *fill* rather than to fit: a crop clipped by the edge of the
    image is not square, and letterboxing it leaves a strip of the widget's
    background in a grid where every millimetre is meant to be picture.
    """
    rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(image.copy())  # detach from the numpy buffer
    return pixmap.scaled(
        size,
        size,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def array_to_pixmap(array: np.ndarray, size: int = 64) -> QPixmap:
    """Renders a 2D array as a grayscale QPixmap scaled to fit within
    `size` x `size`, normalized to its own min/max."""
    if array.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {array.shape}")
    image_data = np.ascontiguousarray((normalize(array) * 255).astype(np.uint8))

    height, width = image_data.shape
    qimage = QImage(image_data.data, width, height, width, QImage.Format.Format_Grayscale8)
    pixmap = QPixmap.fromImage(qimage.copy())  # detach from the numpy buffer before it's freed
    return pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio)
