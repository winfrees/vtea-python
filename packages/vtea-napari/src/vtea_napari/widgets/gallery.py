"""Per-object thumbnail grid: shows cropped image previews for a set of
object ids (typically a gate's members), and reports which one is clicked.

Replaces vtea.exploration.gallery.GalleryViewWindow/GalleryImageProcessor/
GalleryViewDataProvider. Unlike vteaexploration.GateManager/
microGateManager (see gate_table.py's docstring), this is real, working
Java code - right-click a gate -> "Gallery View..." opens exactly this: a
grid of per-object crops around each object's centroid, click one to
highlight it back on the scatter plot. Ported with the same behavior: a
fixed-radius crop around each object's centroid, max-projected to 2D.

**This is the cheapest access pattern a chunked store has**, and the only
view in VTEA that gets *faster* on large data rather than slower: a crop is
a bounding-box read, so a gallery of forty objects reads forty small blocks
whatever the volume weighs. Three things are needed to keep that true, and
all three are about not accidentally touching the whole array:

- Nothing is materialized but the crop. The volume may be a Zarr or Dask
  array that would not fit in memory; `np.asarray` belongs on the result of
  the slice, never on the thing being sliced.
- A pyramid level is chosen per crop. A 64-pixel thumbnail of a 400-pixel
  region does not need level 0 - reading it there is sixteen times the I/O
  for the same picture, and the coarse levels exist precisely for this.
- The z range is cropped like the others. Max-projecting a 2,000-slice
  stack is a reduction over the whole volume wearing a thumbnail's clothes;
  what a thumbnail should show is the object, which is a few slices deep.

What the crops *show* is chosen once for the whole grid, from the controls
at the top:

- **Up to three channels, each in its own LUT**, composited. A cell is not
  one channel, and reading a nucleus against its membrane marker means
  seeing them at once. The same channels for every thumbnail, deliberately:
  a grid where each picture was made differently is not comparable, and
  comparing is what a gallery is for.
- **The segmentation on top**, in a colour and opacity the user picks. A
  dot on the scatter plot is an object id; the outline is what says *which
  cell* that dot is. Opacity because a mask painted opaque hides the
  intensities it is meant to identify.
- **Hovering enlarges one**, at twice its size in each dimension, until the
  pointer moves away - a 64-pixel thumbnail is enough to find a cell in and
  not enough to judge one by.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from vtea_napari.widgets.thumbnail import (
    DEFAULT_LUTS,
    GALLERY_LUTS,
    composite_rgb,
    max_projection,
    overlay_mask,
    resize_nearest,
    rgb_to_pixmap,
)

# Below this, a coarser level would throw away detail the thumbnail can
# still show. Chosen against the thumbnail's own size rather than a fixed
# number of voxels, since that is what decides whether detail survives.
_LEVEL_MARGIN = 1.5

# Yellow, and thick enough to read against a bright crop: a thin outline in
# a mid tone disappears into the very cells the gallery is showing.
SELECTED_STYLE = "border: 3px solid #ffd400;"
# A transparent border of the same width, so selecting a cell doesn't
# reflow the grid by growing it.
UNSELECTED_STYLE = "border: 3px solid transparent;"

# How many channel slots the composite offers. Three is what fits in a
# colour image without the hues becoming impossible to tell apart, and is
# what the request asks for.
CHANNEL_SLOTS = 3
NO_CHANNEL = "(none)"

# The grid is drawn edge to edge: a gallery is a wall of pictures, and every
# millimetre of gap is a millimetre not showing a cell.
GRID_SPACING = 2
# Border (3px each side) plus the grid spacing, which is what one cell costs
# beyond the pixmap itself when working out how many fit across.
CELL_MARGIN = 2 * 3 + GRID_SPACING

# How much bigger the hover preview is, in each dimension.
HOVER_SCALE = 2

# The channel and LUT pickers are capped so six of them plus the overlay
# controls do not set a floor on how narrow this pane can be - the grid is
# the point of it, and a control row that cannot shrink is a grid that
# cannot either.
CONTROL_WIDTH = 92

# What the segmentation overlay starts at: yellow, half transparent, so the
# outline reads without hiding the cell under it.
DEFAULT_MASK_COLOR = "#ffd400"
DEFAULT_MASK_OPACITY = 0.45


class _ClickableThumbnail(QLabel):
    clicked = Signal()
    hovered = Signal()
    unhovered = Signal()

    def __init__(self, object_id, parent=None):
        super().__init__(parent)
        self.object_id = object_id
        # The same crop at hover size, kept so enlarging costs no I/O.
        self.zoom_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(SELECTED_STYLE if selected else UNSELECTED_STYLE)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def enterEvent(self, event):  # Qt's spelling
        self.hovered.emit()
        super().enterEvent(event)

    def leaveEvent(self, event):  # Qt's spelling
        self.unhovered.emit()
        super().leaveEvent(event)


class GalleryWidget(QWidget):
    """A scrollable grid of per-object crops, one of which may be selected."""

    object_selected = Signal(int)
    # Any change to what the crops show, so the owner can remember it.
    view_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(GRID_SPACING)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._thumbnails: list[_ClickableThumbnail] = []
        self.selected_object_id: int | None = None
        # What the last call was asked to show, so a change of channel or
        # overlay redraws the same objects without the caller repeating
        # itself.
        self._last_request: dict | None = None
        self.mask_color = DEFAULT_MASK_COLOR
        self.mask_opacity = DEFAULT_MASK_OPACITY
        self._n_channels = 0
        # Until somebody picks channels, the first slot follows the image -
        # a gallery that opens black because the defaults were chosen before
        # the image was known is a gallery that looks broken.
        self._channels_chosen = False

        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)
        root.addLayout(self._build_controls())

        container = QWidget()
        container.setLayout(self._grid)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setWidget(container)
        root.addWidget(self.scroll, 1)

        # One popup, reused: hovering forty thumbnails should not build
        # forty windows.
        self._zoom = QLabel(None, Qt.WindowType.ToolTip)
        self._zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._zoom.hide()

    # -- controls ---------------------------------------------------------

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(3)
        self.channel_combos: list[QComboBox] = []
        self.lut_combos: list[QComboBox] = []
        for slot in range(CHANNEL_SLOTS):
            channel_combo = QComboBox()
            channel_combo.setToolTip(f"Which channel to show in slot {slot + 1}")
            channel_combo.setMaximumWidth(CONTROL_WIDTH)
            channel_combo.currentTextChanged.connect(lambda _text: self._on_view_changed())
            lut_combo = QComboBox()
            lut_combo.addItems(list(GALLERY_LUTS))
            lut_combo.setCurrentText(DEFAULT_LUTS[slot])
            lut_combo.setToolTip("The colour that channel is shown in")
            lut_combo.setMaximumWidth(CONTROL_WIDTH)
            lut_combo.currentTextChanged.connect(lambda _text: self._on_view_changed())
            row.addWidget(channel_combo)
            row.addWidget(lut_combo)
            self.channel_combos.append(channel_combo)
            self.lut_combos.append(lut_combo)

        row.addSpacing(8)
        row.addWidget(QLabel("Segmentation:"))
        self.mask_color_button = QPushButton("Colour…")
        self.mask_color_button.setToolTip(
            "The colour this object's segmentation is tinted in, so a dot on the plot "
            "can be told from the cell it stands for"
        )
        self.mask_color_button.clicked.connect(self.pick_mask_color)
        row.addWidget(self.mask_color_button)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(DEFAULT_MASK_OPACITY * 100))
        self.opacity_slider.setFixedWidth(90)
        self.opacity_slider.setToolTip("How strongly the segmentation is tinted (0 = off)")
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        row.addWidget(self.opacity_slider)
        row.addStretch()
        self._refresh_channel_choices()
        return row

    def _refresh_channel_choices(self) -> None:
        """Offer one entry per channel of the image being shown.

        The first slot lands on channel 0 and the rest on "(none)", so a
        single-channel acquisition shows the one channel it has and a
        four-channel one does not composite four hues nobody asked for.
        """
        names = [NO_CHANNEL] + [f"Channel {index}" for index in range(self._n_channels)]
        for slot, combo in enumerate(self.channel_combos):
            previous = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if self._channels_chosen and previous in names:
                combo.setCurrentText(previous)
            elif slot == 0 and self._n_channels:
                combo.setCurrentText("Channel 0")
            else:
                combo.setCurrentText(NO_CHANNEL)
            combo.blockSignals(False)

    def selected_channels(self) -> list[tuple[int, str]]:
        """(channel index, LUT) for each slot that has a channel chosen."""
        chosen = []
        for channel_combo, lut_combo in zip(self.channel_combos, self.lut_combos):
            text = channel_combo.currentText()
            if text and text != NO_CHANNEL:
                chosen.append((int(text.rsplit(" ", 1)[-1]), lut_combo.currentText()))
        return chosen

    def set_channels(self, channels) -> None:
        """Choose the channels and LUTs from outside - restoring a
        remembered view, or a script driving this widget."""
        for slot, combo in enumerate(self.channel_combos):
            combo.blockSignals(True)
            self.lut_combos[slot].blockSignals(True)
        for slot in range(CHANNEL_SLOTS):
            channel, lut = channels[slot] if slot < len(channels) else (None, None)
            text = NO_CHANNEL if channel is None else f"Channel {channel}"
            if self.channel_combos[slot].findText(text) != -1:
                self.channel_combos[slot].setCurrentText(text)
            if lut and self.lut_combos[slot].findText(lut) != -1:
                self.lut_combos[slot].setCurrentText(lut)
        for slot, combo in enumerate(self.channel_combos):
            combo.blockSignals(False)
            self.lut_combos[slot].blockSignals(False)
        self._channels_chosen = True
        self.refresh()

    def pick_mask_color(self) -> None:
        from qtpy.QtGui import QColor
        from qtpy.QtWidgets import QColorDialog

        chosen = QColorDialog.getColor(QColor(self.mask_color), self, "Segmentation colour")
        if chosen.isValid():
            self.set_mask_style(color=chosen.name())

    def set_mask_style(self, *, color: str | None = None, opacity: float | None = None) -> None:
        if color is not None:
            self.mask_color = color
        if opacity is not None:
            self.mask_opacity = float(np.clip(opacity, 0.0, 1.0))
            self.opacity_slider.blockSignals(True)
            self.opacity_slider.setValue(int(self.mask_opacity * 100))
            self.opacity_slider.blockSignals(False)
        self.refresh()
        self.view_changed.emit()

    def _on_opacity_changed(self, value: int) -> None:
        self.mask_opacity = value / 100.0
        self.refresh()
        self.view_changed.emit()

    def _on_view_changed(self) -> None:
        self._channels_chosen = True
        self.refresh()
        self.view_changed.emit()

    def view_state(self) -> dict:
        """What the crops are showing, as plain values - kept on the session
        so closing the dock does not cost the settings."""
        return {
            "channels": self.selected_channels(),
            "mask_color": self.mask_color,
            "mask_opacity": self.mask_opacity,
        }

    def apply_view_state(self, state: dict) -> None:
        if not state:
            return
        self.mask_color = state.get("mask_color", self.mask_color)
        opacity = state.get("mask_opacity")
        if opacity is not None:
            self.mask_opacity = float(opacity)
            self.opacity_slider.blockSignals(True)
            self.opacity_slider.setValue(int(self.mask_opacity * 100))
            self.opacity_slider.blockSignals(False)
        channels = state.get("channels")
        if channels:
            self.set_channels(channels)

    # -- the grid ---------------------------------------------------------

    def select(self, object_id) -> None:
        """Outline one object's crop, and clear any previous outline. An id
        that isn't on screen just clears the selection rather than raising -
        a gate can be re-drawn under a selection made against the old one."""
        self.selected_object_id = None
        for thumbnail in self._thumbnails:
            is_selected = thumbnail.object_id == object_id
            thumbnail.set_selected(is_selected)
            if is_selected:
                self.selected_object_id = thumbnail.object_id

    def refresh(self) -> None:
        """Redraw the same objects with the current settings."""
        if self._last_request is not None:
            self.show_objects(**self._last_request)

    def resizeEvent(self, event):  # Qt's spelling
        super().resizeEvent(event)
        # Reflow to the new width: a fixed column count leaves a bar of
        # empty widget beside the grid at every other size.
        if self._last_request is not None and self._columns_changed():
            self.refresh()

    def _columns_for(self, thumbnail_size: int) -> int:
        available = self.scroll.viewport().width() if self.scroll.widget() else self.width()
        return max(1, available // max(thumbnail_size + CELL_MARGIN, 1))

    def _columns_changed(self) -> bool:
        wanted = self._columns_for(self._last_request.get("thumbnail_size", 64))
        return wanted != getattr(self, "_columns", wanted)

    def show_objects(
        self,
        volume,
        frame: pd.DataFrame,
        object_ids,
        *,
        crop_radius: int = 20,
        thumbnail_size: int = 64,
        id_column: str = "object_id",
        prefix: str = "",
        z_radius: int | None = None,
        channel_axis: int | None = None,
        labels=None,
        contrast_limits=None,
    ) -> None:
        """`volume` is an intensity array whose last two axes are (row,
        col); `frame` has an id column and `centroid-*` columns (see
        vtea_core.measurements.extract_measurements) - the last two
        centroid columns are always the array's own (row, col) axes,
        whether the source volume was 2D or 3D.

        `volume` may be a list of arrays, napari's multiscale convention,
        finest level first. Given one, each crop is read from the coarsest
        level that still has the detail the thumbnail can show.

        It may also be anything that slices like an array - a Zarr or Dask
        array larger than memory included. Only the crop is materialized.

        `channel_axis` says which axis holds channels, and is what lets the
        controls above the grid composite up to three of them. Each chosen
        channel is sliced out *before* the crop, so a four-channel volume
        still costs one bounding-box read per channel shown rather than a
        read of all four.

        `contrast_limits` is the (low, high) the image is being displayed at
        in the viewer - one pair, or one per channel index. Given it, every
        crop is scaled the same way, so the grid is comparable and a channel
        with no signal in one cell is dark there rather than a screenful of
        amplified noise. Without it each crop is scaled against itself,
        which is what the greyscale gallery always did.

        `labels` is the segmentation those object ids come from; where it is
        given, each crop is tinted where that object's own voxels are, which
        is what says which cell a dot on the plot stands for.

        A per-cell table names its columns for the segmentation each came
        from (`nuclei_1.centroid-0`), so `prefix` says which of them to crop
        around - the segmentation the cells are rooted on, which is where
        their id points anyway.

        `z_radius` crops the depth around the object as well, instead of
        projecting every slice. `None` keeps the ported behaviour of
        projecting the whole stack, which is right for the twenty-slice
        acquisitions the Java original was written for and wrong for a
        thousand.
        """
        self._last_request = {
            "volume": volume,
            "frame": frame,
            "object_ids": list(object_ids),
            "crop_radius": crop_radius,
            "thumbnail_size": thumbnail_size,
            "id_column": id_column,
            "prefix": prefix,
            "z_radius": z_radius,
            "channel_axis": channel_axis,
            "labels": labels,
            "contrast_limits": contrast_limits,
        }
        self._hide_zoom()
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._thumbnails = []

        if id_column not in frame.columns:
            return
        indexed = frame.set_index(id_column)
        centroid_columns = sorted(
            column for column in frame.columns if column.startswith(f"{prefix}centroid-")
        )
        spatial_columns = centroid_columns[-2:]
        if len(spatial_columns) < 2:
            # Nothing to crop around - a table with no centroids at all, or
            # one whose centroids belong to a different segmentation. An
            # empty gallery is the honest answer; guessing a position would
            # show crops of the wrong things.
            return

        self._set_channel_count(volume, channel_axis)
        levels = pyramid_levels(volume)
        level = choose_level(levels, crop_radius, thumbnail_size)
        source = levels[level]
        scale = level_scale(levels, level)
        depth_column = centroid_columns[-3] if len(centroid_columns) >= 3 else None
        channels = self.selected_channels() or [(None, DEFAULT_LUTS[0])]
        self._columns = self._columns_for(thumbnail_size)

        for position, object_id in enumerate(object_ids):
            if object_id not in indexed.index:
                continue
            row_center, col_center = (
                round(indexed.loc[object_id, column] / scale) for column in spatial_columns
            )
            depth = (
                round(indexed.loc[object_id, depth_column] / scale)
                if depth_column is not None
                else None
            )
            radius = max(1, round(crop_radius / scale))
            planes = []
            for channel, _lut in channels:
                view = select_channel(source, channel_axis, channel)
                crop = crop_around(
                    view,
                    row_center,
                    col_center,
                    radius=radius,
                    depth=depth,
                    z_radius=None if z_radius is None else max(1, round(z_radius / scale)),
                )
                if crop.size == 0:
                    planes = []
                    break
                planes.append(max_projection(crop))
            if not planes:
                continue
            rgb = composite_rgb(
                planes,
                [lut for _channel, lut in channels],
                limits=[limits_for(contrast_limits, channel) for channel, _lut in channels],
            )
            rgb = self._tint_segmentation(
                rgb,
                labels,
                object_id,
                row_center=round(row_center * scale),
                col_center=round(col_center * scale),
                radius=crop_radius,
                depth=None if depth is None else round(depth * scale),
                z_radius=z_radius,
            )

            cell = _ClickableThumbnail(object_id)
            cell.setPixmap(rgb_to_pixmap(rgb, size=thumbnail_size))
            cell.zoom_pixmap = rgb_to_pixmap(rgb, size=thumbnail_size * HOVER_SCALE)
            cell.setFixedSize(thumbnail_size + 6, thumbnail_size + 6)
            cell.setToolTip(f"object {object_id}")
            cell.clicked.connect(lambda oid=object_id: self._on_thumbnail_clicked(oid))
            cell.hovered.connect(lambda one=cell: self._show_zoom(one))
            cell.unhovered.connect(self._hide_zoom)
            self._grid.addWidget(cell, position // self._columns, position % self._columns)
            self._thumbnails.append(cell)

        # Keep the outline on the same object across a refresh where it is
        # still shown; drop it silently where it isn't.
        self.select(self.selected_object_id)

    def _set_channel_count(self, volume, channel_axis) -> None:
        levels = pyramid_levels(volume)
        shape = getattr(levels[0], "shape", ())
        count = shape[channel_axis] if channel_axis is not None and channel_axis < len(shape) else 0
        if count != self._n_channels:
            self._n_channels = int(count)
            self._refresh_channel_choices()

    def _tint_segmentation(
        self, rgb, labels, object_id, *, row_center, col_center, radius, depth, z_radius
    ):
        """Tint this object's own voxels, at level-0 coordinates.

        The segmentation exists at full resolution while the crop may have
        been read from a coarser level, so the mask is cropped in its own
        coordinates and put onto the crop's grid afterwards - see
        thumbnail.resize_nearest.
        """
        if labels is None or self.mask_opacity <= 0:
            return rgb
        mask_crop = crop_around(
            labels,
            row_center,
            col_center,
            radius=radius,
            depth=depth,
            z_radius=z_radius,
        )
        if mask_crop.size == 0:
            return rgb
        mask = max_projection((np.asarray(mask_crop) == object_id).astype(np.uint8)).astype(bool)
        return overlay_mask(rgb, resize_nearest(mask, rgb.shape[:2]), self.mask_color,
                            self.mask_opacity)

    # -- hover ------------------------------------------------------------

    def _show_zoom(self, cell: _ClickableThumbnail) -> None:
        """Show this crop at twice its size, beside the pointer."""
        if cell.zoom_pixmap is None:
            return
        self._zoom.setPixmap(cell.zoom_pixmap)
        self._zoom.resize(cell.zoom_pixmap.size())
        corner = cell.mapToGlobal(cell.rect().topRight())
        self._zoom.move(corner.x() + GRID_SPACING, corner.y())
        self._zoom.show()
        self._zoom.raise_()

    def _hide_zoom(self) -> None:
        self._zoom.hide()

    def hideEvent(self, event):  # Qt's spelling
        # A popup outlives its parent's visibility unless it is told not to.
        self._hide_zoom()
        super().hideEvent(event)

    def _on_thumbnail_clicked(self, object_id) -> None:
        self.select(object_id)
        self.object_selected.emit(object_id)


def limits_for(contrast_limits, channel: int | None):
    """The (low, high) for one channel, from whichever shape the caller had.

    A viewer showing a multi-channel image as one layer has a single pair;
    a caller that knows better can pass one per channel. Neither is worth
    making the caller normalise, so both are read here.
    """
    if contrast_limits is None:
        return None
    if isinstance(contrast_limits, dict):
        return contrast_limits.get(channel)
    values = list(contrast_limits)
    if len(values) == 2 and not hasattr(values[0], "__len__"):
        return tuple(values)
    if channel is not None and channel < len(values):
        return values[channel]
    return None


def select_channel(volume, channel_axis: int | None, channel: int | None):
    """One channel of a volume, without materializing the rest.

    A plain index rather than `np.take`, which on a Zarr array reads every
    channel to return one - the thing this whole module is careful not to
    do.
    """
    if channel is None or channel_axis is None:
        return volume
    ndim = len(getattr(volume, "shape", ()))
    if channel_axis >= ndim:
        return volume
    return volume[(slice(None),) * channel_axis + (channel,)]


def pyramid_levels(volume) -> list:
    """`volume` as a list of levels, finest first.

    napari hands a multiscale layer's data over as a list of arrays; a
    single array is a pyramid of one, so callers need no special case.
    """
    if isinstance(volume, (list, tuple)):
        return list(volume)
    return [volume]


def level_scale(levels: list, level: int) -> float:
    """How many level-0 voxels one voxel of `level` covers.

    Measured from the arrays rather than assumed to be a power of two: a
    store written by another tool is entitled to downsample by three, or by
    different factors per axis, and a centroid divided by the wrong number
    crops the wrong place.
    """
    if level <= 0 or not levels:
        return 1.0
    return float(levels[0].shape[-1]) / float(levels[level].shape[-1])


def choose_level(levels: list, crop_radius: int, thumbnail_size: int) -> int:
    """The coarsest level whose crop still fills the thumbnail.

    A 40-pixel crop shown at 64 pixels wants level 0. A 400-pixel crop shown
    at the same 64 wants a level about four times coarser, and reading it
    from level 0 is sixteen times the I/O for a picture nobody can tell
    apart. Stops before the crop gets smaller than the thumbnail, since
    scaling a 16-pixel crop up to 64 is visibly worse and saves nothing that
    matters.
    """
    chosen = 0
    for level in range(1, len(levels)):
        scale = level_scale(levels, level)
        if (2 * crop_radius) / scale < thumbnail_size * _LEVEL_MARGIN:
            break
        chosen = level
    return chosen


def crop_around(
    volume,
    row_center: int,
    col_center: int,
    *,
    radius: int,
    depth: int | None = None,
    z_radius: int | None = None,
) -> np.ndarray:
    """One object's neighbourhood, materialized and nothing else.

    The slice happens on whatever was passed - Zarr, Dask, NumPy - and
    `np.asarray` is applied to the result, so a volume larger than memory
    costs one bounding-box read rather than a load.
    """
    row0, row1 = max(0, row_center - radius), row_center + radius
    col0, col1 = max(0, col_center - radius), col_center + radius
    window: tuple = (slice(row0, row1), slice(col0, col1))
    if z_radius is not None and depth is not None and len(volume.shape) >= 3:
        window = (slice(max(0, depth - z_radius), depth + z_radius), *window)
        leading = len(volume.shape) - 3
    else:
        leading = len(volume.shape) - 2
    return np.asarray(volume[(slice(None),) * max(leading, 0) + window])
