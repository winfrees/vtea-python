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
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from vtea_napari.widgets.thumbnail import array_to_pixmap, max_projection

_COLUMNS = 6

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


class _ClickableThumbnail(QLabel):
    clicked = Signal()

    def __init__(self, object_id, parent=None):
        super().__init__(parent)
        self.object_id = object_id
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(SELECTED_STYLE if selected else UNSELECTED_STYLE)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class GalleryWidget(QWidget):
    """A scrollable grid of per-object crops, one of which may be selected."""

    object_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._grid = QGridLayout()
        self._thumbnails: list[_ClickableThumbnail] = []
        self.selected_object_id: int | None = None
        container = QWidget()
        container.setLayout(self._grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

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

        levels = pyramid_levels(volume)
        level = choose_level(levels, crop_radius, thumbnail_size)
        source = levels[level]
        scale = level_scale(levels, level)
        depth_column = centroid_columns[-3] if len(centroid_columns) >= 3 else None

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
            crop = crop_around(
                source,
                row_center,
                col_center,
                radius=max(1, round(crop_radius / scale)),
                depth=depth,
                z_radius=None if z_radius is None else max(1, round(z_radius / scale)),
            )
            if crop.size == 0:
                continue
            pixmap = array_to_pixmap(max_projection(crop), size=thumbnail_size)

            cell = _ClickableThumbnail(object_id)
            cell.setPixmap(pixmap)
            cell.setToolTip(f"object {object_id}")
            cell.clicked.connect(lambda oid=object_id: self._on_thumbnail_clicked(oid))
            self._grid.addWidget(cell, position // _COLUMNS, position % _COLUMNS)
            self._thumbnails.append(cell)

        # Keep the outline on the same object across a refresh where it is
        # still shown; drop it silently where it isn't.
        self.select(self.selected_object_id)

    def _on_thumbnail_clicked(self, object_id) -> None:
        self.select(object_id)
        self.object_selected.emit(object_id)


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
