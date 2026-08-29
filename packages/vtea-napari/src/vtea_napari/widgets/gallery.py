"""Per-object thumbnail grid: shows cropped image previews for a set of
object ids (typically a gate's members), and reports which one is clicked.

Replaces vtea.exploration.gallery.GalleryViewWindow/GalleryImageProcessor/
GalleryViewDataProvider. Unlike vteaexploration.GateManager/
microGateManager (see gate_table.py's docstring), this is real, working
Java code - right-click a gate -> "Gallery View..." opens exactly this: a
grid of per-object crops around each object's centroid, click one to
highlight it back on the scatter plot. Ported with the same behavior: a
fixed-radius crop around each object's centroid, max-projected to 2D.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from vtea_napari.widgets.thumbnail import array_to_pixmap, max_projection

_COLUMNS = 6

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
        volume: np.ndarray,
        frame: pd.DataFrame,
        object_ids,
        *,
        crop_radius: int = 20,
        thumbnail_size: int = 64,
    ) -> None:
        """`volume` is an intensity array whose last two axes are (row,
        col); `frame` has `object_id` and `centroid-*` columns (see
        vtea_core.measurements.extract_measurements) - the last two
        centroid columns are always the array's own (row, col) axes,
        whether the source volume was 2D or 3D."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._thumbnails = []

        indexed = frame.set_index("object_id")
        centroid_columns = sorted(c for c in frame.columns if c.startswith("centroid-"))
        spatial_columns = centroid_columns[-2:]

        for position, object_id in enumerate(object_ids):
            if object_id not in indexed.index:
                continue
            row_center, col_center = (round(indexed.loc[object_id, c]) for c in spatial_columns)
            crop = _crop_2d(volume, row_center, col_center, crop_radius)
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


def _crop_2d(volume: np.ndarray, row_center: int, col_center: int, radius: int) -> np.ndarray:
    row0, row1 = max(0, row_center - radius), row_center + radius
    col0, col1 = max(0, col_center - radius), col_center + radius
    return volume[..., row0:row1, col0:col1]
