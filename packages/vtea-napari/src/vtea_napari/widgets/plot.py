"""A scatter plot of a measurement table with click-to-draw polygon gating
and point coloring by a third feature.

Replaces vtea.exploration.plottools.panels' JFreeChart-based XYChartPanel/
XYExplorationPanel plus GateLayer (the JXLayer mouse-interaction overlay
that turned clicks into vtea.exploration.plotgatetools.gates.PolygonGate
instances). Gates there kept vertices in both screen and chart-data space,
with manual java2DToValue/valueToJava2D conversions, purely because
JFreeChart's ChartPanel couples geometry to pixel layout - matplotlib gives
data coordinates directly on every mouse event, so this widget only ever
deals in one coordinate space. Point coloring by a chosen feature replaces
vtea.lut's small LookupPaintScale-based LUT family (discretized into 11
bands); a continuous matplotlib colormap does the same job without the
banding, since nothing here depends on the Cividis/16-color-esque display
tricks JFreeChart's LookupPaintScale needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget
from vtea_core.gates import rectangle_vertices

_NO_COLOR_BY = "(none)"
NO_SIZE_BY = "(none)"

# Replaces vtea.lut's Fire/Black/RedGray/BlueGray/CustomLUT plugin family -
# a plain matplotlib colormap name list instead of separate LUT classes.
_COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "gray"]

# Below this the axes are all margin and no data. The canvas is otherwise
# free to grow with its pane.
_MINIMUM_CANVAS_HEIGHT = 160

POLYGON_MODE = "polygon"
RECTANGLE_MODE = "rectangle"

# Marker shapes offered in the style pane, by the name a person would use
# for them rather than matplotlib's single-character codes.
MARKERS = {
    "circle": "o",
    "square": "s",
    "triangle": "^",
    "diamond": "D",
    "plus": "P",
    "cross": "X",
}

DEFAULT_POINT_SIZE = 12.0
DEFAULT_ALPHA = 1.0
# The span a size-encoded feature is mapped onto, in the same points² units
# matplotlib's `s` takes. Wide enough for the difference to read at a
# glance, floored high enough that the smallest objects stay clickable.
DEFAULT_SIZE_RANGE = (8.0, 120.0)

# Points sampled for the size legend - enough to read the scale off,
# few enough not to crowd the axes.
_SIZE_LEGEND_SAMPLES = 4


def scale_sizes(values, min_size: float, max_size: float):
    """Map a feature's values onto marker areas, plus the inverse.

    Returns `(sizes, to_value)` where `to_value` converts a marker area back
    to the feature value it stands for - which is what lets the size legend
    be labelled in the feature's own units instead of in points².

    A constant feature (or one that is all NaN) has no spread to encode, so
    every point gets the middle of the range and `to_value` reports that
    constant: a legend claiming a gradient over identical values would be a
    lie.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    midpoint = (min_size + max_size) / 2.0
    if not finite.any():
        return np.full(values.shape, midpoint), (lambda sizes: np.full_like(np.asarray(sizes, dtype=float), np.nan))

    low = float(values[finite].min())
    high = float(values[finite].max())
    if high <= low:
        return (
            np.full(values.shape, midpoint),
            lambda sizes: np.full_like(np.asarray(sizes, dtype=float), low),
        )

    normalized = np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0.0, 1.0)
    sizes = min_size + normalized * (max_size - min_size)

    def to_value(size_values):
        fraction = (np.asarray(size_values, dtype=float) - min_size) / (max_size - min_size)
        return low + fraction * (high - low)

    return sizes, to_value


class ScatterPlotWidget(QWidget):
    """A matplotlib scatter plot; click to add a gate vertex, double-click to
    close the polygon and emit it, right-click to cancel the in-progress gate.

    In rectangle mode two clicks - opposite corners - make the gate instead.
    It is still emitted as a 4-vertex polygon, so everything downstream
    (membership, overlays, JSON) handles one kind of gate.
    """

    gate_drawn = Signal(object)  # np.ndarray, shape (N, 2), in data coordinates
    axes_changed = Signal(str, str)  # x_column, y_column
    # Any change to how the plot is set up - axes, encodings, point style -
    # so the owner can keep it somewhere that outlives this widget.
    view_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._frame: pd.DataFrame | None = None
        self.x_column: str | None = None
        self.y_column: str | None = None
        self.color_column: str | None = None
        self.size_column: str | None = None
        self.colormap = _COLORMAPS[0]
        self.gate_mode = POLYGON_MODE
        # Point styling, driven by PlotStylePanel.
        self.point_size = DEFAULT_POINT_SIZE
        self.alpha = DEFAULT_ALPHA
        self.marker = MARKERS["circle"]
        self.size_range = DEFAULT_SIZE_RANGE
        self._pending_vertices: list[tuple[float, float]] = []
        self._gate_overlays: list = []  # vtea_core.gates.Gate instances
        self._colorbar = None

        root = QVBoxLayout(self)

        axis_row = QHBoxLayout()
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.color_combo = QComboBox()
        self.size_combo = QComboBox()
        self.size_combo.setToolTip("Scale each point by a measured feature")
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(_COLORMAPS)
        self.x_combo.currentTextChanged.connect(self._on_axis_combo_changed)
        self.y_combo.currentTextChanged.connect(self._on_axis_combo_changed)
        self.color_combo.currentTextChanged.connect(self._on_color_combo_changed)
        self.size_combo.currentTextChanged.connect(self._on_size_combo_changed)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_combo_changed)
        axis_row.addWidget(QLabel("X:"))
        axis_row.addWidget(self.x_combo)
        axis_row.addWidget(QLabel("Y:"))
        axis_row.addWidget(self.y_combo)
        axis_row.addWidget(QLabel("Color by:"))
        axis_row.addWidget(self.color_combo)
        axis_row.addWidget(QLabel("Size by:"))
        axis_row.addWidget(self.size_combo)
        axis_row.addWidget(QLabel("LUT:"))
        axis_row.addWidget(self.colormap_combo)
        root.addLayout(axis_row)

        # constrained layout keeps the axis labels inside the canvas as the
        # pane is resized; without it the y label is the first thing clipped
        # when the plot is made short, which looked like the y axis simply
        # not resizing.
        self.figure = Figure(figsize=(5, 4), layout="constrained")
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        # A QWidget's default size policy is Preferred, which lets the canvas
        # keep its figsize rather than growing with the splitter pane.
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumHeight(_MINIMUM_CANVAS_HEIGHT)
        root.addWidget(self.canvas, 1)

        self.canvas.mpl_connect("button_press_event", self._on_click)

    def set_data(self, frame: pd.DataFrame, x_column: str | None = None, y_column: str | None = None) -> None:
        """Loads a measurement table; populates the axis/color-by pickers
        from its numeric columns."""
        self._frame = frame
        numeric_columns = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]

        for combo in (self.x_combo, self.y_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(numeric_columns)
            combo.blockSignals(False)
        # Keep the colour and size encodings across a re-run: they are as
        # much part of "the plot I am looking at" as the axes are.
        for combo, placeholder, current in (
            (self.color_combo, _NO_COLOR_BY, self.color_column),
            (self.size_combo, NO_SIZE_BY, self.size_column),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([placeholder] + numeric_columns)
            if current and current in numeric_columns:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        self.color_column = self.color_column if self.color_column in numeric_columns else None
        self.size_column = self.size_column if self.size_column in numeric_columns else None

        if x_column and x_column in numeric_columns:
            self.x_combo.setCurrentText(x_column)
        elif numeric_columns:
            self.x_combo.setCurrentIndex(0)
        if y_column and y_column in numeric_columns:
            self.y_combo.setCurrentText(y_column)
        elif len(numeric_columns) > 1:
            self.y_combo.setCurrentIndex(1)

        self.x_column = self.x_combo.currentText() or None
        self.y_column = self.y_combo.currentText() or None
        self._pending_vertices = []
        self._redraw()

    def view_state(self) -> dict:
        """Everything about how this plot is set up, as plain values.

        Kept on the session so closing the Object Explorer's dock - which
        destroys the widget - doesn't cost the view.
        """
        return {
            "x_column": self.x_column,
            "y_column": self.y_column,
            "color_column": self.color_column,
            "size_column": self.size_column,
            "colormap": self.colormap,
            "point_size": self.point_size,
            "alpha": self.alpha,
            "marker": self.marker,
            "size_range": tuple(self.size_range),
        }

    def apply_view_state(self, state: dict) -> None:
        """Restore what `view_state` recorded. Columns that are no longer in
        the table are skipped rather than forced: the point is to come back
        to the same view where that still means something."""
        if not state:
            return
        self.point_size = float(state.get("point_size", self.point_size))
        self.alpha = float(state.get("alpha", self.alpha))
        self.marker = state.get("marker", self.marker)
        size_range = state.get("size_range")
        if size_range:
            self.size_range = (float(size_range[0]), float(size_range[1]))

        for combo, key, placeholder in (
            (self.x_combo, "x_column", None),
            (self.y_combo, "y_column", None),
            (self.color_combo, "color_column", _NO_COLOR_BY),
            (self.size_combo, "size_column", NO_SIZE_BY),
            (self.colormap_combo, "colormap", None),
        ):
            wanted = state.get(key)
            combo.blockSignals(True)
            if wanted and combo.findText(wanted) != -1:
                combo.setCurrentText(wanted)
            elif placeholder is not None and combo.findText(placeholder) != -1:
                combo.setCurrentText(placeholder)
            combo.blockSignals(False)

        self.x_column = self.x_combo.currentText() or None
        self.y_column = self.y_combo.currentText() or None
        color = self.color_combo.currentText()
        self.color_column = None if color in ("", _NO_COLOR_BY) else color
        size = self.size_combo.currentText()
        self.size_column = None if size in ("", NO_SIZE_BY) else size
        self.colormap = self.colormap_combo.currentText() or self.colormap
        self._redraw()

    def set_gate_mode(self, mode: str) -> None:
        """Switch between polygon and rectangle drawing, discarding whatever
        was half-drawn - a click meant as a polygon vertex should not become
        a rectangle corner."""
        if mode not in (POLYGON_MODE, RECTANGLE_MODE):
            raise ValueError(f"unknown gate mode {mode!r}")
        self.gate_mode = mode
        self._pending_vertices = []
        self._redraw()

    def set_gate_overlays(self, gates: list) -> None:
        """Gate outlines to draw on top of the scatter (only those whose axes
        match the current x/y are actually shown)."""
        self._gate_overlays = list(gates)
        self._redraw()

    def _on_axis_combo_changed(self, _text: str) -> None:
        self.x_column = self.x_combo.currentText() or None
        self.y_column = self.y_combo.currentText() or None
        self._pending_vertices = []
        self._redraw()
        self.view_changed.emit()
        if self.x_column and self.y_column:
            self.axes_changed.emit(self.x_column, self.y_column)

    def _on_color_combo_changed(self, text: str) -> None:
        self.color_column = None if text in ("", _NO_COLOR_BY) else text
        self._redraw()
        self.view_changed.emit()

    def _on_size_combo_changed(self, text: str) -> None:
        self.size_column = None if text in ("", NO_SIZE_BY) else text
        self._redraw()
        self.view_changed.emit()

    def _on_colormap_combo_changed(self, text: str) -> None:
        self.colormap = text
        self._redraw()
        self.view_changed.emit()

    def set_point_style(
        self,
        *,
        point_size: float | None = None,
        alpha: float | None = None,
        marker: str | None = None,
        size_range: tuple[float, float] | None = None,
    ) -> None:
        """How the points are drawn - driven by PlotStylePanel. `point_size`
        is the area every point gets when nothing is encoding size."""
        if point_size is not None:
            self.point_size = float(point_size)
        if alpha is not None:
            self.alpha = float(alpha)
        if marker is not None:
            self.marker = marker
        if size_range is not None:
            self.size_range = (float(size_range[0]), float(size_range[1]))
        self._redraw()
        self.view_changed.emit()

    def _point_sizes(self):
        """(sizes, to_value): the marker area for each point, and how to read
        an area back as a feature value for the legend."""
        if self.size_column is None or self._frame is None:
            return self.point_size, None
        return scale_sizes(
            self._frame[self.size_column].to_numpy(), self.size_range[0], self.size_range[1]
        )

    def _remove_colorbar(self) -> None:
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except (AttributeError, KeyError, ValueError):
                # Already gone with a cleared figure; nothing to undo.
                pass
            self._colorbar = None

    def _redraw(self) -> None:
        # The colorbar lives in its own axes, so it has to go before the
        # main axes are cleared or it accumulates one bar per redraw.
        self._remove_colorbar()
        self.ax.clear()
        if self._frame is not None and self.x_column and self.y_column:
            x = self._frame[self.x_column].to_numpy()
            y = self._frame[self.y_column].to_numpy()
            sizes, to_value = self._point_sizes()
            if self.color_column:
                c = self._frame[self.color_column].to_numpy()
                scatter = self.ax.scatter(
                    x, y, c=c, cmap=self.colormap, s=sizes, alpha=self.alpha, marker=self.marker
                )
                # The LUT bar: a colour gradient means nothing without the
                # values it stands for.
                self._colorbar = self.figure.colorbar(scatter, ax=self.ax)
                self._colorbar.set_label(self.color_column)
            else:
                scatter = self.ax.scatter(
                    x, y, s=sizes, c="tab:blue", alpha=self.alpha, marker=self.marker
                )
            if to_value is not None:
                self._draw_size_legend(scatter, to_value)
            self.ax.set_xlabel(self.x_column)
            self.ax.set_ylabel(self.y_column)

        for gate in self._gate_overlays:
            if gate.x_axis != self.x_column or gate.y_axis != self.y_column or not gate.visible:
                continue
            closed = np.vstack([gate.vertices, gate.vertices[:1]])
            self.ax.plot(closed[:, 0], closed[:, 1], color=gate.color, linewidth=1.5)

        if self._pending_vertices:
            pending = np.array(self._pending_vertices)
            self.ax.plot(
                pending[:, 0], pending[:, 1], color="black", linestyle="--", marker="o", markersize=3
            )

        self.canvas.draw_idle()

    def _draw_size_legend(self, scatter, to_value) -> None:
        """The size equivalent of the colorbar: a few sample dots labelled
        with the feature values they stand for.

        Labelled through `to_value` rather than in matplotlib's points²,
        which would be a number about the drawing rather than about the
        data.
        """
        try:
            handles, labels = scatter.legend_elements(
                prop="sizes", num=_SIZE_LEGEND_SAMPLES, func=to_value, alpha=0.6
            )
        except (ValueError, TypeError):
            # A degenerate spread (every point the same size) has no scale
            # worth showing; the plot is still correct without it.
            return
        if not handles:
            return
        self.ax.legend(
            handles,
            labels,
            title=self.size_column,
            loc="best",
            fontsize="x-small",
            title_fontsize="x-small",
            labelspacing=1.0,
            framealpha=0.7,
        )

    def _on_click(self, event) -> None:
        """`event` is (or duck-types) a matplotlib MouseEvent: .xdata, .ydata,
        .button, .dblclick. Left double-click closes an in-progress polygon
        of >= 3 vertices and emits it; right-click cancels; any other left
        click appends a vertex."""
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 3:
            self._pending_vertices = []
            self._redraw()
            return
        if event.button != 1:
            return
        if self.gate_mode == RECTANGLE_MODE:
            if event.dblclick:
                # The second half of a double-click lands on the same spot as
                # the first, which would close a zero-area rectangle.
                return
            self._pending_vertices.append((event.xdata, event.ydata))
            if len(self._pending_vertices) == 2:
                (x0, y0), (x1, y1) = self._pending_vertices
                self._pending_vertices = []
                self._redraw()
                self.gate_drawn.emit(rectangle_vertices(x0, y0, x1, y1))
            else:
                self._redraw()
            return
        if event.dblclick:
            if len(self._pending_vertices) >= 3:
                vertices = np.array(self._pending_vertices)
                self._pending_vertices = []
                self._redraw()
                self.gate_drawn.emit(vertices)
            else:
                self._pending_vertices = []
                self._redraw()
            return
        self._pending_vertices.append((event.xdata, event.ydata))
        self._redraw()
