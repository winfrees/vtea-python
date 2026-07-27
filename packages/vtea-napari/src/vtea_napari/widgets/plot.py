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
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

_NO_COLOR_BY = "(none)"

# Replaces vtea.lut's Fire/Black/RedGray/BlueGray/CustomLUT plugin family -
# a plain matplotlib colormap name list instead of separate LUT classes.
_COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "gray"]


class ScatterPlotWidget(QWidget):
    """A matplotlib scatter plot; click to add a gate vertex, double-click to
    close the polygon and emit it, right-click to cancel the in-progress gate.
    """

    gate_drawn = Signal(object)  # np.ndarray, shape (N, 2), in data coordinates
    axes_changed = Signal(str, str)  # x_column, y_column

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._frame: pd.DataFrame | None = None
        self.x_column: str | None = None
        self.y_column: str | None = None
        self.color_column: str | None = None
        self.colormap = _COLORMAPS[0]
        self._pending_vertices: list[tuple[float, float]] = []
        self._gate_overlays: list = []  # vtea_core.gates.Gate instances

        root = QVBoxLayout(self)

        axis_row = QHBoxLayout()
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.color_combo = QComboBox()
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(_COLORMAPS)
        self.x_combo.currentTextChanged.connect(self._on_axis_combo_changed)
        self.y_combo.currentTextChanged.connect(self._on_axis_combo_changed)
        self.color_combo.currentTextChanged.connect(self._on_color_combo_changed)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_combo_changed)
        axis_row.addWidget(QLabel("X:"))
        axis_row.addWidget(self.x_combo)
        axis_row.addWidget(QLabel("Y:"))
        axis_row.addWidget(self.y_combo)
        axis_row.addWidget(QLabel("Color by:"))
        axis_row.addWidget(self.color_combo)
        axis_row.addWidget(QLabel("LUT:"))
        axis_row.addWidget(self.colormap_combo)
        root.addLayout(axis_row)

        self.figure = Figure(figsize=(5, 4))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        root.addWidget(self.canvas)

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
        self.color_combo.blockSignals(True)
        self.color_combo.clear()
        self.color_combo.addItems([_NO_COLOR_BY] + numeric_columns)
        self.color_combo.blockSignals(False)

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
        if self.x_column and self.y_column:
            self.axes_changed.emit(self.x_column, self.y_column)

    def _on_color_combo_changed(self, text: str) -> None:
        self.color_column = None if text in ("", _NO_COLOR_BY) else text
        self._redraw()

    def _on_colormap_combo_changed(self, text: str) -> None:
        self.colormap = text
        self._redraw()

    def _redraw(self) -> None:
        self.ax.clear()
        if self._frame is not None and self.x_column and self.y_column:
            x = self._frame[self.x_column].to_numpy()
            y = self._frame[self.y_column].to_numpy()
            if self.color_column:
                c = self._frame[self.color_column].to_numpy()
                self.ax.scatter(x, y, c=c, cmap=self.colormap, s=12)
            else:
                self.ax.scatter(x, y, s=12, c="tab:blue")
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
