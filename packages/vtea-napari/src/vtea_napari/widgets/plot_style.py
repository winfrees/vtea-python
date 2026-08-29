"""The plot's style helper pane: how the points are drawn.

Separate from the axis pickers above the plot, because these are a different
kind of choice. What goes on the axes is part of the analysis; how big the
dots are, how transparent, and what shape is about being able to *see* the
analysis - a few thousand overlapping opaque circles hide the structure
underneath them, and the fix is turning transparency down, not re-running
anything.

Drives a ScatterPlotWidget through `set_point_style`; owns no data of its
own, so the plot stays the single source of truth for what is drawn.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from vtea_napari.widgets.plot import DEFAULT_SIZE_RANGE, MARKERS

# The alpha slider is integer percent; matplotlib wants 0-1.
_ALPHA_STEPS = 100


class PlotStylePanel(QGroupBox):
    """Point size, size range, transparency and marker shape for one plot."""

    def __init__(self, plot, parent: QWidget | None = None):
        super().__init__("Point style", parent)
        self.plot = plot

        grid = QGridLayout()
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setVerticalSpacing(3)

        # Base size: what every point gets when nothing encodes size.
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(1.0, 400.0)
        self.size_spin.setDecimals(0)
        self.size_spin.setValue(plot.point_size)
        self.size_spin.setToolTip("Marker area when 'Size by' is (none)")
        self.size_spin.valueChanged.connect(self._apply)
        grid.addWidget(QLabel("Size:"), 0, 0)
        grid.addWidget(self.size_spin, 0, 1)

        # Transparency, as opacity percent - the control people reach for
        # when the points overlap.
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(5, _ALPHA_STEPS)
        self.alpha_slider.setValue(int(plot.alpha * _ALPHA_STEPS))
        self.alpha_slider.setToolTip("Point opacity - lower it to see through dense clusters")
        self.alpha_slider.valueChanged.connect(self._apply)
        self.alpha_label = QLabel()
        grid.addWidget(QLabel("Opacity:"), 0, 2)
        grid.addWidget(self.alpha_slider, 0, 3)
        grid.addWidget(self.alpha_label, 0, 4)

        self.marker_combo = QComboBox()
        self.marker_combo.addItems(list(MARKERS))
        self.marker_combo.currentTextChanged.connect(self._apply)
        grid.addWidget(QLabel("Shape:"), 1, 0)
        grid.addWidget(self.marker_combo, 1, 1)

        # The span a size-encoded feature is mapped onto. Only meaningful
        # with a "Size by" feature chosen, but harmless otherwise.
        self.min_size_spin = QDoubleSpinBox()
        self.min_size_spin.setRange(1.0, 400.0)
        self.min_size_spin.setDecimals(0)
        self.min_size_spin.setValue(DEFAULT_SIZE_RANGE[0])
        self.max_size_spin = QDoubleSpinBox()
        self.max_size_spin.setRange(1.0, 800.0)
        self.max_size_spin.setDecimals(0)
        self.max_size_spin.setValue(DEFAULT_SIZE_RANGE[1])
        for spin in (self.min_size_spin, self.max_size_spin):
            spin.setToolTip("Smallest and largest marker area when sizing by a feature")
            spin.valueChanged.connect(self._apply)
        grid.addWidget(QLabel("Size range:"), 1, 2)
        grid.addWidget(self.min_size_spin, 1, 3)
        grid.addWidget(self.max_size_spin, 1, 4)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(grid)

        self._refresh_alpha_label()

    def _refresh_alpha_label(self) -> None:
        self.alpha_label.setText(f"{self.alpha_slider.value()}%")

    def _apply(self, *_args) -> None:
        self._refresh_alpha_label()
        low = self.min_size_spin.value()
        high = self.max_size_spin.value()
        if high < low:
            # A range the wrong way round would invert the encoding
            # silently; read it in the order the user meant.
            low, high = high, low
        self.plot.set_point_style(
            point_size=self.size_spin.value(),
            alpha=self.alpha_slider.value() / _ALPHA_STEPS,
            marker=MARKERS[self.marker_combo.currentText()],
            size_range=(low, high),
        )
