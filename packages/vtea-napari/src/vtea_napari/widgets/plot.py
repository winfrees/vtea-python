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

The LUT has two modes, because the table now holds two kinds of number. A
measured feature is continuous and wants a gradient and a colorbar. A
cluster id, a class, an ROI membership - anything the clustering and class
steps produce - is *categorical*, and a gradient over it is actively
misleading: it puts cluster 2 "between" clusters 1 and 3, which is a
statement about a numbering, not about the tissue. Those get a distinct
colour each and a legend. Which one a column is is detected from the column
(see `is_discrete`) and can be overridden, because a manual override is the
honest escape hatch for the one column the heuristic reads wrong.

Rings around points are the plot's half of an image gate: a ring in the ROI
layer's own colour around every object that falls inside a region painted on
the image. Filled colour is already spoken for by the LUT, so membership of
something drawn on the *image* is drawn as an outline instead - the two
encodings can then be read at the same time, which is the whole point of
gating on both at once.

The canvas keeps a 4:3 aspect ratio whatever the pane does (AspectRatioBox).
A scatter plot whose proportions change as a dock is dragged is a plot whose
shape is an artefact of the furniture: two runs look different because the
window was a different size, and a cluster that looks elongated may only be
a narrow pane.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget
from vtea_core.gates import rectangle_vertices

_NO_COLOR_BY = "(none)"
NO_SIZE_BY = "(none)"

# Replaces vtea.lut's Fire/Black/RedGray/BlueGray/CustomLUT plugin family -
# a plain matplotlib colormap name list instead of separate LUT classes.
# The qualitative ones at the end are for the discrete mode: a set of
# distinguishable colours rather than a gradient, which is what a cluster id
# needs.
_COLORMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "turbo",
    "gray",
    "tab10",
    "tab20",
    "Set1",
    "Set2",
    "Dark2",
]

# How the colour encoding reads a column.
LUT_AUTO = "auto"
LUT_CONTINUOUS = "continuous"
LUT_DISCRETE = "discrete"
LUT_MODES = (LUT_AUTO, LUT_CONTINUOUS, LUT_DISCRETE)

# The colormap a discrete LUT falls back to when the chosen one is a
# gradient. Sampling a gradient at N points works and is what happens for
# more levels than this palette has, but ten distinguishable colours beat
# ten samples of viridis every time.
DEFAULT_DISCRETE_COLORMAP = "tab10"

# Above this many distinct values a column is a measurement, not a
# category - and a legend with fifty entries is not a legend. Auto-detection
# stops there; an explicit "Discrete" still draws it, with the legend
# truncated.
MAX_DISCRETE_LEVELS = 20

# How many legend entries a discrete LUT will draw before it stops and says
# how many are left. A legend taller than the axes hides the data.
MAX_LEGEND_ENTRIES = 12

# The plot's shape, kept whatever the pane does - see AspectRatioBox.
PLOT_ASPECT_RATIO = 4 / 3

# How much bigger than its point a ring is drawn, and how thick. Big enough
# to read as a ring around the point rather than a halo on it.
RING_SCALE = 2.6
RING_WIDTH = 1.6

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


def is_discrete(values, mode: str = LUT_AUTO) -> bool:
    """Whether a column should be coloured as categories rather than a range.

    Booleans and strings always are - a gate's membership, a class, a label
    name. A *numeric* column is when it looks like a code rather than a
    measurement: whole numbers, few of them, and starting at 0 or -1. That
    is exactly the shape of what the steps that produce categories produce -
    cluster ids are 0..k-1, an ROI membership is 0 for "in none", a label
    set's code is -1 for "unlabelled" - and it is not the shape of a voxel
    count or a mean intensity, which are whole numbers too but start
    wherever the data starts.

    Deliberately cautious about numbers: colouring a measurement as
    categories loses the ordering that is the whole content of it, while a
    category coloured as a gradient is merely ugly. Where the analysis knows
    better than the values do - the feature catalog records which step
    produced a column - the caller says so through `discrete_columns`, and
    `mode` overrides both.
    """
    if mode == LUT_CONTINUOUS:
        return False
    if mode == LUT_DISCRETE:
        return True
    series = pd.Series(values)
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return True
    finite = series.dropna()
    if finite.empty:
        return False
    numbers = finite.to_numpy()
    if not np.all(np.equal(np.mod(numbers, 1), 0)):
        return False
    if finite.nunique() > MAX_DISCRETE_LEVELS:
        return False
    # bool(), not numpy's: this is a decision the GUI branches on and
    # compares against True, and np.bool_ is a different object.
    return bool(numbers.min() in (-1, 0) and numbers.max() <= MAX_DISCRETE_LEVELS)


def discrete_colors(levels, colormap: str) -> list:
    """One colour per level, from a qualitative palette.

    A gradient asked to colour categories is sampled evenly across its whole
    range rather than at 0, 1, 2..., so ten clusters get ten *different*
    colours instead of ten neighbouring shades of the same one.
    """
    levels = list(levels)
    if not levels:
        return []
    chosen = colormaps[colormap] if colormap in colormaps else colormaps[DEFAULT_DISCRETE_COLORMAP]
    qualitative = getattr(chosen, "colors", None)
    if qualitative is not None and len(qualitative) >= len(levels):
        return [qualitative[index] for index in range(len(levels))]
    if len(levels) == 1:
        return [chosen(0.5)]
    return [chosen(index / (len(levels) - 1)) for index in range(len(levels))]


def _level_label(level) -> str:
    """A legend entry for one discrete value - `3` rather than `3.0`."""
    if isinstance(level, (bool, np.bool_)):
        return str(bool(level))
    if isinstance(level, (int, np.integer)):
        return str(int(level))
    if isinstance(level, (float, np.floating)) and float(level).is_integer():
        return str(int(level))
    return str(level)


def _ring_value(value) -> bool:
    """Whether a ring value means "ring this point"."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return bool(value)
    try:
        return not pd.isna(value) and float(value) != 0.0
    except (TypeError, ValueError):
        return bool(value)


def _as_key(level):
    """The ring colour map is keyed by ROI id; a float 2.0 read off a table
    column has to find the colour stored under 2."""
    if isinstance(level, (float, np.floating)) and float(level).is_integer():
        return int(level)
    if isinstance(level, np.integer):
        return int(level)
    return level


class AspectRatioBox(QWidget):
    """Holds one child at a fixed aspect ratio, centred, whatever its own size.

    Qt has no layout that says "this stays 4:3": `heightForWidth` needs
    cooperation from every parent layout and silently does nothing in a
    splitter. Placing the child by hand in `resizeEvent` is both simpler and
    exactly predictable - the box takes whatever space it is given and the
    plot keeps its proportions inside it, letterboxed.
    """

    def __init__(self, child: QWidget, ratio: float = PLOT_ASPECT_RATIO, parent=None):
        super().__init__(parent)
        self.ratio = float(ratio)
        self.child = child
        child.setParent(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def child_geometry(self, width: int, height: int) -> tuple[int, int, int, int]:
        """(x, y, width, height) for the child inside a `width` x `height`
        box - the largest rectangle of this ratio that fits, centred."""
        width = max(int(width), 1)
        height = max(int(height), 1)
        if width / height > self.ratio:
            # Too wide: the height is the constraint.
            child_height = height
            child_width = round(height * self.ratio)
        else:
            child_width = width
            child_height = round(width / self.ratio)
        return (
            max((width - child_width) // 2, 0),
            max((height - child_height) // 2, 0),
            child_width,
            child_height,
        )

    def resizeEvent(self, event):  # Qt's spelling
        super().resizeEvent(event)
        self.child.setGeometry(*self.child_geometry(self.width(), self.height()))


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
        # How the colour encoding reads its column: auto-detected, or forced
        # one way by the LUT-mode picker.
        self.lut_mode = LUT_AUTO
        # Columns the *analysis* knows are categorical even where the values
        # would not give it away - set by the explorer from the feature
        # catalog (a clustering's output, a class, a label-set code).
        self.discrete_columns: set[str] = set()
        # An image gate's rings: one value per row (0 = no ring) and the
        # colour each value is drawn in - see set_rings.
        self._ring_values = None
        self._ring_colors: dict = {}
        self._ring_label = ""
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
        self.lut_mode_combo = QComboBox()
        self.lut_mode_combo.addItems([mode.capitalize() for mode in LUT_MODES])
        self.lut_mode_combo.setToolTip(
            "How the colour column is read: a gradient for a measurement, "
            "a distinct colour per value for a cluster, class or ROI. "
            "Auto decides from the values."
        )
        self.lut_mode_combo.currentTextChanged.connect(self._on_lut_mode_changed)
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
        axis_row.addWidget(self.lut_mode_combo)
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
        # The canvas grows with the pane, but only in 4:3 - a plot whose
        # proportions follow the furniture is a plot whose shape is an
        # artefact of how somebody dragged a dock.
        self.canvas_box = AspectRatioBox(self.canvas)
        root.addWidget(self.canvas_box, 1)

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
            "lut_mode": self.lut_mode,
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
        mode = state.get("lut_mode")
        if mode in LUT_MODES:
            self.lut_mode = mode
            self.lut_mode_combo.blockSignals(True)
            self.lut_mode_combo.setCurrentText(mode.capitalize())
            self.lut_mode_combo.blockSignals(False)
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

    def color_is_discrete(self) -> bool:
        """Whether the current colour column is being read as categories."""
        if self._frame is None or not self.color_column:
            return False
        if self.color_column in self.discrete_columns and self.lut_mode != LUT_CONTINUOUS:
            return True
        return bool(is_discrete(self._frame[self.color_column], self.lut_mode))

    def set_discrete_columns(self, columns) -> None:
        """Columns the analysis knows are categorical - a clustering's
        output, a class, a label-set code - even where their values alone
        would not say so."""
        self.discrete_columns = set(columns or ())
        self._redraw()

    def set_lut_mode(self, mode: str) -> None:
        if mode not in LUT_MODES:
            raise ValueError(f"unknown LUT mode {mode!r}, expected one of {LUT_MODES}")
        self.lut_mode = mode
        self.lut_mode_combo.blockSignals(True)
        self.lut_mode_combo.setCurrentText(mode.capitalize())
        self.lut_mode_combo.blockSignals(False)
        self._redraw()
        self.view_changed.emit()

    def _on_lut_mode_changed(self, text: str) -> None:
        self.lut_mode = (text or LUT_AUTO).lower()
        self._redraw()
        self.view_changed.emit()

    def set_rings(self, values=None, colors=None, label: str = "") -> None:
        """Ring the points whose `values` entry is non-zero.

        This is an image gate on the plot: `values` is one entry per row -
        the ROI id each object falls in, 0 for none - and `colors` maps an
        id to the colour that region is drawn in on the image, so the same
        population is the same colour in both places. A single colour string
        rings everything in it.
        """
        self._ring_values = None if values is None else np.asarray(values)
        if isinstance(colors, str):
            self._ring_colors = {"*": colors}
        else:
            self._ring_colors = dict(colors or {})
        self._ring_label = label
        self._redraw()

    def _draw_scatter(self, x, y, sizes):
        """The points themselves, coloured however the LUT says."""
        if not self.color_column:
            return self.ax.scatter(
                x, y, s=sizes, c="tab:blue", alpha=self.alpha, marker=self.marker
            )
        values = self._frame[self.color_column]
        if not self.color_is_discrete():
            scatter = self.ax.scatter(
                x,
                y,
                c=values.to_numpy(),
                cmap=self.colormap,
                s=sizes,
                alpha=self.alpha,
                marker=self.marker,
            )
            # The LUT bar: a colour gradient means nothing without the
            # values it stands for.
            self._colorbar = self.figure.colorbar(scatter, ax=self.ax)
            self._colorbar.set_label(self.color_column)
            return scatter
        return self._draw_discrete_scatter(x, y, sizes, values)

    def _draw_discrete_scatter(self, x, y, sizes, values):
        """A colour per distinct value, and a legend saying which is which.

        One scatter for all of them rather than one per level: the size
        encoding, the picking and the ring overlay all stay in step that
        way, and a legend is drawn from proxy handles instead.
        """
        levels = pd.Series(values).dropna().unique().tolist()
        levels.sort(key=lambda value: (isinstance(value, str), value))
        palette = discrete_colors(levels, self.colormap)
        lookup = {level: colour for level, colour in zip(levels, palette)}
        point_colors = [lookup.get(value, (0.6, 0.6, 0.6, 1.0)) for value in values]
        scatter = self.ax.scatter(
            x, y, c=point_colors, s=sizes, alpha=self.alpha, marker=self.marker
        )
        self._draw_level_legend(levels, palette)
        return scatter

    def _draw_level_legend(self, levels, palette) -> None:
        shown = levels[:MAX_LEGEND_ENTRIES]
        handles = [
            Line2D(
                [],
                [],
                marker=self.marker,
                linestyle="none",
                markerfacecolor=colour,
                markeredgecolor="none",
                markersize=6,
            )
            for colour in palette[: len(shown)]
        ]
        labels = [_level_label(level) for level in shown]
        if len(levels) > len(shown):
            handles.append(Line2D([], [], linestyle="none"))
            labels.append(f"+{len(levels) - len(shown)} more")
        self.ax.legend(
            handles,
            labels,
            title=self.color_column,
            loc="best",
            fontsize="x-small",
            title_fontsize="x-small",
            framealpha=0.7,
        )

    def _draw_rings(self, x, y, sizes) -> None:
        """An unfilled marker around every point an image gate selects.

        Drawn on top of the points, in the ROI's own colour, so a napari
        region and the objects inside it read as the same population in both
        windows.
        """
        values = self._ring_values
        if values is None or len(values) != len(x):
            return
        ringed = np.asarray([_ring_value(value) for value in values])
        if not ringed.any():
            return
        areas = np.full(len(x), float(sizes)) if np.isscalar(sizes) else np.asarray(sizes, float)
        single = self._ring_colors.get("*")
        for level in sorted({value for value, keep in zip(values, ringed) if keep}, key=str):
            selected = np.asarray([keep and value == level for value, keep in zip(values, ringed)])
            colour = single or self._ring_colors.get(level) or self._ring_colors.get(
                _as_key(level), "black"
            )
            self.ax.scatter(
                np.asarray(x)[selected],
                np.asarray(y)[selected],
                s=areas[selected] * RING_SCALE,
                facecolors="none",
                edgecolors=colour,
                linewidths=RING_WIDTH,
                marker=self.marker,
            )

    def _redraw(self) -> None:
        # The colorbar lives in its own axes, so it has to go before the
        # main axes are cleared or it accumulates one bar per redraw.
        self._remove_colorbar()
        self.ax.clear()
        if self._frame is not None and self.x_column and self.y_column:
            x = self._frame[self.x_column].to_numpy()
            y = self._frame[self.y_column].to_numpy()
            sizes, to_value = self._point_sizes()
            scatter = self._draw_scatter(x, y, sizes)
            self._draw_rings(x, y, sizes)
            if to_value is not None and not self.color_is_discrete():
                # One legend at a time: the discrete LUT has already claimed
                # it, and two legends over one set of axes is a puzzle
                # rather than a key.
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
