import numpy as np
import pandas as pd
import pytest

from vtea_napari.widgets.plot import ScatterPlotWidget


class FakeMouseEvent:
    """Duck-types matplotlib's MouseEvent: xdata/ydata/button/dblclick are
    all the ScatterPlotWidget click handler reads."""

    def __init__(self, xdata, ydata, button=1, dblclick=False):
        self.xdata = xdata
        self.ydata = ydata
        self.button = button
        self.dblclick = dblclick


def make_frame():
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3],
            "mean": [1.0, 2.0, 3.0],
            "count": [10.0, 20.0, 30.0],
            "sum": [100.0, 200.0, 300.0],
        }
    )


class TestSetData:
    def test_populates_axis_combos_with_numeric_columns(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())

        choices = {widget.x_combo.itemText(i) for i in range(widget.x_combo.count())}
        assert choices == {"object_id", "mean", "count", "sum"}

    def test_defaults_to_first_two_columns(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        assert widget.x_column == "object_id"
        assert widget.y_column == "mean"

    def test_explicit_axes_are_honored(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame(), x_column="count", y_column="sum")
        assert widget.x_column == "count"
        assert widget.y_column == "sum"


class TestGateDrawing:
    def test_three_clicks_and_a_double_click_emits_a_triangle(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())

        received = []
        widget.gate_drawn.connect(lambda vertices: received.append(vertices))

        widget._on_click(FakeMouseEvent(0, 0))
        widget._on_click(FakeMouseEvent(10, 0))
        widget._on_click(FakeMouseEvent(5, 10))
        widget._on_click(FakeMouseEvent(5, 10, dblclick=True))

        assert len(received) == 1
        np.testing.assert_array_equal(received[0], [[0, 0], [10, 0], [5, 10]])

    def test_pending_vertices_cleared_after_emit(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())

        for point in [(0, 0), (10, 0), (5, 10)]:
            widget._on_click(FakeMouseEvent(*point))
        widget._on_click(FakeMouseEvent(5, 10, dblclick=True))

        assert widget._pending_vertices == []

    def test_fewer_than_three_vertices_does_not_emit(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())

        received = []
        widget.gate_drawn.connect(lambda vertices: received.append(vertices))

        widget._on_click(FakeMouseEvent(0, 0))
        widget._on_click(FakeMouseEvent(10, 0, dblclick=True))

        assert received == []

    def test_right_click_cancels_pending_gate(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())

        widget._on_click(FakeMouseEvent(0, 0))
        widget._on_click(FakeMouseEvent(10, 0))
        widget._on_click(FakeMouseEvent(5, 10, button=3))

        assert widget._pending_vertices == []

    def test_click_outside_axes_is_ignored(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())

        widget._on_click(FakeMouseEvent(None, None))

        assert widget._pending_vertices == []


class TestColorBy:
    def test_none_is_default(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        assert widget.color_column is None

    def test_selecting_a_column_sets_color_by(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.color_combo.setCurrentText("sum")
        assert widget.color_column == "sum"


class TestColormap:
    def test_defaults_to_first_option(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        assert widget.colormap == widget.colormap_combo.itemText(0)

    def test_selecting_a_colormap_updates_it(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.colormap_combo.setCurrentText("plasma")
        assert widget.colormap == "plasma"

    def test_redraw_with_color_by_and_colormap_does_not_crash(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.color_combo.setCurrentText("sum")
        widget.colormap_combo.setCurrentText("magma")
        widget._redraw()  # should not raise
