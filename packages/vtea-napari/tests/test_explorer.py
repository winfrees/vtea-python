import numpy as np
import pandas as pd
from qtpy.QtCore import Qt

from vtea_napari.widgets.explorer import ExplorerWidget


def make_frame():
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3, 4],
            "x": [2.0, 7.0, 12.0, 8.0],
            "y": [2.0, 7.0, 12.0, 3.0],
        }
    )


def make_labels():
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[0, 0] = 1
    labels[1, 1] = 2
    labels[2, 2] = 3
    labels[3, 3] = 4
    return labels


class TestSetData:
    def test_loads_frame_into_plot_and_table(self, qtbot):
        widget = ExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        assert widget.plot.x_column == "object_id"
        assert widget.table.rowCount() == 0  # no gates yet

    def test_resets_existing_gates(self, qtbot):
        widget = ExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame(), labels=make_labels())
        widget.plot.set_data(make_frame(), x_column="x", y_column="y")
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
        assert len(widget.gate_set) == 1

        widget.set_data(make_frame())
        assert len(widget.gate_set) == 0


class TestGateDrawing:
    def test_drawing_a_gate_adds_it_to_the_gate_set(self, qtbot):
        widget = ExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame(), labels=make_labels())
        widget.plot.set_data(make_frame(), x_column="x", y_column="y")

        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))

        assert len(widget.gate_set) == 1
        gate = next(iter(widget.gate_set))
        assert gate.name == "gate1"
        assert gate.x_axis == "x"
        assert gate.y_axis == "y"

    def test_drawing_a_gate_populates_the_table(self, qtbot):
        widget = ExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame(), labels=make_labels())
        widget.plot.set_data(make_frame(), x_column="x", y_column="y")

        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))

        assert widget.table.rowCount() == 1
        assert widget.table.item(0, 2).text() == "gate1"

    def test_second_gate_is_numbered_sequentially(self, qtbot):
        widget = ExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame(), labels=make_labels())
        widget.plot.set_data(make_frame(), x_column="x", y_column="y")

        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))

        names = {gate.name for gate in widget.gate_set}
        assert names == {"gate1", "gate2"}


class TestSubgating:
    def test_subgate_checkbox_makes_new_gate_a_child_of_selected(self, qtbot):
        widget = ExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame(), labels=make_labels())
        widget.plot.set_data(make_frame(), x_column="x", y_column="y")

        _draw_triangle(widget.plot, (-100, -100), (100, -100), (0, 100))
        parent = next(iter(widget.gate_set))
        widget._on_gate_selected(parent.id)

        widget.subgate_checkbox.setChecked(True)
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))

        child = [g for g in widget.gate_set if g.id != parent.id][0]
        assert child.parent_id == parent.id


class TestGateManagement:
    def test_renaming_via_table_updates_gate_set(self, qtbot):
        widget = ExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame(), labels=make_labels())
        widget.plot.set_data(make_frame(), x_column="x", y_column="y")
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
        gate = next(iter(widget.gate_set))

        widget.table.item(0, 2).setText("my gate")

        assert widget.gate_set.get(gate.id).name == "my gate"

    def test_deleting_selected_gate_removes_it(self, qtbot):
        widget = ExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame(), labels=make_labels())
        widget.plot.set_data(make_frame(), x_column="x", y_column="y")
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))

        # Select the way a user does - by clicking the gate's row - so the
        # gate manager knows which gate Delete applies to.
        widget.table.cellClicked.emit(0, 2)
        widget.gate_manager.delete_selected_gate()

        assert len(widget.gate_set) == 0
        assert widget.table.rowCount() == 0


class TestImageHighlighting:
    def test_selecting_a_gate_without_a_viewer_does_not_crash(self, qtbot):
        widget = ExplorerWidget(napari_viewer=None)
        qtbot.addWidget(widget)
        widget.set_data(make_frame(), labels=make_labels())
        widget.plot.set_data(make_frame(), x_column="x", y_column="y")
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
        gate = next(iter(widget.gate_set))

        widget._on_gate_selected(gate.id)  # should not raise

    def test_selecting_a_gate_emits_membership_mask(self, qtbot):
        widget = ExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame(), labels=make_labels())
        widget.plot.set_data(make_frame(), x_column="x", y_column="y")
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
        gate = next(iter(widget.gate_set))

        received = []
        widget.gate_membership_changed.connect(lambda gid, mask: received.append((gid, mask)))
        widget._on_gate_selected(gate.id)

        assert received[0][0] == gate.id
        expected = widget.gate_set.mask(gate.id, widget.frame)
        np.testing.assert_array_equal(received[0][1], expected)
        assert expected.any()  # the drawn triangle does contain at least one point

    def test_selecting_a_gate_adds_a_highlight_layer_in_a_real_viewer(self, qtbot):
        import napari

        viewer = napari.Viewer(show=False)
        qtbot.addWidget(viewer.window._qt_window)
        try:
            widget = ExplorerWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.set_data(make_frame(), labels=make_labels())
            widget.plot.set_data(make_frame(), x_column="x", y_column="y")
            _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
            gate = next(iter(widget.gate_set))

            widget._on_gate_selected(gate.id)

            assert "Gate highlight" in viewer.layers
            highlighted = viewer.layers["Gate highlight"].data
            expected_ids = set(make_frame().loc[widget.gate_set.mask(gate.id, widget.frame), "object_id"])
            assert set(np.unique(highlighted)) - {0} == expected_ids
        finally:
            viewer.close()

    def test_selecting_a_second_gate_reuses_the_same_highlight_layer(self, qtbot):
        import napari

        viewer = napari.Viewer(show=False)
        qtbot.addWidget(viewer.window._qt_window)
        try:
            widget = ExplorerWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.set_data(make_frame(), labels=make_labels())
            widget.plot.set_data(make_frame(), x_column="x", y_column="y")
            _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
            _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
            first_gate, second_gate = list(widget.gate_set)

            widget._on_gate_selected(first_gate.id)
            widget._on_gate_selected(second_gate.id)

            assert len([layer for layer in viewer.layers if layer.name == "Gate highlight"]) == 1
        finally:
            viewer.close()


class TestLoadFromActiveLayer:
    def test_loads_features_and_data_from_the_active_labels_layer(self, qtbot):
        import napari

        viewer = napari.Viewer(show=False)
        qtbot.addWidget(viewer.window._qt_window)
        try:
            layer = viewer.add_labels(make_labels(), name="objects")
            layer.features = make_frame()
            viewer.layers.selection.active = layer

            widget = ExplorerWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget._load_from_active_layer()

            assert len(widget.frame) == 4
            assert widget.labels is not None
            np.testing.assert_array_equal(widget.labels, make_labels())
        finally:
            viewer.close()

    def test_no_active_layer_shows_a_status_message_not_a_crash(self, qtbot):
        import napari

        viewer = napari.Viewer(show=False)
        qtbot.addWidget(viewer.window._qt_window)
        try:
            widget = ExplorerWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget._load_from_active_layer()
            assert widget.frame is None
            assert "Labels layer" in widget.status_label.text()
        finally:
            viewer.close()


class _FakeMouseEvent:
    """Duck-types matplotlib's MouseEvent - see test_plot.py's copy of this."""

    def __init__(self, xdata, ydata, button=1, dblclick=False):
        self.xdata = xdata
        self.ydata = ydata
        self.button = button
        self.dblclick = dblclick


def _draw_triangle(plot, p1, p2, p3):
    plot._on_click(_FakeMouseEvent(*p1))
    plot._on_click(_FakeMouseEvent(*p2))
    plot._on_click(_FakeMouseEvent(*p3))
    plot._on_click(_FakeMouseEvent(*p3, dblclick=True))


class TestFloatsByDefault:
    """A scatter plot docked into a narrow side panel is unusable at the
    size napari gives it, and gating means working between the plot and the
    image."""

    def test_it_floats_the_dock_it_is_put_in(self, qtbot):
        from qtpy.QtWidgets import QDockWidget, QMainWindow

        window = QMainWindow()
        qtbot.addWidget(window)
        dock = QDockWidget("Object Explorer", window)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        widget = ExplorerWidget(float_by_default=False)
        dock.setWidget(widget)
        assert widget.float_dock() is True
        assert dock.isFloating()

    def test_no_dock_is_not_an_error(self, qtbot):
        """Constructed standalone, in a script or a test."""
        widget = ExplorerWidget(float_by_default=False)
        qtbot.addWidget(widget)
        assert widget.float_dock() is False


class TestLayout:
    def test_the_plot_takes_two_thirds_of_the_results_row(self, qtbot):
        widget = ExplorerWidget(float_by_default=False)
        qtbot.addWidget(widget)
        widget.resize(900, 600)
        widget.show()
        qtbot.waitExposed(widget)

        plot_width, gate_width = widget.results_splitter.sizes()
        assert plot_width > gate_width
        share = plot_width / (plot_width + gate_width)
        assert 0.55 < share < 0.8, f"plot took {share:.0%} of the results row"

    def test_the_plot_canvas_grows_with_its_pane(self, qtbot):
        """A canvas with matplotlib's default Preferred size policy keeps
        its figsize however the pane is resized, which reads as the y axis
        simply not resizing."""
        from qtpy.QtWidgets import QSizePolicy

        widget = ExplorerWidget(float_by_default=False)
        qtbot.addWidget(widget)
        assert widget.plot.canvas.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding

        widget.resize(900, 500)
        widget.show()
        qtbot.waitExposed(widget)
        short = widget.plot.canvas.height()
        widget.resize(900, 1100)
        qtbot.waitUntil(lambda: widget.plot.canvas.height() > short, timeout=2000)

    def test_the_gate_manager_sits_beside_the_plot(self, qtbot):
        widget = ExplorerWidget(float_by_default=False)
        qtbot.addWidget(widget)
        assert widget.results_splitter.count() == 2
        assert widget.results_splitter.widget(0) is widget.plot
        assert widget.results_splitter.widget(1) is widget.gate_manager

    def test_the_gallery_is_a_second_tab(self, qtbot):
        widget = ExplorerWidget(float_by_default=False)
        qtbot.addWidget(widget)
        assert [widget.tabs.tabText(i) for i in range(widget.tabs.count())] == ["Plot", "Gallery"]
