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

    def test_each_gate_gets_its_own_highlight_layer_in_a_real_viewer(self, qtbot):
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

            name = f"Gate highlight: {gate.name}"
            assert name in viewer.layers
            highlighted = viewer.layers[name].data
            expected_ids = set(
                make_frame().loc[widget.gate_set.mask(gate.id, widget.frame), "object_id"]
            )
            assert set(np.unique(highlighted)) - {0} == expected_ids
        finally:
            viewer.close()

    def test_a_second_gate_adds_a_second_layer_rather_than_replacing_the_first(self, qtbot):
        """Two gates are two populations; showing only the last-selected one
        would make comparing them impossible."""
        import napari

        viewer = napari.Viewer(show=False)
        qtbot.addWidget(viewer.window._qt_window)
        try:
            widget = ExplorerWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.set_data(make_frame(), labels=make_labels())
            widget.plot.set_data(make_frame(), x_column="x", y_column="y")
            _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
            _draw_triangle(widget.plot, (-100, -100), (100, -100), (0, 100))
            first_gate, second_gate = list(widget.gate_set)

            widget._on_gate_selected(first_gate.id)
            widget._on_gate_selected(second_gate.id)

            names = {layer.name for layer in viewer.layers}
            assert f"Gate highlight: {first_gate.name}" in names
            assert f"Gate highlight: {second_gate.name}" in names
        finally:
            viewer.close()

    def test_re_selecting_does_not_stack_up_layers(self, qtbot):
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

            for _ in range(4):
                widget._on_gate_selected(gate.id)

            name = f"Gate highlight: {gate.name}"
            assert len([layer for layer in viewer.layers if layer.name == name]) == 1
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
        # The left pane is the plot with its style helper under it.
        left = widget.results_splitter.widget(0)
        assert widget.plot.parentWidget() is left
        assert widget.style_panel.parentWidget() is left
        assert widget.results_splitter.widget(1) is widget.gate_manager

    def test_the_gallery_is_a_second_tab(self, qtbot):
        widget = ExplorerWidget(float_by_default=False)
        qtbot.addWidget(widget)
        assert [widget.tabs.tabText(i) for i in range(widget.tabs.count())] == ["Plot", "Gallery"]


class TestGateColouredHighlights:
    """A gate's colour identifies it on the plot, so the objects it selects
    carry the same colour on the image - otherwise reading two gates against
    each other means holding a mapping in your head."""

    @staticmethod
    def _prepared(qtbot):
        from napari.components import ViewerModel

        viewer = ViewerModel()
        widget = ExplorerWidget(napari_viewer=viewer, float_by_default=False)
        qtbot.addWidget(widget)
        widget.set_data(make_frame(), labels=make_labels())
        widget.plot.set_data(make_frame(), x_column="x", y_column="y")
        return widget, viewer

    @staticmethod
    def _highlight_layers(viewer):
        return [layer for layer in viewer.layers if layer.name.startswith("Gate highlight")]

    def test_one_layer_per_gate_named_after_it(self, qtbot):
        widget, viewer = self._prepared(qtbot)
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
        _draw_triangle(widget.plot, (-100, -100), (100, -100), (0, 100))

        names = {layer.name for layer in self._highlight_layers(viewer)}
        assert names == {"Gate highlight: gate1", "Gate highlight: gate2"}

    def test_the_layer_holds_only_that_gate_s_objects(self, qtbot):
        widget, viewer = self._prepared(qtbot)
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
        gate = next(iter(widget.gate_set))

        mask = widget.gate_set.mask(gate.id, widget.frame)
        expected = set(widget.frame.loc[mask, "object_id"])
        data = viewer.layers[f"Gate highlight: {gate.name}"].data
        assert set(np.unique(data)) - {0} == expected

    def test_the_layer_is_painted_in_the_gate_s_colour(self, qtbot):
        widget, viewer = self._prepared(qtbot)
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
        gate = next(iter(widget.gate_set))
        widget.gate_manager.set_gate_color(gate.id, "#ff0000")

        layer = viewer.layers[f"Gate highlight: {gate.name}"]
        colours = getattr(layer.colormap, "color_dict", None)
        assert colours is not None, "expected a direct colour mapping"
        painted = {
            str(value) for key, value in colours.items() if key not in (None, 0)
        }
        assert painted  # every gated object, all in one colour
        assert len(set(map(str, painted))) == 1

    def test_hiding_a_gate_hides_its_highlight(self, qtbot):
        widget, viewer = self._prepared(qtbot)
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
        gate = next(iter(widget.gate_set))
        layer = viewer.layers[f"Gate highlight: {gate.name}"]
        assert layer.visible

        # The same checkbox that hides the outline on the plot.
        widget.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)

        assert not layer.visible
        assert gate.visible is False

    def test_showing_it_again_brings_the_highlight_back(self, qtbot):
        widget, viewer = self._prepared(qtbot)
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
        gate = next(iter(widget.gate_set))
        layer = viewer.layers[f"Gate highlight: {gate.name}"]

        widget.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
        widget.table.item(0, 0).setCheckState(Qt.CheckState.Checked)

        assert layer.visible

    def test_deleting_a_gate_removes_its_highlight(self, qtbot):
        widget, viewer = self._prepared(qtbot)
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
        widget.table.cellClicked.emit(0, 2)
        widget.gate_manager.delete_selected_gate()

        assert self._highlight_layers(viewer) == []

    def test_clearing_the_gates_removes_every_highlight(self, qtbot):
        widget, viewer = self._prepared(qtbot)
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
        _draw_triangle(widget.plot, (-100, -100), (100, -100), (0, 100))
        widget.gate_manager.clear_gates()

        assert self._highlight_layers(viewer) == []

    def test_recolouring_repaints_rather_than_adding_a_layer(self, qtbot):
        widget, viewer = self._prepared(qtbot)
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))
        gate = next(iter(widget.gate_set))
        widget.gate_manager.set_gate_color(gate.id, "#ff0000")
        widget.gate_manager.set_gate_color(gate.id, "#00ff00")

        assert len(self._highlight_layers(viewer)) == 1

    def test_no_labels_means_no_highlight_rather_than_an_error(self, qtbot):
        from napari.components import ViewerModel

        viewer = ViewerModel()
        widget = ExplorerWidget(napari_viewer=viewer, float_by_default=False)
        qtbot.addWidget(widget)
        widget.set_data(make_frame())  # a table with no label image
        widget.plot.set_data(make_frame(), x_column="x", y_column="y")
        _draw_triangle(widget.plot, (0, 0), (20, 0), (10, 20))

        assert self._highlight_layers(viewer) == []


def _cell_session():
    """A session holding both an object table and the cell table built from
    it - what a protocol with an association step publishes."""
    from vtea_napari.session import OBJECT_TABLE, AnalysisSession, TableView

    session = AnalysisSession()
    nuclei = np.zeros((6, 12), dtype=np.int32)
    nuclei[1:3, 1:3] = 1
    nuclei[1:3, 5:7] = 2
    nuclei[1:3, 9:11] = 3

    objects = pd.DataFrame(
        {
            "object_id": [1, 2, 3],
            "centroid-0": [1.5, 1.5, 1.5],
            "centroid-1": [1.5, 5.5, 9.5],
            "mean_ch0": [10.0, 20.0, 30.0],
        }
    )
    cells = pd.DataFrame(
        {
            "cell_id": [1, 2, 3],
            "nuclei_1.centroid-0": [1.5, 1.5, 1.5],
            "nuclei_1.centroid-1": [1.5, 5.5, 9.5],
            "nuclei_1.mean_ch0": [10.0, 20.0, 30.0],
            "lysosome_1.n": [4.0, 1.0, 0.0],
        }
    )
    session.set_context(
        {"labels": nuclei, "nuclei_1": nuclei, "intensity": np.ones((6, 12))},
        objects,
        {
            OBJECT_TABLE: TableView(objects),
            "cell_features_1": TableView(
                cells, id_column="cell_id", labels_key="nuclei_1", noun="cells"
            ),
        },
    )
    return session


class TestCellTable:
    """A cell table is not the object table with more columns - its rows are
    cells - so switching to it has to move the axes, the gates and the
    highlighted image together."""

    def test_the_picker_offers_both_tables(self, qtbot):
        widget = ExplorerWidget(session=_cell_session())
        qtbot.addWidget(widget)
        offered = [widget.table_combo.itemText(i) for i in range(widget.table_combo.count())]
        assert offered == ["Objects", "cell_features_1"]

    def test_the_picker_is_hidden_when_there_is_nothing_to_choose(self, qtbot):
        """A control offering one option is furniture."""
        widget = ExplorerWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        assert widget.table_combo.isVisibleTo(widget) is False

    def test_switching_plots_the_cell_features(self, qtbot):
        widget = ExplorerWidget(session=_cell_session())
        qtbot.addWidget(widget)
        widget.table_combo.setCurrentText("cell_features_1")
        assert "lysosome_1.n" in widget.frame.columns
        assert len(widget.frame) == 3

    def test_the_axis_menus_offer_the_cell_features(self, qtbot):
        widget = ExplorerWidget(session=_cell_session())
        qtbot.addWidget(widget)
        widget.table_combo.setCurrentText("cell_features_1")
        offered = [widget.plot.x_combo.itemText(i) for i in range(widget.plot.x_combo.count())]
        assert "lysosome_1.n" in offered
        assert "nuclei_1.mean_ch0" in offered

    def test_a_row_is_a_cell_not_an_object(self, qtbot):
        widget = ExplorerWidget(session=_cell_session())
        qtbot.addWidget(widget)
        widget.table_combo.setCurrentText("cell_features_1")
        assert widget.id_column == "cell_id"
        assert "cells" in widget.status_label.toPlainText()

    def test_gates_belong_to_the_table_they_were_drawn_on(self, qtbot):
        """A polygon over cell features selects nothing on a per-object
        table, so showing it there would be a gate that cannot be what it
        claims."""
        widget = ExplorerWidget(session=_cell_session())
        qtbot.addWidget(widget)
        widget.table_combo.setCurrentText("cell_features_1")
        widget.plot.set_data(widget.frame, x_column="cell_id", y_column="lysosome_1.n")
        _draw_triangle(widget.plot, (0, -1), (4, -1), (2, 6))
        assert len(widget.gate_set) == 1

        widget.table_combo.setCurrentText("Objects")
        assert len(widget.gate_set) == 0

        widget.table_combo.setCurrentText("cell_features_1")
        assert len(widget.gate_set) == 1

    def test_a_gate_on_cells_highlights_the_segmentation_they_are_rooted_on(self, qtbot):
        """The cell ids are nucleus ids, so the nuclei are what can be lit
        up - and the explorer has to look up that image, not the last
        `labels` any step happened to write."""
        session = _cell_session()
        session.context["labels"] = np.zeros((6, 12), dtype=np.int32)  # some later step's output
        widget = ExplorerWidget(session=session)
        qtbot.addWidget(widget)
        widget.table_combo.setCurrentText("cell_features_1")
        assert widget.labels is not None
        assert set(np.unique(widget.labels)) == {0, 1, 2, 3}

    def test_the_gallery_crops_around_the_root_s_centroids(self, qtbot):
        """A cell table's centroid columns are namespaced, so the plain
        `centroid-*` lookup finds nothing and would crop the wrong place."""
        widget = ExplorerWidget(session=_cell_session())
        qtbot.addWidget(widget)
        widget.table_combo.setCurrentText("cell_features_1")
        widget._fill_gallery(np.array([1, 2, 3]))
        assert len(widget.gallery._thumbnails) == 3

    def test_a_table_with_no_usable_centroids_leaves_the_gallery_empty(self, qtbot):
        from vtea_napari.session import OBJECT_TABLE, AnalysisSession, TableView

        session = AnalysisSession()
        frame = pd.DataFrame({"cell_id": [1, 2], "value": [1.0, 2.0]})
        session.set_context(
            {"intensity": np.ones((6, 12))},
            frame,
            {OBJECT_TABLE: TableView(frame, id_column="cell_id")},
        )
        widget = ExplorerWidget(session=session)
        qtbot.addWidget(widget)
        widget._fill_gallery(np.array([1, 2]))
        assert widget.gallery._thumbnails == []

    def test_a_re_run_keeps_the_gates_on_each_table(self, qtbot):
        session = _cell_session()
        widget = ExplorerWidget(session=session)
        qtbot.addWidget(widget)
        widget.table_combo.setCurrentText("cell_features_1")
        widget.plot.set_data(widget.frame, x_column="cell_id", y_column="lysosome_1.n")
        _draw_triangle(widget.plot, (0, -1), (4, -1), (2, 6))

        republished = _cell_session()
        session.set_context(republished.context, republished.results_table("Objects"),
                            dict(republished.tables))
        assert len(session.tables["cell_features_1"].gate_set) == 1
