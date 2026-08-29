"""The gate manager beside the protocol builder's plot: drawing modes, the
gate list, JSON persistence, and per-gate statistics."""

import numpy as np
import pandas as pd
import pytest
from vtea_core.gates import Gate, load_gates, rectangle_vertices

from vtea_napari.widgets.gate_manager import GateManagerWidget
from vtea_napari.widgets.plot import POLYGON_MODE, RECTANGLE_MODE, ScatterPlotWidget


class FakeMouseEvent:
    """Duck-types the matplotlib MouseEvent attributes the plot reads."""

    def __init__(self, x, y, button=1, dblclick=False):
        self.xdata = x
        self.ydata = y
        self.button = button
        self.dblclick = dblclick


def make_frame():
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3, 4],
            "mean_ch0": [1.0, 2.0, 20.0, 30.0],
            "count": [10.0, 20.0, 100.0, 200.0],
        }
    )


def make_manager(qtbot, frame=None):
    plot = ScatterPlotWidget()
    qtbot.addWidget(plot)
    manager = GateManagerWidget(plot)
    qtbot.addWidget(manager)
    frame = make_frame() if frame is None else frame
    plot.set_data(frame, "mean_ch0", "count")
    manager.set_frame(frame)
    return manager


class TestDrawingModes:
    def test_polygon_is_the_default(self, qtbot):
        manager = make_manager(qtbot)
        assert manager.plot.gate_mode == POLYGON_MODE
        assert manager.polygon_button.isChecked()

    def test_the_rectangle_button_switches_the_plot(self, qtbot):
        manager = make_manager(qtbot)
        manager.rectangle_button.click()
        assert manager.plot.gate_mode == RECTANGLE_MODE
        assert manager.rectangle_button.isChecked()
        assert not manager.polygon_button.isChecked()

    def test_two_clicks_make_a_rectangle_gate(self, qtbot):
        manager = make_manager(qtbot)
        manager.set_gate_mode(RECTANGLE_MODE)
        manager.plot._on_click(FakeMouseEvent(0.0, 0.0))
        assert len(manager.gate_set) == 0  # one corner isn't a gate yet
        manager.plot._on_click(FakeMouseEvent(5.0, 50.0))
        assert len(manager.gate_set) == 1
        gate = next(iter(manager.gate_set))
        assert gate.vertices.shape == (4, 2)

    def test_the_rectangle_selects_the_points_inside_it(self, qtbot):
        manager = make_manager(qtbot)
        manager.set_gate_mode(RECTANGLE_MODE)
        manager.plot._on_click(FakeMouseEvent(0.0, 0.0))
        manager.plot._on_click(FakeMouseEvent(5.0, 50.0))
        gate = next(iter(manager.gate_set))
        assert manager.gate_set.summary(gate.id, manager.frame)["n_gated"] == 2

    def test_a_polygon_still_takes_a_double_click_to_close(self, qtbot):
        manager = make_manager(qtbot)
        for x, y in ((0.0, 0.0), (5.0, 0.0), (5.0, 50.0)):
            manager.plot._on_click(FakeMouseEvent(x, y))
        assert len(manager.gate_set) == 0
        manager.plot._on_click(FakeMouseEvent(0.0, 50.0, dblclick=True))
        assert len(manager.gate_set) == 1

    def test_switching_mode_discards_a_half_drawn_shape(self, qtbot):
        """A click meant as a polygon vertex must not become a rectangle corner."""
        manager = make_manager(qtbot)
        manager.plot._on_click(FakeMouseEvent(0.0, 0.0))
        manager.set_gate_mode(RECTANGLE_MODE)
        manager.plot._on_click(FakeMouseEvent(5.0, 50.0))
        assert len(manager.gate_set) == 0

    def test_a_double_click_does_not_close_a_zero_area_rectangle(self, qtbot):
        manager = make_manager(qtbot)
        manager.set_gate_mode(RECTANGLE_MODE)
        manager.plot._on_click(FakeMouseEvent(1.0, 1.0))
        manager.plot._on_click(FakeMouseEvent(1.0, 1.0, dblclick=True))
        assert len(manager.gate_set) == 0

    def test_an_unknown_mode_is_refused(self, qtbot):
        manager = make_manager(qtbot)
        with pytest.raises(ValueError, match="unknown gate mode"):
            manager.plot.set_gate_mode("lasso")


class TestGateList:
    def test_a_drawn_gate_appears_in_the_table(self, qtbot):
        manager = make_manager(qtbot)
        manager.add_gate_from_vertices(rectangle_vertices(0, 0, 5, 50))
        assert manager.table.rowCount() == 1

    def test_the_table_shows_the_axes_and_counts(self, qtbot):
        manager = make_manager(qtbot)
        manager.add_gate_from_vertices(rectangle_vertices(0, 0, 5, 50))
        assert manager.table.item(0, 3).text() == "mean_ch0"
        assert manager.table.item(0, 4).text() == "count"
        assert manager.table.item(0, 5).text() == "2"

    def test_gates_are_numbered_as_they_are_drawn(self, qtbot):
        manager = make_manager(qtbot)
        manager.add_gate_from_vertices(rectangle_vertices(0, 0, 5, 50))
        manager.add_gate_from_vertices(rectangle_vertices(0, 0, 100, 500))
        assert [gate.name for gate in manager.gate_set] == ["gate1", "gate2"]

    def test_deleting_removes_only_the_selected_gate(self, qtbot):
        manager = make_manager(qtbot)
        first = manager.add_gate_from_vertices(rectangle_vertices(0, 0, 5, 50))
        manager.add_gate_from_vertices(rectangle_vertices(0, 0, 100, 500))
        manager.selected_gate_id = first.id
        manager.delete_selected_gate()
        assert [gate.name for gate in manager.gate_set] == ["gate2"]

    def test_clear_removes_every_gate(self, qtbot):
        manager = make_manager(qtbot)
        manager.add_gate_from_vertices(rectangle_vertices(0, 0, 5, 50))
        manager.add_gate_from_vertices(rectangle_vertices(0, 0, 100, 500))
        manager.clear_gates()
        assert len(manager.gate_set) == 0
        assert manager.table.rowCount() == 0

    def test_a_gate_needs_axes_to_be_drawn_against(self, qtbot):
        plot = ScatterPlotWidget()
        qtbot.addWidget(plot)
        manager = GateManagerWidget(plot)
        qtbot.addWidget(manager)
        assert manager.add_gate_from_vertices(rectangle_vertices(0, 0, 1, 1)) is None


class TestPersistence:
    def test_gates_save_and_reopen(self, qtbot, tmp_path):
        manager = make_manager(qtbot)
        manager.add_gate_from_vertices(rectangle_vertices(0, 0, 5, 50))
        path = tmp_path / "gates.json"
        manager.save_gates_to(path)

        manager.clear_gates()
        assert len(manager.gate_set) == 0
        manager.load_gates_from(path)
        assert len(manager.gate_set) == 1
        assert manager.table.rowCount() == 1

    def test_reopened_gates_still_select_the_same_cells(self, qtbot, tmp_path):
        manager = make_manager(qtbot)
        gate = manager.add_gate_from_vertices(rectangle_vertices(0, 0, 5, 50))
        before = manager.gate_set.mask(gate.id, manager.frame)
        path = tmp_path / "gates.json"
        manager.save_gates_to(path)
        manager.load_gates_from(path)
        reopened = next(iter(manager.gate_set))
        np.testing.assert_array_equal(manager.gate_set.mask(reopened.id, manager.frame), before)

    def test_the_saved_file_is_the_core_format(self, qtbot, tmp_path):
        manager = make_manager(qtbot)
        manager.add_gate_from_vertices(rectangle_vertices(0, 0, 5, 50))
        path = tmp_path / "gates.json"
        manager.save_gates_to(path)
        assert len(load_gates(path)) == 1

    def test_new_gates_after_a_load_do_not_reuse_names(self, qtbot, tmp_path):
        manager = make_manager(qtbot)
        manager.add_gate_from_vertices(rectangle_vertices(0, 0, 5, 50))
        path = tmp_path / "gates.json"
        manager.save_gates_to(path)
        manager.load_gates_from(path)
        manager.add_gate_from_vertices(rectangle_vertices(0, 0, 100, 500))
        assert [gate.name for gate in manager.gate_set] == ["gate1", "gate2"]

    def test_opening_a_broken_file_reports_instead_of_raising(self, qtbot, tmp_path, monkeypatch):
        manager = make_manager(qtbot)
        path = tmp_path / "broken.json"
        path.write_text('{"vtea_gates_version": 99, "gates": []}')
        monkeypatch.setattr(
            "vtea_napari.widgets.gate_manager.QFileDialog.getOpenFileName",
            lambda *a, **k: (str(path), ""),
        )
        manager.open_gates_dialog()
        assert "newer than this VTEA" in manager.stats_label.text()


class TestStatistics:
    def test_selecting_a_gate_shows_its_count_and_means(self, qtbot):
        manager = make_manager(qtbot)
        gate = manager.add_gate_from_vertices(rectangle_vertices(0, 0, 5, 50))
        manager._on_gate_selected(gate.id)
        text = manager.stats_label.text()
        assert "2 of 4 cells" in text
        # Means over the gated cells only: mean_ch0 = (1+2)/2, count = (10+20)/2.
        assert "mean mean_ch0 = 1.5" in text
        assert "mean count = 15" in text

    def test_the_means_are_for_the_plotted_axes(self, qtbot):
        manager = make_manager(qtbot)
        gate = manager.add_gate_from_vertices(rectangle_vertices(0, 0, 5, 50))
        manager._on_gate_selected(gate.id)
        assert "mean object_id" not in manager.stats_label.text()

    def test_nothing_selected_says_so(self, qtbot):
        manager = make_manager(qtbot)
        assert "No gate selected" in manager.stats_label.text()

    def test_a_deleted_gate_clears_the_statistics(self, qtbot):
        manager = make_manager(qtbot)
        gate = manager.add_gate_from_vertices(rectangle_vertices(0, 0, 5, 50))
        manager._on_gate_selected(gate.id)
        manager.delete_selected_gate()
        assert "No gate selected" in manager.stats_label.text()

    def test_counts_follow_a_re_run_that_changes_the_table(self, qtbot):
        """Gates are kept across runs, so their numbers have to be
        recomputed against the new table rather than left stale."""
        manager = make_manager(qtbot)
        gate = manager.add_gate_from_vertices(rectangle_vertices(0, 0, 5, 50))
        manager._on_gate_selected(gate.id)
        assert "2 of 4 cells" in manager.stats_label.text()

        smaller = make_frame().iloc[:1]
        manager.plot.set_data(smaller, "mean_ch0", "count")
        manager.set_frame(smaller)
        assert "1 of 1 cells" in manager.stats_label.text()

    def test_no_measurements_yet_does_not_raise(self, qtbot):
        plot = ScatterPlotWidget()
        qtbot.addWidget(plot)
        manager = GateManagerWidget(plot)
        qtbot.addWidget(manager)
        manager.gate_set.add(
            Gate(
                name="g",
                x_axis="mean_ch0",
                y_axis="count",
                vertices=rectangle_vertices(0, 0, 1, 1),
            )
        )
        manager.selected_gate_id = next(iter(manager.gate_set)).id
        manager.refresh()  # must not raise
        assert "no measurements" in manager.stats_label.text()

    def test_a_gate_on_features_this_run_lacks_is_listed_not_crashed(self, qtbot):
        """Reopening gates drawn on a different feature set used to raise a
        KeyError from inside the table refresh, taking the whole pane down."""
        manager = make_manager(qtbot)
        manager.gate_set.add(
            Gate(
                name="from another run",
                x_axis="mean_ch7",
                y_axis="count",
                vertices=rectangle_vertices(0, 0, 1, 1),
            )
        )
        manager.selected_gate_id = list(manager.gate_set)[-1].id
        manager.refresh()  # must not raise
        assert manager.table.rowCount() == 1
        assert manager.table.item(0, 5).text() == "-"
        assert "not in the current data" in manager.stats_label.text()
