"""Saving and opening a protocol, and exporting the table.

The test that matters is the one a lab lives by: a protocol saved in one
napari session, opened in another, re-runs to the same table - and the gates
drawn on the first come back on the second.
"""

import json

import numpy as np
import pandas as pd
import pytest
from vtea_core.data import Spacing
from vtea_core.gates import Gate, rectangle_vertices
from vtea_core.workflow import ProtocolError, Step

from vtea_napari.widgets.explorer import ExplorerWidget
from vtea_napari.widgets.protocol_builder import ProtocolBuilderWidget


def _volume():
    volume = np.zeros((20, 20))
    volume[1:4, 1:4] = 100.0
    volume[8:12, 8:12] = 200.0
    volume[15:18, 2:5] = 150.0
    return volume


def _viewer(scale=None):
    from napari.components import ViewerModel

    viewer = ViewerModel()
    viewer.add_image(_volume(), name="src", **({} if scale is None else {"scale": scale}))
    return viewer


def _builder(qtbot, viewer=None):
    widget = ProtocolBuilderWidget(napari_viewer=viewer if viewer is not None else _viewer())
    qtbot.addWidget(widget)
    return widget


def _build_protocol(widget):
    widget.pipeline.add_step(
        Step.for_function(
            "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
        )
    )
    widget.pipeline.add_step(
        Step.for_function("segmentation", "label_components", taken_names=widget.step_names())
    )
    widget.sync_measurement_steps()
    widget.analysis_pipeline.add_step(
        Step.for_function(
            "clustering",
            "kmeans",
            params={"n_clusters": 2, "random_state": 0},
            taken_names=widget.step_names(),
        )
    )
    widget.refresh_steps()


def _run(widget):
    volume = _volume()
    widget.run_pipeline({"volume": volume, "intensity": volume})
    return widget.results_table()


def _bright_gate():
    return Gate(
        name="bright",
        x_axis="mean",
        y_axis="count",
        vertices=rectangle_vertices(120, 0, 300, 100),
    )


class TestSaveAndOpen:
    def test_a_protocol_re_runs_to_the_same_table_in_a_fresh_session(self, qtbot, tmp_path):
        first = _builder(qtbot)
        _build_protocol(first)
        expected = _run(first)
        path = first.save_protocol_to(tmp_path / "p.vtea.json")

        second = _builder(qtbot)
        assert second.session is not first.session
        second.load_protocol_from(path)
        assert [s.name for s in second.all_steps()] == [s.name for s in first.all_steps()]
        actual = _run(second)
        assert len(expected) == 3
        pd.testing.assert_frame_equal(actual, expected)

    def test_opening_refills_the_pipelines_the_stacks_already_show(self, qtbot, tmp_path):
        first = _builder(qtbot)
        _build_protocol(first)
        path = first.save_protocol_to(tmp_path / "p.vtea.json")

        second = _builder(qtbot)
        pipeline = second.pipeline
        second.load_protocol_from(path)
        assert second.pipeline is pipeline
        assert second.processing_stack.pipeline is pipeline
        assert second.session.processing_pipeline is pipeline
        assert len(second.processing_stack._cards) == 2

    def test_opening_drops_results_the_new_steps_do_not_describe(self, qtbot, tmp_path):
        first = _builder(qtbot)
        _build_protocol(first)
        path = first.save_protocol_to(tmp_path / "p.vtea.json")
        _run(first)
        assert first.session.results_table() is not None

        first.load_protocol_from(path)
        assert first.last_context == {}
        assert first.session.results_table() is None
        assert len(first.session.feature_catalog) == 0

    def test_the_measure_every_segmentation_switch_is_restored(self, qtbot, tmp_path):
        first = _builder(qtbot)
        first.measure_all_check.setChecked(False)
        path = first.save_protocol_to(tmp_path / "p.vtea.json")
        second = _builder(qtbot)
        second.load_protocol_from(path)
        assert not second.measure_all_check.isChecked()

    def test_the_source_image_is_recorded(self, qtbot, tmp_path):
        widget = _builder(qtbot)
        path = widget.save_protocol_to(tmp_path / "p.vtea.json")
        source = json.loads(path.read_text(encoding="utf-8"))["source"]
        assert source["name"] == "src"
        assert source["shape"] == [20, 20]


class TestGates:
    def test_gates_come_back_on_their_table_at_the_next_run(self, qtbot, tmp_path):
        first = _builder(qtbot)
        _build_protocol(first)
        _run(first)
        first.session.gate_set.add(_bright_gate())
        path = first.save_protocol_to(tmp_path / "p.vtea.json")

        second = _builder(qtbot)
        second.load_protocol_from(path)
        _run(second)
        assert [gate.name for gate in second.session.gate_set] == ["bright"]

    def test_gates_opened_but_not_yet_run_are_saved_again(self, qtbot, tmp_path):
        first = _builder(qtbot)
        _build_protocol(first)
        _run(first)
        first.session.gate_set.add(_bright_gate())
        path = first.save_protocol_to(tmp_path / "p.vtea.json")

        second = _builder(qtbot)
        second.load_protocol_from(path)
        again = json.loads(second.save_protocol_to(tmp_path / "again.vtea.json").read_text())
        assert [g["name"] for g in again["gates"]["Objects"]["gates"]] == ["bright"]

    def test_gates_can_be_left_out(self, qtbot, tmp_path):
        widget = _builder(qtbot)
        _build_protocol(widget)
        _run(widget)
        widget.session.gate_set.add(_bright_gate())
        path = widget.save_protocol_to(tmp_path / "p.vtea.json", include_gates=False)
        assert json.loads(path.read_text())["gates"] == {}


class TestVoxelSize:
    def test_an_image_without_one_takes_the_protocols(self, qtbot, tmp_path):
        first = _builder(qtbot)
        first.spacing_control.set_spacing(Spacing((0.5, 0.25), unit="µm"))
        path = first.save_protocol_to(tmp_path / "p.vtea.json")

        second = _builder(qtbot)
        assert not second.spacing_control.spacing().is_known
        second.load_protocol_from(path)
        assert second.spacing_control.spacing().values == (0.5, 0.25)
        assert "from the protocol" in second.status_label.text()

    def test_an_image_that_records_one_keeps_it_and_says_so(self, qtbot, tmp_path):
        first = _builder(qtbot)
        first.spacing_control.set_spacing(Spacing((0.5, 0.25), unit="µm"))
        path = first.save_protocol_to(tmp_path / "p.vtea.json")

        second = _builder(qtbot, _viewer(scale=(2.0, 2.0)))
        second.load_protocol_from(path)
        assert second.spacing_control.spacing().values == (2.0, 2.0)
        assert "Kept this image's voxel size" in second.status_label.text()


class TestRefusals:
    def test_a_protocol_from_a_newer_vtea_is_reported_not_raised(
        self, qtbot, tmp_path, monkeypatch
    ):
        from qtpy.QtWidgets import QFileDialog

        path = tmp_path / "future.vtea.json"
        path.write_text(json.dumps({"vtea_protocol_version": 99}))
        widget = _builder(qtbot)
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
        widget.open_protocol_dialog()
        assert "Could not open the protocol" in widget.status_label.text()
        assert "newer" in widget.status_label.text()

    def test_a_step_that_cannot_be_saved_is_reported(self, qtbot, tmp_path, monkeypatch):
        from qtpy.QtWidgets import QFileDialog

        widget = _builder(qtbot)
        widget.pipeline.add_step(
            Step.for_function("segmentation", "threshold_mask", params={"value": object()})
        )
        target = tmp_path / "p"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
        widget.save_protocol_dialog()
        assert "Could not save the protocol" in widget.status_label.text()
        assert not (tmp_path / "p.vtea.json").exists()

    def test_the_dialog_adds_the_protocol_suffix(self, qtbot, tmp_path, monkeypatch):
        from qtpy.QtWidgets import QFileDialog

        widget = _builder(qtbot)
        target = tmp_path / "mine"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
        widget.save_protocol_dialog()
        assert (tmp_path / "mine.vtea.json").exists()

    def test_loading_directly_raises_protocol_error(self, qtbot, tmp_path):
        path = tmp_path / "not.vtea.json"
        path.write_text("[]")
        with pytest.raises(ProtocolError):
            _builder(qtbot).load_protocol_from(path)


class TestExport:
    def _explorer(self, qtbot, widget):
        explorer = ExplorerWidget(session=widget.session, float_by_default=False)
        qtbot.addWidget(explorer)
        return explorer

    def test_the_table_exports_with_a_column_per_gate(self, qtbot, tmp_path):
        widget = _builder(qtbot)
        _build_protocol(widget)
        table = _run(widget)
        widget.session.gate_set.add(_bright_gate())
        explorer = self._explorer(qtbot, widget)

        written = explorer.export_table_to(tmp_path / "objects.csv")
        exported = pd.read_csv(written[0])
        assert len(exported) == len(table)
        expected = ((table["mean"] >= 120) & (table["mean"] <= 300)).to_numpy()
        np.testing.assert_array_equal(exported["gate_bright"].to_numpy(), expected)

    def test_the_dictionary_describes_every_column_and_the_gate(self, qtbot, tmp_path):
        widget = _builder(qtbot)
        _build_protocol(widget)
        _run(widget)
        widget.session.gate_set.add(_bright_gate())
        explorer = self._explorer(qtbot, widget)

        (table_path, dictionary_path) = explorer.export_table_to(tmp_path / "objects.csv")
        exported = pd.read_csv(table_path)
        dictionary = pd.read_csv(dictionary_path).set_index("column")
        assert list(dictionary.index) == list(exported.columns)
        assert dictionary.loc["gate_bright", "measurement"] == "gate membership"
        assert dictionary.loc["kmeans_1", "measurement"] == "cluster assignment"

    def test_nothing_to_export_is_an_error_not_an_empty_file(self, qtbot, tmp_path):
        explorer = self._explorer(qtbot, _builder(qtbot))
        with pytest.raises(ValueError, match="no table"):
            explorer.export_table_to(tmp_path / "objects.csv")
        assert not (tmp_path / "objects.csv").exists()
