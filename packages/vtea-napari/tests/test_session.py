"""The protocol builder and the Object Explorer are two views of one
analysis, and hiding either pane must not cost anything.

The state lives in an AnalysisSession owned by neither widget, so results
computed while the explorer was closed are waiting when it opens, and gates
drawn in the explorer survive it being hidden.
"""

import numpy as np
import pandas as pd
from vtea_core.gates import rectangle_vertices
from vtea_core.workflow import Step

from vtea_napari.session import AnalysisSession, session_for
from vtea_napari.widgets.explorer import ExplorerWidget
from vtea_napari.widgets.protocol_builder import ProtocolBuilderWidget


def _model_viewer():
    from napari.components import ViewerModel

    return ViewerModel()


def _measured_builder(qtbot, session=None):
    """A builder that has segmented and measured a small image."""
    viewer = _model_viewer()
    volume = np.zeros((12, 12))
    volume[1:4, 1:4] = 100.0
    volume[7:10, 7:10] = 200.0
    viewer.add_image(volume, name="src")
    widget = ProtocolBuilderWidget(napari_viewer=viewer, session=session)
    qtbot.addWidget(widget)
    widget.pipeline.add_step(
        Step.for_function(
            "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
        )
    )
    widget.pipeline.add_step(Step.for_function("segmentation", "label_components"))
    widget.run_processing()
    measure = widget.analysis_pipeline.add_step(
        Step.for_function("measurements", "extract_measurements")
    )
    widget.run_single_step(measure)
    return widget


class TestSessionRegistry:
    def test_widgets_on_one_viewer_share_a_session(self, qtbot):
        viewer = _model_viewer()
        builder = ProtocolBuilderWidget(napari_viewer=viewer)
        explorer = ExplorerWidget(napari_viewer=viewer, float_by_default=False)
        qtbot.addWidget(builder)
        qtbot.addWidget(explorer)
        assert builder.session is explorer.session

    def test_widgets_on_different_viewers_do_not(self, qtbot):
        first = ProtocolBuilderWidget(napari_viewer=_model_viewer())
        second = ProtocolBuilderWidget(napari_viewer=_model_viewer())
        qtbot.addWidget(first)
        qtbot.addWidget(second)
        assert first.session is not second.session

    def test_no_viewer_means_an_independent_session(self, qtbot):
        """Two unrelated standalone widgets sharing global state would be
        worse than not sharing at all."""
        first = ProtocolBuilderWidget()
        second = ProtocolBuilderWidget()
        qtbot.addWidget(first)
        qtbot.addWidget(second)
        assert first.session is not second.session

    def test_an_explicit_session_wins(self, qtbot):
        session = AnalysisSession()
        builder = ProtocolBuilderWidget(session=session)
        qtbot.addWidget(builder)
        assert builder.session is session

    def test_session_for_none_returns_a_usable_session(self):
        assert isinstance(session_for(None), AnalysisSession)


class TestResultsReachTheExplorer:
    def test_a_run_publishes_the_table(self, qtbot):
        session = AnalysisSession()
        builder = _measured_builder(qtbot, session)
        explorer = ExplorerWidget(session=session, float_by_default=False)
        qtbot.addWidget(explorer)

        assert explorer.frame is not None
        assert len(explorer.frame) == len(builder.results_table())

    def test_an_already_open_explorer_updates_live(self, qtbot):
        session = AnalysisSession()
        explorer = ExplorerWidget(session=session, float_by_default=False)
        qtbot.addWidget(explorer)
        assert explorer.frame is None

        _measured_builder(qtbot, session)
        assert explorer.frame is not None
        assert "mean" in _axis_choices(explorer)

    def test_results_computed_before_the_explorer_existed_are_waiting(self, qtbot):
        """The usual case: run the protocol, then open the explorer."""
        session = AnalysisSession()
        _measured_builder(qtbot, session)

        explorer = ExplorerWidget(session=session, float_by_default=False)
        qtbot.addWidget(explorer)
        assert explorer.frame is not None
        assert explorer.plot.x_column is not None

    def test_the_label_image_travels_with_the_table(self, qtbot):
        """Gate highlighting needs the labels the measurements came from."""
        session = AnalysisSession()
        _measured_builder(qtbot, session)
        explorer = ExplorerWidget(session=session, float_by_default=False)
        qtbot.addWidget(explorer)
        assert explorer.labels is not None
        assert explorer.labels.max() == 2

    def test_the_axes_choices_travel_too(self, qtbot):
        session = AnalysisSession()
        _measured_builder(qtbot, session)
        assert session.channel_axis is None
        assert session.source_layer_name == "src"


class TestPersistenceAcrossHiding:
    def test_hiding_and_showing_the_explorer_keeps_its_gates(self, qtbot):
        session = AnalysisSession()
        _measured_builder(qtbot, session)
        explorer = ExplorerWidget(session=session, float_by_default=False)
        qtbot.addWidget(explorer)
        gate = explorer.gate_manager.add_gate_from_vertices(
            _spanning_rectangle(explorer.frame, explorer.plot.x_column, explorer.plot.y_column)
        )

        explorer.hide()
        explorer.show()

        assert gate.id in explorer.gate_set
        assert explorer.table.rowCount() == 1

    def test_a_run_while_the_explorer_is_hidden_is_picked_up_on_show(self, qtbot):
        session = AnalysisSession()
        explorer = ExplorerWidget(session=session, float_by_default=False)
        qtbot.addWidget(explorer)
        explorer.hide()

        _measured_builder(qtbot, session)
        explorer.show()

        assert explorer.frame is not None
        assert "mean" in _axis_choices(explorer)

    def test_a_replacement_explorer_widget_recovers_the_gates(self, qtbot):
        """napari may destroy and rebuild a plugin widget; the gates are on
        the session, not the widget, so they come back."""
        session = AnalysisSession()
        _measured_builder(qtbot, session)
        first = ExplorerWidget(session=session, float_by_default=False)
        qtbot.addWidget(first)
        first.gate_manager.add_gate_from_vertices(
            _spanning_rectangle(first.frame, first.plot.x_column, first.plot.y_column)
        )

        second = ExplorerWidget(session=session, float_by_default=False)
        qtbot.addWidget(second)

        assert len(second.gate_set) == 1
        assert second.table.rowCount() == 1

    def test_gates_survive_a_re_run_and_are_recounted(self, qtbot):
        session = AnalysisSession()
        builder = _measured_builder(qtbot, session)
        explorer = ExplorerWidget(session=session, float_by_default=False)
        qtbot.addWidget(explorer)
        gate = explorer.gate_manager.add_gate_from_vertices(
            _spanning_rectangle(explorer.frame, explorer.plot.x_column, explorer.plot.y_column)
        )

        builder.run_single_step(builder.analysis_pipeline.steps[0])

        assert gate.id in explorer.gate_set
        assert explorer.gate_manager.frame is not None

    def test_a_gate_drawn_in_the_explorer_counts_measured_cells(self, qtbot):
        session = AnalysisSession()
        _measured_builder(qtbot, session)
        explorer = ExplorerWidget(session=session, float_by_default=False)
        qtbot.addWidget(explorer)

        frame = explorer.frame
        gate = explorer.gate_manager.add_gate_from_vertices(
            _spanning_rectangle(frame, explorer.plot.x_column, explorer.plot.y_column)
        )
        explorer.gate_manager._on_gate_selected(gate.id)

        assert f"{len(frame)} of {len(frame)} cells" in explorer.gate_manager.stats_label.text()


class TestBuilderHasNoPlot:
    def test_the_builder_no_longer_owns_a_plot_or_gate_manager(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        assert not hasattr(widget, "plot")
        assert not hasattr(widget, "gate_manager")

    def test_it_offers_a_way_to_open_the_explorer(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        assert widget.explorer_button.text() == "Object Explorer"

    def test_opening_without_a_viewer_reports_rather_than_raising(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        assert widget.open_object_explorer() is None
        assert "needs a napari viewer" in widget.status_label.text()


def _axis_choices(explorer):
    combo = explorer.plot.x_combo
    return {combo.itemText(index) for index in range(combo.count())}


def _spanning_rectangle(frame: pd.DataFrame, x_column: str, y_column: str):
    return rectangle_vertices(
        frame[x_column].min() - 1,
        frame[y_column].min() - 1,
        frame[x_column].max() + 1,
        frame[y_column].max() + 1,
    )


class TestProtocolPersistence:
    def test_a_rebuilt_builder_gets_its_steps_back(self, qtbot):
        """napari may destroy and recreate a plugin widget; the protocol is
        on the session, so it comes back rather than opening empty."""
        session = AnalysisSession()
        first = ProtocolBuilderWidget(session=session)
        qtbot.addWidget(first)
        first.pipeline.add_step(Step.for_function("segmentation", "threshold_mask"))

        second = ProtocolBuilderWidget(session=session)
        qtbot.addWidget(second)

        assert [step.name for step in second.pipeline] == ["threshold_mask_1"]

    def test_two_builders_on_one_viewer_edit_the_same_protocol(self, qtbot):
        viewer = _model_viewer()
        first = ProtocolBuilderWidget(napari_viewer=viewer)
        second = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(first)
        qtbot.addWidget(second)
        first.pipeline.add_step(Step.for_function("segmentation", "threshold_mask"))
        assert len(second.pipeline) == 1

    def test_an_explicitly_passed_pipeline_is_the_one_driven(self, qtbot):
        """Scripts that build a Pipeline and hand it over must keep working."""
        from vtea_core.workflow import Pipeline

        pipeline = Pipeline()
        widget = ProtocolBuilderWidget(pipeline=pipeline)
        qtbot.addWidget(widget)
        assert widget.pipeline is pipeline
        assert widget.session.processing_pipeline is pipeline


class TestHidingTheDock:
    def test_hiding_the_dock_keeps_the_explorer_state(self, qtbot):
        """What the napari Window menu actually does: hide the QDockWidget
        holding the widget, not destroy it."""
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QDockWidget, QMainWindow

        session = AnalysisSession()
        _measured_builder(qtbot, session)

        window = QMainWindow()
        qtbot.addWidget(window)
        dock = QDockWidget("Object Explorer", window)
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        explorer = ExplorerWidget(session=session, float_by_default=False)
        dock.setWidget(explorer)
        window.show()
        qtbot.waitExposed(window)

        gate = explorer.gate_manager.add_gate_from_vertices(
            _spanning_rectangle(explorer.frame, explorer.plot.x_column, explorer.plot.y_column)
        )
        dock.hide()
        dock.show()

        assert gate.id in explorer.gate_set
        assert explorer.frame is not None
        assert explorer.table.rowCount() == 1
