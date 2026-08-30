from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QPushButton

from vtea_napari.widgets.protocol_builder import ProtocolBuilderWidget


def _click_button(qtbot, widget, text):
    for button in widget.findChildren(QPushButton):
        if button.text() == text:
            qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
            return
    raise AssertionError(f"no button with text {text!r} found")


def _explorer_for(widget, qtbot):
    """The Object Explorer viewing the same analysis as `widget`.

    Plotting and gating live there now; the builder publishes results into
    the session they share. Floating is off so the test doesn't need a dock.
    """
    from vtea_napari.widgets.explorer import ExplorerWidget

    explorer = ExplorerWidget(session=widget.session, float_by_default=False)
    qtbot.addWidget(explorer)
    return explorer


def _axis_choices(explorer):
    combo = explorer.plot.x_combo
    return {combo.itemText(index) for index in range(combo.count())}


class TestAddStep:
    def test_starts_empty(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        assert len(widget.pipeline) == 0

    def test_add_step_appends_to_pipeline(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        widget.processing_stack.category_combo.setCurrentText("segmentation")
        widget.processing_stack.function_combo.setCurrentText("threshold_mask")
        _click_button(qtbot, widget.processing_stack, "Add Step")

        assert len(widget.pipeline) == 1
        assert widget.pipeline.steps[0].category == "segmentation"
        assert widget.pipeline.steps[0].function_name == "threshold_mask"

    def test_add_step_creates_a_card(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        widget.processing_stack.category_combo.setCurrentText("clustering")
        widget.processing_stack.function_combo.setCurrentText("kmeans")
        _click_button(qtbot, widget.processing_stack, "Add Step")

        from vtea_napari.widgets.step_card import StepCardWidget

        cards = widget.findChildren(StepCardWidget)
        assert len(cards) == 1

    def _categories(self, stack):
        return {stack.category_combo.itemText(i) for i in range(stack.category_combo.count())}

    def _functions(self, stack):
        return {stack.function_combo.itemText(i) for i in range(stack.function_combo.count())}

    def test_function_choices_follow_category(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        widget.processing_stack.category_combo.setCurrentText("segmentation")
        assert "threshold_mask" in self._functions(widget.processing_stack)

        widget.analysis_stack.category_combo.setCurrentText("clustering")
        assert "kmeans" in self._functions(widget.analysis_stack)

    def test_processing_and_analysis_categories_are_separated(self, qtbot):
        """Image-producing steps go in the top pane, per-object analysis in
        the bottom one - neither should offer the other's categories."""
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        processing = self._categories(widget.processing_stack)
        analysis = self._categories(widget.analysis_stack)

        # "ownership" is a processing category: it reads a label image and
        # a mask and produces something image-shaped.
        assert processing == {"imageprocessing", "segmentation", "ownership"}
        # No "classification": its steps need crops, a model and training
        # labels, none of which any protocol step produces, so every one of
        # them could only ever fail with "needs context key(s) [...]".
        assert analysis == {
            "measurements",
            "association",
            "cells",
            "clustering",
            "reduction",
            "gates",
        }
        assert processing.isdisjoint(analysis)

    def test_classification_is_not_offered(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        assert "classification" not in self._categories(widget.analysis_stack)
        assert "classification" not in self._categories(widget.processing_stack)


class TestDeleteStep:
    def test_delete_removes_from_pipeline_and_ui(self, qtbot):
        from vtea_core.workflow import Step

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        widget.pipeline.add_step(Step(category="segmentation", function_name="threshold_mask"))
        widget.refresh_steps()

        _click_button(qtbot, widget.processing_stack, "Delete")

        assert len(widget.pipeline) == 0
        from vtea_napari.widgets.step_card import StepCardWidget

        assert len(widget.findChildren(StepCardWidget)) == 0

    def test_delete_one_of_two_keeps_the_other(self, qtbot):
        from vtea_core.workflow import Step

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        first = widget.pipeline.add_step(Step(category="segmentation", function_name="threshold_mask"))
        widget.pipeline.add_step(Step(category="segmentation", function_name="label_components"))
        widget.refresh_steps()

        widget.processing_stack._delete_step(first)

        assert len(widget.pipeline) == 1
        assert widget.pipeline.steps[0].function_name == "label_components"


class TestEditStep:
    def test_edit_updates_step_params_on_accept(self, qtbot, monkeypatch):
        from qtpy.QtWidgets import QDialog

        from vtea_core.workflow import Step
        from vtea_napari.widgets.protocol_builder import EditStepDialog

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        step = widget.pipeline.add_step(Step(category="segmentation", function_name="threshold_mask"))
        widget.refresh_steps()

        def fake_exec(self):
            # Simulates the user changing a field and clicking OK, without
            # actually showing a blocking modal dialog in a headless test.
            self.form.set_values({"method": "otsu"})
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(EditStepDialog, "exec", fake_exec)

        widget.processing_stack._edit_step(step)

        assert step.params["method"] == "otsu"

    def test_edit_cancelled_leaves_params_unchanged(self, qtbot, monkeypatch):
        from qtpy.QtWidgets import QDialog

        from vtea_core.workflow import Step
        from vtea_napari.widgets.protocol_builder import EditStepDialog

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        step = widget.pipeline.add_step(
            Step(category="segmentation", function_name="threshold_mask", params={"method": "otsu"})
        )
        widget.refresh_steps()

        def fake_exec(self):
            self.form.set_values({"method": "percentile"})
            return QDialog.DialogCode.Rejected

        monkeypatch.setattr(EditStepDialog, "exec", fake_exec)

        widget.processing_stack._edit_step(step)

        assert step.params["method"] == "otsu"

    def test_edit_dialog_prefills_current_params(self, qtbot):
        from vtea_core.workflow import Step
        from vtea_napari.widgets.protocol_builder import EditStepDialog

        step = Step(category="segmentation", function_name="threshold_mask", params={"method": "percentile"})
        dialog = EditStepDialog(step)
        qtbot.addWidget(dialog)

        assert dialog.form.get_values()["method"] == "percentile"


class TestEndToEnd:
    def test_gui_built_pipeline_produces_correct_results(self, qtbot, monkeypatch):
        """Builds a two-step pipeline entirely through the widget (Add Step,
        edit params via the dialog), then runs the resulting Pipeline
        against synthetic data - verifying the GUI actually wires into the
        headless engine correctly, not just that widgets don't crash."""
        import numpy as np
        from qtpy.QtWidgets import QDialog

        from vtea_napari.widgets.protocol_builder import EditStepDialog

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        widget.processing_stack.category_combo.setCurrentText("segmentation")
        widget.processing_stack.function_combo.setCurrentText("threshold_mask")
        _click_button(qtbot, widget.processing_stack, "Add Step")

        widget.processing_stack.category_combo.setCurrentText("segmentation")
        widget.processing_stack.function_combo.setCurrentText("label_components")
        _click_button(qtbot, widget.processing_stack, "Add Step")

        threshold_step, label_step = widget.pipeline.steps
        threshold_step.input_keys = {"volume": "volume"}
        threshold_step.output_key = "mask"
        label_step.input_keys = {"mask": "mask"}
        label_step.output_key = "labels"

        def fake_exec(self):
            self.form.set_values({"method": "fixed", "value": "50"})
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(EditStepDialog, "exec", fake_exec)
        widget.processing_stack._edit_step(threshold_step)
        assert threshold_step.params["value"] == 50.0

        volume = np.zeros((10, 10))
        volume[1:3, 1:3] = 100.0  # object 1
        volume[6:9, 6:9] = 100.0  # object 2

        result = widget.pipeline.run({"volume": volume})

        assert result["mask"].sum() == 13  # 4 + 9 above-threshold pixels
        assert result["labels"].max() == 2


class TestRunPipelineThumbnails:
    def test_run_pipeline_attaches_thumbnails_to_cards(self, qtbot):
        import numpy as np

        from vtea_core.workflow import Step
        from vtea_napari.widgets.step_card import StepCardWidget

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        step = widget.pipeline.add_step(
            Step(
                category="segmentation",
                function_name="threshold_mask",
                input_keys={"volume": "volume"},
                output_key="mask",
                params={"method": "fixed", "value": 50.0},
            )
        )
        widget.refresh_steps()

        volume = np.zeros((10, 10))
        volume[1:3, 1:3] = 100.0
        result = widget.run_pipeline({"volume": volume})

        assert result["mask"].sum() == 4
        cards = widget.findChildren(StepCardWidget)
        assert len(cards) == 1
        assert not cards[0].thumbnail_label.pixmap().isNull()
        assert step.output_key in widget.last_context

    def test_run_button_not_shown_without_a_viewer(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        buttons = [b.text() for b in widget.findChildren(QPushButton)]
        assert "Run Processing" not in buttons
        assert widget.processing_stack.action_button is None

    def test_run_button_pulls_volume_from_active_layer(self, qtbot):
        import napari
        import numpy as np

        from vtea_core.workflow import Step

        viewer = napari.Viewer(show=False)
        qtbot.addWidget(viewer.window._qt_window)
        try:
            volume = np.zeros((10, 10))
            volume[1:3, 1:3] = 100.0
            # add_labels rather than add_image: this container's offscreen
            # GL setup can't create an Image layer's vispy visual (same
            # class of headless-OpenGL gap as release.yml's Windows smoke
            # test - see packaging/pyinstaller/README.md), but the widget
            # code under test only reads `layer.data`, which works
            # identically regardless of layer type.
            layer = viewer.add_labels(volume.astype("int32"), name="input")
            viewer.layers.selection.active = layer

            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.pipeline.add_step(
                Step(
                    category="segmentation",
                    function_name="threshold_mask",
                    input_keys={"volume": "volume"},
                    output_key="mask",
                    params={"method": "fixed", "value": 50.0},
                )
            )
            widget.refresh_steps()

            # Each card runs its own step; there is no pane-level Run button.
            _click_button(qtbot, widget.processing_stack, "Run")

            assert widget.last_context["mask"].sum() == 4
        finally:
            viewer.close()


class TestGuiBuiltPipelineRuns:
    """The bug behind 'cellpose_segmentation() missing 1 required positional
    argument: volume': steps added through the GUI had no input_keys, so the
    data was never passed. Earlier tests wired steps by hand, which a GUI
    user cannot do, so they passed over a completely broken Run button."""

    def test_add_step_wires_inputs_and_output(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        widget.processing_stack.category_combo.setCurrentText("segmentation")
        widget.processing_stack.function_combo.setCurrentText("threshold_mask")
        _click_button(qtbot, widget.processing_stack, "Add Step")

        step = widget.pipeline.steps[0]
        assert step.input_keys == {"volume": "volume"}
        assert step.output_key == "mask"

    def test_cellpose_step_added_from_the_menu_receives_its_volume(self, qtbot):
        """The exact step the bug was reported against. Cellpose itself needs
        the deeplearning extra, so this checks the wiring reaches the
        function rather than running the model."""
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        widget.processing_stack.category_combo.setCurrentText("segmentation")
        widget.processing_stack.function_combo.setCurrentText("cellpose_segmentation")
        _click_button(qtbot, widget.processing_stack, "Add Step")

        step = widget.pipeline.steps[0]
        assert step.input_keys == {"volume": "volume"}

        import numpy as np

        class StubModel:
            def eval(self, x, **kwargs):
                return np.ones(x.shape[:-1], dtype=np.int32), None, None

        step.params["model"] = StubModel()
        result = widget.run_pipeline({"volume": np.zeros((4, 4)), "intensity": np.zeros((4, 4))})
        assert result["labels"].shape == (4, 4)

    def test_two_gui_added_steps_chain_without_manual_wiring(self, qtbot):
        import numpy as np

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        widget.processing_stack.category_combo.setCurrentText("segmentation")
        widget.processing_stack.function_combo.setCurrentText("threshold_mask")
        _click_button(qtbot, widget.processing_stack, "Add Step")
        widget.processing_stack.function_combo.setCurrentText("label_components")
        _click_button(qtbot, widget.processing_stack, "Add Step")

        widget.pipeline.steps[0].params = {"method": "fixed", "value": 50.0}

        volume = np.zeros((10, 10))
        volume[1:3, 1:3] = 100.0
        volume[6:9, 6:9] = 100.0
        result = widget.run_pipeline({"volume": volume, "intensity": volume})

        assert result["mask"].sum() == 13
        assert result["labels"].max() == 2


class TestChannelSelectionUI:
    def _viewer_with_multichannel_image(self, qtbot):
        import napari
        import numpy as np

        viewer = napari.Viewer(show=False)
        qtbot.addWidget(viewer.window._qt_window)
        # (Z, C, Y, X), matching the shape a 4-channel hyperstack loads as.
        volume = np.zeros((6, 4, 16, 16), dtype="int32")
        volume[:, 2, 2:6, 2:6] = 100
        layer = viewer.add_labels(volume, name="image")
        viewer.layers.selection.active = layer
        return viewer, volume

    def test_channel_axis_choices_come_from_the_active_image(self, qtbot):
        viewer, _ = self._viewer_with_multichannel_image(qtbot)
        try:
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            labels = [
                widget.channel_axis_combo.itemText(i)
                for i in range(widget.channel_axis_combo.count())
            ]
            assert labels[0].startswith("None")
            assert "axis 1 (size 4)" in labels
        finally:
            viewer.close()

    def test_selecting_a_channel_axis_sets_it_on_the_pipeline(self, qtbot):
        viewer, _ = self._viewer_with_multichannel_image(qtbot)
        try:
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.channel_axis_combo.setCurrentIndex(widget.channel_axis_combo.findData(1))
            assert widget.pipeline.channel_axis == 1
            assert widget.n_channels() == 4
        finally:
            viewer.close()

    def test_edit_dialog_offers_one_entry_per_channel(self, qtbot):
        from vtea_core.workflow import Step
        from vtea_napari.widgets.protocol_builder import ALL_CHANNELS, EditStepDialog

        step = Step.for_function("segmentation", "threshold_mask")
        dialog = EditStepDialog(step, n_channels=4)
        qtbot.addWidget(dialog)

        entries = [dialog.channel_combo.itemText(i) for i in range(dialog.channel_combo.count())]
        assert entries == [ALL_CHANNELS, "Channel 0", "Channel 1", "Channel 2", "Channel 3"]
        assert dialog.updated_channel() is None

    def test_edit_dialog_keeps_a_channel_the_image_no_longer_has(self, qtbot):
        from vtea_core.workflow import Step
        from vtea_napari.widgets.protocol_builder import EditStepDialog

        step = Step.for_function("segmentation", "threshold_mask")
        step.channel = 7
        dialog = EditStepDialog(step, n_channels=2)
        qtbot.addWidget(dialog)

        assert dialog.updated_channel() == 7

    def test_editing_a_step_stores_the_chosen_channel(self, qtbot, monkeypatch):
        from qtpy.QtWidgets import QDialog

        from vtea_core.workflow import Step
        from vtea_napari.widgets.protocol_builder import EditStepDialog

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        step = widget.pipeline.add_step(Step.for_function("segmentation", "threshold_mask"))
        widget.refresh_steps()

        def fake_exec(self):
            self.channel_combo.setCurrentIndex(self.channel_combo.findData(2))
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(EditStepDialog, "exec", fake_exec)
        # n_channels() is None without a viewer, so seed the combo directly
        monkeypatch.setattr(ProtocolBuilderWidget, "n_channels", lambda self: 4)
        widget.processing_stack._edit_step(step)

        assert step.channel == 2

    def test_end_to_end_channel_selection_through_the_gui(self, qtbot):
        """A 4-channel image, signal only in channel 2 - the pipeline should
        find it when that channel is selected and nothing otherwise."""
        viewer, _ = self._viewer_with_multichannel_image(qtbot)
        try:
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.channel_axis_combo.setCurrentIndex(widget.channel_axis_combo.findData(1))

            widget.processing_stack.category_combo.setCurrentText("segmentation")
            widget.processing_stack.function_combo.setCurrentText("threshold_mask")
            _click_button(qtbot, widget.processing_stack, "Add Step")
            step = widget.pipeline.steps[0]
            step.params = {"method": "fixed", "value": 50.0}

            step.channel = 2
            widget._run_pipeline_from_active_layer()
            assert widget.last_context["mask"].shape == (6, 16, 16)
            assert widget.last_context["mask"].sum() == 6 * 16

            step.channel = 0
            widget._run_pipeline_from_active_layer()
            assert widget.last_context["mask"].sum() == 0
        finally:
            viewer.close()


class TestShowStepResult:
    """Each card's Show button puts that step's output into the viewer."""

    def _viewer(self, qtbot):
        import napari

        viewer = napari.Viewer(show=False)
        qtbot.addWidget(viewer.window._qt_window)
        return viewer

    def test_every_card_has_its_own_run_button(self, qtbot):
        """Analysis steps form a graph rather than a chain, so each step has
        to be runnable on its own."""
        from vtea_core.workflow import Step
        from vtea_napari.widgets.step_card import StepCardWidget

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        widget.pipeline.add_step(Step.for_function("segmentation", "threshold_mask"))
        widget.analysis_pipeline.add_step(
            Step.for_function("measurements", "extract_measurements")
        )
        widget.refresh_steps()

        for stack in (widget.processing_stack, widget.analysis_stack):
            card = stack.findChildren(StepCardWidget)[0]
            assert card.run_button.text() == "Run"
            assert card.run_button.isEnabled()

    def test_show_adds_a_labels_layer_for_an_integer_result(self, qtbot):
        import numpy as np

        from vtea_core.workflow import Step

        viewer = self._viewer(qtbot)
        try:
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            step = widget.pipeline.add_step(
                Step.for_function(
                    "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
                )
            )
            widget.pipeline.add_step(Step.for_function("segmentation", "label_components"))

            volume = np.zeros((10, 10))
            volume[1:3, 1:3] = 100.0
            widget.run_pipeline({"volume": volume, "intensity": volume})

            labels_step = widget.pipeline.steps[1]
            widget.show_step_result(labels_step)

            names = [layer.name for layer in viewer.layers]
            assert any("label_components" in name for name in names)
            # A boolean mask should also land as Labels, not Image.
            widget.show_step_result(step)
            mask_layer = [ly for ly in viewer.layers if "threshold_mask" in ly.name][0]
            assert type(mask_layer).__name__ == "Labels"
        finally:
            viewer.close()

    def test_show_is_idempotent_rather_than_stacking_duplicates(self, qtbot):
        import numpy as np

        from vtea_core.workflow import Step

        viewer = self._viewer(qtbot)
        try:
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            step = widget.pipeline.add_step(
                Step.for_function(
                    "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
                )
            )
            volume = np.zeros((8, 8))
            volume[1:3, 1:3] = 100.0
            widget.run_pipeline({"volume": volume, "intensity": volume})

            widget.show_step_result(step)
            widget.show_step_result(step)
            matching = [ly for ly in viewer.layers if "threshold_mask" in ly.name]
            assert len(matching) == 1
        finally:
            viewer.close()

    def test_non_image_result_reports_instead_of_crashing(self, qtbot):
        import numpy as np

        from vtea_core.workflow import Step

        viewer = self._viewer(qtbot)
        try:
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.pipeline.add_step(
                Step.for_function(
                    "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
                )
            )
            widget.pipeline.add_step(Step.for_function("segmentation", "label_components"))
            measure = widget.analysis_pipeline.add_step(
                Step.for_function("measurements", "extract_measurements")
            )
            volume = np.zeros((10, 10))
            volume[1:3, 1:3] = 100.0
            widget.run_pipeline({"volume": volume, "intensity": volume})

            before = len(viewer.layers)
            widget.show_step_result(measure)  # a DataFrame, not an image
            assert len(viewer.layers) == before
            assert "nothing to show" in widget.status_label.text()
        finally:
            viewer.close()


class TestAnalysisPaneAndPlot:
    def _measured(self, widget):
        import numpy as np

        from vtea_core.workflow import Step

        widget.pipeline.add_step(
            Step.for_function(
                "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
            )
        )
        widget.pipeline.add_step(Step.for_function("segmentation", "label_components"))
        widget.analysis_pipeline.add_step(
            Step.for_function("measurements", "extract_measurements")
        )
        volume = np.zeros((12, 12))
        volume[1:4, 1:4] = 100.0
        volume[7:10, 7:10] = 200.0
        return widget.run_pipeline({"volume": volume, "intensity": volume})

    def test_analysis_steps_run_after_processing(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        result = self._measured(widget)
        assert result["labels"].max() == 2
        assert len(result["measurements"]) == 2

    def test_plot_is_populated_with_one_point_per_object(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        self._measured(widget)

        frame = widget.results_table()
        assert frame is not None
        assert len(frame) == 2
        # Axes offered are the measured values - in the explorer, which is
        # where the plot lives.
        assert {"mean", "count", "object_id"} <= _axis_choices(_explorer_for(widget, qtbot))

    def test_per_object_analysis_output_becomes_a_plot_column(self, qtbot):
        """Cluster ids and similar per-object results should be selectable as
        axes/colours alongside the raw measurements, named after the step
        that produced them."""
        import numpy as np

        from vtea_core.workflow import Step

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        self._measured(widget)
        step = widget.analysis_pipeline.add_step(
            Step.for_function("clustering", "kmeans", params={"n_clusters": 2})
        )
        widget.last_context[step.name] = np.array([0, 1])

        frame = widget.results_table()
        assert step.name == "kmeans_1"
        assert "kmeans_1" in frame.columns

    def test_no_measurements_leaves_the_plot_empty_without_error(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        assert widget.results_table() is None
        widget._publish_results()  # must not raise


class TestPaneSizing:
    def test_the_panes_share_the_height(self, qtbot):
        """The steps list used to be squeezed to about one visible card."""
        from vtea_napari.widgets.step_stack import MINIMUM_STACK_HEIGHT

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        # Processing and analysis; the plot lives in the Object Explorer.
        assert widget.splitter.count() == 2
        assert widget.processing_stack.scroll.minimumHeight() >= MINIMUM_STACK_HEIGHT
        assert not widget.splitter.childrenCollapsible()

        # The real requirement: at a realistic dock height, no pane is
        # squeezed to a sliver - each should get roughly a third.
        widget.resize(500, 900)
        widget.show()
        qtbot.waitExposed(widget)
        sizes = widget.splitter.sizes()
        assert len(sizes) == 2
        total = sum(sizes)
        assert total > 0
        assert min(sizes) / total > 0.3, f"panes split unevenly: {sizes}"

    def test_the_dock_is_capped_at_thirty_percent_of_the_screen(self, qtbot):
        from qtpy.QtWidgets import QApplication

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry().width()
        assert widget.maximumWidth() <= int(available * 0.30) + 1


class TestLogView:
    def test_long_messages_wrap_instead_of_widening_the_dock(self, qtbot):
        from qtpy.QtWidgets import QPlainTextEdit

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        assert widget.status_label.lineWrapMode() == QPlainTextEdit.LineWrapMode.WidgetWidth

    def test_it_keeps_earlier_messages(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        widget.status_label.setText("first")
        widget.status_label.setText("second")
        assert "first" in widget.status_label.text()
        assert "second" in widget.status_label.text()

    def test_it_is_capped_at_a_tenth_of_the_dock(self, qtbot):
        from vtea_napari.widgets.log_view import MINIMUM_HEIGHT

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        widget.resize(500, 1000)
        widget.show()
        qtbot.waitExposed(widget)
        assert widget.status_label.maximumHeight() <= max(100, MINIMUM_HEIGHT)

    def test_it_scrolls_rather_than_growing(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        widget.resize(500, 1000)
        widget.show()
        qtbot.waitExposed(widget)
        capped = widget.status_label.maximumHeight()
        for index in range(40):
            widget.status_label.setText(f"message {index}")
        assert widget.status_label.maximumHeight() == capped
        assert widget.status_label.verticalScrollBar().maximum() > 0


class TestSourceAndAxisPickers:
    def _viewer_with_layers(self, qtbot):
        import napari
        import numpy as np

        viewer = napari.Viewer(show=False)
        qtbot.addWidget(viewer.window._qt_window)
        viewer.add_labels(np.zeros((6, 4, 16, 16), dtype="int32"), name="stack-a")
        viewer.add_labels(np.zeros((3, 8, 8), dtype="int32"), name="stack-b")
        return viewer

    def test_image_picker_lists_loaded_layers(self, qtbot):
        viewer = self._viewer_with_layers(qtbot)
        try:
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            names = {widget.layer_combo.itemText(i) for i in range(widget.layer_combo.count())}
            assert {"stack-a", "stack-b"} <= names
        finally:
            viewer.close()

    def test_picking_a_layer_selects_what_gets_processed(self, qtbot):
        viewer = self._viewer_with_layers(qtbot)
        try:
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.layer_combo.setCurrentIndex(widget.layer_combo.findData("stack-b"))
            assert widget.active_image().shape == (3, 8, 8)
            widget.layer_combo.setCurrentIndex(widget.layer_combo.findData("stack-a"))
            assert widget.active_image().shape == (6, 4, 16, 16)
        finally:
            viewer.close()

    def test_axis_pickers_follow_the_selected_layer(self, qtbot):
        viewer = self._viewer_with_layers(qtbot)
        try:
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.layer_combo.setCurrentIndex(widget.layer_combo.findData("stack-a"))
            z_axes = {widget.z_axis_combo.itemText(i) for i in range(widget.z_axis_combo.count())}
            assert "axis 0 (size 6)" in z_axes
            assert "axis 1 (size 4)" in z_axes

            widget.layer_combo.setCurrentIndex(widget.layer_combo.findData("stack-b"))
            z_axes = {widget.z_axis_combo.itemText(i) for i in range(widget.z_axis_combo.count())}
            assert "axis 0 (size 3)" in z_axes
            assert "axis 1 (size 4)" not in z_axes
        finally:
            viewer.close()

    def test_new_layers_appear_in_the_picker(self, qtbot):
        import numpy as np

        viewer = self._viewer_with_layers(qtbot)
        try:
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            viewer.add_labels(np.zeros((2, 4, 4), dtype="int32"), name="added-later")
            names = {widget.layer_combo.itemText(i) for i in range(widget.layer_combo.count())}
            assert "added-later" in names
        finally:
            viewer.close()

    def test_z_axis_selection_is_remembered(self, qtbot):
        viewer = self._viewer_with_layers(qtbot)
        try:
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.z_axis_combo.setCurrentIndex(widget.z_axis_combo.findData(0))
            assert widget.z_axis == 0
        finally:
            viewer.close()


class TestResultsFollowTheSourceAxes:
    """A channel-selected result loses the channel axis, and napari
    right-aligns arrays of differing ndim - so without padding, the result's
    z would land on the source's channel axis and the slider would show the
    wrong section."""

    def test_channel_sliced_result_is_padded_back_to_source_ndim(self, qtbot):
        import napari
        import numpy as np

        viewer = napari.Viewer(show=False)
        qtbot.addWidget(viewer.window._qt_window)
        try:
            source = np.zeros((6, 4, 16, 16), dtype="int32")  # (Z, C, Y, X)
            viewer.add_labels(source, name="src")
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.channel_axis_combo.setCurrentIndex(widget.channel_axis_combo.findData(1))

            aligned = widget.align_to_source(np.zeros((6, 16, 16), dtype="int32"))

            assert aligned.shape == (6, 1, 16, 16)
            assert aligned.ndim == source.ndim
        finally:
            viewer.close()

    def test_unsliced_result_is_left_alone(self, qtbot):
        import napari
        import numpy as np

        viewer = napari.Viewer(show=False)
        qtbot.addWidget(viewer.window._qt_window)
        try:
            viewer.add_labels(np.zeros((6, 4, 16, 16), dtype="int32"), name="src")
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.channel_axis_combo.setCurrentIndex(widget.channel_axis_combo.findData(1))

            same = np.zeros((6, 4, 16, 16), dtype="int32")
            assert widget.align_to_source(same).shape == same.shape
        finally:
            viewer.close()

    def test_shown_result_keeps_the_full_z_stack_and_aligns_with_the_source(self, qtbot):
        import napari
        import numpy as np

        from vtea_core.workflow import Step

        viewer = napari.Viewer(show=False)
        qtbot.addWidget(viewer.window._qt_window)
        try:
            volume = np.zeros((6, 4, 16, 16), dtype="float32")
            volume[:, 2, 2:6, 2:6] = 100.0
            viewer.add_labels(volume.astype("int32"), name="src")
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.channel_axis_combo.setCurrentIndex(widget.channel_axis_combo.findData(1))
            widget.z_axis_combo.setCurrentIndex(widget.z_axis_combo.findData(0))

            step = widget.pipeline.add_step(
                Step.for_function(
                    "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
                )
            )
            step.channel = 2
            widget.run_pipeline({"volume": volume, "intensity": volume})
            widget.show_step_result(step)

            layer = [ly for ly in viewer.layers if "threshold_mask" in ly.name][0]
            # Full depth retained, and same ndim as the source so napari's
            # z slider drives both together.
            assert layer.data.shape == (6, 1, 16, 16)
            assert layer.extent.world[1][0] == 5  # z extent, not a channel extent

            # 3D view should display (z, y, x), not (channel, y, x).
            assert tuple(viewer.dims.order)[-3:] == (0, 2, 3)
        finally:
            viewer.close()


class TestLayerTypeFollowsStepCategory:
    """Uses ViewerModel rather than napari.Viewer: it has the same
    add_image/add_labels/layers/dims API but builds no vispy visuals, and
    creating an Image visual needs a real GL context that CI and this
    container don't have."""

    def _viewer(self):
        from napari.components import ViewerModel

        return ViewerModel()

    def test_image_processing_result_is_an_image_not_labels(self, qtbot):
        """gaussian_blur on integer data returns integers, but it is not a
        label image - adding it as Labels renders it as random colours,
        which is what prompted this."""
        import numpy as np

        from vtea_core.workflow import Step

        viewer = self._viewer()
        volume = (np.random.default_rng(0).random((8, 8)) * 1000).astype("uint16")
        viewer.add_image(volume, name="src")
        widget = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(widget)

        step = widget.pipeline.add_step(
            Step.for_function("imageprocessing", "gaussian_blur", params={"sigma": 1.0})
        )
        widget.run_pipeline({"volume": volume, "intensity": volume})
        # The premise: the result really is an integer array.
        assert np.issubdtype(widget.last_context["volume"].dtype, np.integer)

        widget.show_step_result(step)
        layer = [ly for ly in viewer.layers if "gaussian_blur" in ly.name][0]
        assert type(layer).__name__ == "Image"

    def test_segmentation_result_is_still_labels(self, qtbot):
        import numpy as np

        from vtea_core.workflow import Step

        viewer = self._viewer()
        volume = np.zeros((10, 10))
        volume[1:4, 1:4] = 100.0
        viewer.add_image(volume, name="src")
        widget = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(widget)

        widget.pipeline.add_step(
            Step.for_function(
                "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
            )
        )
        labels_step = widget.pipeline.add_step(
            Step.for_function("segmentation", "label_components")
        )
        widget.run_pipeline({"volume": volume, "intensity": volume})
        widget.show_step_result(labels_step)

        layer = [ly for ly in viewer.layers if "label_components" in ly.name][0]
        assert type(layer).__name__ == "Labels"


def _model_viewer():
    """ViewerModel has the same layers/dims/add_* API as napari.Viewer but
    builds no vispy visuals, which need a GL context this container lacks."""
    from napari.components import ViewerModel

    return ViewerModel()


class TestRunProcessingButton:
    def test_neither_pane_has_a_run_button_of_its_own(self, qtbot):
        """Per-step Run buttons replaced it: they are finer-grained and
        unambiguous about what will run."""
        viewer = _model_viewer()
        widget = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(widget)

        assert widget.processing_stack.action_button is None
        assert widget.analysis_stack.action_button is None
        for button in widget.findChildren(QPushButton):
            assert button.text() != "Run Processing"

    def test_the_method_still_runs_the_processing_steps(self, qtbot):
        import numpy as np

        from vtea_core.workflow import Step

        viewer = _model_viewer()
        volume = np.zeros((10, 10))
        volume[1:4, 1:4] = 100.0
        viewer.add_image(volume, name="src")
        widget = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(widget)

        widget.pipeline.add_step(
            Step.for_function(
                "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
            )
        )
        widget.pipeline.add_step(Step.for_function("segmentation", "label_components"))
        widget.analysis_pipeline.add_step(
            Step.for_function("measurements", "extract_measurements")
        )

        widget.run_processing()

        assert widget.last_context["labels"].max() == 1
        # The analysis step is run from its own card, not by this button.
        assert "measurements" not in widget.last_context


class TestPerStepRun:
    def _prepared(self, qtbot):
        import numpy as np

        from vtea_core.workflow import Step

        viewer = _model_viewer()
        volume = np.zeros((12, 12))
        volume[1:4, 1:4] = 100.0
        volume[7:10, 7:10] = 200.0
        viewer.add_image(volume, name="src")
        widget = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(widget)
        widget.pipeline.add_step(
            Step.for_function(
                "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
            )
        )
        widget.pipeline.add_step(Step.for_function("segmentation", "label_components"))
        widget.run_processing()
        return widget

    def test_running_one_analysis_step_does_not_need_the_others(self, qtbot):
        from vtea_core.workflow import Step

        widget = self._prepared(qtbot)
        measure = widget.analysis_pipeline.add_step(
            Step.for_function("measurements", "extract_measurements")
        )
        widget.run_single_step(measure)
        assert len(widget.last_context["measurements"]) == 2

    def test_measurements_can_feed_several_steps_independently(self, qtbot):
        """One-to-many: clustering and reduction both consume measurements,
        neither depends on the other."""
        import numpy as np

        from vtea_core.workflow import Step

        widget = self._prepared(qtbot)
        measure = widget.analysis_pipeline.add_step(
            Step.for_function("measurements", "extract_measurements")
        )
        widget.run_single_step(measure)

        features = widget.last_context["measurements"][["mean", "count"]].to_numpy()
        widget.last_context["data"] = features

        cluster = widget.analysis_pipeline.add_step(
            Step.for_function("clustering", "kmeans", params={"n_clusters": 2})
        )
        reduce_step = widget.analysis_pipeline.add_step(
            Step.for_function("reduction", "pca", params={"n_components": 2})
        )
        widget.run_single_step(cluster)
        widget.run_single_step(reduce_step)

        assert len(widget.last_context["clusters"]) == 2
        assert np.asarray(widget.last_context["reduced"]).shape[0] == 2

    def test_cluster_result_feeds_back_as_a_measurement_feature(self, qtbot):
        """A clustering result becomes a column of the measurement table, so
        it can be used as a plot axis or by a later step."""
        from vtea_core.workflow import Step

        widget = self._prepared(qtbot)
        measure = widget.analysis_pipeline.add_step(
            Step.for_function("measurements", "extract_measurements")
        )
        widget.run_single_step(measure)
        widget.last_context["data"] = widget.last_context["measurements"][
            ["mean", "count"]
        ].to_numpy()

        cluster = widget.analysis_pipeline.add_step(
            Step.for_function("clustering", "kmeans", params={"n_clusters": 2})
        )
        widget.run_single_step(cluster)

        # Named after the step, not its shared "clusters" output key, so a
        # second clustering doesn't overwrite the first one's column.
        assert cluster.name in widget.last_context["measurements"].columns
        axes = _axis_choices(_explorer_for(widget, qtbot))
        assert cluster.name in axes

    def test_a_failing_step_reports_instead_of_raising(self, qtbot):
        from vtea_core.workflow import Step

        widget = self._prepared(qtbot)
        # kmeans has no "data" in the context yet.
        cluster = widget.analysis_pipeline.add_step(
            Step.for_function("clustering", "kmeans", params={"n_clusters": 2})
        )
        widget.run_single_step(cluster)
        assert "kmeans" in widget.status_label.text()


class TestPlotIsFedByMeasurements:
    def test_measured_features_become_plot_axes_on_channel_sliced_data(self, qtbot):
        """The reported symptom: nothing populated the x/y axes. The cause
        was extract_measurements raising, because the segmentation step was
        channel-sliced to 3D while intensity stayed 4D - so the run aborted
        before producing any measurements."""
        import numpy as np

        viewer = _model_viewer()
        volume = np.zeros((6, 4, 16, 16), dtype="float32")
        volume[:, 2, 2:6, 2:6] = 100.0
        viewer.add_image(volume, name="src")
        widget = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(widget)
        widget.channel_axis_combo.setCurrentIndex(widget.channel_axis_combo.findData(1))

        widget.processing_stack.category_combo.setCurrentText("segmentation")
        widget.processing_stack.function_combo.setCurrentText("threshold_mask")
        widget.processing_stack._add_step_from_selection()
        threshold = widget.pipeline.steps[0]
        threshold.params = {"method": "fixed", "value": 50.0}
        threshold.channel = 2

        widget.processing_stack.function_combo.setCurrentText("label_components")
        widget.processing_stack._add_step_from_selection()
        widget.analysis_stack.category_combo.setCurrentText("measurements")
        widget.analysis_stack.function_combo.setCurrentText("extract_measurements")
        widget.analysis_stack._add_step_from_selection()

        widget.run_processing()
        widget.run_single_step(widget.analysis_pipeline.steps[0])

        axes = _axis_choices(_explorer_for(widget, qtbot))
        assert {"mean", "count", "sum", "stddev"} <= axes

    def test_new_steps_inherit_the_channel_already_in_use(self, qtbot):
        import numpy as np

        viewer = _model_viewer()
        viewer.add_image(np.zeros((6, 4, 16, 16), dtype="float32"), name="src")
        widget = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(widget)
        widget.channel_axis_combo.setCurrentIndex(widget.channel_axis_combo.findData(1))

        widget.processing_stack.category_combo.setCurrentText("segmentation")
        widget.processing_stack.function_combo.setCurrentText("threshold_mask")
        widget.processing_stack._add_step_from_selection()
        widget.pipeline.steps[0].channel = 2

        widget.analysis_stack.category_combo.setCurrentText("measurements")
        widget.analysis_stack.function_combo.setCurrentText("extract_measurements")
        widget.analysis_stack._add_step_from_selection()

        assert widget.analysis_pipeline.steps[0].channel == 2


class TestCompactStyling:
    def test_text_is_scaled_down(self, qtbot):
        from qtpy.QtWidgets import QApplication

        from vtea_napari.widgets.protocol_builder import COMPACT_FONT_SCALE

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        expected = QApplication.font().pointSizeF() * COMPACT_FONT_SCALE
        assert f"{expected:.1f}pt" in widget.styleSheet()
        assert COMPACT_FONT_SCALE == 0.75

    def test_layout_padding_is_tightened(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        margins = widget.layout().contentsMargins()
        assert margins.top() <= 4 and margins.left() <= 4
        assert widget.layout().spacing() <= 3


def _measured_multichannel(qtbot):
    """A builder that has segmented one channel of a (z, c, y, x) image and
    measured that segmentation against every channel - the state the last
    five UI items are all about."""
    import numpy as np

    from vtea_core.workflow import Step

    viewer = _model_viewer()
    image = np.zeros((4, 3, 12, 12))
    for channel in range(3):
        image[:, channel, 1:5, 1:5] = 100.0 * (channel + 1)
        image[:, channel, 8:11, 8:11] = 50.0 * (channel + 1)
    viewer.add_image(image, name="src")
    widget = ProtocolBuilderWidget(napari_viewer=viewer)
    qtbot.addWidget(widget)
    widget.channel_axis_combo.setCurrentIndex(widget.channel_axis_combo.findData(1))

    widget.pipeline.add_step(
        Step.for_function(
            "segmentation",
            "threshold_mask",
            params={"method": "fixed", "value": 50.0},
            channel=0,
        )
    )
    labeler = widget.pipeline.add_step(
        Step.for_function(
            "segmentation",
            "label_components",
            available={"mask"},
            taken_names=["threshold_mask_1"],
        )
    )
    widget.run_processing()

    measure = widget.analysis_pipeline.add_step(
        Step.for_function(
            "measurements",
            "extract_measurements_by_channel",
            available=set(widget.last_context) | {"channel_axis"},
            taken_names=widget.step_names(),
        )
    )
    measure.input_keys["labels"] = labeler.name
    widget.run_single_step(measure)
    return widget


class TestNamedSegmentations:
    """Item 1: every step's result carries a unique, editable name, so a
    later step can say which segmentation it means."""

    def test_a_step_added_from_the_gui_gets_a_default_name(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        stack = widget.processing_stack
        stack.category_combo.setCurrentText("segmentation")
        stack.function_combo.setCurrentText("watershed_split")
        _click_button(qtbot, stack, "Add Step")
        assert widget.pipeline.steps[0].name == "watershed_split_1"

    def test_two_of_the_same_step_get_different_names(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        stack = widget.processing_stack
        stack.category_combo.setCurrentText("segmentation")
        stack.function_combo.setCurrentText("watershed_split")
        _click_button(qtbot, stack, "Add Step")
        _click_button(qtbot, stack, "Add Step")
        names = [step.name for step in widget.pipeline.steps]
        assert names == ["watershed_split_1", "watershed_split_2"]

    def test_names_are_unique_across_both_panes(self, qtbot):
        """The two panes share one run context, so a name used in the
        processing pane must not be reused in the analysis pane."""
        from vtea_core.workflow import Step

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        widget.pipeline.add_step(Step.for_function("measurements", "extract_measurements"))
        stack = widget.analysis_stack
        stack.category_combo.setCurrentText("measurements")
        stack.function_combo.setCurrentText("extract_measurements")
        _click_button(qtbot, stack, "Add Step")
        assert widget.analysis_pipeline.steps[0].name == "extract_measurements_2"

    def test_the_name_is_shown_on_the_card(self, qtbot):
        from vtea_core.workflow import Step

        from vtea_napari.widgets.step_card import StepCardWidget

        step = Step.for_function("segmentation", "watershed_split")
        card = StepCardWidget(1, step)
        qtbot.addWidget(card)
        assert card.name_label.text() == "watershed_split_1"

    def test_renaming_keeps_names_unique(self, qtbot):
        from vtea_core.workflow import Step

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        first = widget.pipeline.add_step(Step.for_function("segmentation", "watershed_split"))
        second = widget.pipeline.add_step(
            Step.for_function("segmentation", "label_components")
        )
        widget.processing_stack.rename_step(second, first.name)
        assert second.name != first.name

    def test_a_blank_name_falls_back_to_a_default(self, qtbot):
        from vtea_core.workflow import Step

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        step = widget.pipeline.add_step(Step.for_function("segmentation", "watershed_split"))
        widget.processing_stack.rename_step(step, "")
        assert step.name == "watershed_split_1"

    def test_renaming_repoints_steps_that_referred_to_the_old_name(self, qtbot):
        """A rename that silently broke a downstream measurement step would
        be worse than not allowing renames at all."""
        from vtea_core.workflow import Step

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        segmentation = widget.pipeline.add_step(
            Step.for_function("segmentation", "label_components")
        )
        measure = widget.analysis_pipeline.add_step(
            Step.for_function("measurements", "extract_measurements")
        )
        measure.input_keys["labels"] = segmentation.name

        widget.processing_stack.rename_step(segmentation, "nuclei")
        assert measure.input_keys["labels"] == "nuclei"

    def test_each_card_previews_its_own_result_not_the_last_one(self, qtbot):
        """Both segmentations write "labels"; without per-name lookup every
        card would show whichever ran last."""
        import numpy as np

        from vtea_core.workflow import Step

        from vtea_napari.widgets.step_card import StepCardWidget

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        first = widget.pipeline.add_step(Step.for_function("segmentation", "label_components"))
        second = widget.pipeline.add_step(
            Step.for_function("segmentation", "filter_by_size", params={"min_size": 2})
        )
        widget.last_context = {
            first.name: np.ones((4, 4), dtype=np.int32),
            second.name: np.zeros((4, 4), dtype=np.int32),
            "labels": np.zeros((4, 4), dtype=np.int32),
        }
        widget.refresh_steps()
        cards = widget.processing_stack.findChildren(StepCardWidget)
        assert len(cards) == 2
        # The first card previews the non-empty first segmentation; the
        # second previews the emptied one.
        assert cards[0].thumbnail_label.pixmap() is not None


class TestMeasurementsPickASegmentation:
    """Item 2: a measurement step measures a *named* segmentation, across
    every channel unless one is chosen."""

    def _widget_with_two_segmentations(self, qtbot):
        from vtea_core.workflow import Step

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        widget.pipeline.add_step(Step.for_function("segmentation", "label_components"))
        widget.pipeline.add_step(
            Step.for_function(
                "segmentation", "filter_by_size", params={"min_size": 2}, taken_names=["label_components_1"]
            )
        )
        return widget

    def test_input_candidates_list_every_named_producer(self, qtbot):
        widget = self._widget_with_two_segmentations(qtbot)
        assert widget.input_candidates("labels") == [
            "labels",
            "label_components_1",
            "filter_by_size_1",
        ]

    def test_the_edit_dialog_offers_them(self, qtbot):
        from vtea_napari.widgets.protocol_builder import EditStepDialog

        widget = self._widget_with_two_segmentations(qtbot)
        from vtea_core.workflow import Step

        measure = Step.for_function("measurements", "extract_measurements_by_channel")
        dialog = EditStepDialog(measure, input_candidates=widget.input_candidates)
        qtbot.addWidget(dialog)
        combo = dialog.input_combos["labels"]
        choices = [combo.itemData(i) for i in range(combo.count())]
        assert "label_components_1" in choices

    def test_choosing_one_rewires_the_step(self, qtbot):
        from vtea_core.workflow import Step

        from vtea_napari.widgets.protocol_builder import EditStepDialog

        widget = self._widget_with_two_segmentations(qtbot)
        measure = Step.for_function("measurements", "extract_measurements_by_channel")
        dialog = EditStepDialog(measure, input_candidates=widget.input_candidates)
        qtbot.addWidget(dialog)
        dialog.input_combos["labels"].setCurrentIndex(
            dialog.input_combos["labels"].findData("label_components_1")
        )
        assert dialog.updated_input_keys()["labels"] == "label_components_1"

    def test_the_measurements_category_defaults_to_the_multichannel_step(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        stack = widget.analysis_stack
        stack.category_combo.setCurrentText("measurements")
        assert stack.function_combo.currentText() == "extract_measurements_by_channel"

    def test_a_measurement_step_starts_on_all_channels(self, qtbot):
        """Even when an earlier segmentation picked one - measuring every
        channel is the point of the multi-channel step."""
        from vtea_core.workflow import Step

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        widget.pipeline.add_step(
            Step.for_function("segmentation", "threshold_mask", channel=2)
        )
        stack = widget.analysis_stack
        stack.category_combo.setCurrentText("measurements")
        stack.function_combo.setCurrentText("extract_measurements_by_channel")
        _click_button(qtbot, stack, "Add Step")
        assert widget.analysis_pipeline.steps[0].channel is None

    def test_running_it_produces_channel_tagged_features(self, qtbot):
        """Items 2 and 4 together: one table, one row per object, feature
        names carrying the channel they were measured on."""
        widget = _measured_multichannel(qtbot)

        frame = widget.last_context["measurements"]
        assert len(frame) == 2
        assert {"mean_ch0", "mean_ch1", "mean_ch2"} <= set(frame.columns)
        assert frame.loc[0, "mean_ch1"] == 200.0
        # Geometry is not repeated per channel.
        assert list(frame.columns).count("count") == 1

    def test_the_channel_tagged_features_reach_the_plot_axes(self, qtbot):
        """Item 3: those names are what the X/Y menus offer."""
        widget = _measured_multichannel(qtbot)
        axes = _axis_choices(_explorer_for(widget, qtbot))
        assert {"mean_ch0", "mean_ch1", "mean_ch2"} <= axes

    def test_the_channel_axis_is_seeded_for_the_step_to_read(self, qtbot):
        """Nothing in a protocol produces "channel_axis"; the widget has to
        put it in the context or the step measures one channel only."""
        widget = _measured_multichannel(qtbot)
        assert widget.last_context["channel_axis"] == 1


class TestAnalysisResultsBecomeFeatures:
    """Item 5: PCA/t-SNE/clustering outputs join the same table under
    unique names, so they are plottable from the X/Y menus."""

    def test_a_clustering_run_from_its_card_adds_a_named_column(self, qtbot):
        from vtea_core.workflow import Step

        widget = _measured_multichannel(qtbot)
        cluster = widget.analysis_pipeline.add_step(
            Step.for_function(
                "clustering", "kmeans", params={"n_clusters": 2}, taken_names=widget.step_names()
            )
        )
        widget.run_single_step(cluster)

        assert cluster.name == "kmeans_1"
        assert "kmeans_1" in widget.results_table().columns
        axes = _axis_choices(_explorer_for(widget, qtbot))
        assert "kmeans_1" in axes

    def test_a_reduction_adds_one_column_per_component(self, qtbot):
        from vtea_core.workflow import Step

        widget = _measured_multichannel(qtbot)
        reduce_step = widget.analysis_pipeline.add_step(
            Step.for_function(
                "reduction", "pca", params={"n_components": 1}, taken_names=widget.step_names()
            )
        )
        widget.run_single_step(reduce_step)
        assert f"{reduce_step.name}_1" in widget.results_table().columns

    def test_two_clusterings_do_not_overwrite_each_other(self, qtbot):
        from vtea_core.workflow import Step

        widget = _measured_multichannel(qtbot)
        for _ in range(2):
            widget.analysis_pipeline.add_step(
                Step.for_function(
                    "clustering",
                    "kmeans",
                    params={"n_clusters": 2},
                    taken_names=widget.step_names(),
                )
            )
        for step in list(widget.analysis_pipeline.steps):
            if step.category == "clustering":
                widget.run_single_step(step)

        columns = widget.results_table().columns
        assert "kmeans_1" in columns
        assert "kmeans_2" in columns

    def test_data_is_built_from_the_table_so_the_step_can_run_at_all(self, qtbot):
        """Nothing in a protocol produces a "data" key; without the widget
        deriving it, every clustering/reduction step fails with
        "needs context key(s) ['data']"."""
        from vtea_core.workflow import Step

        widget = _measured_multichannel(qtbot)
        cluster = widget.analysis_pipeline.add_step(
            Step.for_function(
                "clustering", "kmeans", params={"n_clusters": 2}, taken_names=widget.step_names()
            )
        )
        widget.run_single_step(cluster)
        assert "needs context key" not in widget.status_label.text()
        assert cluster.name in widget.last_context

    def test_identifiers_and_centroids_are_left_out_of_the_features(self, qtbot):
        widget = _measured_multichannel(qtbot)
        frame = widget.results_table()
        expected = [
            column
            for column in frame.columns
            if column != "object_id" and not column.startswith("centroid-")
        ]
        assert widget.available_features() == expected

    def test_the_whole_table_is_seeded_so_each_step_can_narrow_it(self, qtbot):
        """`data` is the table, not a pre-built matrix: the step decides
        which of its columns it is built from."""
        import pandas as pd

        widget = _measured_multichannel(qtbot)
        context = {}
        widget._seed_feature_matrix(context)
        assert isinstance(context["data"], pd.DataFrame)


class TestTabularStepsShowNoChannel:
    """A clustering or reduction step reads the measured feature table, not
    the image. Offering it a channel picker says something untrue about what
    it does - every channel is already a column of that table."""

    def test_the_edit_dialog_hides_the_channel_row(self, qtbot):
        from vtea_core.workflow import Step

        from vtea_napari.widgets.protocol_builder import EditStepDialog

        dialog = EditStepDialog(
            Step.for_function("clustering", "kmeans"), n_channels=3
        )
        qtbot.addWidget(dialog)
        # Only the "All channels" placeholder; no per-channel entries, and
        # no visible row offering them. What it shows instead is the feature
        # picker, since that is the choice this step actually has.
        assert dialog.channel_combo.count() == 1
        assert dialog.updated_channel() is None
        assert dialog.feature_select is not None

    def test_an_image_step_still_shows_it(self, qtbot):
        from vtea_core.workflow import Step

        from vtea_napari.widgets.protocol_builder import EditStepDialog

        dialog = EditStepDialog(
            Step.for_function("segmentation", "threshold_mask"), n_channels=3
        )
        qtbot.addWidget(dialog)
        assert dialog.channel_combo.count() == 4  # All channels + 3

    def test_adding_one_does_not_inherit_the_segmentation_channel(self, qtbot):
        from vtea_core.workflow import Step

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        widget.pipeline.add_step(
            Step.for_function("segmentation", "threshold_mask", channel=2)
        )
        stack = widget.analysis_stack
        stack.category_combo.setCurrentText("clustering")
        stack.function_combo.setCurrentText("kmeans")
        _click_button(qtbot, stack, "Add Step")

        assert widget.analysis_pipeline.steps[0].channel is None

    def test_the_card_says_which_features_it_uses(self, qtbot):
        """Not a channel - what a clustering or reduction step is built
        from is a set of features, and the card should say how many."""
        from vtea_core.workflow import Step

        from vtea_napari.widgets.step_card import StepCardWidget, summarize_channel

        step = Step.for_function("reduction", "pca")
        card = StepCardWidget(1, step)
        qtbot.addWidget(card)
        assert summarize_channel(step) == "all features"
        assert "all features" in card.channel_label.text()

        step.features = ["mean_ch0", "mean_ch1", "count"]
        assert summarize_channel(step) == "3 feature(s)"

    def test_a_step_with_no_channel_and_no_features_says_no_channel(self, qtbot):
        """A gate reads a table and a derived segmentation reads a label
        image; neither has a channel, and neither reads "the feature
        table" in the sense the clustering steps do."""
        from vtea_core.workflow import Step

        from vtea_napari.widgets.step_card import summarize_channel

        assert summarize_channel(Step.for_function("gates", "polygon_gate")) == "no channel"
        assert summarize_channel(Step.for_function("segmentation", "label_ring")) == "no channel"

    def test_a_clustering_step_runs_on_the_feature_table_from_every_channel(self, qtbot):
        """The end of the chain: features measured on three channels feed
        one clustering, with no channel selection anywhere in between."""
        from vtea_core.workflow import Step

        widget = _measured_multichannel(qtbot)
        cluster = widget.analysis_pipeline.add_step(
            Step.for_function(
                "clustering", "kmeans", params={"n_clusters": 2}, taken_names=widget.step_names()
            )
        )
        widget.run_single_step(cluster)

        assert cluster.channel is None
        assert cluster.name in widget.last_context
        # It clustered on features from all three channels, not one.
        columns = widget.results_table().columns
        assert {"mean_ch0", "mean_ch1", "mean_ch2"} <= set(columns)


class TestFeatureSelectionThroughTheGui:
    """A clustering step is built from the features chosen for it, and that
    choice is recorded on the step so the protocol carries it."""

    def _clustering(self, widget, **kwargs):
        from vtea_core.workflow import Step

        return widget.analysis_pipeline.add_step(
            Step.for_function(
                "clustering",
                "kmeans",
                params={"n_clusters": 2},
                taken_names=widget.step_names(),
                **kwargs,
            )
        )

    def test_the_builder_offers_every_measured_feature(self, qtbot):
        widget = _measured_multichannel(qtbot)
        features = widget.available_features()
        assert {"mean_ch0", "mean_ch1", "mean_ch2", "count"} <= set(features)
        # Identifiers and centroids are not features to cluster on.
        assert "object_id" not in features
        assert not any(name.startswith("centroid-") for name in features)

    def test_the_edit_dialog_lists_them(self, qtbot):
        from vtea_napari.widgets.protocol_builder import EditStepDialog

        widget = _measured_multichannel(qtbot)
        step = self._clustering(widget)
        dialog = EditStepDialog(
            step,
            available_features=widget.available_features(),
            feature_catalog=widget.feature_catalog(),
        )
        qtbot.addWidget(dialog)
        assert dialog.feature_select is not None
        assert set(dialog.feature_select.selected()) == set(widget.available_features())

    def test_a_selection_made_in_the_dialog_lands_on_the_step(self, qtbot, monkeypatch):
        from qtpy.QtWidgets import QDialog

        from vtea_napari.widgets.protocol_builder import EditStepDialog

        widget = _measured_multichannel(qtbot)
        step = self._clustering(widget)
        widget.refresh_steps()

        def fake_exec(self):
            self.feature_select.select_none()
            self.feature_select.filter_edit.setText("mean_ch")
            self.feature_select.select_all()
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(EditStepDialog, "exec", fake_exec)
        widget.analysis_stack._edit_step(step)

        # A plain substring filter, so "mean_ch" catches threshold_mean_ch*
        # too - the point is that one gesture selected a family of features.
        assert step.features == [
            name for name in widget.available_features() if "mean_ch" in name
        ]
        assert "mean_ch0" in step.features
        assert "count" not in step.features

    def test_the_step_runs_on_only_those_features(self, qtbot):
        widget = _measured_multichannel(qtbot)
        step = self._clustering(widget, features=["mean_ch0"])
        widget.run_single_step(step)

        assert step.name in widget.last_context
        assert widget.last_context[step.name].shape == (len(widget.results_table()),)

    def test_a_selection_of_everything_is_stored_as_no_selection(self, qtbot, monkeypatch):
        """So the protocol doesn't pin a list that should grow when a later
        measurement step adds features."""
        from qtpy.QtWidgets import QDialog

        from vtea_napari.widgets.protocol_builder import EditStepDialog

        widget = _measured_multichannel(qtbot)
        step = self._clustering(widget, features=["mean_ch0"])
        widget.refresh_steps()

        monkeypatch.setattr(
            EditStepDialog,
            "exec",
            lambda self: (self.feature_select.select_all(), QDialog.DialogCode.Accepted)[1],
        )
        widget.analysis_stack._edit_step(step)

        assert step.features == []

    def test_the_card_shows_how_many_features(self, qtbot):
        from vtea_napari.widgets.step_card import StepCardWidget

        widget = _measured_multichannel(qtbot)
        self._clustering(widget, features=["mean_ch0", "mean_ch1"])
        widget.refresh_steps()
        cards = widget.analysis_stack.findChildren(StepCardWidget)
        assert "2 feature(s)" in cards[-1].channel_label.text()


class TestFeatureProvenanceIsRecorded:
    """Every column of the data table records what it is and how it was
    produced - which is what makes the numbers interpretable later."""

    def test_measured_features_are_catalogued(self, qtbot):
        widget = _measured_multichannel(qtbot)
        catalog = widget.feature_catalog()
        assert set(catalog.names()) == set(widget.results_table().columns)

    def test_a_measured_feature_records_its_channel_and_segmentation(self, qtbot):
        widget = _measured_multichannel(qtbot)
        descriptor = widget.feature_catalog().get("mean_ch2")
        assert descriptor.measurement == "mean"
        assert descriptor.channel == 2
        assert descriptor.segmentation == "label_components_1"
        assert descriptor.function == "measurements.extract_measurements_by_channel"

    def test_a_clustering_records_the_features_it_was_built_from(self, qtbot):
        from vtea_core.workflow import Step

        widget = _measured_multichannel(qtbot)
        step = widget.analysis_pipeline.add_step(
            Step.for_function(
                "clustering",
                "kmeans",
                params={"n_clusters": 2},
                features=["mean_ch0", "mean_ch1"],
                taken_names=widget.step_names(),
            )
        )
        widget.run_single_step(step)

        descriptor = widget.feature_catalog().get(step.name)
        assert descriptor.kind == "derived"
        assert descriptor.source_features == ["mean_ch0", "mean_ch1"]
        assert descriptor.params == {"n_clusters": 2}
        assert descriptor.function == "clustering.kmeans"

    def test_a_step_with_no_selection_records_the_features_it_actually_used(self, qtbot):
        from vtea_core.workflow import Step

        widget = _measured_multichannel(qtbot)
        step = widget.analysis_pipeline.add_step(
            Step.for_function(
                "clustering",
                "kmeans",
                params={"n_clusters": 2},
                taken_names=widget.step_names(),
            )
        )
        widget.run_single_step(step)

        sources = widget.feature_catalog().get(step.name).source_features
        assert set(sources) == set(widget.available_features()) - {step.name}
        # It must not record itself as one of its own inputs.
        assert step.name not in sources

    def test_a_reduction_records_one_entry_per_component(self, qtbot):
        from vtea_core.workflow import Step

        widget = _measured_multichannel(qtbot)
        step = widget.analysis_pipeline.add_step(
            Step.for_function(
                "reduction",
                "pca",
                params={"n_components": 2},
                features=["mean_ch0", "mean_ch1", "mean_ch2"],
                taken_names=widget.step_names(),
            )
        )
        widget.run_single_step(step)

        catalog = widget.feature_catalog()
        for component in (1, 2):
            descriptor = catalog.get(f"{step.name}_{component}")
            assert descriptor.measurement == "reduced dimension"
            assert descriptor.source_features == ["mean_ch0", "mean_ch1", "mean_ch2"]

    def test_a_re_measurement_forgets_the_old_columns(self, qtbot):
        """A stale catalog entry is worse than a missing one - it looks
        authoritative."""
        widget = _measured_multichannel(qtbot)
        catalog = widget.feature_catalog()
        catalog.record_measured(["mean_ch9"], produced_by="from_another_run")
        assert "mean_ch9" in catalog

        widget.run_single_step(widget.analysis_pipeline.steps[0])
        assert "mean_ch9" not in widget.feature_catalog()

    def test_the_catalog_renders_as_a_data_dictionary(self, qtbot):
        widget = _measured_multichannel(qtbot)
        dictionary = widget.feature_catalog().to_dataframe()
        assert len(dictionary) == len(widget.results_table().columns)
        row = dictionary.set_index("column").loc["mean_ch1"]
        assert row["channel"] == 1
        assert row["segmentation"] == "label_components_1"

    def test_the_catalog_lives_on_the_shared_session(self, qtbot):
        """So the explorer sees it, and so it survives a pane closing."""
        widget = _measured_multichannel(qtbot)
        assert widget.feature_catalog() is widget.session.feature_catalog


class TestDerivedSegmentationWorkflow:
    """The half of cell association that needs no inference: a nucleus, an
    envelope derived from it, a cytosol band outside that, and an exact
    statement of which belongs to which."""

    def _builder(self, qtbot):
        import numpy as np

        from vtea_core.workflow import Step

        viewer = _model_viewer()
        volume = np.zeros((16, 16))
        volume[3:7, 3:7] = 200.0
        volume[10:14, 10:14] = 200.0
        viewer.add_image(volume, name="dapi", scale=(0.5, 0.5))
        widget = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(widget)
        widget.pipeline.add_step(
            Step.for_function(
                "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
            )
        )
        widget.pipeline.add_step(
            Step.for_function("segmentation", "label_components", available={"mask"})
        )
        return widget

    def test_the_derived_steps_are_offered_alongside_the_others(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        stack = widget.processing_stack
        stack.category_combo.setCurrentText("segmentation")
        offered = {stack.function_combo.itemText(i) for i in range(stack.function_combo.count())}
        assert {"expand_labels", "label_ring", "label_shell"} <= offered

    def test_association_is_its_own_analysis_category(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        stack = widget.analysis_stack
        stack.category_combo.setCurrentText("association")
        offered = {stack.function_combo.itemText(i) for i in range(stack.function_combo.count())}
        assert "associate_by_identity" in offered

    def test_a_derived_step_gets_no_channel_and_the_spacing(self, qtbot):
        widget = self._builder(qtbot)
        widget.run_processing()
        stack = widget.processing_stack
        stack.category_combo.setCurrentText("segmentation")
        stack.function_combo.setCurrentText("label_ring")
        _click_button(qtbot, stack, "Add Step")

        step = widget.pipeline.steps[-1]
        assert step.channel_applies is False
        assert step.channel is None
        assert "spacing" in step.input_keys

    def test_a_cytosol_ring_runs_and_keeps_its_parent_s_ids(self, qtbot):
        import numpy as np

        from vtea_core.workflow import Step

        widget = self._builder(qtbot)
        widget.run_processing()
        nuclei = widget.last_context["labels"]

        ring = widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "label_ring",
                available=set(widget.last_context) | {"spacing"},
                params={"thickness": 1.0},
            )
        )
        widget.run_single_step(ring)

        rings = widget.last_context[ring.name]
        assert set(np.unique(rings)) == set(np.unique(nuclei))
        assert (rings[nuclei != 0] == 0).all()

    def test_the_association_step_links_them_by_name(self, qtbot):
        from vtea_core.objects import ObjectRef
        from vtea_core.workflow import Step

        widget = self._builder(qtbot)
        widget.run_processing()
        nuclei_step = widget.pipeline.steps[-1]

        ring = widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "label_ring",
                available=set(widget.last_context) | {"spacing"},
                params={"thickness": 1.0},
                taken_names=widget.step_names(),
            )
        )
        widget.run_single_step(ring)

        associate = widget.analysis_pipeline.add_step(
            Step.for_function(
                "association",
                "associate_by_identity",
                available=set(widget.last_context),
                # No names passed: they come from the wiring below.
                taken_names=widget.step_names(),
            )
        )
        associate.input_keys["child_labels"] = ring.name
        associate.input_keys["parent_labels"] = nuclei_step.name
        widget.run_single_step(associate)

        associations = widget.last_context[associate.name]
        assert len(associations) == 2
        assert associations.parent_of(ObjectRef(ring.name, 1)) == ObjectRef(nuclei_step.name, 1)
        assert all(link.is_certain for link in associations)

    def test_an_association_step_offers_every_segmentation_for_its_inputs(self, qtbot):
        """Its inputs are named for their role, not for a context key, so
        they still need the by-name picker measurement steps get."""
        from vtea_core.workflow import Step

        widget = self._builder(qtbot)
        widget.pipeline.add_step(
            Step.for_function("segmentation", "label_ring", taken_names=widget.step_names())
        )
        candidates = widget.input_candidates("child_labels")
        assert "label_components_1" in candidates
        assert "label_ring_1" in candidates


class TestProbabilisticAssociation:
    """Two channels segmented independently, and the question that follows:
    which cytoplasm belongs to which nucleus. Unlike the derived case
    nothing in the data answers it, so the step has to be run and its
    posterior read."""

    def _builder(self, qtbot):
        """A field of four cells: a bright nucleus inside each of four
        cytoplasm blocks, in two channels of one image."""
        import numpy as np

        viewer = _model_viewer()
        volume = np.zeros((2, 20, 80))
        for index in range(4):
            left = index * 20 + 2
            volume[0, 3:17, left : left + 16] = 200.0  # cytoplasm
            centre = left + 8
            volume[1, 8:12, centre - 2 : centre + 2] = 200.0  # nucleus
        viewer.add_image(volume, name="two channel")
        widget = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(widget)
        widget.channel_axis_combo.setCurrentIndex(1)  # axis 0
        return widget

    def _segment(self, widget, channel, name):
        from vtea_core.workflow import Step

        threshold = widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "threshold_mask",
                params={"method": "fixed", "value": 50.0},
                channel=channel,
                taken_names=widget.step_names(),
            )
        )
        widget.run_single_step(threshold)
        labels = widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "label_components",
                available=set(widget.last_context),
                name=name,
                taken_names=widget.step_names(),
            )
        )
        labels.input_keys["mask"] = threshold.name
        widget.run_single_step(labels)
        return labels

    def test_the_new_steps_are_offered_in_their_menus(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        widget.processing_stack.category_combo.setCurrentText("segmentation")
        combo = widget.processing_stack.function_combo
        assert "watershed_ownership" in {combo.itemText(i) for i in range(combo.count())}

        widget.analysis_stack.category_combo.setCurrentText("association")
        combo = widget.analysis_stack.function_combo
        assert "associate_objects" in {combo.itemText(i) for i in range(combo.count())}

    def test_it_links_each_cytoplasm_to_the_nucleus_inside_it(self, qtbot):
        import numpy as np
        from vtea_core.workflow import Step

        widget = self._builder(qtbot)
        cytoplasm = self._segment(widget, 0, "cytoplasm")
        nuclei = self._segment(widget, 1, "nuclei")

        associate = widget.analysis_pipeline.add_step(
            Step.for_function(
                "association",
                "associate_objects",
                available=set(widget.last_context),
                params={"method": "containment", "mode": "one_to_one"},
                taken_names=widget.step_names(),
            )
        )
        associate.input_keys["child_labels"] = cytoplasm.name
        associate.input_keys["parent_labels"] = nuclei.name
        widget.run_single_step(associate)

        links = widget.last_context[associate.name]
        assert len(links) == 4
        assert links.unassigned == []
        # The two segmentations number their objects independently, so the
        # pairing has to be checked against the images, not against the ids:
        # every linked nucleus lies inside the cytoplasm it was given to.
        cytoplasms = widget.last_context[cytoplasm.name]
        nucleus_labels = widget.last_context[nuclei.name]
        for link in links:
            inside = cytoplasms[nucleus_labels == link.parent.object_id]
            assert set(np.unique(inside)) == {link.child.object_id}

    def test_the_links_are_named_for_the_steps_they_were_wired_to(self, qtbot):
        """Without the wiring filling these in, every ObjectRef would say
        `child#3` - the function's own default - and the record would not
        say which segmentation an object came from."""
        from vtea_core.workflow import Step

        widget = self._builder(qtbot)
        cytoplasm = self._segment(widget, 0, "cytoplasm")
        nuclei = self._segment(widget, 1, "nuclei")

        associate = widget.analysis_pipeline.add_step(
            Step.for_function(
                "association",
                "associate_objects",
                available=set(widget.last_context),
                taken_names=widget.step_names(),
            )
        )
        associate.input_keys["child_labels"] = cytoplasm.name
        associate.input_keys["parent_labels"] = nuclei.name
        widget.run_single_step(associate)

        link = next(iter(widget.last_context[associate.name]))
        assert link.child.segmentation == "cytoplasm"
        assert link.parent.segmentation == "nuclei"

    def test_rewiring_the_step_renames_the_segmentations_too(self, qtbot):
        """The names are the wiring, so pointing the step somewhere else has
        to move them with it."""
        from vtea_core.workflow import Step

        widget = self._builder(qtbot)
        cytoplasm = self._segment(widget, 0, "cytoplasm")
        nuclei = self._segment(widget, 1, "nuclei")

        associate = Step.for_function(
            "association", "associate_objects", available=set(widget.last_context)
        )
        associate.input_keys["child_labels"] = nuclei.name
        associate.input_keys["parent_labels"] = cytoplasm.name
        widget.analysis_pipeline.add_step(associate)
        widget.run_single_step(associate)

        link = next(iter(widget.last_context[associate.name]))
        assert link.child.segmentation == "nuclei"
        assert link.parent.segmentation == "cytoplasm"

    def test_the_log_says_how_much_of_it_worked(self, qtbot):
        """An association draws nothing, so "it ran" would hide the one
        number that says whether the parameters were right."""
        import numpy as np
        from vtea_core.workflow import Step

        widget = self._builder(qtbot)
        cytoplasm = self._segment(widget, 0, "cytoplasm")
        nuclei = self._segment(widget, 1, "nuclei")
        # Take one nucleus away: that cytoplasm now has nothing to link to.
        widget.last_context[nuclei.name] = np.where(
            widget.last_context[nuclei.name] == 3, 0, widget.last_context[nuclei.name]
        )

        associate = Step.for_function(
            "association",
            "associate_objects",
            available=set(widget.last_context),
            params={"mode": "one_to_one"},
        )
        associate.input_keys["child_labels"] = cytoplasm.name
        associate.input_keys["parent_labels"] = nuclei.name
        widget.analysis_pipeline.add_step(associate)
        widget.run_single_step(associate)

        logged = widget.status_label.toPlainText()
        assert "3 linked" in logged
        assert "1 unassigned" in logged

    def test_ownership_splits_a_shared_region_between_two_nuclei(self, qtbot):
        """The deterministic answer to a contested area, run from the
        builder: one cytoplasm mask, two nuclei, two territories."""
        import numpy as np
        from vtea_core.workflow import Step

        viewer = _model_viewer()
        volume = np.zeros((2, 16, 40))
        volume[0, 3:13, 2:38] = 200.0  # one connected cytoplasm
        volume[1, 7:9, 6:9] = 200.0
        volume[1, 7:9, 31:34] = 200.0
        viewer.add_image(volume, name="two channel")
        widget = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(widget)
        widget.channel_axis_combo.setCurrentIndex(1)

        cytoplasm_mask = widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "threshold_mask",
                params={"method": "fixed", "value": 50.0},
                channel=0,
                taken_names=widget.step_names(),
            )
        )
        widget.run_single_step(cytoplasm_mask)
        nuclei = self._segment(widget, 1, "nuclei")

        ownership = widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "watershed_ownership",
                available=set(widget.last_context) | {"spacing"},
                taken_names=widget.step_names(),
            )
        )
        ownership.input_keys["labels"] = nuclei.name
        ownership.input_keys["mask"] = cytoplasm_mask.name
        widget.run_single_step(ownership)

        territories = widget.last_context[ownership.name]
        assert set(np.unique(territories)) == {0, 1, 2}
        assert territories[8, 7] != territories[8, 32]

    def test_a_choice_of_methods_is_a_dropdown_not_a_text_field(self, qtbot):
        """A typed "one_to_1" would fail at run time; a dropdown cannot be
        typed wrong at all."""
        from qtpy.QtWidgets import QComboBox
        from vtea_core.workflow import Step
        from vtea_napari.widgets.protocol_builder import EditStepDialog

        dialog = EditStepDialog(Step.for_function("association", "associate_objects"))
        qtbot.addWidget(dialog)

        mode = dialog.form._field_widgets["mode"]
        assert isinstance(mode, QComboBox)
        assert {mode.itemText(i) for i in range(mode.count())} == {"many_to_one", "one_to_one"}

    def test_the_segmentation_names_are_not_offered_as_form_fields(self, qtbot):
        """They come from the wiring, so a text field for them could only
        disagree with the step the input is pointed at."""
        from vtea_core.workflow import Step
        from vtea_napari.widgets.protocol_builder import EditStepDialog

        dialog = EditStepDialog(Step.for_function("association", "associate_objects"))
        qtbot.addWidget(dialog)
        assert "child_name" not in dialog.form._field_widgets
        assert "parent_name" not in dialog.form._field_widgets

    def test_the_threshold_method_is_still_a_dropdown(self, qtbot):
        """It used to be one by a hard-coded special case; it should now be
        one because its own annotation says so."""
        from qtpy.QtWidgets import QComboBox
        from vtea_core.workflow import Step
        from vtea_napari.widgets.protocol_builder import EditStepDialog

        dialog = EditStepDialog(Step.for_function("segmentation", "threshold_mask"))
        qtbot.addWidget(dialog)
        assert isinstance(dialog.form._field_widgets["method"], QComboBox)


class TestCellsFromTheBuilder:
    """Nucleus, cytoplasm, and one row per cell: the point of associating
    segmentations at all."""

    def _builder(self, qtbot):
        import numpy as np

        viewer = _model_viewer()
        volume = np.zeros((2, 20, 80))
        for index in range(4):
            left = index * 20 + 2
            # Each cell a different size and a different nuclear brightness,
            # so a per-cell table that joined the wrong rows together would
            # show it rather than looking plausible.
            volume[0, 3 : 17 - index, left : left + 16] = 200.0
            centre = left + 8
            volume[1, 8:12, centre - 2 : centre + 2] = 100.0 * (index + 1)
        viewer.add_image(volume, name="two channel")
        widget = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(widget)
        widget.channel_axis_combo.setCurrentIndex(1)
        return widget

    def _segment(self, widget, channel, name, value=50.0):
        from vtea_core.workflow import Step

        threshold = widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "threshold_mask",
                params={"method": "fixed", "value": value},
                channel=channel,
                taken_names=widget.step_names(),
            )
        )
        widget.run_single_step(threshold)
        labels = widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "label_components",
                available=set(widget.last_context),
                name=name,
                taken_names=widget.step_names(),
            )
        )
        labels.input_keys["mask"] = threshold.name
        widget.run_single_step(labels)
        return labels

    def _measure(self, widget, segmentation, name):
        from vtea_core.workflow import Step

        measure = widget.analysis_pipeline.add_step(
            Step.for_function(
                "measurements",
                "extract_measurements_by_channel",
                available=set(widget.last_context),
                name=name,
                taken_names=widget.step_names(),
            )
        )
        measure.input_keys["labels"] = segmentation
        widget.run_single_step(measure)
        return measure

    def _cells(self, widget):
        from vtea_core.workflow import Step

        cytoplasm = self._segment(widget, 0, "cytoplasm")
        nuclei = self._segment(widget, 1, "nuclei")

        associate = widget.analysis_pipeline.add_step(
            Step.for_function(
                "association",
                "associate_objects",
                available=set(widget.last_context),
                params={"method": "containment", "mode": "one_to_one"},
                taken_names=widget.step_names(),
            )
        )
        associate.input_keys["child_labels"] = cytoplasm.name
        associate.input_keys["parent_labels"] = nuclei.name
        widget.run_single_step(associate)

        cells = widget.analysis_pipeline.add_step(
            Step.for_function(
                "cells",
                "build_cells",
                available=set(widget.last_context),
                taken_names=widget.step_names(),
            )
        )
        cells.input_keys["associations"] = associate.name
        cells.input_keys["root_labels"] = nuclei.name
        widget.run_single_step(cells)
        return cytoplasm, nuclei, associate, cells

    def test_cells_is_an_analysis_category(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        stack = widget.analysis_stack
        stack.category_combo.setCurrentText("cells")
        offered = {stack.function_combo.itemText(i) for i in range(stack.function_combo.count())}
        assert {"build_cells", "cell_features"} <= offered

    def test_a_cell_is_built_for_every_nucleus(self, qtbot):
        widget = self._builder(qtbot)
        _cytoplasm, _nuclei, _associate, cells = self._cells(widget)
        assert len(widget.last_context[cells.name]) == 4

    def test_the_root_segmentation_comes_from_the_wiring(self, qtbot):
        """`root` is not a field to type: it is the step the root input is
        pointed at, so it can never disagree with what was actually read."""
        widget = self._builder(qtbot)
        _cytoplasm, nuclei, _associate, cells = self._cells(widget)
        assert widget.last_context[cells.name].root_segmentation == nuclei.name

    def test_the_log_says_how_complete_the_cells_are(self, qtbot):
        widget = self._builder(qtbot)
        self._cells(widget)
        assert "4 cells" in widget.status_label.toPlainText()

    def test_the_per_cell_table_has_a_row_for_each_cell(self, qtbot):
        from vtea_core.workflow import Step

        widget = self._builder(qtbot)
        cytoplasm, nuclei, _associate, cells = self._cells(widget)
        self._measure(widget, nuclei.name, "nucleus_measurements")
        self._measure(widget, cytoplasm.name, "cytoplasm_measurements")

        table_step = widget.analysis_pipeline.add_step(
            Step.for_function(
                "cells",
                "cell_features",
                available=set(widget.last_context) | {"measurement_tables"},
                taken_names=widget.step_names(),
            )
        )
        table_step.input_keys["cells"] = cells.name
        widget.run_single_step(table_step)

        table = widget.last_context[table_step.name]
        assert len(table) == 4
        assert list(table["cell_id"]) == [1, 2, 3, 4]

    def test_the_per_cell_table_namespaces_each_segmentation_s_features(self, qtbot):
        """The whole point: a nucleus's brightness and its cytoplasm's are
        two columns of one row rather than two rows of one column."""
        from vtea_core.workflow import Step

        widget = self._builder(qtbot)
        cytoplasm, nuclei, _associate, cells = self._cells(widget)
        self._measure(widget, nuclei.name, "nucleus_measurements")
        self._measure(widget, cytoplasm.name, "cytoplasm_measurements")

        table_step = widget.analysis_pipeline.add_step(
            Step.for_function(
                "cells",
                "cell_features",
                available=set(widget.last_context) | {"measurement_tables"},
                taken_names=widget.step_names(),
            )
        )
        table_step.input_keys["cells"] = cells.name
        widget.run_single_step(table_step)

        table = widget.last_context[table_step.name]
        assert f"{nuclei.name}.mean_ch1" in table.columns
        assert f"{cytoplasm.name}.count" in table.columns
        # Nucleus brightness rises across the four cells and cytoplasm size
        # falls, both by construction - so a table that paired a nucleus with
        # somebody else's cytoplasm would break one of these.
        brightness = list(table[f"{nuclei.name}.mean_ch1"])
        sizes = list(table[f"{cytoplasm.name}.count"])
        assert brightness == sorted(brightness)
        assert sizes == sorted(sizes, reverse=True)

    def test_the_measurement_tables_are_keyed_by_the_segmentation_measured(self, qtbot):
        widget = self._builder(qtbot)
        cytoplasm, nuclei, _associate, _cells = self._cells(widget)
        self._measure(widget, nuclei.name, "nucleus_measurements")
        self._measure(widget, cytoplasm.name, "cytoplasm_measurements")

        context = {}
        widget._seed_measurement_tables(context)
        assert set(context["measurement_tables"]) == {nuclei.name, cytoplasm.name}


class TestProbabilisticOwnership:
    """A boundary the stain never resolved: which cell owns the voxels
    between two nuclei, and how sure the answer is."""

    def _builder(self, qtbot):
        import numpy as np

        viewer = _model_viewer()
        volume = np.zeros((2, 16, 40))
        volume[0, 3:13, 2:38] = 200.0  # one connected cytoplasm
        volume[1, 7:9, 6:9] = 200.0  # two nuclei inside it
        volume[1, 7:9, 31:34] = 200.0
        viewer.add_image(volume, name="two channel")
        widget = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(widget)
        widget.channel_axis_combo.setCurrentIndex(1)
        return widget

    def _pieces(self, widget):
        from vtea_core.workflow import Step

        cytoplasm = widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "threshold_mask",
                params={"method": "fixed", "value": 50.0},
                channel=0,
                taken_names=widget.step_names(),
            )
        )
        widget.run_single_step(cytoplasm)
        nuclei_mask = widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "threshold_mask",
                params={"method": "fixed", "value": 50.0},
                channel=1,
                taken_names=widget.step_names(),
            )
        )
        widget.run_single_step(nuclei_mask)
        nuclei = widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "label_components",
                available=set(widget.last_context),
                name="nuclei",
                taken_names=widget.step_names(),
            )
        )
        nuclei.input_keys["mask"] = nuclei_mask.name
        widget.run_single_step(nuclei)
        return cytoplasm, nuclei

    def _own(self, widget, cytoplasm, nuclei, falloff=6.0):
        from vtea_core.workflow import Step

        ownership = widget.pipeline.add_step(
            Step.for_function(
                "ownership",
                "distance_ownership",
                available=set(widget.last_context) | {"spacing"},
                params={"falloff": falloff},
                taken_names=widget.step_names(),
            )
        )
        ownership.input_keys["labels"] = nuclei.name
        ownership.input_keys["mask"] = cytoplasm.name
        widget.run_single_step(ownership)
        return ownership

    def test_ownership_is_offered_in_the_processing_menu(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        stack = widget.processing_stack
        stack.category_combo.setCurrentText("ownership")
        offered = {stack.function_combo.itemText(i) for i in range(stack.function_combo.count())}
        assert "distance_ownership" in offered

    def test_it_divides_the_region_between_the_two_nuclei(self, qtbot):
        import numpy as np

        widget = self._builder(qtbot)
        cytoplasm, nuclei = self._pieces(widget)
        step = self._own(widget, cytoplasm, nuclei)

        ownership = widget.last_context[step.name]
        assert set(np.unique(ownership.hard())) == {0, 1, 2}
        assert ownership.hard()[8, 7] != ownership.hard()[8, 32]

    def test_the_midline_comes_out_as_a_close_call(self, qtbot):
        """The point of the whole phase: the voxel a watershed hands to one
        cell without comment is reported as a coin toss."""
        widget = self._builder(qtbot)
        cytoplasm, nuclei = self._pieces(widget)
        step = self._own(widget, cytoplasm, nuclei)

        ownership = widget.last_context[step.name]
        assert ownership.confidence()[8, 20] < 0.7
        assert ownership.confidence()[8, 7] > 0.95

    def test_the_ids_are_the_segmentation_it_was_built_from(self, qtbot):
        widget = self._builder(qtbot)
        cytoplasm, nuclei = self._pieces(widget)
        step = self._own(widget, cytoplasm, nuclei)
        assert widget.last_context[step.name].segmentation == nuclei.name

    def test_it_adds_both_the_answer_and_the_confidence_as_layers(self, qtbot):
        """The hard argmax on its own is indistinguishable from a watershed,
        which is the problem the confidence map exists to solve."""
        widget = self._builder(qtbot)
        cytoplasm, nuclei = self._pieces(widget)
        step = self._own(widget, cytoplasm, nuclei)

        names = [layer.name for layer in widget.viewer.layers]
        assert step.name in names
        assert f"{step.name} confidence" in names

    def test_the_log_says_how_much_was_contested(self, qtbot):
        widget = self._builder(qtbot)
        cytoplasm, nuclei = self._pieces(widget)
        self._own(widget, cytoplasm, nuclei)
        assert "contested" in widget.status_label.toPlainText()

    def test_a_weighted_measurement_step_runs_on_it(self, qtbot):
        from vtea_core.workflow import Step

        widget = self._builder(qtbot)
        cytoplasm, nuclei = self._pieces(widget)
        step = self._own(widget, cytoplasm, nuclei)

        measure = widget.analysis_pipeline.add_step(
            Step.for_function(
                "measurements",
                "weighted_measurements_by_channel",
                available=set(widget.last_context),
                taken_names=widget.step_names(),
            )
        )
        measure.input_keys["ownership"] = step.name
        widget.run_single_step(measure)

        table = widget.last_context[measure.name]
        assert list(table["object_id"]) == [1, 2]
        assert "mean_ch0" in table.columns
        # An expected volume, so the contested middle is split between them
        # rather than counted twice or given wholly to one.
        assert table["count"].sum() < (widget.last_context[cytoplasm.name] != 0).sum() + 1

    def test_the_weighted_table_knows_which_segmentation_it_measured(self, qtbot):
        """Its rows are objects of the markers the ownership was built from,
        and the ownership records that - so a per-cell table can line it up
        with the rest without guessing from the step graph."""
        from vtea_core.workflow import Step

        widget = self._builder(qtbot)
        cytoplasm, nuclei = self._pieces(widget)
        step = self._own(widget, cytoplasm, nuclei)

        measure = widget.analysis_pipeline.add_step(
            Step.for_function(
                "measurements",
                "weighted_measurements_by_channel",
                available=set(widget.last_context),
                taken_names=widget.step_names(),
            )
        )
        measure.input_keys["ownership"] = step.name
        widget.run_single_step(measure)

        context = {}
        widget._seed_measurement_tables(context)
        assert nuclei.name in context["measurement_tables"]

    def test_a_wider_falloff_leaves_more_of_the_field_contested(self, qtbot):
        widget = self._builder(qtbot)
        cytoplasm, nuclei = self._pieces(widget)

        # Each step is run before its result is looked up: running one
        # rebinds `last_context` to a new dict rather than mutating it.
        sharp_step = self._own(widget, cytoplasm, nuclei, falloff=1.0)
        sharp = widget.last_context[sharp_step.name]
        broad_step = self._own(widget, cytoplasm, nuclei, falloff=10.0)
        broad = widget.last_context[broad_step.name]
        assert sharp.contested(0.9).sum() < broad.contested(0.9).sum()
