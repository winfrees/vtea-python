from qtpy.QtCore import Qt
from qtpy.QtWidgets import QPushButton

from vtea_napari.widgets.protocol_builder import ProtocolBuilderWidget


def _click_button(qtbot, widget, text):
    for button in widget.findChildren(QPushButton):
        if button.text() == text:
            qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
            return
    raise AssertionError(f"no button with text {text!r} found")


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

        assert processing == {"imageprocessing", "segmentation"}
        assert analysis == {"measurements", "clustering", "reduction", "gates", "classification"}
        assert processing.isdisjoint(analysis)


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
        assert "Run pipeline" not in buttons

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

            _click_button(qtbot, widget, "Run pipeline")

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

    def test_show_button_is_disabled_until_the_step_has_run(self, qtbot):
        from vtea_core.workflow import Step
        from vtea_napari.widgets.step_card import StepCardWidget

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        widget.pipeline.add_step(Step.for_function("segmentation", "threshold_mask"))
        widget.refresh_steps()

        card = widget.processing_stack.findChildren(StepCardWidget)[0]
        assert not card.show_button.isEnabled()

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
        # Axes offered are the measured values.
        axes = {widget.plot.x_combo.itemText(i) for i in range(widget.plot.x_combo.count())}
        assert {"mean", "count", "object_id"} <= axes

    def test_per_object_analysis_output_becomes_a_plot_column(self, qtbot):
        """Cluster ids and similar per-object results should be selectable as
        axes/colours alongside the raw measurements."""
        import numpy as np

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        self._measured(widget)
        widget.last_context["clusters"] = np.array([0, 1])

        frame = widget.results_table()
        assert "clusters" in frame.columns

    def test_no_measurements_leaves_the_plot_empty_without_error(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        assert widget.results_table() is None
        widget._refresh_plot()  # must not raise


class TestPaneSizing:
    def test_both_panes_share_the_height(self, qtbot):
        """The steps list used to be squeezed to about one visible card."""
        from vtea_napari.widgets.step_stack import MINIMUM_STACK_HEIGHT

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        assert widget.splitter.count() == 2
        assert widget.processing_stack.scroll.minimumHeight() >= MINIMUM_STACK_HEIGHT
        assert not widget.splitter.childrenCollapsible()

        # The real requirement: at a realistic dock height, neither pane is
        # squeezed to a sliver - each should get a substantial share.
        widget.resize(500, 900)
        widget.show()
        qtbot.waitExposed(widget)
        sizes = widget.splitter.sizes()
        assert len(sizes) == 2
        total = sum(sizes)
        assert total > 0
        assert min(sizes) / total > 0.3, f"panes split unevenly: {sizes}"


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
