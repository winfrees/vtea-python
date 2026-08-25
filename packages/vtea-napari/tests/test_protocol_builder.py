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

        widget.category_combo.setCurrentText("segmentation")
        widget.function_combo.setCurrentText("threshold_mask")
        _click_button(qtbot, widget, "Add Step")

        assert len(widget.pipeline) == 1
        assert widget.pipeline.steps[0].category == "segmentation"
        assert widget.pipeline.steps[0].function_name == "threshold_mask"

    def test_add_step_creates_a_card(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        widget.category_combo.setCurrentText("clustering")
        widget.function_combo.setCurrentText("kmeans")
        _click_button(qtbot, widget, "Add Step")

        from vtea_napari.widgets.step_card import StepCardWidget

        cards = widget.findChildren(StepCardWidget)
        assert len(cards) == 1

    def test_function_choices_follow_category(self, qtbot):
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        widget.category_combo.setCurrentText("clustering")
        clustering_choices = {widget.function_combo.itemText(i) for i in range(widget.function_combo.count())}
        assert "kmeans" in clustering_choices
        assert "threshold_mask" not in clustering_choices

        widget.category_combo.setCurrentText("segmentation")
        segmentation_choices = {widget.function_combo.itemText(i) for i in range(widget.function_combo.count())}
        assert "threshold_mask" in segmentation_choices


class TestDeleteStep:
    def test_delete_removes_from_pipeline_and_ui(self, qtbot):
        from vtea_core.workflow import Step

        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)
        widget.pipeline.add_step(Step(category="segmentation", function_name="threshold_mask"))
        widget.refresh_steps()

        _click_button(qtbot, widget, "Delete")

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

        widget._delete_step(first)

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

        widget._edit_step(step)

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

        widget._edit_step(step)

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

        widget.category_combo.setCurrentText("segmentation")
        widget.function_combo.setCurrentText("threshold_mask")
        _click_button(qtbot, widget, "Add Step")

        widget.category_combo.setCurrentText("segmentation")
        widget.function_combo.setCurrentText("label_components")
        _click_button(qtbot, widget, "Add Step")

        threshold_step, label_step = widget.pipeline.steps
        threshold_step.input_keys = {"volume": "volume"}
        threshold_step.output_key = "mask"
        label_step.input_keys = {"mask": "mask"}
        label_step.output_key = "labels"

        def fake_exec(self):
            self.form.set_values({"method": "fixed", "value": "50"})
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(EditStepDialog, "exec", fake_exec)
        widget._edit_step(threshold_step)
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

        widget.category_combo.setCurrentText("segmentation")
        widget.function_combo.setCurrentText("threshold_mask")
        _click_button(qtbot, widget, "Add Step")

        step = widget.pipeline.steps[0]
        assert step.input_keys == {"volume": "volume"}
        assert step.output_key == "mask"

    def test_cellpose_step_added_from_the_menu_receives_its_volume(self, qtbot):
        """The exact step the bug was reported against. Cellpose itself needs
        the deeplearning extra, so this checks the wiring reaches the
        function rather than running the model."""
        widget = ProtocolBuilderWidget()
        qtbot.addWidget(widget)

        widget.category_combo.setCurrentText("segmentation")
        widget.function_combo.setCurrentText("cellpose_segmentation")
        _click_button(qtbot, widget, "Add Step")

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

        widget.category_combo.setCurrentText("segmentation")
        widget.function_combo.setCurrentText("threshold_mask")
        _click_button(qtbot, widget, "Add Step")
        widget.function_combo.setCurrentText("label_components")
        _click_button(qtbot, widget, "Add Step")

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
        widget._edit_step(step)

        assert step.channel == 2

    def test_end_to_end_channel_selection_through_the_gui(self, qtbot):
        """A 4-channel image, signal only in channel 2 - the pipeline should
        find it when that channel is selected and nothing otherwise."""
        viewer, _ = self._viewer_with_multichannel_image(qtbot)
        try:
            widget = ProtocolBuilderWidget(napari_viewer=viewer)
            qtbot.addWidget(widget)
            widget.channel_axis_combo.setCurrentIndex(widget.channel_axis_combo.findData(1))

            widget.category_combo.setCurrentText("segmentation")
            widget.function_combo.setCurrentText("threshold_mask")
            _click_button(qtbot, widget, "Add Step")
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
