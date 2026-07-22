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
