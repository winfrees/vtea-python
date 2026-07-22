from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QPushButton

from vtea_core.workflow import Step
from vtea_napari.widgets.step_card import StepCardWidget, summarize_params


class TestSummarizeParams:
    def test_no_params(self):
        step = Step(category="segmentation", function_name="label_components")
        assert summarize_params(step) == "(default parameters)"

    def test_with_params(self):
        step = Step(category="segmentation", function_name="threshold_mask", params={"method": "otsu"})
        assert "method=otsu" in summarize_params(step)


class TestStepCardWidget:
    def test_shows_category_and_function_name(self, qtbot):
        step = Step(category="segmentation", function_name="threshold_mask")
        card = StepCardWidget(1, step)
        qtbot.addWidget(card)
        texts = _collect_label_texts(card)
        assert any("segmentation.threshold_mask" in t for t in texts)

    def test_shows_position_number(self, qtbot):
        step = Step(category="segmentation", function_name="threshold_mask")
        card = StepCardWidget(3, step)
        qtbot.addWidget(card)
        texts = _collect_label_texts(card)
        assert any(t == "3." for t in texts)

    def test_edit_button_emits_signal(self, qtbot):
        step = Step(category="segmentation", function_name="threshold_mask")
        card = StepCardWidget(1, step)
        qtbot.addWidget(card)
        with qtbot.waitSignal(card.edit_requested, timeout=1000):
            _click_button(qtbot, card, "Edit")

    def test_delete_button_emits_signal(self, qtbot):
        step = Step(category="segmentation", function_name="threshold_mask")
        card = StepCardWidget(1, step)
        qtbot.addWidget(card)
        with qtbot.waitSignal(card.delete_requested, timeout=1000):
            _click_button(qtbot, card, "Delete")


def _collect_label_texts(widget):
    return [child.text() for child in widget.findChildren(QLabel)]


def _click_button(qtbot, widget, text):
    for button in widget.findChildren(QPushButton):
        if button.text() == text:
            qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
            return
    raise AssertionError(f"no button with text {text!r} found")
