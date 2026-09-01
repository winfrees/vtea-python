import numpy as np
import pandas as pd
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


class TestThumbnail:
    def test_no_thumbnail_by_default(self, qtbot):
        step = Step(category="segmentation", function_name="threshold_mask")
        card = StepCardWidget(1, step)
        qtbot.addWidget(card)
        assert card.thumbnail_label.pixmap() is None or card.thumbnail_label.pixmap().isNull()

    def test_2d_array_thumbnail_is_shown(self, qtbot):
        step = Step(category="segmentation", function_name="threshold_mask")
        card = StepCardWidget(1, step, thumbnail=np.zeros((10, 10)))
        qtbot.addWidget(card)
        assert not card.thumbnail_label.pixmap().isNull()

    def test_boolean_mask_thumbnail_is_shown(self, qtbot):
        step = Step(category="segmentation", function_name="threshold_mask")
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 2:5] = True
        card = StepCardWidget(1, step, thumbnail=mask)
        qtbot.addWidget(card)
        assert not card.thumbnail_label.pixmap().isNull()

    def test_3d_volume_thumbnail_is_shown(self, qtbot):
        step = Step(category="segmentation", function_name="threshold_mask")
        card = StepCardWidget(1, step, thumbnail=np.zeros((3, 10, 10)))
        qtbot.addWidget(card)
        assert not card.thumbnail_label.pixmap().isNull()

    def test_non_array_output_shows_no_thumbnail(self, qtbot):
        step = Step(category="measurements", function_name="extract_measurements")
        card = StepCardWidget(1, step, thumbnail=pd.DataFrame({"a": [1, 2]}))
        qtbot.addWidget(card)
        assert card.thumbnail_label.pixmap() is None or card.thumbnail_label.pixmap().isNull()

    def test_set_thumbnail_updates_an_existing_card(self, qtbot):
        step = Step(category="segmentation", function_name="threshold_mask")
        card = StepCardWidget(1, step)
        qtbot.addWidget(card)
        card.set_thumbnail(np.zeros((10, 10)))
        assert not card.thumbnail_label.pixmap().isNull()
        card.set_thumbnail(None)
        assert card.thumbnail_label.pixmap() is None or card.thumbnail_label.pixmap().isNull()


class TestProgressBar:
    """Item 2: every step card carries a progress bar, and it stays out of
    the way - no wider than the three buttons together, no taller than one
    of them."""

    def _card(self, qtbot, category="segmentation", function_name="threshold_mask"):
        card = StepCardWidget(1, Step(category=category, function_name=function_name))
        qtbot.addWidget(card)
        card.show()
        return card

    def test_every_card_has_one(self, qtbot):
        assert self._card(qtbot).progress_bar is not None

    def test_it_is_hidden_until_the_step_runs(self, qtbot):
        card = self._card(qtbot)
        assert not card.progress_bar.isVisible()

    def test_it_is_no_wider_than_the_three_buttons_together(self, qtbot):
        card = self._card(qtbot)
        buttons = (card.run_button, card.edit_button, card.delete_button)
        combined = sum(button.sizeHint().width() for button in buttons) + 4
        assert card.progress_bar.maximumWidth() <= combined

    def test_it_is_no_taller_than_a_button(self, qtbot):
        card = self._card(qtbot)
        assert card.progress_bar.height() <= card.run_button.sizeHint().height()

    def test_a_step_with_an_estimate_gets_a_measured_bar(self, qtbot):
        card = self._card(qtbot)
        card.begin_progress(12.0)
        assert card.progress_bar.isVisible()
        assert card.progress_bar.maximum() > 0
        assert "12" in card.progress_bar.toolTip()

    def test_it_advances_with_the_fraction(self, qtbot):
        card = self._card(qtbot)
        card.begin_progress(10.0)
        card.set_progress(0.5)
        assert card.progress_bar.value() == card.progress_bar.maximum() // 2

    def test_it_says_how_long_is_left(self, qtbot):
        card = self._card(qtbot)
        card.begin_progress(10.0)
        card.set_progress(0.5, remaining=5.0)
        assert "5 s" in card.progress_bar.toolTip()

    def test_a_step_with_no_estimate_gets_a_continuous_bar(self, qtbot):
        """t-SNE's runtime depends on how its optimisation converges. A
        fraction for it would have to be invented, so it gets Qt's busy
        indicator instead - minimum and maximum both zero."""
        card = self._card(qtbot, "reduction", "tsne")
        card.begin_progress(None)
        assert (card.progress_bar.minimum(), card.progress_bar.maximum()) == (0, 0)

    def test_a_continuous_bar_ignores_a_fraction(self, qtbot):
        card = self._card(qtbot, "reduction", "tsne")
        card.begin_progress(None)
        card.set_progress(0.5)
        assert card.progress_bar.maximum() == 0

    def test_a_step_that_learns_its_fraction_switches_to_a_measured_bar(self, qtbot):
        """A tiled run counts tiles, which beats any estimate from a clock."""
        card = self._card(qtbot)
        card.begin_progress(None)
        card.set_determinate()
        card.set_progress(0.25)
        assert card.progress_bar.value() == card.progress_bar.maximum() // 4

    def test_it_goes_away_when_the_step_finishes(self, qtbot):
        card = self._card(qtbot)
        card.begin_progress(5.0)
        card.end_progress()
        assert not card.progress_bar.isVisible()
        assert card.progress_bar.value() == 0


def _collect_label_texts(widget):
    return [child.text() for child in widget.findChildren(QLabel)]


def _click_button(qtbot, widget, text):
    for button in widget.findChildren(QPushButton):
        if button.text() == text:
            qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
            return
    raise AssertionError(f"no button with text {text!r} found")
