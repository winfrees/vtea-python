"""One scrollable stack of pipeline steps, with its own Add-step pickers.

Factored out of ProtocolBuilderWidget so the builder can show two of them:
image processing/segmentation on top, and the per-object analysis steps
(measurements, clustering, reduction, gates, classification) below, each
editing its own Pipeline. That split matches how the work actually runs -
processing produces a label image, analysis consumes it - and keeps a long
segmentation protocol from pushing the analysis steps out of view.
"""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from vtea_core.workflow import STEP_REGISTRY, Pipeline, Step

from vtea_napari.widgets.step_card import StepCardWidget

# A stack showing only one card at a time is unusable, which is what a
# bare QScrollArea inside a narrow dock gives you. This floors each pane at
# roughly three cards; the splitter in ProtocolBuilderWidget then divides
# the real estate evenly and lets the user re-drag it.
MINIMUM_STACK_HEIGHT = 240


class StepStackWidget(QWidget):
    """Add/edit/delete/show for the steps of one Pipeline, limited to
    `categories` of the step registry."""

    steps_changed = Signal()
    run_step_requested = Signal(object)  # vtea_core.workflow.Step

    def __init__(
        self,
        categories: tuple[str, ...],
        pipeline: Pipeline,
        *,
        title: str = "",
        seed_keys: set[str] | None = None,
        n_channels_provider: Callable[[], int | None] | None = None,
        results_provider: Callable[[], dict] | None = None,
        default_channel_provider: Callable[[], int | None] | None = None,
        action_text: str = "",
        action_style: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.pipeline = pipeline
        self.categories = tuple(c for c in categories if c in STEP_REGISTRY)
        self.seed_keys = set(seed_keys or {"volume", "intensity"})
        self._n_channels_provider = n_channels_provider or (lambda: None)
        self._results_provider = results_provider or dict
        self._default_channel_provider = default_channel_provider or (lambda: None)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.action_button: QPushButton | None = None
        if title or action_text:
            title_row = QHBoxLayout()
            if title:
                heading = QLabel(title)
                heading.setStyleSheet("font-weight: bold;")
                title_row.addWidget(heading)
            if action_text:
                # The pane's own run button sits with its heading, so it is
                # obvious which set of steps it applies to.
                self.action_button = QPushButton(action_text)
                if action_style:
                    self.action_button.setStyleSheet(action_style)
                title_row.addWidget(self.action_button)
            title_row.addStretch()
            root.addLayout(title_row)

        add_row = QHBoxLayout()
        self.category_combo = QComboBox()
        self.category_combo.addItems(sorted(self.categories))
        self.category_combo.currentTextChanged.connect(self._refresh_function_choices)
        self.function_combo = QComboBox()
        add_button = QPushButton("Add Step")
        add_button.clicked.connect(self._add_step_from_selection)
        add_row.addWidget(QLabel("Category:"))
        add_row.addWidget(self.category_combo)
        add_row.addWidget(QLabel("Step:"))
        add_row.addWidget(self.function_combo, 1)
        add_row.addWidget(add_button)
        root.addLayout(add_row)

        self._steps_container = QWidget()
        self._steps_layout = QVBoxLayout(self._steps_container)
        self._steps_layout.addStretch()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self._steps_container)
        self.scroll.setMinimumHeight(MINIMUM_STACK_HEIGHT)
        root.addWidget(self.scroll, 1)

        self._refresh_function_choices(self.category_combo.currentText())
        self.refresh_steps()

    def _refresh_function_choices(self, category: str) -> None:
        self.function_combo.clear()
        if category:
            self.function_combo.addItems(sorted(STEP_REGISTRY[category]))

    def _add_step_from_selection(self) -> None:
        category = self.category_combo.currentText()
        function_name = self.function_combo.currentText()
        if not category or not function_name:
            return
        available = self.pipeline.available_keys(self.seed_keys)
        # Inherit the channel already in use: picking channel 2 for
        # segmentation and leaving a later step on "all channels" fed
        # mismatched shapes into it and aborted the run.
        self.pipeline.add_step(
            Step.for_function(
                category,
                function_name,
                available=available,
                channel=self._default_channel_provider(),
            )
        )
        self.refresh_steps()
        self.steps_changed.emit()

    def refresh_steps(self) -> None:
        while self._steps_layout.count() > 1:
            item = self._steps_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        results = self._results_provider()
        for position, step in enumerate(self.pipeline, start=1):
            result = results.get(step.output_key)
            card = StepCardWidget(position, step, thumbnail=result)
            card.edit_requested.connect(lambda s=step: self._edit_step(s))
            card.delete_requested.connect(lambda s=step: self._delete_step(s))
            card.run_requested.connect(lambda s=step: self.run_step_requested.emit(s))
            self._steps_layout.insertWidget(self._steps_layout.count() - 1, card)

    def _edit_step(self, step: Step) -> None:
        from vtea_napari.widgets.protocol_builder import EditStepDialog

        dialog = EditStepDialog(step, parent=self, n_channels=self._n_channels_provider())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            step.params = dialog.updated_params()
            step.channel = dialog.updated_channel()
            self.refresh_steps()
            self.steps_changed.emit()

    def _delete_step(self, step: Step) -> None:
        self.pipeline.remove_step(self.pipeline.steps.index(step))
        self.refresh_steps()
        self.steps_changed.emit()
