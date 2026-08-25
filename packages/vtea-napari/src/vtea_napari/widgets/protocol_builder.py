"""The protocol builder dock widget: Option A, a fully-functional GUI clone
of vtea.protocol's step-stack pipeline builder (see PORT_PLAN.md).

Owns a vtea_core.workflow.Pipeline and renders its steps as an ordered
stack of cards (StepCardWidget), matching ProtocolManagerMulti/
blockstepgui's actual layout (a plain top-to-bottom stack built by adding
steps from a category menu - not a node-graph editor, see PORT_PLAN.md's
"Protocol builder: Option A" section for why). Steps are added via category
+ function pickers, edited via a ParameterForm dialog, and removed with a
button on each card - the same operations the Java UI exposed, execution
handled by the shared Pipeline engine either here or from a script/notebook.

run_pipeline() also threads each step's last-run output back onto its card
as a thumbnail (StepCardWidget.set_thumbnail) - the Java UI had per-step
previews and this first pass initially didn't; skip it when there's no
`napari_viewer` (headless/script use) rather than requiring one.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from vtea_core.workflow import STEP_REGISTRY, Pipeline, Step

from vtea_napari.widgets.param_form import ParameterForm
from vtea_napari.widgets.step_card import StepCardWidget

ALL_CHANNELS = "All channels"


class EditStepDialog(QDialog):
    """A modal dialog wrapping a ParameterForm, pre-filled with the step's
    current params, plus the channel this step should work on."""

    def __init__(self, step: Step, parent=None, n_channels: int | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit {step.category}.{step.function_name}")
        self.step = step

        layout = QVBoxLayout(self)

        # Channel first: which channel a step runs on is usually the most
        # consequential choice for a multi-channel acquisition.
        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItem(ALL_CHANNELS, None)
        for index in range(n_channels or 0):
            self.channel_combo.addItem(f"Channel {index}", index)
        if step.channel is not None:
            position = self.channel_combo.findData(step.channel)
            if position == -1:
                # The step remembers a channel the current image doesn't
                # have (different file loaded since). Keep it visible rather
                # than silently resetting it to "All channels".
                self.channel_combo.addItem(f"Channel {step.channel} (not in image)", step.channel)
                position = self.channel_combo.count() - 1
            self.channel_combo.setCurrentIndex(position)
        channel_row.addWidget(self.channel_combo)
        channel_row.addStretch()
        layout.addLayout(channel_row)

        self.form = ParameterForm(step.category, step.function_name)
        self.form.set_values(step.params)
        layout.addWidget(self.form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def updated_params(self) -> dict:
        return self.form.get_values()

    def updated_channel(self) -> int | None:
        return self.channel_combo.currentData()


class ProtocolBuilderWidget(QWidget):
    """The napari dock widget: a Pipeline plus the UI to build it.
    `napari_viewer` is auto-injected by napari's plugin engine when opened
    from the Plugins menu; pass None for standalone/script/test use (the
    "Run pipeline" button is only shown when a viewer is available, since
    it needs a layer to pull the initial volume from)."""

    def __init__(self, pipeline: Pipeline | None = None, napari_viewer=None, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline if pipeline is not None else Pipeline()
        self.viewer = napari_viewer
        self.last_context: dict = {}

        root = QVBoxLayout(self)

        add_row = QHBoxLayout()
        self.category_combo = QComboBox()
        self.category_combo.addItems(sorted(STEP_REGISTRY))
        self.category_combo.currentTextChanged.connect(self._refresh_function_choices)
        self.function_combo = QComboBox()
        add_button = QPushButton("Add Step")
        add_button.clicked.connect(self._add_step_from_selection)
        add_row.addWidget(QLabel("Category:"))
        add_row.addWidget(self.category_combo)
        add_row.addWidget(QLabel("Step:"))
        add_row.addWidget(self.function_combo)
        add_row.addWidget(add_button)
        if self.viewer is not None:
            run_button = QPushButton("Run pipeline")
            run_button.clicked.connect(self._run_pipeline_from_active_layer)
            add_row.addWidget(run_button)
        root.addLayout(add_row)

        # Which axis holds channels is a property of the loaded image, so
        # it's set once here; which channel each step uses is per-step (in
        # that step's Edit dialog).
        channel_axis_row = QHBoxLayout()
        channel_axis_row.addWidget(QLabel("Channel axis:"))
        self.channel_axis_combo = QComboBox()
        self.channel_axis_combo.currentIndexChanged.connect(self._on_channel_axis_changed)
        channel_axis_row.addWidget(self.channel_axis_combo)
        channel_axis_row.addStretch()
        root.addLayout(channel_axis_row)
        self._refresh_channel_axis_choices()

        self._steps_container = QWidget()
        self._steps_layout = QVBoxLayout(self._steps_container)
        self._steps_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._steps_container)
        root.addWidget(scroll)

        self._refresh_function_choices(self.category_combo.currentText())
        self.refresh_steps()

    def active_image(self) -> np.ndarray | None:
        """The selected layer's data, or None when there's nothing to run on."""
        if self.viewer is None:
            return None
        layer = self.viewer.layers.selection.active
        if layer is None:
            return None
        return np.asarray(layer.data)

    def n_channels(self) -> int | None:
        """How many channels the active image has along the chosen channel
        axis, or None if no channel axis is set."""
        axis = self.pipeline.channel_axis
        image = self.active_image()
        if axis is None or image is None or axis >= image.ndim:
            return None
        return image.shape[axis]

    def run_pipeline(self, context: dict) -> dict:
        """Runs the pipeline against `context` (e.g. {"volume": array}),
        keeps the result to drive step-card thumbnails, and returns it."""
        self.last_context = self.pipeline.run(context)
        self.refresh_steps()
        return self.last_context

    def _run_pipeline_from_active_layer(self) -> None:
        image = self.active_image()
        if image is None:
            return
        # "intensity" is the untouched original, kept separate from "volume"
        # so that a preprocessing step (which writes back to "volume") does
        # not change what measurement steps read intensities from.
        self.run_pipeline({"volume": image, "intensity": image})

    def _refresh_channel_axis_choices(self) -> None:
        image = self.active_image()
        self.channel_axis_combo.blockSignals(True)
        self.channel_axis_combo.clear()
        self.channel_axis_combo.addItem("None (not multi-channel)", None)
        if image is not None:
            for axis, size in enumerate(image.shape):
                self.channel_axis_combo.addItem(f"axis {axis} (size {size})", axis)
        position = self.channel_axis_combo.findData(self.pipeline.channel_axis)
        self.channel_axis_combo.setCurrentIndex(max(position, 0))
        self.channel_axis_combo.blockSignals(False)

    def _on_channel_axis_changed(self, _index: int) -> None:
        self.pipeline.channel_axis = self.channel_axis_combo.currentData()
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
        # Step.for_function derives input_keys/output_key from the
        # function's declared I/O. Building a bare Step(...) here instead is
        # what made every GUI-built pipeline fail on Run with
        # "missing 1 required positional argument" - nothing passed the data.
        available = self.pipeline.available_keys({"volume", "intensity"})
        self.pipeline.add_step(
            Step.for_function(category, function_name, available=available)
        )
        self.refresh_steps()

    def refresh_steps(self) -> None:
        # The active layer can change between refreshes, so keep the axis
        # choices in step with whatever image is currently selected.
        self._refresh_channel_axis_choices()

        while self._steps_layout.count() > 1:
            item = self._steps_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for position, step in enumerate(self.pipeline, start=1):
            thumbnail = self.last_context.get(step.output_key)
            card = StepCardWidget(position, step, thumbnail=thumbnail)
            card.edit_requested.connect(lambda s=step: self._edit_step(s))
            card.delete_requested.connect(lambda s=step: self._delete_step(s))
            self._steps_layout.insertWidget(self._steps_layout.count() - 1, card)

    def _edit_step(self, step: Step) -> None:
        dialog = EditStepDialog(step, parent=self, n_channels=self.n_channels())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            step.params = dialog.updated_params()
            step.channel = dialog.updated_channel()
            self.refresh_steps()

    def _delete_step(self, step: Step) -> None:
        index = self.pipeline.steps.index(step)
        self.pipeline.remove_step(index)
        self.refresh_steps()
