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
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from vtea_core.workflow import Pipeline, Step

from vtea_napari.widgets.param_form import ParameterForm
from vtea_napari.widgets.plot import ScatterPlotWidget
from vtea_napari.widgets.step_stack import StepStackWidget

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
    """The napari dock widget: two step stacks plus the results plot.

    Split into a processing pane (image processing, segmentation - the steps
    that turn an image into labels) and an analysis pane (measurements,
    clustering, reduction, gates, classification - the steps that turn
    labels into per-object numbers). They run as two Pipelines threading one
    shared context, and sit in a vertical splitter so each gets half the
    height by default and can be re-dragged.

    The analysis pane also plots the result: one point per segmented object,
    with the axes chosen from whatever measurements were computed.

    `napari_viewer` is auto-injected by napari's plugin engine when opened
    from the Plugins menu; pass None for standalone/script/test use (the
    "Run" button needs a layer to pull the initial volume from).
    """

    PROCESSING_CATEGORIES = ("imageprocessing", "segmentation")
    ANALYSIS_CATEGORIES = ("measurements", "clustering", "reduction", "gates", "classification")

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        napari_viewer=None,
        analysis_pipeline: Pipeline | None = None,
        parent=None,
    ):
        super().__init__(parent)
        # `pipeline` stays the processing pipeline so existing callers and
        # scripts keep working.
        self.pipeline = pipeline if pipeline is not None else Pipeline()
        self.analysis_pipeline = analysis_pipeline if analysis_pipeline is not None else Pipeline()
        self.viewer = napari_viewer
        self.last_context: dict = {}

        root = QVBoxLayout(self)

        top_row = QHBoxLayout()
        if self.viewer is not None:
            run_button = QPushButton("Run pipeline")
            run_button.clicked.connect(self._run_pipeline_from_active_layer)
            top_row.addWidget(run_button)
        top_row.addWidget(QLabel("Channel axis:"))
        self.channel_axis_combo = QComboBox()
        self.channel_axis_combo.currentIndexChanged.connect(self._on_channel_axis_changed)
        top_row.addWidget(self.channel_axis_combo)
        top_row.addStretch()
        root.addLayout(top_row)
        self._refresh_channel_axis_choices()

        self.processing_stack = StepStackWidget(
            self.PROCESSING_CATEGORIES,
            self.pipeline,
            title="Processing",
            seed_keys={"volume", "intensity"},
            n_channels_provider=lambda: self.n_channels(),
            results_provider=lambda: self.last_context,
        )
        self.processing_stack.show_result_requested.connect(self.show_step_result)

        self.analysis_stack = StepStackWidget(
            self.ANALYSIS_CATEGORIES,
            self.analysis_pipeline,
            title="Analysis",
            # Analysis steps consume what processing produced.
            seed_keys={"volume", "intensity", "labels", "mask"},
            n_channels_provider=lambda: self.n_channels(),
            results_provider=lambda: self.last_context,
        )
        self.analysis_stack.show_result_requested.connect(self.show_step_result)

        analysis_pane = QWidget()
        analysis_layout = QVBoxLayout(analysis_pane)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.addWidget(self.analysis_stack)
        self.plot = ScatterPlotWidget()
        analysis_layout.addWidget(self.plot, 1)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.processing_stack)
        self.splitter.addWidget(analysis_pane)
        self.splitter.setChildrenCollapsible(False)
        # Equal stretch: each pane takes half the dock's height rather than
        # the steps list being squeezed to a single visible card.
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        root.addWidget(self.splitter, 1)

        self.status_label = QLabel("")
        root.addWidget(self.status_label)

    # -- data -------------------------------------------------------------

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

    # -- running ----------------------------------------------------------

    def run_pipeline(self, context: dict) -> dict:
        """Runs processing then analysis over one shared context, keeps the
        result for step thumbnails/Show buttons, and updates the plot."""
        result = self.pipeline.run(context)
        self.analysis_pipeline.channel_axis = self.pipeline.channel_axis
        result = self.analysis_pipeline.run(result)
        self.last_context = result
        self.refresh_steps()
        self._refresh_plot()
        return result

    def _run_pipeline_from_active_layer(self) -> None:
        image = self.active_image()
        if image is None:
            self.status_label.setText("Select an image layer to run on.")
            return
        # "intensity" is the untouched original, kept separate from "volume"
        # so a preprocessing step (which writes back to "volume") doesn't
        # change what measurement steps read intensities from.
        try:
            self.run_pipeline({"volume": image, "intensity": image})
        except Exception as exc:  # noqa: BLE001 - surface it in the UI, don't crash napari
            self.status_label.setText(f"{type(exc).__name__}: {exc}")
            return
        self.status_label.setText("Pipeline finished.")

    def refresh_steps(self) -> None:
        self._refresh_channel_axis_choices()
        self.processing_stack.refresh_steps()
        self.analysis_stack.refresh_steps()

    # -- results ----------------------------------------------------------

    def show_step_result(self, step: Step) -> None:
        """Add one step's result to the viewer as a layer."""
        if self.viewer is None:
            return
        result = self.last_context.get(step.output_key)
        if not isinstance(result, np.ndarray) or result.ndim < 2:
            self.status_label.setText(f"{step.output_key}: not an image, nothing to show")
            return
        name = f"{step.function_name} ({step.output_key})"
        for existing in list(self.viewer.layers):
            if existing.name == name:
                self.viewer.layers.remove(existing)
        # Integer arrays are object labels; booleans are masks, which napari
        # also renders best as labels. Anything else is intensity.
        if result.dtype == bool:
            self.viewer.add_labels(result.astype(np.uint8), name=name)
        elif np.issubdtype(result.dtype, np.integer):
            self.viewer.add_labels(result.astype(np.int32), name=name)
        else:
            self.viewer.add_image(result, name=name)
        self.status_label.setText(f"Added layer '{name}'")

    def results_table(self) -> pd.DataFrame | None:
        """The per-object measurement table, with any per-object analysis
        outputs (cluster ids, reduced dimensions) joined on as extra columns
        so they can be used as plot axes or colours."""
        frame = self.last_context.get("measurements")
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None
        frame = frame.copy()
        for key, value in self.last_context.items():
            if key == "measurements" or not isinstance(value, np.ndarray):
                continue
            if value.ndim == 1 and len(value) == len(frame):
                frame[key] = value
            elif value.ndim == 2 and value.shape[0] == len(frame) and value.shape[1] <= 8:
                for column in range(value.shape[1]):
                    frame[f"{key}{column + 1}"] = value[:, column]
        return frame

    def _refresh_plot(self) -> None:
        frame = self.results_table()
        if frame is None:
            return
        self.plot.set_data(frame)

    # -- channel axis -----------------------------------------------------

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
        self.analysis_pipeline.channel_axis = self.pipeline.channel_axis
        self.refresh_steps()
