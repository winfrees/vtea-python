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

Each step's last-run output is threaded back onto its card as a thumbnail
(StepCardWidget.set_thumbnail) - the Java UI had per-step previews and this
first pass initially didn't; skip it when there's no `napari_viewer`
(headless/script use) rather than requiring one.

Results are published to a shared vtea_napari.session.AnalysisSession rather
than displayed here: the Object Explorer is the pane that plots and gates
them. That split keeps this dock about building and running a protocol, and
means closing either pane loses nothing.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from vtea_core.blocked import format_bytes
from vtea_core.measurements import FeatureCatalog, feature_matrix
from vtea_core.objects import AssociationSet, CellCollection, Ownership
from vtea_core.workflow import Calibration, Pipeline, Step, estimate_seconds, format_duration
from vtea_core.workflow import rename_segmentation as rename_measured_segmentation
from vtea_core.workflow import sync_measurement_steps as sync_measurements

from vtea_napari.session import AnalysisSession, TableView, session_for
from vtea_napari.widgets.feature_select import FeatureSelectWidget
from vtea_napari.widgets.log_view import LogView
from vtea_napari.widgets.memory_control import MemoryControl
from vtea_napari.widgets.param_form import ParameterForm
from vtea_napari.widgets.roi_preview import PREVIEW_PREFIX, PreviewControl, visible_region
from vtea_napari.widgets.run_control import ProgressRelay, RunControl
from vtea_napari.widgets.spacing_control import SpacingControl
from vtea_napari.widgets.step_stack import StepStackWidget

ALL_CHANNELS = "All channels"

# The dock was taking more screen than its contents warranted; scale text
# to 75% of the application font and tighten the surrounding padding.
COMPACT_FONT_SCALE = 0.75

RUN_BUTTON_STYLE = (
    "QPushButton { background-color: #f0a500; color: #202020; font-weight: bold; "
    "border: 1px solid #c78500; border-radius: 3px; padding: 2px 8px; }"
    "QPushButton:hover { background-color: #ffc233; }"
    "QPushButton:pressed { background-color: #d99000; }"
)

# A 2D per-object result wider than this is a crop stack or a distance
# matrix, not a handful of features to plot against each other.
MAX_DERIVED_FEATURES = 8

# What a preview block is assumed to cost per voxel: the input plus a couple
# of intermediates, which is what a short protocol actually holds. Only used
# to refuse a view too large to preview, so an approximation that errs
# towards refusing is the right kind of wrong.
PREVIEW_BYTES_PER_VOXEL = 24

# A dock this wide already shows every control; past that it is just taking
# screen away from the image canvas, which is the thing being analysed.
MAX_WIDTH_SCREEN_FRACTION = 0.30


def feature_columns(name: str, result, n_objects: int) -> dict[str, np.ndarray]:
    """The columns a step's result contributes to the measurement table.

    A per-object vector (cluster ids) becomes one column named after the
    step; a per-object matrix (PCA/t-SNE coordinates) becomes one column per
    component, `name_1`, `name_2`, ... Anything that isn't one row per
    object contributes nothing. Naming after the step is what keeps a second
    reduction from overwriting the first one's columns, and is what makes
    those columns findable in the plot's X/Y menus.
    """
    if not isinstance(result, np.ndarray) or result.shape[:1] != (n_objects,):
        return {}
    if result.ndim == 1:
        return {name: result}
    if result.ndim == 2 and result.shape[1] <= MAX_DERIVED_FEATURES:
        return {f"{name}_{index + 1}": result[:, index] for index in range(result.shape[1])}
    return {}


# How far a determinate progress bar may run before its step actually
# finishes. A bar that sat at 100% while the step ran on would be a lie
# told at exactly the moment the user is deciding whether it has hung.
MAX_ESTIMATED_FRACTION = 0.99


class StepProgressDisplay:
    """Drives whichever step card is currently running.

    Every method here runs on the GUI thread - it is called from
    RunControl's pump loop, which is the one place during a run that is
    both regular and safely on the thread that owns the widgets. The
    running step's *name* comes from the worker thread through a
    ProgressRelay; nothing else crosses.

    A step with an estimate gets a bar that advances with the clock and a
    tooltip counting down; one without gets Qt's continuous bar. The
    advancing bar stops just short of full: it is an estimate, and the step
    finishing is what completes it.
    """

    def __init__(self, card_for):
        self._card_for = card_for
        self._card = None
        self._name = ""
        self._estimate: float | None = None
        self._started_at = 0.0
        self._determinate = False

    def show(self, name: str, estimate: float | None = None) -> None:
        """Report that the step called `name` is the one running now."""
        if name == self._name:
            self._advance()
            return
        self.finish()
        if not name:
            return
        self._name = name
        self._estimate = estimate
        self._determinate = estimate is not None
        self._started_at = time.monotonic()
        self._card = self._card_for(name)
        if self._card is not None:
            self._card.begin_progress(estimate)

    def report_fraction(self, name: str, fraction: float) -> None:
        """Report a step that knows exactly how far along it is.

        A tiled run counts tiles, which beats any estimate from the clock -
        so the bar switches from continuous to measured the moment a real
        fraction arrives.
        """
        self.show(name)
        if self._card is None:
            return
        if not self._determinate:
            self._card.set_determinate()
            self._determinate = True
        self._card.set_progress(fraction)

    def _advance(self) -> None:
        if self._card is None or not self._estimate:
            return
        elapsed = time.monotonic() - self._started_at
        self._card.set_progress(
            min(elapsed / self._estimate, MAX_ESTIMATED_FRACTION),
            remaining=max(self._estimate - elapsed, 0.0),
        )

    def finish(self) -> None:
        """Nothing is running any more: put every bar away."""
        if self._card is not None:
            self._card.end_progress()
        self._card = None
        self._name = ""
        self._estimate = None
        self._determinate = False


class EditStepDialog(QDialog):
    """A modal dialog wrapping a ParameterForm, pre-filled with the step's
    current params, plus the step's name, the channel it should work on, and
    which named upstream result each of its data inputs reads from.

    `input_candidates` answers "what could this input be wired to?" for one
    input name (e.g. "labels" -> ["labels", "watershed_split_1",
    "cellpose_segmentation_1"]); without it the input rows are omitted.
    """

    def __init__(
        self,
        step: Step,
        parent=None,
        n_channels: int | None = None,
        input_candidates=None,
        available_features=(),
        feature_catalog=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Edit {step.category}.{step.function_name}")
        self.step = step
        self.feature_select: FeatureSelectWidget | None = None

        layout = QVBoxLayout(self)

        # The step's name is what other steps use to refer to its result, so
        # it's the first thing to be able to change.
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit(step.name)
        self.name_edit.setToolTip(
            "How other steps refer to this step's result (e.g. the segmentation "
            "a measurement step measures)."
        )
        name_row.addWidget(self.name_edit, 1)
        layout.addLayout(name_row)

        # Which upstream result each data input reads from - this is how a
        # measurement step picks one of several segmentations.
        self.input_combos: dict[str, QComboBox] = {}
        for argument, key in step.input_keys.items():
            choices = list(input_candidates(argument)) if input_candidates else []
            if key not in choices:
                choices.insert(0, key)
            if len(choices) < 2:
                continue
            input_row = QHBoxLayout()
            input_row.addWidget(QLabel(f"{argument}:"))
            combo = QComboBox()
            for choice in choices:
                combo.addItem(choice, choice)
            combo.setCurrentIndex(max(combo.findData(key), 0))
            input_row.addWidget(combo, 1)
            layout.addLayout(input_row)
            self.input_combos[argument] = combo

        # Channel next: which channel a step runs on is usually the most
        # consequential choice for a multi-channel acquisition - but only
        # for the steps that read the image. A clustering or reduction step
        # reads the measured feature table, which has no channel axis (every
        # channel is already there as its own column), so offering it a
        # channel picker would say something untrue about what it does.
        self.channel_combo = QComboBox()
        self.channel_combo.addItem(ALL_CHANNELS, None)
        if step.channel_applies:
            for index in range(n_channels or 0):
                self.channel_combo.addItem(f"Channel {index}", index)
            if step.channel is not None:
                position = self.channel_combo.findData(step.channel)
                if position == -1:
                    # The step remembers a channel the current image doesn't
                    # have (different file loaded since). Keep it visible
                    # rather than silently resetting it to "All channels".
                    self.channel_combo.addItem(
                        f"Channel {step.channel} (not in image)", step.channel
                    )
                    position = self.channel_combo.count() - 1
                self.channel_combo.setCurrentIndex(position)
            channel_row = QHBoxLayout()
            channel_row.addWidget(QLabel("Channel:"))
            channel_row.addWidget(self.channel_combo)
            channel_row.addStretch()
            layout.addLayout(channel_row)
        elif step.feature_input is not None:
            # Which of the measured features this step is built from. The
            # most consequential choice for a clustering or reduction, and
            # the one that has to be recorded for the result to mean
            # anything later.
            source = QLabel("Features (from the measured data table):")
            source.setStyleSheet("color: gray;")
            layout.addWidget(source)
            self.feature_select = FeatureSelectWidget(
                available_features, step.features, feature_catalog
            )
            layout.addWidget(self.feature_select, 1)
        else:
            source = QLabel("No channel to choose: this step reads a label image or a table.")
            source.setStyleSheet("color: gray;")
            source.setWordWrap(True)
            layout.addWidget(source)

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

    def updated_name(self) -> str:
        return self.name_edit.text().strip()

    def updated_features(self) -> list[str]:
        """The chosen features, or [] for "all of them" - including when the
        step has no feature input at all."""
        if self.feature_select is None:
            return list(self.step.features)
        return self.feature_select.selected_or_all()

    def updated_input_keys(self) -> dict[str, str]:
        keys = dict(self.step.input_keys)
        for argument, combo in self.input_combos.items():
            keys[argument] = combo.currentData()
        return keys


class ProtocolBuilderWidget(QWidget):
    """The napari dock widget: the two step stacks that build a protocol.

    Split into a processing pane (image processing, segmentation - the steps
    that turn an image into labels) and an analysis pane (measurements,
    clustering, reduction, gates, classification - the steps that turn
    labels into per-object numbers). They run as two Pipelines threading one
    shared context, and sit in a vertical splitter so each gets half the
    height by default and can be re-dragged.

    Plotting and gating are *not* here: they live in the Object Explorer,
    which reads the same AnalysisSession this widget writes its results
    into. That keeps this pane about building and running a protocol, and
    lets the explorer float over the image where a scatter plot is actually
    usable.

    `napari_viewer` is auto-injected by napari's plugin engine when opened
    from the Plugins menu; pass None for standalone/script/test use (the
    "Run" button needs a layer to pull the initial volume from).
    """

    # "ownership" sits with the image steps rather than the analysis ones:
    # it reads a label image and a mask and produces something
    # image-shaped, next to the watershed_ownership it is the
    # probabilistic counterpart of.
    PROCESSING_CATEGORIES = ("imageprocessing", "segmentation", "ownership")
    # "classification" is deliberately absent. Its steps need `crops`,
    # `model`, `object_ids` and `class_labels` - none of which any step in a
    # protocol produces - so every one of them could only ever fail with
    # "needs context key(s) [...]". The functions stay in vtea_core and work
    # from a script; putting them back in this menu needs a crop-extraction
    # step and a way to label training objects, neither of which exists yet.
    ANALYSIS_CATEGORIES = (
        "measurements",
        "association",
        "cells",
        "clustering",
        "reduction",
        "gates",
    )

    def __init__(
        self,
        pipeline: Pipeline | None = None,
        napari_viewer=None,
        analysis_pipeline: Pipeline | None = None,
        parent=None,
        session: AnalysisSession | None = None,
    ):
        super().__init__(parent)
        self.viewer = napari_viewer
        # Results and the protocol itself go here rather than staying in
        # this widget, so the Object Explorer sees them and so closing
        # either pane - or napari rebuilding this one - loses nothing.
        self.session = session if session is not None else session_for(napari_viewer)
        # `pipeline` stays the processing pipeline so existing callers and
        # scripts keep working; an explicitly passed one takes over the
        # session's, since the caller means to drive that object.
        if pipeline is not None:
            self.session.processing_pipeline = pipeline
        if analysis_pipeline is not None:
            self.session.analysis_pipeline = analysis_pipeline
        self.pipeline = self.session.processing_pipeline
        self.analysis_pipeline = self.session.analysis_pipeline
        self.last_context: dict = {}
        # Set by an out-of-core run: the scratch store holding its results,
        # and the ledger saying how each object a tile boundary cut was put
        # back together. Both outlive the run because the results live in
        # the store rather than in memory.
        self._scratch = None
        self.blocked_ledgers: dict = {}
        # Which axis is depth; used to present results as full z-stacks.
        self.z_axis: int | None = None
        # What the steps actually cost on this machine, learned as they run,
        # so the progress bars stop being wrong in the same direction every
        # time - see vtea_core.workflow.cost.
        self.calibration = Calibration()
        # What a running step has to say, written on the worker thread and
        # read on this one. Nothing else crosses that boundary.
        self.progress = ProgressRelay()
        self.step_progress = StepProgressDisplay(self.card_for_name)
        # The signatures the current results were computed from, so an edit
        # that changes a setting can be told from one that only renames -
        # see recalculate_from.
        self._computed_signatures: dict[str, tuple] = {}

        root = QVBoxLayout(self)

        # Built before anything that might report into it - reading a voxel
        # size off the first layer happens during construction, and a
        # message with nowhere to go would take the whole widget down.
        self.status_label = LogView()

        # Source, then how to read its axes, then run - left to right in the
        # order you have to decide them.
        top_row = QHBoxLayout()

        top_row.addWidget(QLabel("Image:"))
        self.layer_combo = QComboBox()
        self.layer_combo.setToolTip("Which loaded layer the pipeline runs on")
        self.layer_combo.currentIndexChanged.connect(self._on_source_layer_changed)
        top_row.addWidget(self.layer_combo, 1)

        top_row.addWidget(QLabel("Channel axis:"))
        self.channel_axis_combo = QComboBox()
        self.channel_axis_combo.currentIndexChanged.connect(self._on_channel_axis_changed)
        top_row.addWidget(self.channel_axis_combo)

        top_row.addWidget(QLabel("Z axis:"))
        self.z_axis_combo = QComboBox()
        self.z_axis_combo.setToolTip(
            "Which axis is depth. Used to show results as full z-stacks that "
            "follow napari's slider and render in 3D."
        )
        self.z_axis_combo.currentIndexChanged.connect(self._on_z_axis_changed)
        top_row.addWidget(self.z_axis_combo)

        # The physical voxel size, read off the image where the file
        # recorded one. Sits with the axis pickers because it is the same
        # kind of fact: how to read the array as a specimen.
        self.spacing_control = SpacingControl()
        self.spacing_control.spacing_changed.connect(self._on_spacing_changed)
        top_row.addWidget(self.spacing_control)

        # How much memory this run may use, and what that divides the data
        # into. Shown rather than assumed: at a laptop's budget a large
        # volume becomes hundreds of tiles, and a user who cannot see that
        # cannot tell a slow run from a stuck one.
        self.memory_control = MemoryControl()
        self.memory_control.budget_changed.connect(lambda _budget: self.refresh_plan())
        top_row.addWidget(self.memory_control)

        # Only visible while an out-of-core run is going. A run measured in
        # hours has to be stoppable, and a window that has stopped
        # repainting is indistinguishable from one that has crashed.
        self.run_control = RunControl(self)
        top_row.addWidget(self.run_control.widget())

        # A protocol usually has more than one segmentation - a nucleus, a
        # ring derived from it - and every one of them is a population worth
        # measuring. On by default because the alternative is an analysis
        # silently missing the ring; a switch rather than a rule because a
        # protocol that segments an intermediate mask it does not care about
        # should be able to say so.
        self.measure_all_check = QCheckBox("Measure every segmentation")
        self.measure_all_check.setToolTip(
            "Keep one measurement step per segmentation, named after it "
            "(measure_<segmentation>). Renaming a segmentation renames its "
            "measurement; deleting one removes it."
        )
        self.measure_all_check.setChecked(True)
        self.measure_all_check.toggled.connect(lambda _on: self.sync_measurement_steps())
        top_row.addWidget(self.measure_all_check)

        # Run the protocol over the part of the image on screen, at the
        # resolution on screen. On a dataset where a full run is measured in
        # hours, this is the only way to tune a parameter at all.
        self.preview_control = PreviewControl()
        self.preview_control.requested.connect(self.run_preview)
        top_row.addWidget(self.preview_control)

        top_row.addStretch()

        # The explorer is where the results are looked at, so it is worth
        # being able to open it from here rather than hunting the Plugins
        # menu after every run.
        self.explorer_button = QPushButton("Object Explorer")
        self.explorer_button.setToolTip("Open the plot and gate manager for these results")
        self.explorer_button.clicked.connect(self.open_object_explorer)
        top_row.addWidget(self.explorer_button)

        root.addLayout(top_row)

        if self.viewer is not None:
            # Keep the picker in step with what's loaded.
            self.viewer.layers.events.inserted.connect(lambda _e: self.refresh_sources())
            self.viewer.layers.events.removed.connect(lambda _e: self.refresh_sources())
            self.preview_control.attach(self.viewer)
        self.refresh_sources()

        # No pane-level Run button: every step card has its own, which is
        # both finer-grained and unambiguous about what will run.
        self.processing_stack = StepStackWidget(
            self.PROCESSING_CATEGORIES,
            self.pipeline,
            title="Processing",
            # "spacing" is here because the derived segmentation steps take
            # a physical thickness: without it they would wire up happily
            # and silently work in voxels.
            seed_keys={"volume", "intensity", "spacing"},
            n_channels_provider=lambda: self.n_channels(),
            results_provider=lambda: self.last_context,
            default_channel_provider=lambda: self.default_channel(),
            taken_names_provider=lambda: self.step_names(),
            input_candidates_provider=self.input_candidates,
            available_features_provider=self.available_features,
            feature_catalog_provider=self.feature_catalog,
        )
        self.processing_stack.run_step_requested.connect(self.run_single_step)
        self.processing_stack.step_renamed.connect(self.repoint_inputs)
        # A segmentation added or deleted changes what there is to measure.
        self.processing_stack.steps_changed.connect(self.sync_measurement_steps)
        self.processing_stack.step_settings_changed.connect(self.recalculate_from)

        self.analysis_stack = StepStackWidget(
            self.ANALYSIS_CATEGORIES,
            self.analysis_pipeline,
            title="Analysis",
            # Analysis steps consume what processing produced, plus the two
            # things the widget itself supplies: how to read the channel axis,
            # and the measurement table flattened into a feature matrix.
            seed_keys={"volume", "intensity", "labels", "mask", "channel_axis", "spacing", "data"},
            n_channels_provider=lambda: self.n_channels(),
            results_provider=lambda: self.last_context,
            default_channel_provider=lambda: self.default_channel(),
            taken_names_provider=lambda: self.step_names(),
            input_candidates_provider=self.input_candidates,
            available_features_provider=self.available_features,
            feature_catalog_provider=self.feature_catalog,
        )
        self.analysis_stack.run_step_requested.connect(self.run_single_step)
        self.analysis_stack.step_renamed.connect(self.repoint_inputs)
        self.analysis_stack.step_settings_changed.connect(self.recalculate_from)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.processing_stack)
        self.splitter.addWidget(self.analysis_stack)
        self.splitter.setChildrenCollapsible(False)
        # Equal stretch: each pane takes half the dock's height rather than
        # the steps list being squeezed to a single visible card.
        for index in range(self.splitter.count()):
            self.splitter.setStretchFactor(index, 1)
        root.addWidget(self.splitter, 1)

        root.addWidget(self.status_label)

        self._apply_compact_style(root)
        self._apply_width_budget()

    def _apply_compact_style(self, root: QVBoxLayout) -> None:
        """Shrink text ~25% and tighten the padding around everything.

        The dock was claiming a lot of screen for the amount it shows. Scaled
        off the application font rather than a hard-coded point size, so it
        stays proportional on a high-DPI display or when the user has already
        changed napari's font size.
        """
        base = QApplication.font().pointSizeF()
        if base > 0:
            self.setStyleSheet(
                f"QWidget {{ font-size: {base * COMPACT_FONT_SCALE:.1f}pt; }}"
                f"{RUN_BUTTON_STYLE}"
            )
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(3)
        for layout in self.findChildren(QHBoxLayout) + self.findChildren(QVBoxLayout):
            layout.setSpacing(3)
        for card_layout in self.findChildren(QFormLayout):
            card_layout.setVerticalSpacing(2)

    def _apply_width_budget(self) -> None:
        """Cap the dock at a fraction of the screen. Everything inside wraps
        or scrolls, so a long message can no longer push the dock wider than
        the image it is meant to sit next to."""
        screen = QApplication.primaryScreen()
        if screen is None:  # no display (headless tests, offscreen platform)
            return
        available = screen.availableGeometry().width()
        if available > 0:
            self.setMaximumWidth(int(available * MAX_WIDTH_SCREEN_FRACTION))

    def resizeEvent(self, event):  # Qt's spelling
        super().resizeEvent(event)
        self.status_label.apply_height_budget(self.height())

    # -- data -------------------------------------------------------------

    def source_layer(self):
        """The layer chosen in the Image picker, falling back to the
        selected layer so the widget still works before anything is picked."""
        if self.viewer is None:
            return None
        name = self.layer_combo.currentData()
        if name is not None:
            for layer in self.viewer.layers:
                if layer.name == name:
                    return layer
        return self.viewer.layers.selection.active

    def source_data(self):
        """The chosen layer's data *as the layer holds it*.

        Not materialized. napari renders a Dask- or Zarr-backed layer by
        pulling the chunks it needs, and calling `np.asarray` here would
        undo that at the moment a protocol runs - reading a 40 GB
        acquisition into memory to decide whether it fits in memory. What
        happens to it next is `run_processing`'s decision, and it depends on
        the tile plan.
        """
        layer = self.source_layer()
        return None if layer is None else layer.data

    def active_image(self) -> np.ndarray | None:
        """The chosen layer's data in memory, or None when there is nothing
        to run on.

        Materializes. Correct for data that fits, which is what the
        in-memory path needs; `source_data` is the one to use when the
        answer might be that it does not.
        """
        data = self.source_data()
        return None if data is None else np.asarray(data)

    def memory_budget(self):
        """How much memory a run here may use - see MemoryControl."""
        return self.memory_control.budget()

    def spatial_axes(self, ndim: int) -> tuple[int, ...]:
        """Which axes a tile plan may divide.

        Everything but the channel axis. A tile holding one channel of a
        four-channel volume would have to read the other three to get it,
        since they interleave, and a step that measures every channel needs
        them together anyway.
        """
        channel_axis = self.pipeline.channel_axis
        return tuple(axis for axis in range(ndim) if axis != channel_axis)

    def tile_plan(self):
        """How this protocol divides this image, or None without one.

        Computed from the steps rather than from the image alone: the tile
        size is set by the protocol's heaviest step, which is why the
        control can say *what* bounded it.
        """
        from vtea_core.blocked import plan_for_steps

        data = self.source_data()
        if data is None:
            return None
        try:
            return plan_for_steps(
                self.all_steps(),
                tuple(data.shape),
                budget=self.memory_budget(),
                spacing=self.spacing_control.spacing(),
                tiled_axes=self.spatial_axes(len(data.shape)),
            )
        except Exception:  # noqa: BLE001 - an unplannable protocol is not a crash
            return None

    def refresh_plan(self) -> None:
        """Show what the current budget and protocol imply, before running."""
        plan = self.tile_plan()
        self.memory_control.set_plan(plan)
        return plan

    def refresh_sources(self) -> None:
        """Repopulate the Image picker and the axis pickers from the viewer."""
        if self.viewer is not None:
            previous = self.layer_combo.currentData()
            self.layer_combo.blockSignals(True)
            self.layer_combo.clear()
            for layer in self.viewer.layers:
                # Only things with pixels are candidates to process, and a
                # preview is not one: it is a partial answer over part of
                # the image, and running a protocol on it would compound
                # the approximation rather than show anything.
                if layer.name.startswith(PREVIEW_PREFIX):
                    continue
                if hasattr(layer, "data") and getattr(layer.data, "ndim", 0) >= 2:
                    self.layer_combo.addItem(layer.name, layer.name)
            position = self.layer_combo.findData(previous)
            self.layer_combo.setCurrentIndex(max(position, 0))
            self.layer_combo.blockSignals(False)
        self._refresh_channel_axis_choices()
        self._refresh_z_axis_choices()
        if hasattr(self, "spacing_control"):
            self.spacing_control.read_from_layer(self.source_layer())

    def _on_source_layer_changed(self, _index: int) -> None:
        # A different image can have a different shape, so the axis choices
        # have to be rebuilt against it.
        self._refresh_channel_axis_choices()
        self._refresh_z_axis_choices()
        self.spacing_control.read_from_layer(self.source_layer())

    def _on_spacing_changed(self, spacing) -> None:
        self.session.set_spacing(spacing)
        if spacing is not None and spacing.is_known:
            self.status_label.setText(f"Voxel size: {spacing.describe()}")

    def all_steps(self) -> list[Step]:
        return list(self.pipeline.steps) + list(self.analysis_pipeline.steps)

    def step_names(self) -> list[str]:
        """Every step name in use, across both pipelines - names have to be
        unique between them because they share one run context."""
        return [step.name for step in self.all_steps() if step.name]

    def input_candidates(self, argument: str) -> list[str]:
        """What an input called `argument` could be wired to: the shared key
        of the same name, plus the name of every step that produces it.

        This is what turns "measure the labels" into "measure
        watershed_split_2" - with two segmentations in a protocol, the shared
        "labels" key holds whichever ran last, which isn't a choice.
        """
        # An association step's inputs are named for their role
        # ("child_labels", "parent_labels"), not for the context key they
        # read, so they are offered every segmentation rather than only the
        # steps that happen to write a key of that name.
        produces = "labels" if argument.endswith("labels") else argument
        choices = [produces]
        for step in self.all_steps():
            if step.name and step.output_key == produces and step.name not in choices:
                choices.append(step.name)
        return choices

    def repoint_inputs(self, old_name: str, new_name: str) -> None:
        """Follow a rename through the whole protocol.

        Three things carry a step's name and all three have to move with it:
        the inputs of every step wired to it, the result it has already
        published (under the old name, in the run context and in the feature
        catalog), and - when a segmentation is what was renamed - the
        measurement step raised for it, which is named after it.

        A rename that left any of those behind would be worse than not
        allowing renames at all: the protocol would still look right and
        would measure the wrong thing, or nothing.
        """
        self._repoint(old_name, new_name)
        self._rename_result(old_name, new_name)
        for step, previous in rename_measured_segmentation(
            self.analysis_pipeline, old_name, new_name, taken_names=self.step_names()
        ):
            if previous and previous != step.name:
                self._repoint(previous, step.name)
                self._rename_result(previous, step.name)
        self.refresh_steps()

    def _repoint(self, old_name: str, new_name: str) -> None:
        for step in self.all_steps():
            for parameter, key in list(step.input_keys.items()):
                if key == old_name:
                    step.input_keys[parameter] = new_name

    def _rename_result(self, old_name: str, new_name: str) -> None:
        """Move what a step has already produced onto its new name."""
        if old_name in self.last_context:
            self.last_context[new_name] = self.last_context.pop(old_name)
        if old_name in self._computed_signatures:
            self._computed_signatures[new_name] = self._computed_signatures.pop(old_name)
        self.session.feature_catalog.rename_source(old_name, new_name)

    # -- measuring every segmentation -------------------------------------

    def sync_measurement_steps(self) -> list[Step]:
        """Keep one measurement step per segmentation - see
        vtea_core.workflow.measure.

        Returns the steps it raised. A no-op when the switch is off, and
        idempotent otherwise, so it can be wired straight to "the steps
        changed" without having to work out what changed.
        """
        if not getattr(self, "measure_all_check", None) or not self.measure_all_check.isChecked():
            return []
        added, removed = sync_measurements(
            self.pipeline,
            self.analysis_pipeline,
            available=set(self.analysis_stack.seed_keys) | set(self.last_context),
            taken_names=self.step_names(),
        )
        if not added and not removed:
            return []
        for step in removed:
            self._computed_signatures.pop(step.name, None)
        self.refresh_steps()
        parts = []
        if added:
            parts.append(f"measuring {', '.join(step.auto_for for step in added)}")
        if removed:
            parts.append(f"dropped {', '.join(step.name for step in removed)}")
        self.status_label.setText("; ".join(parts))
        return added

    # -- estimating and re-running ----------------------------------------

    def card_for(self, step: Step):
        """The card showing `step`, from whichever pane holds it."""
        for stack in (self.processing_stack, self.analysis_stack):
            card = stack.card_for(step)
            if card is not None:
                return card
        return None

    def card_for_name(self, name: str):
        """The card of the step whose result is called `name`."""
        for step in self.all_steps():
            if step.result_key == name:
                return self.card_for(step)
        return None

    def estimate_for(self, step: Step) -> float | None:
        """How long `step` should take on this data, or None for a step
        whose duration cannot honestly be predicted - see
        vtea_core.workflow.estimate_seconds.

        Sized from the tile the protocol would actually run in where there
        is a plan for one, and from the whole image otherwise, because those
        are the same number of voxels only when the data fits in memory.
        """
        voxels, tiles = self._work_size()
        frame = self.results_table()
        n_objects = 0 if frame is None else len(frame)
        n_features = 0
        if frame is not None and step.feature_input is not None:
            n_features = len(step.selected_features(frame))
        return estimate_seconds(
            step,
            voxels=voxels,
            n_objects=n_objects,
            n_features=n_features,
            tiles=tiles,
            calibration=self.calibration,
        )

    def _work_size(self) -> tuple[int, int]:
        """(voxels per tile, number of tiles) for the current source."""
        data = self.source_data()
        shape = getattr(data, "shape", None)
        if not shape:
            return 0, 1
        plan = self.tile_plan()
        if plan is not None and not plan.is_single_tile:
            return int(np.prod(plan.padded_tile)), plan.n_tiles
        return int(np.prod(shape)), 1

    def stale_steps(self, step: Step) -> list[Step]:
        """`step` and everything downstream of it, in protocol order.

        Downstream means reading this step's result: by its name, or - for
        the last step to write a shared key like "labels" - by that key,
        since that is the one a later step wired to the default is actually
        reading. Anything else would sweep in the steps that read a
        *different* segmentation's labels and re-run half the protocol.
        """
        order = self.all_steps()
        if step not in order:
            return []
        tainted = {step.result_key}
        if self._is_last_producer(step, order):
            tainted.add(step.output_key)
        affected = [step]
        for other in order[order.index(step) + 1 :]:
            if any(key in tainted for key in other.input_keys.values()):
                affected.append(other)
                tainted.add(other.result_key)
                if self._is_last_producer(other, order):
                    tainted.add(other.output_key)
        return affected

    @staticmethod
    def _is_last_producer(step: Step, order: list[Step]) -> bool:
        producers = [other for other in order if other.output_key == step.output_key]
        return bool(producers) and producers[-1] is step

    def recalculate_from(self, step: Step) -> list[Step]:
        """Re-run `step` and everything downstream that has already run.

        This is what a changed setting means: the result on the card was
        computed from parameters that are no longer this step's, and leaving
        it there would show a segmentation nobody asked for next to the
        settings that did not produce it. Steps that have never run are left
        alone - they have nothing stale to correct, and running them
        unasked is how a click on "Edit" turns into an hour of watershed.
        """
        if self.run_control.busy:
            return []
        if self._computed_signatures.get(step.result_key) == step.settings_signature:
            # Whatever changed, it was not a setting this step computes from
            # - a rename, or a call made for a step nobody has touched.
            return []
        affected = [
            other for other in self.stale_steps(step) if other.result_key in self.last_context
        ]
        if not affected:
            return []
        self.status_label.setText(
            f"{step.result_key} changed - recalculating "
            f"{', '.join(other.result_key for other in affected)}"
        )
        for other in affected:
            self.run_single_step(other, show_result=False)
        return affected

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
        result for step thumbnails/Show buttons, and updates the plot.

        The analysis steps are stepped through one at a time rather than
        handed to Pipeline.run, because `data` - the feature matrix the
        clustering and reduction steps consume - has to be rebuilt from the
        measurement table between them; a clustering that follows a
        reduction should see the reduced dimensions as features.
        """
        self.last_context = self.pipeline.run(context)
        self.analysis_pipeline.channel_axis = self.pipeline.channel_axis
        for step in self.analysis_pipeline.steps:
            working = dict(self.last_context)
            self._seed_axes(working)
            self._seed_feature_matrix(working)
            self._seed_measurement_tables(working)
            result = step.run(
                working,
                channel_axis=self.pipeline.channel_axis,
                full_ndim=self._seed_ndim(working),
            )
            working[step.output_key] = result
            if step.name:
                working[step.name] = result
            self.last_context = working
            self._merge_into_measurements(step, result)
        self._record_signatures(self.all_steps())
        self.refresh_steps()
        self._publish_results()
        return self.last_context

    def default_channel(self) -> int | None:
        """The channel a newly added step should start on: whichever one the
        protocol is already using. Leaving a later step on "all channels"
        while an earlier one selected channel 2 fed a 4D intensity array and
        a 3D label array into the same function, which aborted the run."""
        for pipeline in (self.analysis_pipeline, self.pipeline):
            for step in reversed(pipeline.steps):
                if step.channel is not None:
                    return step.channel
        return None

    def _seed_axes(self, context: dict) -> None:
        """Supply the two facts no step produces and the builder owns: which
        axis is the channel axis, and how big a voxel is.

        Seeded rather than assumed present because a measurement step is
        wired to both, and a run driven from a script or a test starts from
        whatever context it was handed. Absent, the step aborts with "needs
        context key(s) ['channel_axis']" - which is true and useless.
        """
        context.setdefault("channel_axis", self.pipeline.channel_axis)
        context.setdefault("spacing", self.spacing_control.spacing())

    def seed_context(self) -> dict:
        """Starting context for a run: the chosen image under both the key
        processing consumes and the untouched one measurements read, plus
        the channel axis, which a measurement step needs in order to measure
        every channel and label its features with the channel each came
        from."""
        image = self.active_image()
        if image is None:
            return {}
        context = {
            "volume": image,
            "intensity": image,
            "channel_axis": self.pipeline.channel_axis,
        }
        spacing = self.spacing_control.spacing()
        if spacing is not None:
            context["spacing"] = spacing
        return context

    def run_processing(self) -> dict:
        """Run only the processing pipeline. Analysis steps are run
        individually from their own cards - they are not a chain.

        The plan decides how. Data that fits the budget runs in memory,
        which is what it has always done and is faster for the sizes that
        allow it; data that does not runs a tile at a time through
        `vtea_core.blocked`. The user sees which, because a run that takes
        four hours should say why before it starts rather than after.
        """
        context = dict(self.last_context) or {}
        plan = self.refresh_plan()
        if plan is not None and not plan.is_single_tile:
            return self._run_processing_blocked(plan, context)

        seed = self.seed_context()
        if not seed:
            self.status_label.setText("Select an image layer to run on.")
            return context
        context.update(seed)
        # An in-memory run has no seams. Clearing here rather than on
        # success keeps a failed run from publishing the previous run's
        # ledger against this run's table.
        self.blocked_ledgers = {}
        try:
            self.last_context = self._run_off_thread(self.pipeline, context)
        except Exception as exc:  # noqa: BLE001 - report in the UI, don't crash napari
            self.status_label.setText(f"{type(exc).__name__}: {exc}")
            return self.last_context
        self._record_signatures(self.pipeline.steps)
        self.refresh_steps()
        self._publish_results()
        self.status_label.setText("Processing finished.")
        return self.last_context

    def _run_off_thread(self, pipeline: Pipeline, context: dict) -> dict:
        """Run `pipeline` on a worker thread, driving the cards' progress
        bars from this one.

        Even an in-memory run is minutes on a real acquisition, and a window
        that has stopped repainting is indistinguishable from one that has
        crashed. Nothing about the pipeline changes: it is the same
        `Pipeline.run`, and its result comes back to this thread, which is
        the only one that touches a layer. Re-entrancy - a step already
        running - falls back to running here rather than refusing, since the
        caller is then already off the GUI thread.
        """
        if self.run_control.busy:
            return pipeline.run(context)
        estimates = {step.result_key: self.estimate_for(step) for step in pipeline.steps}
        total = self._describe_total(pipeline.steps, estimates)
        if total:
            self.status_label.setText(f"Running {len(pipeline.steps)} step(s), {total}")
        started = {}

        def report(step, done, count):
            # Worker thread: writes to the relay, touches nothing.
            name = "" if step is None else step.result_key
            self.progress.report(
                f"step {done + 1} of {count}: {name}" if step is not None else "",
                fraction=done / max(count, 1),
                name=name,
            )

        def on_tick(_elapsed):
            message, _fraction, name = self.progress.snapshot()
            if message:
                self.status_label.setText(message)
            started.setdefault(name, time.monotonic())
            self.step_progress.show(name, estimates.get(name))

        self.progress.clear()
        try:
            result = self.run_control.run(
                lambda _should_stop: pipeline.run(context, progress=report),
                on_tick=on_tick,
            )
        finally:
            self.step_progress.finish()
            self.progress.clear()
        self._observe_durations(pipeline.steps, started, estimates)
        return result

    def _describe_total(self, steps, estimates: dict) -> str:
        """"about 4 min" for a whole protocol, or the fact that it cannot be
        said - which is itself worth saying before somebody commits an
        afternoon to a run."""
        known = [estimates.get(step.result_key) for step in steps]
        timed = [value for value in known if value is not None]
        if not timed:
            return ""
        total = format_duration(sum(timed))
        if len(timed) < len(known):
            return f"at least {total}"
        return total

    def _observe_durations(self, steps, started: dict, estimates: dict) -> None:
        """Feed what the steps actually took back into the estimates.

        Timed from when each step was first seen running to when the next
        one was, which is as fine-grained as the pump loop allows and quite
        enough to catch an estimate that is out by a factor of four.
        """
        marks = sorted(started.items(), key=lambda item: item[1])
        for index, (name, start) in enumerate(marks):
            end = marks[index + 1][1] if index + 1 < len(marks) else time.monotonic()
            step = next((one for one in steps if one.result_key == name), None)
            if step is not None:
                self.calibration.observe(
                    step, seconds=end - start, predicted=estimates.get(name)
                )

    def _run_processing_blocked(self, plan, context: dict) -> dict:
        """Run the protocol out of core, a tile at a time.

        The results stay in the scratch store rather than being copied into
        memory - that is the point - so the scratch store outlives this
        call and is closed when the next run replaces it or the widget goes
        away. napari renders the stored arrays the same way it renders any
        other chunked layer.
        """
        from vtea_core.blocked import BlockedPipeline, Cancelled, ZarrScratch

        data = self.source_data()
        if data is None:
            self.status_label.setText("Select an image layer to run on.")
            return context
        seed = {
            "volume": data,
            "intensity": data,
            "channel_axis": self.pipeline.channel_axis,
        }
        spacing = self.spacing_control.spacing()
        if spacing is not None:
            seed["spacing"] = spacing

        if self.run_control.busy:
            self.status_label.setText("A run is already in progress.")
            return context

        self.status_label.setText(f"Running out of core: {plan.describe()}")
        self._close_scratch()
        self._scratch = ZarrScratch()
        runner = BlockedPipeline(
            self.pipeline, plan=plan, scratch=self._scratch, spacing=spacing
        )
        try:
            # Off the Qt thread, with the event loop pumped meanwhile - see
            # RunControl. Nothing here touches a napari layer; publishing
            # happens below, back on the thread that called this.
            self.last_context = self.run_control.run(
                lambda should_stop: runner.run(
                    seed, progress=self._on_block_progress, should_stop=should_stop
                ),
                on_tick=self._on_blocked_tick,
            )
            self.blocked_ledgers = dict(runner.ledgers)
        except Cancelled:
            # A cancelled run has a partial result in the scratch store,
            # which must not be published as a finished one. It is kept
            # rather than deleted: paired with a manifest it is the start of
            # a resume, and either way it is the user's to discard.
            self.status_label.setText(
                f"Cancelled - {plan.n_tiles:,} tiles planned, partial result not published."
            )
            return self.last_context
        except Exception as exc:  # noqa: BLE001 - report in the UI, don't crash napari
            self._close_scratch()
            self.status_label.setText(f"{type(exc).__name__}: {exc}")
            return self.last_context
        self.refresh_steps()
        self._publish_results()
        self.status_label.setText(f"Processing finished - {plan.n_tiles:,} tiles.")
        return self.last_context

    def _on_block_progress(self, name: str, done: int, total: int) -> None:
        """Say which step is running and how far in.

        A blocked run is measured in tiles and can be measured in hours, so
        the status line is the difference between a progress report and an
        apparently frozen window.

        Called on the worker thread, so it writes to the relay rather than
        to the status label: setting a widget's text from a worker thread is
        not a cosmetic bug in Qt, it is a crash waiting for a repaint.
        """
        self.progress.report(
            f"{name}: tile {done:,} of {total:,}",
            fraction=done / max(total, 1),
            name=name,
        )

    def _on_blocked_tick(self, _elapsed: float) -> None:
        """Draw what the blocked run has reported. On the GUI thread."""
        message, fraction, name = self.progress.snapshot()
        if message:
            self.status_label.setText(message)
        if not name:
            return
        # A tiled run knows exactly how far along it is - tiles done over
        # tiles planned - which beats any estimate from the clock.
        if fraction is None:
            self.step_progress.show(name)
        else:
            self.step_progress.report_fraction(name, fraction)

    def _close_scratch(self) -> None:
        scratch = getattr(self, "_scratch", None)
        if scratch is not None:
            scratch.close()
            self._scratch = None

    def run_single_step(self, step: Step, *, show_result: bool = True) -> None:
        """Run one step against everything computed so far and show what it
        produced.

        This is what lets the analysis steps be a graph rather than a chain:
        measurements can feed clustering, reduction and gating
        independently, and a clustering result can be folded back into the
        measurement table as another feature.

        The step itself runs on a worker thread with its card's progress bar
        going, so a ten-minute clustering leaves the window usable instead
        of leaving the user to guess whether napari has hung. The result
        comes back here, on the GUI thread, and everything that touches a
        layer happens after that.
        """
        context = dict(self.last_context)
        for key, value in self.seed_context().items():
            context.setdefault(key, value)
        self._seed_axes(context)
        # Rebuild `data` from the current table each time, so a clustering
        # step run after a reduction step sees the reduced dimensions as
        # features too - that feedback is the point of running these
        # individually.
        self._seed_feature_matrix(context)
        self._seed_measurement_tables(context)

        try:
            result = self._run_step_off_thread(step, context)
        except Exception as exc:  # noqa: BLE001 - report in the UI, don't crash napari
            self.status_label.setText(f"{step.function_name}: {type(exc).__name__}: {exc}")
            return

        context[step.output_key] = result
        if step.name:
            context[step.name] = result
        self.last_context = context
        self._record_signatures([step])
        self._merge_into_measurements(step, result)
        self.refresh_steps()
        self._publish_results()

        if not show_result:
            return
        if step.produces_image and isinstance(result, np.ndarray) and result.ndim >= 2:
            self.show_step_result(step)
        elif isinstance(result, np.ndarray):
            # A clustering or a reduction. Its result is per object, not per
            # voxel, so it belongs in the table and on the plot's axes - not
            # in the layer list, where a (n_objects, 2) t-SNE embedding
            # would sit as a two-pixel-wide stripe that means nothing.
            self.status_label.setText(f"{step.result_key}: {self._describe_features(step)}")
        elif isinstance(result, Ownership):
            self.show_ownership(step, result)
        elif isinstance(result, CellCollection):
            # Same reason as an association: nothing to draw, and how many
            # cells are missing a part is the number worth seeing.
            self.status_label.setText(f"{step.result_key}: {result.summary()}")
        elif isinstance(result, AssociationSet):
            # An association has nothing to draw, and how many objects were
            # left unlinked is the whole question - "it ran" would hide the
            # one number that says whether the parameters were right.
            #
            # Any decisions a person has already made about these objects are
            # re-applied first: re-running with different parameters should
            # correct the automated answers, not quietly undo the settled
            # ones.
            restored = self.session.apply_manual_links(result)
            summary = result.summary()
            if restored:
                summary += f"; {restored} manual decision(s) kept"
            self.status_label.setText(f"{step.result_key}: {summary}")
        else:
            self.status_label.setText(f"Ran {step.function_name} -> '{step.result_key}'")

    def _run_step_off_thread(self, step: Step, context: dict):
        """Run one step on a worker thread, driving its card's bar from here.

        Falls back to running in place when a run is already in progress -
        which happens when a recalculation is triggered from inside one -
        since the caller is then already off the GUI thread and a second
        worker would buy nothing.
        """

        def run():
            return step.run(
                context,
                channel_axis=self.pipeline.channel_axis,
                full_ndim=self._seed_ndim(context),
            )

        if self.run_control.busy:
            return run()

        estimate = self.estimate_for(step)
        self.step_progress.show(step.result_key, estimate)
        started = time.monotonic()
        try:
            result = self.run_control.run(
                lambda _should_stop: run(),
                on_tick=lambda _elapsed: self.step_progress.show(step.result_key, estimate),
                show_cancel=False,
            )
        finally:
            self.step_progress.finish()
        self.calibration.observe(step, seconds=time.monotonic() - started, predicted=estimate)
        return result

    def _describe_features(self, step: Step) -> str:
        """What a per-object step contributed to the table, for the log."""
        frame = self.last_context.get("measurements")
        if not isinstance(frame, pd.DataFrame):
            return "per-object result added to the data"
        prefix = f"{step.result_key}_"
        columns = [
            name for name in frame.columns if name == step.result_key or name.startswith(prefix)
        ]
        if not columns:
            return "per-object result added to the data"
        return f"added {', '.join(columns)} to the data (no image layer)"

    def _record_signatures(self, steps) -> None:
        """Remember what each of these steps was computed from.

        The record is per step name, and it is what "a setting changed"
        means later: the signature on the card no longer matches the one the
        result in hand was produced with.
        """
        for step in steps:
            self._computed_signatures[step.result_key] = step.settings_signature

    def _seed_ndim(self, context: dict) -> int | None:
        arrays = [value for value in context.values() if isinstance(value, np.ndarray)]
        return max((value.ndim for value in arrays), default=None)

    def _seed_feature_matrix(self, context: dict) -> None:
        """Put the measurement table into the context as `data`, which is
        what clustering and reduction steps consume.

        Nothing in a protocol produces a `data` key, so without this every
        clustering/reduction step fails with "needs context key(s) ['data']"
        the moment it's run. The *table* is seeded rather than a ready-made
        matrix so each step can narrow it to its own chosen features - see
        Step.selected_features.
        """
        frame = self.results_table()
        if frame is not None:
            context["data"] = frame

    def _seed_measurement_tables(self, context: dict) -> None:
        """Put each measurement step's table into the context under the
        segmentation it measured, which is what a per-cell table is built
        from.

        A cell's parts come from several segmentations, so measuring them is
        several steps producing several tables - and unlike `data`, they
        cannot be one flat frame, because they have different rows. Keying
        them by segmentation is what lets `cell_features` line up a nucleus
        with the cytoplasm it belongs to.
        """
        tables = {}
        for step in self.all_steps():
            if step.output_key != "measurements" or not step.name:
                continue
            frame = self.last_context.get(step.name)
            role = self._segmentation_measured_by(step)
            if isinstance(frame, pd.DataFrame) and role:
                tables[role] = frame
        context["measurement_tables"] = tables

    def _segmentation_measured_by(self, step: Step) -> str:
        """Which segmentation a measurement step's rows are objects of.

        Usually the step the `labels` input points at. A weighted step reads
        an ownership instead, and the ids in that ownership belong to the
        markers it was built from - which the ownership itself records, so
        the answer does not depend on guessing from the step graph.
        """
        labels = step.input_keys.get("labels", "")
        if labels:
            return labels
        ownership = self.last_context.get(step.input_keys.get("ownership", ""))
        return ownership.segmentation if isinstance(ownership, Ownership) else ""

    def available_features(self) -> list[str]:
        """Every numeric feature a clustering or reduction step could be
        built from."""
        frame = self.results_table()
        return [] if frame is None else feature_matrix(frame)[1]

    def feature_catalog(self) -> FeatureCatalog:
        return self.session.feature_catalog

    def _record_features(self, step: Step, columns, source_features=()) -> None:
        """Record where a step's columns came from.

        This is the difference between a table of numbers and a table anyone
        can interpret later: what was measured, on which channel and
        segmentation, by which step and with what parameters - and, for a
        clustering or reduction, exactly which features were fed to it.
        """
        catalog = self.session.feature_catalog
        function = f"{step.category}.{step.function_name}"
        segmentation = self._segmentation_measured_by(step)
        if step.output_key == "measurements":
            catalog.record_measured(
                columns,
                produced_by=step.result_key,
                function=function,
                params=step.params,
                segmentation=segmentation,
            )
        else:
            catalog.record_derived(
                columns,
                produced_by=step.result_key,
                function=function,
                params=step.params,
                source_features=source_features,
                segmentation=self._segmentation_behind_the_table(),
            )

    def _segmentation_behind_the_table(self):
        """Which segmentation the rows of the current table are objects of -
        recorded on a derived feature, whose own step never names it."""
        for step in self.analysis_pipeline.steps:
            if step.output_key == "measurements":
                return self._segmentation_measured_by(step)
        return ""

    def _merge_into_measurements(self, step: Step, result) -> None:
        """Fold a per-object result into the measurement table so it becomes
        a feature - a cluster id or a reduced dimension is then available to
        later steps and as a plot axis, which is the feedback the analysis
        steps are meant to support.

        Columns are named after the *step*, not its shared output key, so two
        clusterings give `kmeans_1` and `kmeans_2` instead of overwriting one
        `clusters` column.
        """
        frame = self.last_context.get("measurements")
        if step.output_key == "measurements":
            if isinstance(result, pd.DataFrame):
                self.session.feature_catalog.drop_missing(result.columns)
                self._record_features(step, list(result.columns))
            return
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return
        # Read the inputs off the table *before* adding this step's own
        # columns to it, or a step with no explicit selection would record
        # itself as one of its own inputs.
        sources = step.selected_features(frame) if step.feature_input else []
        columns = feature_columns(step.result_key, result, len(frame))
        for column, values in columns.items():
            frame[column] = values
        if columns:
            self._record_features(step, list(columns), sources)

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

    def align_to_source(self, result: np.ndarray) -> np.ndarray:
        """Give a result the source image's dimensionality again.

        A channel-selecting step drops the channel axis, so its result has
        one dimension fewer than the source. napari right-aligns arrays of
        differing ndim, which silently maps that result's leading axis onto
        the *channel* axis of the world instead of z - so scrolling z showed
        the wrong thing. Re-inserting the channel axis as a singleton makes
        the result line up axis-for-axis with the image it came from, and
        keeps the full z-stack intact.
        """
        image = self.active_image()
        channel_axis = self.pipeline.channel_axis
        if image is None or channel_axis is None:
            return result
        if result.ndim == image.ndim - 1 and channel_axis <= result.ndim:
            return np.expand_dims(result, channel_axis)
        return result

    def _order_dims_for_z(self) -> None:
        """Put (z, y, x) in napari's displayed block so 3D view renders the
        whole stack and the sliders are the remaining axes."""
        if self.viewer is None or self.z_axis is None:
            return
        ndim = self.viewer.dims.ndim
        if self.z_axis >= ndim:
            return
        spatial = [self.z_axis, ndim - 2, ndim - 1]
        # Anything that isn't z/y/x becomes a slider, ahead of them.
        order = [axis for axis in range(ndim) if axis not in spatial] + spatial
        if len(set(order)) == ndim:
            self.viewer.dims.order = tuple(order)

    def show_step_result(self, step: Step) -> None:
        """Add one step's result to the viewer as a layer."""
        if self.viewer is None:
            return
        if not step.produces_image:
            # A clustering, a reduction, a gate: per-object numbers, however
            # array-shaped. They are features of the data and are shown on
            # the plot's axes; a layer for one would be a picture of a table.
            self.status_label.setText(
                f"{step.result_key}: a {step.category} result is per-object data, not an "
                f"image - nothing to show as a layer; plot it in the Object Explorer"
            )
            return
        result = self.last_context.get(step.result_key, self.last_context.get(step.output_key))
        if not isinstance(result, np.ndarray) or result.ndim < 2:
            self.status_label.setText(f"{step.result_key}: not an image, nothing to show")
            return

        result = self.align_to_source(result)
        name = step.result_key
        for existing in list(self.viewer.layers):
            if existing.name == name:
                self.viewer.layers.remove(existing)

        # Layer type follows what the step *is*, not what dtype it happens
        # to return: a blurred uint16 image is integer but is not a label
        # image, and adding it as Labels renders it as random colours.
        if step.category == "imageprocessing":
            self.viewer.add_image(result, name=name)
        elif result.dtype == bool:
            self.viewer.add_labels(result.astype(np.uint8), name=name)
        elif np.issubdtype(result.dtype, np.integer):
            self.viewer.add_labels(result.astype(np.int32), name=name)
        else:
            self.viewer.add_image(result, name=name)

        self._order_dims_for_z()
        self.status_label.setText(f"Added layer '{name}'")

    def run_preview(self) -> np.ndarray | None:
        """Run the processing protocol over the region on screen.

        As one tile of the protocol's own tiling: the visible box grown by
        the same halo every other tile gets, run in memory, and trimmed
        back. So the preview is what a full run would write there rather
        than what a filter computes when it can see nothing past the edge
        of the view - and a preview that disagreed with the run at the
        edges would be worse than none, since the edges are what gets
        looked at.

        Read at the level the viewer is displaying. Reading full resolution
        for a view showing every eighth voxel is exactly the I/O a pyramid
        exists to avoid, and the layer name says which level it used so a
        coarse preview is never mistaken for the answer.
        """
        from vtea_core.blocked import read_block, tile_for_region

        layer = self.source_layer()
        if layer is None or not self.pipeline.steps:
            self.preview_control.set_status("nothing to preview")
            return None
        region = visible_region(layer, whole_axes=self._whole_axes(layer))
        if region is None:
            self.preview_control.set_status("nothing on screen")
            return None

        levels = getattr(layer, "data", None)
        source = (
            levels[region.level]
            if isinstance(levels, (list, tuple))
            else levels
        )
        tile = tile_for_region(region.core, tuple(source.shape), self._preview_halo())
        budget = self.memory_budget()
        cost = int(np.prod(tile.padded_shape)) * PREVIEW_BYTES_PER_VOXEL
        if budget is not None and cost > budget.usable_bytes:
            self.preview_control.set_status(
                f"the view is {format_bytes(cost)} at level {region.level}, over the "
                f"{format_bytes(budget.usable_bytes)} budget - zoom in to preview it"
            )
            return None

        block = read_block(source, tile)
        context = {
            "volume": block,
            "intensity": block,
            "channel_axis": self.pipeline.channel_axis,
        }
        spacing = self.spacing_control.spacing()
        if spacing is not None:
            context["spacing"] = spacing
        try:
            # Deliberately not `self.last_context`: a preview is not a
            # result, and a step's Show button, the plot and the tables must
            # go on reading whatever the last real run produced.
            result = self.pipeline.run(context)
        except Exception as exc:  # noqa: BLE001 - report in the UI, don't crash napari
            self.preview_control.set_status(f"{type(exc).__name__}: {exc}")
            return None

        array = result.get(self.pipeline.steps[-1].output_key)
        if not isinstance(array, np.ndarray) or array.ndim != len(tile.core):
            self.preview_control.set_status("the last step produced no image to preview")
            return None
        trimmed = array[tile.inner]
        self._show_preview(trimmed, region)
        self.preview_control.set_status(
            "; ".join([f"preview: {region.describe()}", *self._preview_caveats()])
        )
        return trimmed

    def _preview_caveats(self) -> list[str]:
        """What this preview cannot promise, said out loud.

        A step whose parameter comes from a statistic over the whole image -
        an Otsu threshold, a percentile - computes that statistic over the
        region on screen instead, because that is all it was given. Often
        that is exactly what a user tuning it wants to see; it is never what
        the full run will do, and the difference is invisible unless it is
        stated.
        """
        from vtea_core.blocked import GLOBAL_STAT
        from vtea_core.workflow.wiring import scaling_for

        names = [
            step.name or step.function_name
            for step in self.pipeline.steps
            # Resolved against the step's own parameters: `threshold_mask`
            # is elementwise at a fixed value and a global statistic at
            # otsu, which is the whole question here.
            if scaling_for(step.category, step.function_name)
            .resolve(step.params)
            .mode
            == GLOBAL_STAT
        ]
        if not names:
            return []
        return [f"{', '.join(names)} measured on the view, not the whole image"]

    def _whole_axes(self, layer) -> tuple[int, ...]:
        """Axes a preview must take entire rather than crop: the channel
        axis, because slicing one channel out of a protocol that measures
        all of them changes what it computes rather than only where."""
        ndim = getattr(getattr(layer, "data", None), "ndim", None)
        channel_axis = self.pipeline.channel_axis
        if channel_axis is None:
            return ()
        return (channel_axis,) if ndim is None or channel_axis < ndim else ()

    def _preview_halo(self) -> tuple[int, ...]:
        """The halo this protocol needs, from the plan it would run under.

        In level-0 voxels, and used unchanged at whatever level is on
        screen. That over-reads by the level's downsample factor, and over
        is the safe direction: the halo a filter needs at a coarse level is
        smaller in voxels than the one it needs at full resolution, so a
        level-0 halo always covers it. Scaling it down to save the reads
        would be an optimisation with a correctness question attached.

        Zero when the protocol cannot be planned - the preview is then only
        approximate at its edges, which is still better than nothing and is
        the one case where it can differ from a run.
        """
        plan = self.tile_plan()
        return tuple(plan.halo) if plan is not None else ()

    def _show_preview(self, array: np.ndarray, region) -> None:
        """Put the preview on the image, placed where it was computed.

        Named so it cannot be taken for a result, scaled by the level it was
        read at so it sits over the data it describes, and replaced rather
        than accumulated - a layer list with forty previews in it is its own
        kind of unusable.
        """
        if self.viewer is None:
            return
        name = f"{PREVIEW_PREFIX}{self.pipeline.steps[-1].result_key}"
        for existing in list(self.viewer.layers):
            if existing.name.startswith(PREVIEW_PREFIX):
                self.viewer.layers.remove(existing)
        # The layer's own scale as well as the level's: a source layer
        # carrying a physical voxel size places its data in microns, and a
        # preview scaled only by the pyramid factor would sit somewhere the
        # user is not looking on exactly the anisotropic stacks this is for.
        source_scale = self._source_scale(len(region.core))
        placement = {
            "name": name,
            "scale": [level * own for level, own in zip(region.scale, source_scale)],
            "translate": [
                start * level * own
                for start, level, own in zip(
                    [part.start for part in region.core], region.scale, source_scale
                )
            ],
        }
        if array.dtype == bool:
            self.viewer.add_labels(array.astype(np.uint8), **placement)
        elif np.issubdtype(array.dtype, np.integer) and self.pipeline.steps[
            -1
        ].category != "imageprocessing":
            self.viewer.add_labels(array.astype(np.int32), **placement)
        else:
            self.viewer.add_image(array, **placement)

    def _source_scale(self, ndim: int) -> tuple[float, ...]:
        """The source layer's own per-axis scale, or ones."""
        layer = self.source_layer()
        try:
            scale = tuple(float(value) for value in layer.scale)
        except (AttributeError, TypeError, ValueError):
            return (1.0,) * ndim
        if len(scale) != ndim:
            return (1.0,) * ndim
        return scale

    def show_ownership(self, step: Step, ownership) -> None:
        """Show a probabilistic ownership as two layers: who won, and how
        sure that was.

        The hard argmax on its own is indistinguishable from a watershed -
        which is the problem the confidence map exists to solve. Seeing the
        two together is what makes a boundary the method was unsure about
        visible instead of merely present, so both go on at once.
        """
        self.status_label.setText(f"{step.result_key}: {ownership.summary()}")
        if self.viewer is None:
            return
        for suffix, array, adder in (
            ("", ownership.hard().astype(np.int32), self.viewer.add_labels),
            (" confidence", ownership.confidence(), self.viewer.add_image),
        ):
            name = f"{step.result_key}{suffix}"
            for existing in list(self.viewer.layers):
                if existing.name == name:
                    self.viewer.layers.remove(existing)
            adder(self.align_to_source(array), name=name)
        self._order_dims_for_z()

    def results_table(self) -> pd.DataFrame | None:
        """The per-object "data" table: the measurements, with every
        per-object analysis output (cluster ids, PCA/t-SNE coordinates)
        joined on as extra columns under the producing step's name.

        This one flat table is what the plot's X and Y menus list, so a
        feature measured on channel 2 (`mean_ch2`) and a cluster assignment
        (`kmeans_1`) are plottable against each other without the user
        having to join anything themselves.
        """
        frame = self.last_context.get("measurements")
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return None
        frame = frame.copy()
        # Only named step results, so a result stored under both its name and
        # its shared output key doesn't produce two identical columns.
        for step in self.all_steps():
            if not step.name or step.output_key == "measurements":
                continue
            result = self.last_context.get(step.name)
            for column, values in feature_columns(step.name, result, len(frame)).items():
                frame[column] = values
        return frame

    def open_object_explorer(self):
        """Open (or raise) the Object Explorer on the same session.

        Returns the explorer widget, or None with a message when there's no
        viewer to dock it into.
        """
        if self.viewer is None:
            self.status_label.setText("The Object Explorer needs a napari viewer.")
            return None
        from vtea_napari.widgets.explorer import ExplorerWidget

        for widget in QApplication.topLevelWidgets():
            for existing in widget.findChildren(ExplorerWidget):
                if existing.session is self.session:
                    existing.show()
                    existing.raise_()
                    return existing

        explorer = ExplorerWidget(napari_viewer=self.viewer, session=self.session)
        self.viewer.window.add_dock_widget(explorer, name="Object Explorer", area="right")
        return explorer

    def _publish_results(self) -> None:
        """Hand the run context and the flat feature table to the shared
        session, which is what the Object Explorer plots and gates. Doing it
        here rather than pushing directly at the explorer means the results
        survive the explorer being closed, and are already waiting when it
        is opened."""
        self.session.set_axes(
            source_layer_name=self.layer_combo.currentData(),
            channel_axis=self.pipeline.channel_axis,
            z_axis=self.z_axis,
        )
        self.session.set_spacing(self.spacing_control.spacing())
        self.session.set_ledger(self.measurement_ledger())
        self.session.set_context(
            self.last_context,
            self.results_table(),
            {**self.measurement_tables(), **self.cell_tables()},
        )

    def measurement_ledger(self):
        """The seam ledger behind the published measurement table, if any.

        There is one ledger per blocked segmentation, so which one belongs
        to the table is the one for the labels that table was measured on.
        `None` for an in-memory run, which is how the explorer knows there
        are no seams to review.
        """
        if not self.blocked_ledgers:
            return None
        for step in self.pipeline.steps:
            if step.output_key != "measurements":
                continue
            key = step.input_keys.get("labels", "labels")
            if key in self.blocked_ledgers:
                return self.blocked_ledgers[key]
        return self.blocked_ledgers.get("labels")

    def measurement_tables(self) -> dict:
        """One table per segmentation this protocol measured.

        A second segmentation is not extra columns, it is different rows -
        a ring has one row per ring, and joining it onto the nuclei would be
        a claim about which ring belongs to which nucleus that only an
        association step is entitled to make. So each measurement step's
        table travels on its own, named after the step (which is named after
        its segmentation), with the label image its rows are objects of, so
        a gate drawn on the rings lights up rings.

        Only when there is more than one: with a single measurement step the
        per-object table already *is* that table, and offering it twice
        under two names would be a choice with no difference behind it.
        """
        published = {}
        for step in self.all_steps():
            if step.output_key != "measurements" or not step.name:
                continue
            frame = self.last_context.get(step.name)
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            published[step.name] = TableView(
                frame=frame,
                id_column="object_id",
                labels_key=self._segmentation_measured_by(step) or "labels",
                noun="objects",
            )
        return published if len(published) > 1 else {}

    def cell_tables(self) -> dict:
        """The per-cell tables this protocol produced, ready to plot.

        A cell table is not the object table with extra columns - its rows
        are cells - so it travels with what its rows are: `cell_id` rather
        than `object_id`, and the segmentation the cells are rooted on, so a
        gate drawn on cells lights up the nuclei that identify them rather
        than nothing at all.
        """
        tables = {}
        for step in self.all_steps():
            if step.output_key != "cell_table" or not step.name:
                continue
            frame = self.last_context.get(step.name)
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            cells = self.last_context.get(step.input_keys.get("cells", ""))
            root = cells.root_segmentation if isinstance(cells, CellCollection) else ""
            tables[step.name] = TableView(
                frame=frame,
                id_column="cell_id",
                labels_key=root or "labels",
                noun="cells",
            )
        return tables

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

    def _refresh_z_axis_choices(self) -> None:
        image = self.active_image()
        self.z_axis_combo.blockSignals(True)
        previous = self.z_axis_combo.currentData()
        self.z_axis_combo.clear()
        self.z_axis_combo.addItem("None (2D)", None)
        if image is not None:
            for axis, size in enumerate(image.shape):
                self.z_axis_combo.addItem(f"axis {axis} (size {size})", axis)
        position = self.z_axis_combo.findData(previous if previous is not None else self.z_axis)
        self.z_axis_combo.setCurrentIndex(max(position, 0))
        self.z_axis_combo.blockSignals(False)
        self.z_axis = self.z_axis_combo.currentData()

    def _on_z_axis_changed(self, _index: int) -> None:
        self.z_axis = self.z_axis_combo.currentData()
