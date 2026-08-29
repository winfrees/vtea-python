"""The Object Explorer dock widget: scatter plot + gate manager + gallery +
image highlighting, the vtea-python equivalent of vteaexploration.MicroExplorer.

Reads and writes the shared vtea_napari.session.AnalysisSession rather than
owning its own copy of the analysis. The protocol builder publishes each run
into that session; this widget plots it. Neither pane has to be open for the
other to work, and hiding one (the napari Window menu, or a dock's close
button) loses nothing - on re-show this widget reads the session back.

It floats by default. A scatter plot docked into a narrow side panel is
unusable at the size napari gives it, and gating means working between the
plot and the image, so the natural place for it is over the canvas where it
can be moved and resized freely.

The plot and gate manager are the same widgets the protocol builder briefly
carried, moved here where they belong: ScatterPlotWidget (click-to-draw
polygon or two-click rectangle gates, colour-by/LUT) and GateManagerWidget
(the gate list, JSON save/open, per-gate statistics). What MicroExplorer/
XYExplorationPanel/TableWindow did together in Java is these three
cooperating pieces connected by Qt signals instead of that subsystem's ~25
single-method listener interfaces.

"Subgating" (vtea's SubGateListener, which opened a whole new MicroExplorer
window over a pre-filtered dataset) is real gate hierarchy here instead:
check "Gate within selection", select a gate, then draw - new gates get that
gate as their parent_id and GateSet already restricts a child's membership
to its parent's (see vtea_core.gates.gate).

Selecting a gate highlights its members as a napari Labels layer (only the
gated object ids kept, background elsewhere) - the closest napari-native
analog of vtea's colorized ImagePlus overlay repaint - and fills the gallery
with a crop around each gated object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vtea_napari.session import AnalysisSession, session_for
from vtea_napari.widgets.gallery import GalleryWidget
from vtea_napari.widgets.gate_manager import GateManagerWidget
from vtea_napari.widgets.log_view import LogView
from vtea_napari.widgets.plot import ScatterPlotWidget

# The plot is the point of this pane; the gate manager beside it is
# controls. Same 2:1 split the protocol builder used before this moved here.
PLOT_WIDTH_SHARE = 2
GATE_WIDTH_SHARE = 1

# Big enough that the axes and the gate table are both usable when it first
# appears floating over the canvas.
DEFAULT_FLOATING_SIZE = (900, 560)

HIGHLIGHT_LAYER_NAME = "Gate highlight"


class ExplorerWidget(QWidget):
    """A napari dock widget: `napari_viewer` is auto-injected by napari's
    plugin engine when opened from the Plugins menu; pass None to use
    standalone (no image-highlighting) from a script or in tests.

    `session` is the shared analysis state; when omitted it is looked up
    from the viewer, which is what makes this widget and the protocol
    builder two views of one analysis.
    """

    gate_membership_changed = Signal(str, object)  # gate id, boolean mask (np.ndarray)

    def __init__(
        self,
        napari_viewer=None,
        parent: QWidget | None = None,
        session: AnalysisSession | None = None,
        float_by_default: bool = True,
    ):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.session = session if session is not None else session_for(napari_viewer)
        self._highlight_layer = None

        root = QVBoxLayout(self)

        header = QHBoxLayout()
        self.subgate_checkbox = QCheckBox("Gate within selection")
        self.subgate_checkbox.setToolTip(
            "New gates become subgates of the selected one: their membership is "
            "restricted to their parent's."
        )
        header.addWidget(self.subgate_checkbox)
        header.addStretch()
        refresh_button = QPushButton("Refresh")
        refresh_button.setToolTip("Re-read the latest results from the protocol builder")
        refresh_button.clicked.connect(self.reload_from_session)
        header.addWidget(refresh_button)
        if self.viewer is not None:
            load_button = QPushButton("Load from active Labels layer")
            load_button.setToolTip("Use a Labels layer's own .features table instead")
            load_button.clicked.connect(self._load_from_active_layer)
            header.addWidget(load_button)
        root.addLayout(header)

        self.plot = ScatterPlotWidget()
        self.gate_manager = GateManagerWidget(self.plot, parent_id_provider=self._parent_gate_id)
        # One GateSet, owned by the session, so gates outlive this widget.
        self.gate_manager.gate_set = self.session.gate_set
        self.gate_manager.gate_selected.connect(self._on_gate_selected)
        self.gate_manager.gates_changed.connect(self._on_gates_changed)

        self.results_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.results_splitter.addWidget(self.plot)
        self.results_splitter.addWidget(self.gate_manager)
        self.results_splitter.setChildrenCollapsible(False)
        self.results_splitter.setStretchFactor(0, PLOT_WIDTH_SHARE)
        self.results_splitter.setStretchFactor(1, GATE_WIDTH_SHARE)

        # The gallery is a second view of the same selection, not a third
        # column: it needs the full width to show a useful number of crops.
        self.gallery = GalleryWidget()
        self.gallery.object_selected.connect(self._on_object_selected)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.results_splitter, "Plot")
        self.tabs.addTab(self.gallery, "Gallery")
        root.addWidget(self.tabs, 1)

        self.status_label = LogView()
        root.addWidget(self.status_label)

        # Re-read on every publish from the builder, and once now in case
        # results already exist (the usual case: this pane is opened after a
        # run, or reopened after being closed).
        self.session.data_changed.connect(self.reload_from_session)
        self.reload_from_session()

        if float_by_default:
            # The dock doesn't exist yet - napari adds this widget to one
            # after constructing it - so ask again once the event loop has
            # run.
            QTimer.singleShot(0, self.float_dock)

    # -- session ----------------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame | None:
        return self.session.results_table()

    @property
    def labels(self) -> np.ndarray | None:
        return self.session.labels()

    @property
    def gate_set(self):
        return self.gate_manager.gate_set

    @property
    def table(self):
        """The gate list. Lives in the gate manager; exposed here because it
        is part of what this pane *is*."""
        return self.gate_manager.table

    def reload_from_session(self) -> None:
        """Pull the current table and gates out of the shared session.

        Called on every publish from the builder and whenever this pane is
        shown, which is what makes closing and reopening it free.
        """
        self.gate_manager.gate_set = self.session.gate_set
        frame = self.session.results_table()
        if frame is None:
            self.gate_manager.set_frame(None)
            self.status_label.setText("No measurements yet - run a measurement step.")
            return
        # Keep the axes on screen across a re-run.
        self.plot.set_data(frame, self.plot.x_column, self.plot.y_column)
        self.gate_manager.set_frame(frame)
        self.status_label.setText(f"{len(frame)} objects, {len(frame.columns)} features.")

    def set_data(self, frame: pd.DataFrame, labels: np.ndarray | None = None) -> None:
        """Load a table directly, bypassing the protocol builder - used by
        the Labels-layer button, and by scripts driving this widget alone.

        Unlike a re-run, which keeps the gates (they are drawn on features
        that still exist), this is a different dataset, so the gates go: a
        polygon over one image's populations means nothing over another's.
        """
        context = dict(self.session.context)
        context["measurements"] = frame
        if labels is not None:
            context["labels"] = labels
        self.gate_manager.clear_gates()
        self.session.set_gate_set(self.gate_manager.gate_set)
        self.session.set_context(context, frame)

    def showEvent(self, event):  # Qt's spelling
        super().showEvent(event)
        # Hiding a dock (napari's Window menu, or the dock's close button)
        # doesn't destroy the widget, but the session may have moved on
        # while it was hidden.
        self.reload_from_session()

    def float_dock(self) -> bool:
        """Float the QDockWidget napari put this widget in, and give it a
        usable size. Returns whether a dock was found to float."""
        dock = self._dock_widget()
        if dock is None:
            return False
        dock.setFloating(True)
        dock.resize(*DEFAULT_FLOATING_SIZE)
        return True

    def _dock_widget(self):
        from qtpy.QtWidgets import QDockWidget

        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QDockWidget):
                return parent
            parent = parent.parentWidget()
        return None

    def _load_from_active_layer(self) -> None:
        if self.viewer is None:
            return
        layer = self.viewer.layers.selection.active
        if (
            layer is None
            or not hasattr(layer, "features")
            or layer.features is None
            or layer.features.empty
        ):
            self.status_label.setText("Select a Labels layer with a `.features` table first.")
            return
        self.set_data(layer.features, labels=np.asarray(layer.data))

    # -- gates ------------------------------------------------------------

    def _parent_gate_id(self) -> str | None:
        """Which gate a newly drawn one should be a subgate of - asked by the
        gate manager at the moment it creates the Gate."""
        selected = self.gate_manager.selected_gate_id
        return selected if (self.subgate_checkbox.isChecked() and selected) else None

    def _on_gates_changed(self) -> None:
        self.session.set_gate_set(self.gate_manager.gate_set)

    def _on_gate_selected(self, gate_id: str) -> None:
        if self.frame is None or gate_id not in self.gate_set:
            return
        gate = self.gate_set.get(gate_id)
        if gate.x_axis not in self.frame.columns or gate.y_axis not in self.frame.columns:
            return
        mask = self.gate_set.mask(gate_id, self.frame)
        self.gate_membership_changed.emit(gate_id, mask)
        gated_ids = self.frame.loc[mask, "object_id"].to_numpy()
        self._highlight(gated_ids)
        self._fill_gallery(gated_ids)
        self.status_label.setText(f"{gate.name}: {len(gated_ids)} of {len(self.frame)} objects.")

    def _highlight(self, gated_ids: np.ndarray) -> None:
        labels = self.labels
        if self.viewer is None or labels is None:
            return
        highlighted = np.where(np.isin(labels, gated_ids), labels, 0)
        if self._highlight_layer is not None and self._highlight_layer in self.viewer.layers:
            self._highlight_layer.data = highlighted
        else:
            self._highlight_layer = self.viewer.add_labels(
                highlighted, name=HIGHLIGHT_LAYER_NAME
            )

    def _fill_gallery(self, gated_ids: np.ndarray) -> None:
        intensity = self.session.intensity()
        if intensity is None or self.frame is None:
            return
        # The gallery crops in (row, col), so a channel axis has to go first.
        channel_axis = self.session.channel_axis
        if channel_axis is not None and intensity.ndim > 2 and channel_axis < intensity.ndim:
            intensity = np.take(intensity, 0, axis=channel_axis)
        self.gallery.show_objects(intensity, self.frame, gated_ids)

    def _on_object_selected(self, object_id: int) -> None:
        self.status_label.setText(f"Object {object_id} selected.")
        self._highlight(np.array([object_id]))
