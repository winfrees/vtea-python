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
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from vtea_core.gates import SEAM_GATE_COLOR, has_seam_columns, seam_gate

from vtea_napari.session import AnalysisSession, session_for
from vtea_napari.widgets.association_review import AssociationReviewWidget
from vtea_napari.widgets.gallery import GalleryWidget
from vtea_napari.widgets.gate_manager import GateManagerWidget
from vtea_napari.widgets.log_view import LogView
from vtea_napari.widgets.plot import ScatterPlotWidget
from vtea_napari.widgets.plot_style import PlotStylePanel
from vtea_napari.widgets.seam_review import SeamReviewWidget

# How many slices either side of an object's centroid the gallery projects.
# Wide enough to hold a nucleus at ordinary z-steps, and bounded so the cost
# of a thumbnail does not grow with the depth of the acquisition.
GALLERY_Z_RADIUS = 8

# What the seam tab is called, and where it sits: after the review panes a
# user reaches for on every run, since most runs have no seams at all.
SEAM_TAB_NAME = "Seams"

# The plot is the point of this pane; the gate manager beside it is
# controls. Same 2:1 split the protocol builder used before this moved here.
PLOT_WIDTH_SHARE = 2
GATE_WIDTH_SHARE = 1

# Big enough that the axes and the gate table are both usable when it first
# appears floating over the canvas.
DEFAULT_FLOATING_SIZE = (900, 560)

HIGHLIGHT_LAYER_NAME = "Gate highlight"
REVIEW_LAYER_NAME = "Reviewing"


def _solid_label_colormap(label_ids, color: str):
    """Paint every gated object in one colour - the gate's - instead of
    napari's per-label palette, so a gate reads the same on the image as it
    does on the plot. None when this napari has no direct-colour support."""
    try:
        from napari.utils.colormaps import DirectLabelColormap
    except ImportError:
        return None
    color_dict = {int(label): color for label in np.unique(label_ids) if int(label) != 0}
    color_dict[None] = "transparent"
    try:
        return DirectLabelColormap(color_dict=color_dict)
    except (TypeError, ValueError):
        return None


def _apply_gate_color(layer, colormap) -> bool:
    """Paint a highlight layer in its gate's colour, if this napari can.

    Applied to an already-added layer rather than passed to `add_labels`, so
    a failure leaves a working highlight in napari's own label colours
    instead of a half-added layer. It can fail for reasons that have nothing
    to do with the data - building the colour texture needs a GL context,
    which a headless session may not have - and a highlight that is the
    wrong colour is much better than a pane that falls over.
    """
    if colormap is None:
        return False
    try:
        layer.colormap = colormap
    except Exception:  # noqa: BLE001 - display-only; see the docstring
        return False
    return True


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
        # True while the pane is loading a table and restoring the saved
        # view, so the defaults picked on the way through don't overwrite
        # the view being restored.
        self._restoring = False
        # gate id -> (layer, signature); the signature is what the layer
        # was built from, so it is only rebuilt when that changes.
        self._highlight_layers: dict[str, tuple] = {}

        root = QVBoxLayout(self)

        header = QHBoxLayout()
        # Which table is being plotted. A protocol that builds cells has two
        # - its objects and its cells - and they are not two views of one
        # thing: their rows are different, so the axes, the gates and the
        # highlighted image all change with the choice.
        self.table_combo = QComboBox()
        self.table_combo.setToolTip("Which table to plot: the objects, or the cells built from them")
        self.table_combo.currentTextChanged.connect(self._on_table_changed)
        self.table_label = QLabel("Table:")
        header.addWidget(self.table_label)
        header.addWidget(self.table_combo)
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

        # The style pane sits under the plot rather than beside it: it is
        # about how the points are drawn, and it belongs to the plot the way
        # the axis pickers above it do.
        self.style_panel = PlotStylePanel(self.plot)
        plot_pane = QWidget()
        plot_layout = QVBoxLayout(plot_pane)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.plot, 1)
        plot_layout.addWidget(self.style_panel)

        self.results_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.results_splitter.addWidget(plot_pane)
        self.results_splitter.addWidget(self.gate_manager)
        self.results_splitter.setChildrenCollapsible(False)
        self.results_splitter.setStretchFactor(0, PLOT_WIDTH_SHARE)
        self.results_splitter.setStretchFactor(1, GATE_WIDTH_SHARE)

        # The gallery is a second view of the same selection, not a third
        # column: it needs the full width to show a useful number of crops.
        self.gallery = GalleryWidget()
        self.gallery.object_selected.connect(self._on_object_selected)
        # Reviewing the association is a third view of the same analysis, not
        # a third column: the few percent of links worth a person's attention
        # need the width to show what the alternatives were.
        self.review = AssociationReviewWidget()
        self.review.link_selected.connect(self._on_link_selected)
        self.review.associations_changed.connect(self._on_associations_changed)

        # And a fourth: the objects a tile boundary cut, when the run was a
        # blocked one. The tab only exists when there is something in it -
        # an in-memory run has no seams, and a permanently empty tab reads
        # as a broken feature rather than an absent condition.
        self.seam_review = SeamReviewWidget()
        self.seam_review.object_selected.connect(self._on_object_selected)
        self.seam_review.objects_rejected.connect(self._on_objects_rejected)
        self.seam_review.gate_requested.connect(self.add_seam_gate)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.results_splitter, "Plot")
        self.tabs.addTab(self.gallery, "Gallery")
        self.tabs.addTab(self.review, "Associations")
        root.addWidget(self.tabs, 1)

        self.status_label = LogView()
        root.addWidget(self.status_label)

        # Re-read on every publish from the builder, and once now in case
        # results already exist (the usual case: this pane is opened after a
        # run, or reopened after being closed).
        self.session.data_changed.connect(self.reload_from_session)
        self.plot.view_changed.connect(self._remember_view)
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
    def id_column(self) -> str:
        """What names a row of the current table - `object_id` for objects,
        `cell_id` for cells. Everything that maps a plotted point back to the
        image goes through this rather than assuming one of them."""
        return self.session.id_column()

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
        """Pull the current table, gates and view out of the shared session.

        Called on every publish from the builder and whenever this pane is
        shown, which is what makes closing and reopening it free. The view is
        restored at the end rather than in the constructor because loading a
        table picks default axes on the way through, and those defaults would
        otherwise be remembered over the view being restored.
        """
        self._restoring = True
        try:
            self._refresh_table_choices()
            self.review.set_sources(self.session.associations())
            self.seam_review.set_source(self.session.results_table(), self.session.ledger)
            self._refresh_seam_tab()
            self.gate_manager.gate_set = self.session.gate_set
            frame = self.session.results_table()
            if frame is None:
                self.gate_manager.set_frame(None)
                self.status_label.setText("No measurements yet - run a measurement step.")
                return
            # Keep the axes on screen across a re-run.
            self.plot.set_data(frame, self.plot.x_column, self.plot.y_column)
            self.gate_manager.set_frame(frame)
            self.refresh_highlights()
            self.plot.apply_view_state(self.session.view_state)
            self.style_panel.read_from_plot()
            noun = self.session.row_noun()
            self.status_label.setText(f"{len(frame)} {noun}, {len(frame.columns)} features.")
        finally:
            self._restoring = False

    def _on_link_selected(self, child, parent) -> None:
        """Show the two objects a contested link is between.

        Reading a posterior off a table is not how anybody decides which
        nucleus a cytoplasm belongs to - looking at them is - so selecting a
        row puts both on the image.
        """
        self.status_label.setText(f"{child} -> {parent}")
        if self.viewer is None:
            return
        for ref, colour in ((child, "yellow"), (parent, "cyan")):
            labels = self.session.context.get(ref.segmentation)
            if not isinstance(labels, np.ndarray):
                continue
            name = f"{REVIEW_LAYER_NAME}: {ref.segmentation}"
            for existing in list(self.viewer.layers):
                if existing.name == name:
                    self.viewer.layers.remove(existing)
            layer = self.viewer.add_labels(
                np.where(labels == ref.object_id, labels, 0).astype(np.int32), name=name
            )
            _apply_gate_color(layer, _solid_label_colormap([ref.object_id], colour))

    def _refresh_seam_tab(self) -> None:
        """Show the seam tab exactly when the current table has seams."""
        frame = self.session.results_table()
        wanted = frame is not None and has_seam_columns(frame)
        index = self.tabs.indexOf(self.seam_review)
        if wanted and index < 0:
            self.tabs.addTab(self.seam_review, SEAM_TAB_NAME)
        elif not wanted and index >= 0:
            self.tabs.removeTab(index)
            # removeTab reparents the page to nothing but keeps it alive, so
            # the same widget goes back on the next blocked run with its
            # threshold and its selection intact.
            self.seam_review.setParent(self)
            self.seam_review.hide()

    def _on_objects_rejected(self, object_id) -> None:
        """A reviewer excluded a seam object. Recorded on the ledger by the
        review pane itself; here it only has to be said out loud."""
        ledger = self.session.ledger
        excluded = len(ledger.dropped) if ledger is not None else 0
        self.status_label.setText(
            f"Object {object_id} excluded; {excluded} excluded in total."
        )

    def add_seam_gate(self, threshold: float) -> object:
        """Put the seam objects on the plot as an ordinary gate.

        Switches the axes to the pair the gate is drawn over, because a gate
        whose outline is invisible on the axes in front of you is indistinguishable
        from one that was never added.
        """
        frame = self.frame
        if frame is None or not has_seam_columns(frame):
            return None
        gate = seam_gate(frame, threshold=threshold, parent_id=self._parent_gate_id())
        gate.color = SEAM_GATE_COLOR
        self.gate_manager.gate_set.add(gate)
        self.gate_manager.selected_gate_id = gate.id
        self.plot.set_data(frame, gate.x_axis, gate.y_axis)
        self.gate_manager.refresh()
        self._on_gates_changed()
        self._on_gate_selected(gate.id)
        self.tabs.setCurrentWidget(self.results_splitter)
        return gate

    def _on_associations_changed(self) -> None:
        """A hand-made decision: remember it on the session so re-running the
        association step corrects the automated answers without undoing it."""
        associations = self.review.associations
        if associations is None:
            return
        for child in associations.edited_by_hand:
            self.session.record_manual_link(child, associations.parent_of(child))
        self.status_label.setText(associations.summary())

    def _refresh_table_choices(self) -> None:
        names = self.session.table_names()
        # One table is the ordinary case; a picker offering a single choice
        # is furniture, so it only appears once there is a choice to make.
        visible = len(names) > 1
        self.table_combo.setVisible(visible)
        self.table_label.setVisible(visible)
        if names == [self.table_combo.itemText(i) for i in range(self.table_combo.count())]:
            self.table_combo.setCurrentText(self.session.active_table)
            return
        self.table_combo.blockSignals(True)
        self.table_combo.clear()
        self.table_combo.addItems(names)
        self.table_combo.setCurrentText(self.session.active_table)
        self.table_combo.blockSignals(False)

    def _on_table_changed(self, name: str) -> None:
        if self._restoring or not name:
            return
        # The gates belong to the table, so switching hides the ones drawn on
        # the other and shows this table's own - including on the image.
        self.session.set_active_table(name)

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

    def _remember_view(self) -> None:
        if self._restoring:
            return
        self.session.remember_view(self.plot.view_state())

    def _on_gates_changed(self) -> None:
        self.session.set_gate_set(self.gate_manager.gate_set)
        self.refresh_highlights()

    def _on_gate_selected(self, gate_id: str) -> None:
        if self.frame is None or gate_id not in self.gate_set:
            return
        gate = self.gate_set.get(gate_id)
        if gate.x_axis not in self.frame.columns or gate.y_axis not in self.frame.columns:
            return
        mask = self.gate_set.mask(gate_id, self.frame)
        self.gate_membership_changed.emit(gate_id, mask)
        gated_ids = self.frame.loc[mask, self.id_column].to_numpy()
        self.refresh_highlights()
        self._fill_gallery(gated_ids)
        self.status_label.setText(f"{gate.name}: {len(gated_ids)} of {len(self.frame)} objects.")

    def refresh_highlights(self) -> None:
        """One napari Labels layer per gate, painted in that gate's colour
        and shown only while the gate is visible.

        A gate's colour is how it is identified on the plot, so the objects
        it selects should carry the same colour on the image - otherwise
        reading two gates against each other means holding a mapping in your
        head. Unchecking a gate's Visible box hides its layer, which is the
        same gesture that hides its outline.

        A layer is rebuilt rather than re-filled when its membership
        changes: the gate's colour is applied at creation, and reassigning
        `.data` afterwards makes napari rebuild the colour texture, which is
        an operation that can fail for reasons unrelated to the data. Only
        visibility, which touches nothing else, is updated in place.
        """
        if self.viewer is None:
            return
        labels = self.labels
        frame = self.frame
        for gate in self.gate_set:
            usable = (
                labels is not None
                and frame is not None
                and gate.x_axis in frame.columns
                and gate.y_axis in frame.columns
            )
            if not usable:
                self._remove_highlight(gate.id)
                continue
            mask = self.gate_set.mask(gate.id, frame)
            gated_ids = frame.loc[mask, self.id_column].to_numpy()
            name = f"{HIGHLIGHT_LAYER_NAME}: {gate.name}"
            signature = (name, gate.color, frozenset(int(i) for i in gated_ids))

            layer, previous = self._highlight_layers.get(gate.id, (None, None))
            if layer is None or layer not in self.viewer.layers or previous != signature:
                self._remove_highlight(gate.id)
                data = np.where(np.isin(labels, gated_ids), labels, 0)
                layer = self.viewer.add_labels(data, name=name)
                _apply_gate_color(layer, _solid_label_colormap(gated_ids, gate.color))
                self._highlight_layers[gate.id] = (layer, signature)
            layer.visible = gate.visible

        for gate_id in [gid for gid in self._highlight_layers if gid not in self.gate_set]:
            self._remove_highlight(gate_id)

    def _remove_highlight(self, gate_id: str) -> None:
        layer, _signature = self._highlight_layers.pop(gate_id, (None, None))
        if layer is not None and self.viewer is not None and layer in self.viewer.layers:
            self.viewer.layers.remove(layer)

    def _fill_gallery(self, gated_ids: np.ndarray) -> None:
        intensity = self.session.intensity()
        if intensity is None or self.frame is None:
            return
        # The gallery crops in (row, col), so a channel axis has to go first.
        # Sliced rather than np.take: `take` on a Zarr array reads the whole
        # channel into memory, which for a stored volume is the one thing
        # the gallery is careful not to do. A plain index works lazily on
        # Zarr and Dask alike and is the same operation on NumPy.
        channel_axis = self.session.channel_axis
        ndim = len(intensity.shape)
        if channel_axis is not None and ndim > 2 and channel_axis < ndim:
            intensity = intensity[(slice(None),) * channel_axis + (0,)]
        view = self.session.table_view()
        # A per-cell table's centroids are namespaced by the segmentation
        # they were measured on, so crop around the one the cells are rooted
        # on - the same objects their ids name.
        prefix = f"{view.labels_key}." if view is not None and view.id_column != "object_id" else ""
        self.gallery.show_objects(
            intensity,
            self.frame,
            gated_ids,
            id_column=self.id_column,
            prefix=prefix,
            # Crop the depth as well as the plane. Projecting every slice of
            # a thousand-slice stack is a reduction over the volume rather
            # than a thumbnail of an object, and the object is a few slices
            # deep.
            z_radius=GALLERY_Z_RADIUS,
        )

    def _on_object_selected(self, object_id: int) -> None:
        """A gallery crop was clicked: outline it there, and show only that
        object on the image."""
        self.status_label.setText(f"Object {object_id} selected.")
