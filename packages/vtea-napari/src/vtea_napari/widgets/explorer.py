"""The Object Explorer dock widget: scatter plot + gate table + image
highlighting, the vtea-python equivalent of vteaexploration.MicroExplorer.

Owns a vtea_core.gates.GateSet against a measurement DataFrame (typically a
napari Labels layer's `.features`, the idiomatic napari place for a
per-label table - no custom SQL-backed store needed, unlike
vtea.jdbc.H2DatabaseEngine). Wires ScatterPlotWidget's gate_drawn signal to
GateSet.add() and GateTableWidget's edits back onto the same GateSet,
matching what MicroExplorer/XYExplorationPanel/TableWindow did together in
Java but as three cooperating pieces connected by four Qt signals instead
of that subsystem's ~25 single-method listener interfaces.

"Subgating" (vtea's SubGateListener, which opened a whole new MicroExplorer
window over a pre-filtered dataset) is real gate hierarchy here instead:
check "Gate within selection", select a gate, then draw - new gates get
that gate as their parent_id and GateSet already restricts a child's
membership to its parent's (see vtea_core.gates.gate).

Selecting a gate highlights its members as a napari Labels layer (only the
gated object ids kept, background elsewhere) - the closest napari-native
analog of vtea's colorized ImagePlus overlay repaint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from vtea_core.gates import Gate, GateSet

from vtea_napari.widgets.gate_table import GateTableWidget
from vtea_napari.widgets.plot import ScatterPlotWidget

_GATE_COUNTER_START = 1


class ExplorerWidget(QWidget):
    """A napari dock widget: `napari_viewer` is auto-injected by napari's
    plugin engine when opened from the Plugins menu; pass None to use
    standalone (no image-highlighting) from a script or in tests."""

    gate_membership_changed = Signal(str, object)  # gate id, boolean mask (np.ndarray)

    def __init__(self, napari_viewer=None, parent: QWidget | None = None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.gate_set = GateSet()
        self.frame: pd.DataFrame | None = None
        self.labels: np.ndarray | None = None
        self._selected_gate_id: str | None = None
        self._next_gate_number = _GATE_COUNTER_START
        self._highlight_layer = None

        root = QVBoxLayout(self)

        if self.viewer is not None:
            load_row = QHBoxLayout()
            load_button = QPushButton("Load from active Labels layer")
            load_button.clicked.connect(self._load_from_active_layer)
            load_row.addWidget(load_button)
            root.addLayout(load_row)

        self.plot = ScatterPlotWidget()
        self.plot.gate_drawn.connect(self._on_gate_drawn)
        self.plot.axes_changed.connect(lambda *_: self.plot.set_gate_overlays(list(self.gate_set)))
        root.addWidget(self.plot)

        subgate_row = QHBoxLayout()
        self.subgate_checkbox = QCheckBox("Gate within selection")
        subgate_row.addWidget(self.subgate_checkbox)
        subgate_row.addStretch()
        root.addLayout(subgate_row)

        self.table = GateTableWidget()
        self.table.gate_selected.connect(self._on_gate_selected)
        self.table.gate_visibility_changed.connect(self._on_gate_visibility_changed)
        self.table.gate_renamed.connect(self._on_gate_renamed)
        root.addWidget(self.table)

        button_row = QHBoxLayout()
        delete_button = QPushButton("Delete selected gate")
        delete_button.clicked.connect(self._delete_selected_gate)
        button_row.addWidget(delete_button)
        button_row.addStretch()
        root.addLayout(button_row)

        self.status_label = QLabel("No data loaded.")
        root.addWidget(self.status_label)

    def set_data(self, frame: pd.DataFrame, labels: np.ndarray | None = None) -> None:
        """Loads a per-object measurement table (and optionally its source
        label mask, to enable gate -> image highlighting)."""
        self.frame = frame
        self.labels = labels
        self.gate_set = GateSet()
        self._selected_gate_id = None
        self._next_gate_number = _GATE_COUNTER_START
        self.plot.set_data(frame)
        self.table.refresh(self.gate_set, frame)
        self.status_label.setText(f"{len(frame)} objects loaded.")

    def _load_from_active_layer(self) -> None:
        if self.viewer is None:
            return
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "features") or layer.features is None or layer.features.empty:
            self.status_label.setText("Select a Labels layer with a `.features` table first.")
            return
        self.set_data(layer.features, labels=np.asarray(layer.data))

    def _on_gate_drawn(self, vertices: np.ndarray) -> None:
        if self.frame is None or self.plot.x_column is None or self.plot.y_column is None:
            return
        parent_id = self._selected_gate_id if self.subgate_checkbox.isChecked() else None
        gate = Gate(
            name=f"gate{self._next_gate_number}",
            x_axis=self.plot.x_column,
            y_axis=self.plot.y_column,
            vertices=vertices,
            parent_id=parent_id,
        )
        self._next_gate_number += 1
        self.gate_set.add(gate)
        self._refresh_views()

    def _on_gate_selected(self, gate_id: str) -> None:
        self._selected_gate_id = gate_id
        gate = self.gate_set.get(gate_id)
        self.plot.set_data(self.frame, x_column=gate.x_axis, y_column=gate.y_axis)
        self.plot.set_gate_overlays(list(self.gate_set))
        self._highlight_gate(gate_id)

    def _on_gate_visibility_changed(self, gate_id: str, visible: bool) -> None:
        self.gate_set.get(gate_id).visible = visible
        self.plot.set_gate_overlays(list(self.gate_set))

    def _on_gate_renamed(self, gate_id: str, name: str) -> None:
        self.gate_set.get(gate_id).name = name
        self.table.refresh(self.gate_set, self.frame)

    def _delete_selected_gate(self) -> None:
        if self._selected_gate_id is None or self._selected_gate_id not in self.gate_set:
            return
        self.gate_set.remove(self._selected_gate_id)
        self._selected_gate_id = None
        self._refresh_views()

    def _refresh_views(self) -> None:
        self.table.refresh(self.gate_set, self.frame)
        self.plot.set_gate_overlays(list(self.gate_set))

    def _highlight_gate(self, gate_id: str) -> None:
        mask = self.gate_set.mask(gate_id, self.frame)
        self.gate_membership_changed.emit(gate_id, mask)
        if self.viewer is None or self.labels is None:
            return
        gated_ids = self.frame.loc[mask, "object_id"].to_numpy()
        highlighted = np.where(np.isin(self.labels, gated_ids), self.labels, 0)
        gate_name = self.gate_set.get(gate_id).name
        layer_name = "Gate highlight"
        if self._highlight_layer is not None and self._highlight_layer in self.viewer.layers:
            self._highlight_layer.data = highlighted
            self._highlight_layer.name = layer_name
        else:
            self._highlight_layer = self.viewer.add_labels(highlighted, name=layer_name)
        self.status_label.setText(f"{gate_name}: {len(gated_ids)} of {len(self.frame)} objects.")
