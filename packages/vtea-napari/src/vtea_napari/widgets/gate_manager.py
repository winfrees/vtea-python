"""The gate manager: draw, list, persist and summarize gates over a scatter
plot.

Sits beside the protocol builder's plot and owns the `vtea_core.gates.GateSet`
drawn on it. What the Java UI split across MicroExplorer's toolbar, the
"Gate Management" TableWindow and a set of ad-hoc dialogs is one pane here:

- Rectangle / Polygon buttons put the plot into the matching drawing mode.
- The gate table lists every gate with its axes and counts.
- Clear / Save / Open persist the set as plain JSON
  (`vtea_core.gates.io`), so the gates behind a figure can be re-opened and
  archived with it.
- The statistics box answers what a gate is drawn to ask: how many objects
  are in it, and what their mean is on each plotted axis.

The widget never computes membership itself - GateSet does, including
parent-gate chaining - so a gate means the same thing here, in a script, and
in a re-opened JSON file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from vtea_core.gates import Gate, GateSet, load_gates, save_gates

from vtea_napari.widgets.gate_table import GateTableWidget
from vtea_napari.widgets.plot import POLYGON_MODE, RECTANGLE_MODE

GATE_FILE_FILTER = "VTEA gates (*.json);;All files (*)"


class GateManagerWidget(QWidget):
    """Gate drawing controls, the gate list, JSON persistence, and per-gate
    statistics for one ScatterPlotWidget.

    `plot` is the scatter plot to draw on; the manager connects to its
    `gate_drawn` signal and pushes drawing modes and overlays back to it.
    """

    gate_selected = Signal(str)  # gate id
    gates_changed = Signal()

    def __init__(self, plot, parent: QWidget | None = None, parent_id_provider=None):
        super().__init__(parent)
        self.plot = plot
        self.gate_set = GateSet()
        self.frame: pd.DataFrame | None = None
        self.selected_gate_id: str | None = None
        # Asked, at the moment a gate is drawn, which gate the new one
        # should be a subgate of - the Object Explorer's "Gate within
        # selection" checkbox. A provider rather than a signal because both
        # this widget and the explorer listen to the plot's gate_drawn, and
        # depending on which slot Qt happens to call first would be a bug
        # waiting to happen.
        self._parent_id_provider = parent_id_provider or (lambda: None)
        self._next_gate_number = 1

        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)

        heading = QLabel("Gates")
        heading.setStyleSheet("font-weight: bold;")
        root.addWidget(heading)

        draw_row = QHBoxLayout()
        self.rectangle_button = QPushButton("Rectangle")
        self.rectangle_button.setToolTip("Click two opposite corners on the plot")
        self.rectangle_button.setCheckable(True)
        self.rectangle_button.clicked.connect(lambda: self.set_gate_mode(RECTANGLE_MODE))
        self.polygon_button = QPushButton("Polygon")
        self.polygon_button.setToolTip(
            "Click to add vertices, double-click to close, right-click to cancel"
        )
        self.polygon_button.setCheckable(True)
        self.polygon_button.setChecked(True)
        self.polygon_button.clicked.connect(lambda: self.set_gate_mode(POLYGON_MODE))
        draw_row.addWidget(self.rectangle_button)
        draw_row.addWidget(self.polygon_button)
        root.addLayout(draw_row)

        self.table = GateTableWidget()
        self.table.gate_selected.connect(self._on_gate_selected)
        self.table.gate_visibility_changed.connect(self._on_gate_visibility_changed)
        self.table.gate_renamed.connect(self._on_gate_renamed)
        root.addWidget(self.table, 1)

        file_row = QHBoxLayout()
        self.delete_button = QPushButton("Delete")
        self.delete_button.setToolTip("Delete the selected gate (and its subgates)")
        self.delete_button.clicked.connect(self.delete_selected_gate)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Remove every gate")
        self.clear_button.clicked.connect(self.clear_gates)
        self.save_button = QPushButton("Save…")
        self.save_button.clicked.connect(self.save_gates_dialog)
        self.open_button = QPushButton("Open…")
        self.open_button.clicked.connect(self.open_gates_dialog)
        for button in (self.delete_button, self.clear_button, self.save_button, self.open_button):
            file_row.addWidget(button)
        root.addLayout(file_row)

        stats_box = QGroupBox("Selected gate")
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setContentsMargins(4, 4, 4, 4)
        self.stats_label = QLabel("No gate selected.")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        root.addWidget(stats_box)

        self.plot.gate_drawn.connect(self.add_gate_from_vertices)
        self.plot.axes_changed.connect(lambda *_: self.refresh())

    # -- data -------------------------------------------------------------

    def set_frame(self, frame: pd.DataFrame | None) -> None:
        """Point the gates at a measurement table. Existing gates are kept:
        re-running a step produces a new table for the same objects, and
        throwing the gates away on every run would make them unusable."""
        self.frame = frame
        self.refresh()

    def set_gate_mode(self, mode: str) -> None:
        self.plot.set_gate_mode(mode)
        self.rectangle_button.setChecked(mode == RECTANGLE_MODE)
        self.polygon_button.setChecked(mode == POLYGON_MODE)

    # -- gates ------------------------------------------------------------

    def add_gate_from_vertices(self, vertices: np.ndarray, parent_id: str | None = None) -> Gate | None:
        """Turn a shape drawn on the plot into a named gate on the current axes."""
        if parent_id is None:
            parent_id = self._parent_id_provider()
        if parent_id is not None and parent_id not in self.gate_set:
            parent_id = None
        if self.plot.x_column is None or self.plot.y_column is None:
            return None
        gate = Gate(
            name=f"gate{self._next_gate_number}",
            x_axis=self.plot.x_column,
            y_axis=self.plot.y_column,
            vertices=np.asarray(vertices, dtype=float),
            parent_id=parent_id,
        )
        self._next_gate_number += 1
        self.gate_set.add(gate)
        self.selected_gate_id = gate.id
        self.refresh()
        self.gates_changed.emit()
        return gate

    def delete_selected_gate(self) -> None:
        if self.selected_gate_id is None or self.selected_gate_id not in self.gate_set:
            return
        self.gate_set.remove(self.selected_gate_id)
        self.selected_gate_id = None
        self.refresh()
        self.gates_changed.emit()

    def clear_gates(self) -> None:
        self.gate_set = GateSet()
        self.selected_gate_id = None
        self._next_gate_number = 1
        self.refresh()
        self.gates_changed.emit()

    # -- persistence ------------------------------------------------------

    def save_gates_to(self, path) -> None:
        save_gates(self.gate_set, path)

    def load_gates_from(self, path) -> None:
        self.gate_set = load_gates(path)
        self.selected_gate_id = None
        # Keep generated names from colliding with the ones just loaded.
        self._next_gate_number = len(self.gate_set) + 1
        self.refresh()
        self.gates_changed.emit()

    def save_gates_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save gates", "gates.json", GATE_FILE_FILTER)
        if not path:
            return
        self.save_gates_to(path)
        self.stats_label.setText(f"Saved {len(self.gate_set)} gate(s).")

    def open_gates_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open gates", "", GATE_FILE_FILTER)
        if not path:
            return
        try:
            self.load_gates_from(path)
        except (OSError, ValueError, KeyError) as exc:  # report, don't crash napari
            self.stats_label.setText(f"{type(exc).__name__}: {exc}")

    # -- views ------------------------------------------------------------

    def refresh(self) -> None:
        frame = self.frame if self.frame is not None else pd.DataFrame()
        self.table.refresh(self.gate_set, frame)
        self.plot.set_gate_overlays(list(self.gate_set))
        self._refresh_statistics()

    def _refresh_statistics(self) -> None:
        if self.selected_gate_id is None or self.selected_gate_id not in self.gate_set:
            self.stats_label.setText("No gate selected.")
            return
        gate = self.gate_set.get(self.selected_gate_id)
        if self.frame is None or self.frame.empty:
            self.stats_label.setText(f"{gate.name}: no measurements to gate yet.")
            return
        if gate.x_axis not in self.frame.columns or gate.y_axis not in self.frame.columns:
            # A gate reopened from JSON, or drawn before a re-run changed
            # which features exist. It is still listed; it just can't be
            # counted against this table.
            self.stats_label.setText(
                f"{gate.name}: '{gate.x_axis}' / '{gate.y_axis}' are not in the current data."
            )
            return
        # The axes on screen, not the ones the gate was drawn on, so the
        # numbers describe the plot the user is looking at.
        columns = [
            column
            for column in (self.plot.x_column, self.plot.y_column)
            if column is not None
        ] or [gate.x_axis, gate.y_axis]
        stats = self.gate_set.statistics(self.selected_gate_id, self.frame, columns)
        lines = [
            f"<b>{gate.name}</b>",
            f"{stats['n_gated']} of {stats['n_total']} cells ({stats['percent']:.1f}%)",
        ]
        lines += [f"mean {column} = {value:.4g}" for column, value in stats["means"].items()]
        self.stats_label.setText("<br>".join(lines))

    def _on_gate_selected(self, gate_id: str) -> None:
        self.selected_gate_id = gate_id
        self._refresh_statistics()
        self.gate_selected.emit(gate_id)

    def _on_gate_visibility_changed(self, gate_id: str, visible: bool) -> None:
        self.gate_set.get(gate_id).visible = visible
        self.plot.set_gate_overlays(list(self.gate_set))

    def _on_gate_renamed(self, gate_id: str, name: str) -> None:
        self.gate_set.get(gate_id).name = name
        self.refresh()
