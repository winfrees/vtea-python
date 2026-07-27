"""The gate list/management table: visibility, color, name, axes, and
per-gate object counts.

Replaces vteaexploration.TableWindow ("Gate Management" - a JTable with
View/Color/Name/XAxis/YAxis/Gated/Total/% columns editable in place).
vteaexploration.GateManager.java and microGateManager.java, despite the
similar names, aren't ported: GateManager is instantiated once and never
shown or populated anywhere in the Java codebase, and microGateManager is
an ~2800-line near-verbatim fork of ImageJ's RoiManager operating on
ij.gui.Roi (not VTEA's gate model at all) that's likewise never
instantiated - both are dead code, confirmed by grep across the codebase.
"""

from __future__ import annotations

import pandas as pd
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem

from vtea_core.gates import GateSet

_COLUMNS = ["Visible", "Color", "Name", "X axis", "Y axis", "Gated", "Total", "%"]
_VISIBLE_COL, _COLOR_COL, _NAME_COL = 0, 1, 2


class GateTableWidget(QTableWidget):
    """Emits gate_selected/gate_visibility_changed/gate_renamed; the parent
    widget owns the GateSet and applies the change, then calls refresh()."""

    gate_selected = Signal(str)
    gate_visibility_changed = Signal(str, bool)
    gate_renamed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(0, len(_COLUMNS), parent)
        self.setHorizontalHeaderLabels(_COLUMNS)
        self._gate_ids: list[str] = []
        self.cellClicked.connect(self._on_cell_clicked)
        self.itemChanged.connect(self._on_item_changed)

    def refresh(self, gate_set: GateSet, frame: pd.DataFrame) -> None:
        self.blockSignals(True)
        self.setRowCount(0)
        self._gate_ids = []
        for gate in gate_set:
            row = self.rowCount()
            self.insertRow(row)
            self._gate_ids.append(gate.id)

            visible_item = QTableWidgetItem()
            visible_item.setFlags(
                (visible_item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable
            )
            visible_item.setCheckState(Qt.CheckState.Checked if gate.visible else Qt.CheckState.Unchecked)
            self.setItem(row, _VISIBLE_COL, visible_item)

            color_item = QTableWidgetItem()
            color_item.setBackground(QColor(gate.color))
            color_item.setFlags(color_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row, _COLOR_COL, color_item)

            self.setItem(row, _NAME_COL, QTableWidgetItem(gate.name))

            summary = gate_set.summary(gate.id, frame)
            for column, text in (
                (3, gate.x_axis),
                (4, gate.y_axis),
                (5, str(summary["n_gated"])),
                (6, str(summary["n_total"])),
                (7, f"{summary['percent']:.1f}"),
            ):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.setItem(row, column, item)
        self.blockSignals(False)

    def _on_cell_clicked(self, row: int, column: int) -> None:
        if column == _VISIBLE_COL:
            return  # handled by _on_item_changed, to react to the actual check-state flip
        self.gate_selected.emit(self._gate_ids[row])

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        gate_id = self._gate_ids[item.row()]
        if item.column() == _VISIBLE_COL:
            self.gate_visibility_changed.emit(gate_id, item.checkState() == Qt.CheckState.Checked)
        elif item.column() == _NAME_COL:
            self.gate_renamed.emit(gate_id, item.text())
