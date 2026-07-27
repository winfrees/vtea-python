import numpy as np
import pandas as pd
from qtpy.QtCore import Qt

from vtea_core.gates import Gate, GateSet
from vtea_napari.widgets.gate_table import GateTableWidget


def make_frame():
    return pd.DataFrame({"object_id": [1, 2, 3, 4], "x": [2, 7, 12, 8], "y": [2, 7, 12, 3]})


def make_gate_set():
    gates = GateSet()
    gates.add(
        Gate(name="square1", x_axis="x", y_axis="y", vertices=np.array([[0, 0], [0, 10], [10, 10], [10, 0]]))
    )
    return gates


class TestRefresh:
    def test_one_row_per_gate(self, qtbot):
        widget = GateTableWidget()
        qtbot.addWidget(widget)
        widget.refresh(make_gate_set(), make_frame())
        assert widget.rowCount() == 1

    def test_shows_name_and_axes(self, qtbot):
        widget = GateTableWidget()
        qtbot.addWidget(widget)
        widget.refresh(make_gate_set(), make_frame())
        assert widget.item(0, 2).text() == "square1"
        assert widget.item(0, 3).text() == "x"
        assert widget.item(0, 4).text() == "y"

    def test_shows_gated_total_percent(self, qtbot):
        widget = GateTableWidget()
        qtbot.addWidget(widget)
        widget.refresh(make_gate_set(), make_frame())
        assert widget.item(0, 5).text() == "3"
        assert widget.item(0, 6).text() == "4"
        assert widget.item(0, 7).text() == "75.0"


class TestInteraction:
    def test_clicking_name_cell_emits_gate_selected(self, qtbot):
        widget = GateTableWidget()
        qtbot.addWidget(widget)
        gate_set = make_gate_set()
        gate = next(iter(gate_set))
        widget.refresh(gate_set, make_frame())

        received = []
        widget.gate_selected.connect(received.append)
        widget.cellClicked.emit(0, 2)

        assert received == [gate.id]

    def test_unchecking_visible_emits_visibility_changed(self, qtbot):
        widget = GateTableWidget()
        qtbot.addWidget(widget)
        gate_set = make_gate_set()
        gate = next(iter(gate_set))
        widget.refresh(gate_set, make_frame())

        received = []
        widget.gate_visibility_changed.connect(lambda gate_id, visible: received.append((gate_id, visible)))
        widget.item(0, 0).setCheckState(Qt.CheckState.Unchecked)

        assert received == [(gate.id, False)]

    def test_editing_name_cell_emits_gate_renamed(self, qtbot):
        widget = GateTableWidget()
        qtbot.addWidget(widget)
        gate_set = make_gate_set()
        gate = next(iter(gate_set))
        widget.refresh(gate_set, make_frame())

        received = []
        widget.gate_renamed.connect(lambda gate_id, name: received.append((gate_id, name)))
        widget.item(0, 2).setText("renamed")

        assert received == [(gate.id, "renamed")]

    def test_color_and_count_cells_are_not_editable(self, qtbot):
        widget = GateTableWidget()
        qtbot.addWidget(widget)
        widget.refresh(make_gate_set(), make_frame())
        assert not (widget.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable)
        assert not (widget.item(0, 5).flags() & Qt.ItemFlag.ItemIsEditable)
