import numpy as np
import pandas as pd
import pytest

from vtea_core.gates import Gate, GateSet


def make_frame():
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3, 4],
            "x": [2, 7, 12, 8],
            "y": [2, 7, 12, 3],
        }
    )


class TestGateSet:
    def test_add_and_mask(self):
        gates = GateSet()
        square = Gate(name="square1", x_axis="x", y_axis="y", vertices=np.array([[0, 0], [0, 10], [10, 10], [10, 0]]))
        gates.add(square)
        mask = gates.mask(square.id, make_frame())
        np.testing.assert_array_equal(mask, [True, True, False, True])

    def test_summary_counts(self):
        gates = GateSet()
        square = Gate(name="square1", x_axis="x", y_axis="y", vertices=np.array([[0, 0], [0, 10], [10, 10], [10, 0]]))
        gates.add(square)
        summary = gates.summary(square.id, make_frame())
        assert summary == {"n_gated": 3, "n_total": 4, "percent": pytest.approx(75.0)}

    def test_child_gate_is_intersected_with_parent(self):
        gates = GateSet()
        parent = Gate(name="parent", x_axis="x", y_axis="y", vertices=np.array([[0, 0], [0, 10], [10, 10], [10, 0]]))
        gates.add(parent)
        # child covers everything, but should still be restricted to parent's members
        child = Gate(
            name="child",
            x_axis="x",
            y_axis="y",
            vertices=np.array([[-100, -100], [-100, 100], [100, 100], [100, -100]]),
            parent_id=parent.id,
        )
        gates.add(child)
        mask = gates.mask(child.id, make_frame())
        np.testing.assert_array_equal(mask, gates.mask(parent.id, make_frame()))

    def test_adding_gate_with_unknown_parent_raises(self):
        gates = GateSet()
        orphan = Gate(name="orphan", x_axis="x", y_axis="y", vertices=np.zeros((3, 2)), parent_id="nonexistent")
        with pytest.raises(KeyError):
            gates.add(orphan)

    def test_children(self):
        gates = GateSet()
        parent = Gate(name="parent", x_axis="x", y_axis="y", vertices=np.zeros((3, 2)))
        gates.add(parent)
        child = Gate(name="child", x_axis="x", y_axis="y", vertices=np.zeros((3, 2)), parent_id=parent.id)
        gates.add(child)
        assert [g.id for g in gates.children(parent.id)] == [child.id]

    def test_remove_cascades_to_children(self):
        gates = GateSet()
        parent = Gate(name="parent", x_axis="x", y_axis="y", vertices=np.zeros((3, 2)))
        gates.add(parent)
        child = Gate(name="child", x_axis="x", y_axis="y", vertices=np.zeros((3, 2)), parent_id=parent.id)
        gates.add(child)
        gates.remove(parent.id)
        assert len(gates) == 0
        assert child.id not in gates

    def test_iteration_preserves_insertion_order(self):
        gates = GateSet()
        first = gates.add(Gate(name="a", x_axis="x", y_axis="y", vertices=np.zeros((3, 2))))
        second = gates.add(Gate(name="b", x_axis="x", y_axis="y", vertices=np.zeros((3, 2))))
        assert [g.id for g in gates] == [first.id, second.id]
