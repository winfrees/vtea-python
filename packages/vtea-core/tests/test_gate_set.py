import numpy as np
import pandas as pd
import pytest

from vtea_core.gates import Gate, GateSet, rectangle_vertices


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


class TestStatistics:
    """What a gate is drawn to ask: how many cells, and how bright are they?"""

    @staticmethod
    def make_frame():
        return pd.DataFrame(
            {
                "object_id": [1, 2, 3, 4],
                "mean_ch0": [1.0, 2.0, 20.0, 30.0],
                "count": [10.0, 20.0, 100.0, 200.0],
            }
        )

    @staticmethod
    def make_gate_set():
        gate_set = GateSet()
        gate_set.add(
            Gate(
                name="dim",
                x_axis="mean_ch0",
                y_axis="count",
                vertices=rectangle_vertices(0, 0, 5, 50),
            )
        )
        return gate_set

    def test_counts_match_summary(self):
        frame = self.make_frame()
        gate_set = self.make_gate_set()
        gate_id = next(iter(gate_set)).id
        stats = gate_set.statistics(gate_id, frame)
        assert stats["n_gated"] == 2
        assert stats["n_total"] == 4
        assert stats["percent"] == pytest.approx(50.0)

    def test_means_default_to_the_gate_axes(self):
        frame = self.make_frame()
        gate_set = self.make_gate_set()
        gate_id = next(iter(gate_set)).id
        means = gate_set.statistics(gate_id, frame)["means"]
        assert set(means) == {"mean_ch0", "count"}
        # Only the two gated objects, not all four.
        assert means["mean_ch0"] == pytest.approx(1.5)
        assert means["count"] == pytest.approx(15.0)

    def test_explicit_columns_are_used(self):
        frame = self.make_frame()
        gate_set = self.make_gate_set()
        gate_id = next(iter(gate_set)).id
        means = gate_set.statistics(gate_id, frame, ["object_id"])["means"]
        assert means["object_id"] == pytest.approx(1.5)

    def test_an_empty_gate_gives_nan_rather_than_raising(self):
        frame = self.make_frame()
        gate_set = GateSet()
        gate = gate_set.add(
            Gate(
                name="nothing",
                x_axis="mean_ch0",
                y_axis="count",
                vertices=rectangle_vertices(-10, -10, -5, -5),
            )
        )
        stats = gate_set.statistics(gate.id, frame)
        assert stats["n_gated"] == 0
        assert np.isnan(stats["means"]["mean_ch0"])

    def test_a_column_missing_from_the_table_gives_nan(self):
        frame = self.make_frame()
        gate_set = self.make_gate_set()
        gate_id = next(iter(gate_set)).id
        means = gate_set.statistics(gate_id, frame, ["not_measured"])["means"]
        assert np.isnan(means["not_measured"])

    def test_a_subgate_counts_only_within_its_parent(self):
        frame = self.make_frame()
        gate_set = GateSet()
        parent = gate_set.add(
            Gate(
                name="parent",
                x_axis="mean_ch0",
                y_axis="count",
                vertices=rectangle_vertices(0, 0, 5, 50),
            )
        )
        child = gate_set.add(
            Gate(
                name="child",
                x_axis="mean_ch0",
                y_axis="count",
                vertices=rectangle_vertices(0, 0, 100, 500),
                parent_id=parent.id,
            )
        )
        assert gate_set.statistics(child.id, frame)["n_gated"] == 2
