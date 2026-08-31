"""The ready-made gate over seam-crossing objects (phase L6/L8 review).

Three columns on the measurement table make a seam object selectable with
the gating machinery that already exists. These tests pin the claim that
one call turns them into a real gate selecting the right rows, and that the
rectangle is built from the data rather than from a guess about it.
"""

import numpy as np
import pandas as pd
import pytest

from vtea_core.gates import (
    DEFAULT_THRESHOLD,
    SEAM_GATE_COLOR,
    GateSet,
    has_seam_columns,
    seam_gate,
    seam_table,
)


def make_frame():
    """Four objects: two clean, two the reconciler was unsure about."""
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3, 4],
            "n_fragments": [1, 2, 3, 1],
            "seam_rule": ["uncut", "overlap", "overlap", "uncut"],
            "seam_confidence": [1.0, 0.42, 0.0, 1.0],
            "area": [100.0, 240.0, 900.0, 110.0],
        }
    )


class TestHasSeamColumns:
    def test_a_blocked_table_has_them(self):
        assert has_seam_columns(make_frame())

    def test_an_in_memory_table_does_not(self):
        """No seams to review is not an error - it is the ordinary case for
        an image that fitted in memory."""
        assert not has_seam_columns(pd.DataFrame({"object_id": [1], "area": [1.0]}))


class TestSeamGate:
    def test_it_selects_the_uncertain_objects(self):
        frame = make_frame()
        gate_set = GateSet()
        gate = seam_gate(frame)
        gate_set.add(gate)
        selected = frame.loc[gate_set.mask(gate.id, frame), "object_id"].tolist()
        assert selected == [2, 3]

    def test_confidence_zero_is_inside_not_on_the_edge(self):
        """The object no tile managed to contain scores 0.0, and it is the
        one a reviewer most needs to see - a rectangle starting exactly at
        zero would put it on the boundary."""
        frame = make_frame()
        gate_set = GateSet()
        gate = seam_gate(frame)
        gate_set.add(gate)
        mask = gate_set.mask(gate.id, frame)
        assert bool(np.asarray(mask)[frame["object_id"].to_numpy() == 3][0])

    def test_the_ceiling_comes_from_the_data(self):
        """A vessel that ended up in nine tiles is exactly the object worth
        looking at, so a fixed ceiling would miss the most interesting row."""
        frame = make_frame()
        frame.loc[frame["object_id"] == 3, "n_fragments"] = 9
        gate_set = GateSet()
        gate = seam_gate(frame)
        gate_set.add(gate)
        selected = frame.loc[gate_set.mask(gate.id, frame), "object_id"].tolist()
        assert 3 in selected

    def test_it_is_drawn_over_the_seam_columns(self):
        gate = seam_gate(make_frame())
        assert (gate.x_axis, gate.y_axis) == ("seam_confidence", "n_fragments")
        assert gate.color == SEAM_GATE_COLOR

    def test_a_tighter_threshold_selects_fewer(self):
        frame = make_frame()
        gate_set = GateSet()
        gate = seam_gate(frame, threshold=0.2)
        gate_set.add(gate)
        assert frame.loc[gate_set.mask(gate.id, frame), "object_id"].tolist() == [3]

    def test_it_can_be_a_subgate(self):
        gate = seam_gate(make_frame(), parent_id="parent")
        assert gate.parent_id == "parent"

    def test_a_table_with_no_seams_refuses(self):
        """Rather than returning a gate that selects nothing, which reads as
        'no seam objects' when the truth is 'this run had no seams'."""
        with pytest.raises(ValueError, match="no seam columns"):
            seam_gate(pd.DataFrame({"object_id": [1]}))

    def test_an_impossible_threshold_refuses(self):
        with pytest.raises(ValueError, match="threshold"):
            seam_gate(make_frame(), threshold=0.0)


class TestSeamTable:
    def test_worst_first(self):
        listed = seam_table(make_frame())
        assert listed["object_id"].tolist() == [3, 2]
        assert np.all(np.diff(listed["seam_confidence"].to_numpy()) >= 0)

    def test_it_carries_the_rule_that_decided_each(self):
        listed = seam_table(make_frame())
        assert set(listed.columns) == {
            "object_id",
            "n_fragments",
            "seam_rule",
            "seam_confidence",
        }

    def test_the_same_rows_the_gate_selects(self):
        frame = make_frame()
        gate_set = GateSet()
        gate = seam_gate(frame, threshold=DEFAULT_THRESHOLD)
        gate_set.add(gate)
        gated = set(frame.loc[gate_set.mask(gate.id, frame), "object_id"])
        assert gated == set(seam_table(frame, threshold=DEFAULT_THRESHOLD)["object_id"])

    def test_an_in_memory_table_lists_nothing(self):
        assert seam_table(pd.DataFrame({"object_id": [1]})).empty
