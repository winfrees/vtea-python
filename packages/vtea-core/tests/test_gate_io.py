"""Gates round-trip through plain JSON.

A polygon someone drew around a population is the one part of an analysis
that can't be recomputed from parameters, so it has to be saveable
alongside the figure it produced.
"""

import json

import numpy as np
import pandas as pd
import pytest

from vtea_core.gates import (
    GATES_FORMAT_VERSION,
    Gate,
    GateSet,
    gate_set_from_dict,
    gate_set_to_dict,
    load_gates,
    rectangle_vertices,
    save_gates,
)


def make_gate_set():
    gate_set = GateSet()
    parent = gate_set.add(
        Gate(
            name="bright",
            x_axis="mean_ch0",
            y_axis="count",
            vertices=rectangle_vertices(0, 0, 10, 10),
            color="#ff0000",
        )
    )
    gate_set.add(
        Gate(
            name="bright and big",
            x_axis="mean_ch0",
            y_axis="count",
            vertices=rectangle_vertices(2, 2, 8, 8),
            parent_id=parent.id,
            visible=False,
        )
    )
    return gate_set, parent


class TestRoundTrip:
    def test_names_axes_and_vertices_survive(self):
        gate_set, _ = make_gate_set()
        restored = gate_set_from_dict(gate_set_to_dict(gate_set))
        original, loaded = list(gate_set), list(restored)
        assert [gate.name for gate in loaded] == [gate.name for gate in original]
        assert [gate.x_axis for gate in loaded] == [gate.x_axis for gate in original]
        np.testing.assert_allclose(loaded[0].vertices, original[0].vertices)

    def test_hierarchy_survives(self):
        """Ids have to be preserved, or parent_id points at nothing."""
        gate_set, parent = make_gate_set()
        restored = gate_set_from_dict(gate_set_to_dict(gate_set))
        child = list(restored)[1]
        assert child.parent_id == parent.id
        assert restored.get(child.parent_id).name == "bright"

    def test_color_and_visibility_survive(self):
        gate_set, _ = make_gate_set()
        restored = list(gate_set_from_dict(gate_set_to_dict(gate_set)))
        assert restored[0].color == "#ff0000"
        assert restored[1].visible is False

    def test_membership_is_unchanged_after_a_round_trip(self):
        gate_set, parent = make_gate_set()
        frame = pd.DataFrame({"mean_ch0": [1.0, 5.0, 20.0], "count": [1.0, 5.0, 20.0]})
        restored = gate_set_from_dict(gate_set_to_dict(gate_set))
        np.testing.assert_array_equal(
            gate_set.mask(parent.id, frame), restored.mask(parent.id, frame)
        )

    def test_a_file_round_trips(self, tmp_path):
        gate_set, _ = make_gate_set()
        path = save_gates(gate_set, tmp_path / "gates.json")
        assert len(load_gates(path)) == 2

    def test_the_file_is_readable_json_with_a_version(self, tmp_path):
        """Plain, inspectable JSON - not a pickle - so it can be archived
        with a figure and read by anything."""
        gate_set, _ = make_gate_set()
        path = save_gates(gate_set, tmp_path / "gates.json")
        data = json.loads(path.read_text())
        assert data["vtea_gates_version"] == GATES_FORMAT_VERSION
        assert data["gates"][0]["name"] == "bright"


class TestLoadingEdgeCases:
    def test_children_listed_before_parents_still_load(self):
        gate_set, parent = make_gate_set()
        data = gate_set_to_dict(gate_set)
        data["gates"].reverse()
        assert len(gate_set_from_dict(data)) == 2

    def test_a_missing_parent_is_reported(self):
        data = {
            "vtea_gates_version": GATES_FORMAT_VERSION,
            "gates": [
                {
                    "name": "orphan",
                    "x_axis": "a",
                    "y_axis": "b",
                    "vertices": [[0, 0], [1, 0], [1, 1]],
                    "parent_id": "does-not-exist",
                }
            ],
        }
        with pytest.raises(ValueError, match="missing or circular parents"):
            gate_set_from_dict(data)

    def test_a_newer_file_version_is_refused_clearly(self):
        with pytest.raises(ValueError, match="newer than this VTEA"):
            gate_set_from_dict({"vtea_gates_version": GATES_FORMAT_VERSION + 1, "gates": []})

    def test_an_empty_set_round_trips(self):
        assert len(gate_set_from_dict(gate_set_to_dict(GateSet()))) == 0


class TestRectangleVertices:
    def test_four_corners_in_order(self):
        vertices = rectangle_vertices(0, 0, 2, 3)
        assert vertices.shape == (4, 2)
        np.testing.assert_allclose(vertices, [[0, 0], [2, 0], [2, 3], [0, 3]])

    def test_corners_given_backwards_still_enclose_the_interior(self):
        from vtea_core.gates import polygon_gate

        vertices = rectangle_vertices(2, 3, 0, 0)
        assert polygon_gate(np.array([1.0]), np.array([1.5]), vertices)[0]
