"""Contexts: levels from pieces of cells up to neighbourhoods of
neighbourhoods, the two operators that move features along them, the rule
that keeps a level from being typed on its own type, and the level
definitions a protocol saves."""

import json

import numpy as np
import pandas as pd
import pytest
from vtea_core.context import (
    CELLULAR,
    MEMBER_OF,
    MISSING_CATEGORY,
    NEIGHBORHOOD,
    PART_OF,
    SUBCELLULAR,
    ContextGraph,
    ContextSpec,
    Level,
    LevelDisplay,
    LevelSpec,
    build_context,
)
from vtea_core.objects import Cell, CellSet, ObjectRef
from vtea_core.workflow import Protocol, load_protocol, save_protocol


def small_graph():
    """Four lysosomes in two cells in one neighbourhood."""
    graph = ContextGraph()
    graph.add_level(
        Level(
            "lysosomes",
            SUBCELLULAR,
            pd.DataFrame({"object_id": [1, 2, 3, 4], "mean": [1.0, 3.0, 10.0, np.nan]}),
            "object_id",
            rank=0,
        )
    )
    graph.add_level(
        Level("cells", CELLULAR, pd.DataFrame({"cell_id": [10, 20], "kind": [0, 1]}), "cell_id", rank=1)
    )
    graph.add_level(
        Level(
            "nbhd",
            NEIGHBORHOOD,
            pd.DataFrame({"neighborhood_id": [10, 20], "neighborhood_type": [5, 7], "density": [2.0, 4.0]}),
            "neighborhood_id",
            rank=2,
        )
    )
    graph.link(
        "lysosomes", "cells", pd.DataFrame({"child": [1, 2, 3, 4], "parent": [10, 10, 20, 20]}), PART_OF
    )
    # neighbourhood 10 (seeded on cell 10) holds both cells; 20 only cell 20
    graph.link(
        "cells",
        "nbhd",
        pd.DataFrame(
            {"child": [10, 20, 20], "parent": [10, 10, 20], "is_seed": [True, False, True]}
        ),
        MEMBER_OF,
    )
    return graph


class TestStructure:
    def test_the_stack_runs_lowest_first(self):
        assert [level.name for level in small_graph().stack()] == ["lysosomes", "cells", "nbhd"]

    def test_a_path_chains_links(self):
        chain = small_graph().path("lysosomes", "nbhd")
        assert [(link.child, link.parent) for link in chain] == [("lysosomes", "cells"), ("cells", "nbhd")]

    def test_no_path_downward(self):
        with pytest.raises(ValueError, match="not above"):
            small_graph().path("nbhd", "lysosomes")

    def test_a_link_must_run_upward(self):
        graph = small_graph()
        with pytest.raises(ValueError, match="does not sit above"):
            graph.link("nbhd", "cells", pd.DataFrame({"child": [10], "parent": [10]}), PART_OF)

    def test_membership_through_two_links(self):
        matrix = small_graph().membership_matrix("lysosomes", "nbhd", relation="member").toarray()
        # lysosomes of cell 20 are in both neighbourhoods, of cell 10 only in 10
        np.testing.assert_array_equal(matrix, [[1, 0], [1, 0], [1, 1], [1, 1]])

    def test_own_follows_only_the_seeded_neighbourhood(self):
        matrix = small_graph().membership_matrix("lysosomes", "nbhd", relation="own").toarray()
        np.testing.assert_array_equal(matrix, [[1, 0], [1, 0], [0, 1], [0, 1]])


class TestAggregateUp:
    def test_counts_and_means_ignoring_missing_values(self):
        result = small_graph().aggregate_up(
            "lysosomes", "cells", features="mean", aggregations="n,mean,sum,max"
        )
        assert list(result["lysosomes.n"]) == [2, 2]
        np.testing.assert_allclose(result["lysosomes.mean_mean"], [2.0, 10.0])
        np.testing.assert_allclose(result["lysosomes.sum_mean"], [4.0, 10.0])
        np.testing.assert_allclose(result["lysosomes.max_mean"], [3.0, 10.0])

    def test_through_two_links(self):
        result = small_graph().aggregate_up("lysosomes", "nbhd", features="mean", aggregations="n")
        assert list(result["lysosomes.n"]) == [4, 2]

    def test_refuses_an_unknown_reduction(self):
        with pytest.raises(ValueError, match="aggregation"):
            small_graph().aggregate_up("lysosomes", "cells", aggregations="mode")


class TestReflectDown:
    def test_a_piece_of_a_cell_takes_on_its_cells_class(self):
        result = small_graph().reflect_down("cells", "lysosomes", features="kind", categorical="kind")
        assert list(result["cells.kind"]) == [0, 0, 1, 1]

    def test_through_two_links_to_its_own_neighbourhood(self):
        result = small_graph().reflect_down(
            "nbhd", "lysosomes", features="neighborhood_type,density", categorical="neighborhood_type"
        )
        assert list(result["nbhd.neighborhood_type"]) == [5, 5, 7, 7]
        np.testing.assert_allclose(result["nbhd.density"], [2.0, 2.0, 4.0, 4.0])

    def test_membership_averages_and_votes(self):
        result = small_graph().reflect_down(
            "nbhd", "cells", features="neighborhood_type,density", categorical="neighborhood_type", relation="member"
        )
        np.testing.assert_allclose(result["nbhd.density"], [2.0, 3.0])
        # cell 20 is in 10 (type 5) and 20 (type 7): a tie goes to the smaller
        assert list(result["nbhd.neighborhood_type"]) == [5, 5]
        assert list(result["nbhd.n_memberships"]) == [1, 2]

    def test_an_entity_in_nothing_gets_missing_values(self):
        graph = small_graph()
        graph.levels["lysosomes"].table.loc[4] = {"object_id": 99, "mean": 0.0}
        result = graph.reflect_down("nbhd", "lysosomes", features="neighborhood_type", categorical="neighborhood_type")
        assert result["nbhd.neighborhood_type"].iloc[-1] == MISSING_CATEGORY

    def test_own_without_seeds_is_refused(self):
        with pytest.raises(ValueError, match="relation='member'"):
            small_graph().reflect_down("cells", "lysosomes", relation="own")


class TestCircularity:
    def test_a_level_is_not_typed_on_what_came_down_from_it(self):
        graph = small_graph()
        reflected = graph.reflect_down("nbhd", "cells", features="neighborhood_type", categorical="neighborhood_type")
        graph.apply_reflection("cells", "nbhd", reflected)
        assert "nbhd.neighborhood_type" in graph.level("cells").table.columns
        # cells may be typed on their own features, not on the neighbourhood's
        assert "kind" in graph.typing_features("cells")
        assert "nbhd.neighborhood_type" not in graph.typing_features("cells")


def two_halves(n_side=10, spacing=10.0):
    """Nuclei on a grid, each with one lysosome; left half class 0, right 1."""
    rows, lysosomes, cells = [], [], []
    object_id = 0
    for i in range(n_side):
        for j in range(n_side):
            object_id += 1
            y, x = i * spacing, j * spacing
            rows.append({"object_id": object_id, "centroid-0": y, "centroid-1": x, "kmeans_1": int(j >= n_side // 2)})
            lysosomes.append({"object_id": 1000 + object_id, "centroid-0": y + 1, "centroid-1": x + 1, "mean": float(j)})
            cells.append(
                Cell(
                    object_id,
                    ObjectRef("nuclei", object_id),
                    {"lysosomes": (ObjectRef("lysosomes", 1000 + object_id),)},
                )
            )
    cell_set = CellSet(cells, single_roles=["lysosomes"])
    return pd.DataFrame(rows), pd.DataFrame(lysosomes), cell_set


def three_level_spec(**neighborhood):
    return ContextSpec(
        [
            LevelSpec("nuclei", SUBCELLULAR, "nuclei"),
            LevelSpec("lysosomes", SUBCELLULAR, "lysosomes"),
            LevelSpec("cells", CELLULAR, "build_cells_1"),
            LevelSpec(
                "nbhd_1",
                NEIGHBORHOOD,
                "cells",
                build={"method": "radius", "radius": 15.0},
                measure={"class_column": "nuclei.kmeans_1"},
                classify={"n_clusters": 2},
                **neighborhood,
            ),
            LevelSpec(
                "nbhd_2",
                NEIGHBORHOOD,
                "nbhd_1",
                build={"method": "radius", "radius": 25.0},
                measure={"class_column": "neighborhood_type"},
                classify={"n_clusters": 2},
            ),
        ]
    )


def build_three_levels():
    nuclei, lysosomes, cell_set = two_halves()
    cell_table = pd.DataFrame(
        {
            "cell_id": nuclei["object_id"],
            "nuclei.centroid-0": nuclei["centroid-0"],
            "nuclei.centroid-1": nuclei["centroid-1"],
            "nuclei.kmeans_1": nuclei["kmeans_1"],
        }
    )
    return build_context(
        three_level_spec(),
        measurement_tables={"nuclei": nuclei, "lysosomes": lysosomes},
        cells={"build_cells_1": cell_set},
        cell_tables={"build_cells_1": cell_table},
    ), nuclei


class TestBuildContext:
    def test_builds_every_level_in_order(self):
        graph, _nuclei = build_three_levels()
        assert [level.name for level in graph.stack()] == ["lysosomes", "nuclei", "cells", "nbhd_1", "nbhd_2"]
        assert graph.level("nbhd_2").rank == 3

    def test_pieces_of_cells_are_linked_to_them(self):
        graph, _nuclei = build_three_levels()
        assert set(graph.parents_of("lysosomes")) == {"cells"}
        assert set(graph.parents_of("nuclei")) == {"cells"}

    def test_a_lysosome_takes_on_its_neighbourhood_of_neighbourhoods(self):
        graph, nuclei = build_three_levels()
        lysosomes = graph.level("lysosomes").table
        assert "nbhd_1.neighborhood_type" in lysosomes.columns
        assert "nbhd_2.neighborhood_type" in lysosomes.columns
        # the first-level type splits left from right
        halves = nuclei["kmeans_1"].to_numpy()
        agreement = pd.crosstab(halves, lysosomes["nbhd_1.neighborhood_type"].to_numpy())
        assert (agreement.max(axis=1) / agreement.sum(axis=1)).min() > 0.9

    def test_neighbourhoods_of_neighbourhoods_are_measured_on_the_types_below(self):
        graph, _nuclei = build_three_levels()
        table = graph.level("nbhd_2").table
        assert {"Class_0_ClassFraction", "Class_1_ClassFraction"} <= set(table.columns)

    def test_a_centred_neighbourhood_is_drawn_from_its_sources_labels(self):
        graph, _nuclei = build_three_levels()
        assert graph.level("nbhd_1").labels_key == "nuclei"
        assert graph.level("nbhd_1").neighborhoods is not None

    def test_a_level_defined_on_its_own_type_is_refused(self):
        nuclei, lysosomes, cell_set = two_halves()
        spec = ContextSpec(
            [
                LevelSpec("nuclei", SUBCELLULAR, "nuclei"),
                LevelSpec("n1", NEIGHBORHOOD, "nuclei", build={"radius": 15.0}, measure={"class_column": "kmeans_1"}, classify={"n_clusters": 2}),
                # a sibling at the same rank, defined on n1's type handed down
                LevelSpec("n1b", NEIGHBORHOOD, "nuclei", build={"radius": 30.0}, measure={"class_column": "n1.neighborhood_type"}),
            ]
        )
        with pytest.raises(ValueError, match="own type"):
            build_context(spec, measurement_tables={"nuclei": nuclei})

    def test_a_missing_source_is_named(self):
        with pytest.raises(ValueError, match="was not measured"):
            build_context(ContextSpec([LevelSpec("x", SUBCELLULAR, "nope")]))


class TestSpec:
    def test_a_neighbourhood_needs_a_level_before_it(self):
        with pytest.raises(ValueError, match="not a level defined before"):
            ContextSpec([LevelSpec("n", NEIGHBORHOOD, "cells")])

    def test_only_neighbourhoods_are_built_by_the_context(self):
        with pytest.raises(ValueError, match="only a neighborhood level"):
            LevelSpec("cells", CELLULAR, "build_cells_1", build={"radius": 1.0})

    def test_names_are_unique(self):
        with pytest.raises(ValueError, match="two levels"):
            ContextSpec([LevelSpec("a", SUBCELLULAR, "x"), LevelSpec("a", SUBCELLULAR, "y")])

    def test_ranks(self):
        spec = three_level_spec()
        assert [spec.rank(level.name) for level in spec] == [0, 0, 1, 2, 3]

    def test_display_defaults_to_gated_hatched_outlines(self):
        display = LevelDisplay()
        assert display.draw == "gated"
        assert display.fill_pattern == "hatch"
        with pytest.raises(ValueError, match="fill pattern"):
            LevelDisplay(fill_pattern="plaid")
        with pytest.raises(ValueError, match="draw mode"):
            LevelDisplay(draw="everything")


class TestInTheProtocol:
    def test_round_trips_through_a_protocol_file(self, tmp_path):
        spec = three_level_spec(display=LevelDisplay(outline_color="#00ffcc", fill_pattern="crosshatch"))
        path = save_protocol(Protocol(context=spec), tmp_path / "p.vtea.json")
        data = json.loads(path.read_text())
        assert data["vtea_protocol_version"] == 2
        assert [level["tier"] for level in data["context"]["levels"]] == [
            "subcellular", "subcellular", "cellular", "neighborhood", "neighborhood"
        ]
        loaded = load_protocol(path).context
        assert [level.to_dict() for level in loaded] == [level.to_dict() for level in spec]
        assert loaded.get("nbhd_1").display.fill_pattern == "crosshatch"

    def test_a_protocol_without_levels_has_no_context_section(self, tmp_path):
        path = save_protocol(Protocol(), tmp_path / "p.vtea.json")
        data = json.loads(path.read_text())
        assert "context" not in data
        assert data["vtea_protocol_version"] == 1
        assert not load_protocol(path).context
