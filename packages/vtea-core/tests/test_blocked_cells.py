"""Composing and measuring cells as SQL, and the claim it changes nothing.

`build_cells` builds a Cell object per cell with an ObjectRef per part,
which at ten million cells is several gigabytes of Python objects before a
single measurement is joined to any of them. The same two operations as a
recursive join and a group-by give the same answers, and most of these
tests say exactly that by comparing against the in-memory form.

Two things had to survive the port precisely, and they have their own
classes below: which roles are single - because that decides the *shape* of
the table and a shape that varied by field could not be pooled - and NaN
for a missing part, because "this cell has no lysosomes" is a finding.
"""

import numpy as np
import pandas as pd
import pytest
from vtea_core.blocked.cells import (
    CellMembership,
    build_cells_blocked,
    cell_features_blocked,
    single_roles_from_links,
)
from vtea_core.measurements import write_measurements
from vtea_core.objects import (
    DERIVED,
    Association,
    AssociationSet,
    CellCollection,
    CellSet,
    ObjectRef,
    build_cells,
    cell_features,
)


def link(child_segmentation, child_id, parent_segmentation, parent_id, mode="one_to_one"):
    return Association(
        child=ObjectRef(child_segmentation, child_id),
        parent=ObjectRef(parent_segmentation, parent_id),
        method="test",
        params={"mode": mode},
    )


def one_cell_each():
    return AssociationSet([link("cytoplasm", index, "nuclei", index) for index in (1, 2, 3)])


def a_hierarchy():
    """Three nuclei; a cytoplasm each; lysosomes in two of the cytoplasms;
    one lysosome whose chain never reaches a root."""
    return AssociationSet(
        [
            link("cytoplasm", 1, "nuclei", 1),
            link("cytoplasm", 2, "nuclei", 2),
            link("cytoplasm", 3, "nuclei", 3),
            link("lysosome", 10, "cytoplasm", 1, mode="many_to_one"),
            link("lysosome", 11, "cytoplasm", 1, mode="many_to_one"),
            link("lysosome", 12, "cytoplasm", 2, mode="many_to_one"),
            link("lysosome", 13, "cytoplasm", 99, mode="many_to_one"),
        ]
    )


def tables():
    return {
        "nuclei": pd.DataFrame(
            {"object_id": [1, 2, 3, 4], "mean": [10.0, 20.0, 30.0, 40.0], "count": [5, 6, 7, 8]}
        ),
        "cytoplasm": pd.DataFrame(
            {"object_id": [1, 2, 3], "mean": [1.0, 2.0, 3.0], "count": [50, 60, 70]}
        ),
        "lysosome": pd.DataFrame(
            {"object_id": [10, 11, 12, 13], "mean": [0.5, 1.5, 2.5, 3.5], "count": [1, 2, 3, 4]}
        ),
    }


class TestTheLinksAsATable:
    def test_every_link_becomes_a_row(self):
        frame = a_hierarchy().to_frame()
        assert len(frame) == 7
        assert set(frame["child_segmentation"]) == {"cytoplasm", "lysosome"}

    def test_an_unassigned_child_is_a_row_with_no_parent(self):
        """Not an absent row: a result that silently omits the children
        nothing claimed looks exactly like one where everything was."""
        links = one_cell_each()
        links.add_unassigned(ObjectRef("cytoplasm", 9))
        frame = links.to_frame()
        orphan = frame[frame["child_id"] == 9].iloc[0]
        assert pd.isna(orphan["parent_id"])
        assert len(frame) == 4

    def test_object_ids_stay_integers(self):
        """A parent id that came back as 12.0 joins against nothing, and
        looks like a rounding question rather than a type one."""
        frame = a_hierarchy().to_frame()
        assert frame["child_id"].dtype == np.int64
        assert str(frame["parent_id"].dtype) == "Int64"

    def test_how_the_link_was_made_travels_with_it(self):
        frame = a_hierarchy().to_frame()
        by_role = frame.groupby("child_segmentation")["at_most_one"].all()
        assert bool(by_role["cytoplasm"])
        assert not bool(by_role["lysosome"])


class TestBuildCells:
    def test_the_membership_matches_the_object_form(self):
        membership = build_cells_blocked(a_hierarchy(), [1, 2, 3, 4], root="nuclei")
        cells = build_cells(a_hierarchy(), [1, 2, 3, 4], root="nuclei")
        expected = {
            (cell.cell_id, role, ref.object_id)
            for cell in cells
            for role in cells.roles()
            for ref in cell.objects(role)
        }
        got = set(membership.frame.itertuples(index=False, name=None))
        assert got == expected

    def test_a_root_object_with_nothing_attached_is_still_a_cell(self):
        """Dropping exactly the cells that lost a part biases every per-cell
        statistic that follows."""
        membership = build_cells_blocked(a_hierarchy(), [1, 2, 3, 4], root="nuclei")
        assert len(membership) == 4
        assert 4 in membership.cell_ids.tolist()

    def test_the_chain_is_followed_through_the_cytoplasm(self):
        membership = build_cells_blocked(a_hierarchy(), None, root="nuclei")
        cell_one = membership.frame[membership.frame["cell_id"] == 1]
        assert set(cell_one["role"]) == {"nuclei", "cytoplasm", "lysosome"}
        assert sorted(cell_one[cell_one["role"] == "lysosome"]["object_id"]) == [10, 11]

    def test_an_object_whose_chain_never_reaches_a_root_is_unclaimed(self):
        membership = build_cells_blocked(a_hierarchy(), None, root="nuclei")
        assert membership.unclaimed["object_id"].tolist() == [13]

    def test_a_cycle_is_refused_rather_than_followed(self):
        looped = AssociationSet(
            [
                link("cytoplasm", 1, "nuclei", 1),
                link("lysosome", 10, "cytoplasm", 1),
                link("cytoplasm", 1, "lysosome", 10),
            ]
        )
        with pytest.raises(ValueError, match="cycle"):
            build_cells_blocked(looped, [1], root="nuclei")

    def test_the_two_link_cycle_the_object_form_refuses_is_refused_here_too(self):
        """Parity with `build_cells`, which is the point: a cytoplasm that
        is both the parent and the child of a nucleus is a contradiction
        whichever code notices it."""
        looped = AssociationSet(
            [link("cytoplasm", 1, "nuclei", 1), link("nuclei", 1, "cytoplasm", 1)]
        )
        with pytest.raises(ValueError, match="cycle"):
            build_cells_blocked(looped, None, root="nuclei")

    def test_a_long_chain_that_does_not_loop_is_allowed(self):
        """The cycle check follows parent pointers a bounded number of
        times, so a legitimately deep hierarchy must not trip it."""
        deep = AssociationSet(
            [
                link("b", 1, "a", 1),
                link("c", 1, "b", 1),
                link("d", 1, "c", 1),
                link("e", 1, "d", 1),
            ]
        )
        membership = build_cells_blocked(deep, [1], root="a")
        assert set(membership.frame["role"]) == {"a", "b", "c", "d", "e"}

    def test_an_object_that_is_its_own_parent_is_refused(self):
        looped = AssociationSet([link("cytoplasm", 1, "cytoplasm", 1)])
        with pytest.raises(ValueError, match="cycle"):
            build_cells_blocked(looped, [1], root="cytoplasm")

    def test_a_root_that_is_only_ever_a_child_is_refused_clearly(self):
        with pytest.raises(ValueError, match="only ever a child"):
            build_cells_blocked(one_cell_each(), None, root="cytoplasm")

    def test_the_root_segmentation_has_to_be_named(self):
        with pytest.raises(ValueError, match="identifies a cell"):
            build_cells_blocked(one_cell_each(), None)

    def test_a_cytoplasm_can_be_the_root(self):
        """Nothing makes nuclei special, which is how the multinucleate case
        is expressed."""
        links = AssociationSet(
            [
                link("nuclei", 1, "cytoplasm", 1),
                link("nuclei", 2, "cytoplasm", 1),
            ]
        )
        membership = build_cells_blocked(links, [1], root="cytoplasm")
        assert len(membership) == 1
        assert sorted(membership.objects("nuclei")["object_id"]) == [1, 2]

    def test_it_answers_the_questions_any_cell_result_answers(self):
        membership = build_cells_blocked(a_hierarchy(), [1, 2, 3, 4], root="nuclei")
        assert isinstance(membership, CellCollection)
        assert membership.root_segmentation == "nuclei"
        assert membership.part_roles() == ("cytoplasm", "lysosome")
        assert "4 cells" in membership.summary()
        assert "lysosome: 2/4" in membership.summary()

    def test_it_converts_back_to_the_object_form(self):
        """Reading a table is not how anybody inspects one cell."""
        membership = build_cells_blocked(a_hierarchy(), [1, 2, 3, 4], root="nuclei")
        cells = membership.to_cell_set()
        assert isinstance(cells, CellSet)
        expected = build_cells(a_hierarchy(), [1, 2, 3, 4], root="nuclei")
        assert {cell.cell_id: cell.parts for cell in cells} == {
            cell.cell_id: cell.parts for cell in expected
        }
        assert cells.unclaimed == expected.unclaimed
        assert cells.single_roles == expected.single_roles


class TestWhichRolesAreSingle:
    """The shape of the feature table, and the one thing that must not come
    from the data: a field that happens to hold no cell with two lysosomes
    has to produce the same columns as one that does."""

    def test_a_one_to_one_assignment_is_single(self):
        assert "cytoplasm" in single_roles_from_links(one_cell_each().to_frame())

    def test_a_many_to_one_assignment_is_not(self):
        assert "lysosome" not in single_roles_from_links(a_hierarchy().to_frame())

    def test_a_derived_part_is_single_by_construction(self):
        derived = AssociationSet(
            [
                Association(
                    child=ObjectRef("envelope", 1),
                    parent=ObjectRef("nuclei", 1),
                    relationship=DERIVED,
                    method="identity",
                )
            ]
        )
        assert "envelope" in single_roles_from_links(derived.to_frame())

    def test_one_many_to_one_link_makes_the_whole_role_aggregated(self):
        mixed = AssociationSet(
            [
                link("cytoplasm", 1, "nuclei", 1),
                link("cytoplasm", 2, "nuclei", 2, mode="many_to_one"),
            ]
        )
        assert "cytoplasm" not in single_roles_from_links(mixed.to_frame())

    def test_an_unassigned_child_says_nothing_about_the_shape(self):
        """It is not a link, so it cannot be evidence about how links of
        that role were made."""
        links = one_cell_each()
        links.add_unassigned(ObjectRef("cytoplasm", 9))
        assert "cytoplasm" in single_roles_from_links(links.to_frame())

    def test_it_agrees_with_the_in_memory_form(self):
        links = a_hierarchy()
        assert build_cells_blocked(links, [1, 2, 3, 4], root="nuclei").single_roles == (
            build_cells(links, [1, 2, 3, 4], root="nuclei").single_roles
        )


class TestCellFeatures:
    def test_the_table_is_identical_to_the_in_memory_one(self):
        """Values, column names, column order and dtypes - a column that
        changes dtype depending on which path produced it cannot be pooled
        with one that did not."""
        links = a_hierarchy()
        got = cell_features_blocked(
            build_cells_blocked(links, [1, 2, 3, 4], root="nuclei"), tables()
        )
        expected = cell_features(build_cells(links, [1, 2, 3, 4], root="nuclei"), tables())
        pd.testing.assert_frame_equal(got, expected)

    def test_a_cell_with_none_of_a_role_counts_zero_rather_than_vanishing(self):
        links = a_hierarchy()
        got = cell_features_blocked(
            build_cells_blocked(links, [1, 2, 3, 4], root="nuclei"), tables()
        )
        assert got.loc[got["cell_id"] == 3, "lysosome.n"].item() == 0
        assert len(got) == 4

    def test_a_cell_missing_a_part_gets_nan_not_a_dropped_row(self):
        links = a_hierarchy()
        got = cell_features_blocked(
            build_cells_blocked(links, [1, 2, 3, 4], root="nuclei"), tables()
        )
        row = got[got["cell_id"] == 4].iloc[0]
        assert pd.isna(row["cytoplasm.mean"])
        assert row["nuclei.mean"] == 40.0

    def test_a_single_role_keeps_its_own_column_names(self):
        links = a_hierarchy()
        got = cell_features_blocked(
            build_cells_blocked(links, [1, 2, 3, 4], root="nuclei"), tables()
        )
        assert "cytoplasm.mean" in got.columns
        assert "cytoplasm.n" not in got.columns

    def test_a_one_to_many_role_is_counted_and_aggregated(self):
        links = a_hierarchy()
        got = cell_features_blocked(
            build_cells_blocked(links, [1, 2, 3, 4], root="nuclei"), tables()
        )
        first = got[got["cell_id"] == 1].iloc[0]
        assert first["lysosome.n"] == 2
        assert first["lysosome.mean_mean"] == pytest.approx(1.0)
        assert first["lysosome.sum_count"] == pytest.approx(3.0)

    def test_a_median_is_available_too(self):
        links = a_hierarchy()
        membership = build_cells_blocked(links, [1, 2, 3, 4], root="nuclei")
        got = cell_features_blocked(membership, tables(), aggregations=("n", "median"))
        expected = cell_features(
            build_cells(links, [1, 2, 3, 4], root="nuclei"),
            tables(),
            aggregations=("n", "median"),
        )
        pd.testing.assert_frame_equal(got, expected)

    def test_an_unknown_aggregation_is_refused(self):
        membership = build_cells_blocked(a_hierarchy(), None, root="nuclei")
        with pytest.raises(ValueError, match="unknown aggregation"):
            cell_features_blocked(membership, tables(), aggregations=("n", "vibes"))

    def test_a_role_with_no_table_is_simply_absent(self):
        membership = build_cells_blocked(a_hierarchy(), [1, 2, 3, 4], root="nuclei")
        got = cell_features_blocked(membership, {"nuclei": tables()["nuclei"]})
        assert [column for column in got.columns if column.startswith("lysosome")] == []
        assert len(got) == 4

    def test_non_numeric_columns_are_left_out(self):
        membership = build_cells_blocked(one_cell_each(), [1, 2, 3], root="nuclei")
        with_text = tables()
        with_text["nuclei"]["note"] = ["a", "b", "c", "d"]
        got = cell_features_blocked(membership, with_text)
        assert "nuclei.note" not in got.columns

    def test_a_measurement_table_can_be_a_parquet_file(self, tmp_path):
        """The point of doing this in a database: a table of ten million
        objects is read as the query needs it rather than loaded."""
        links = a_hierarchy()
        paths = {
            role: write_measurements(frame, tmp_path / f"{role}.parquet")
            for role, frame in tables().items()
        }
        got = cell_features_blocked(
            build_cells_blocked(links, [1, 2, 3, 4], root="nuclei"), paths
        )
        expected = cell_features(build_cells(links, [1, 2, 3, 4], root="nuclei"), tables())
        pd.testing.assert_frame_equal(got, expected)

    def test_the_shape_does_not_depend_on_how_many_children_this_field_had(self):
        """One field with one lysosome per cell and one with three have to
        produce the same columns, or the two cannot be pooled."""
        sparse = AssociationSet(
            [
                link("cytoplasm", 1, "nuclei", 1),
                link("lysosome", 10, "cytoplasm", 1, mode="many_to_one"),
            ]
        )
        one = cell_features_blocked(build_cells_blocked(sparse, [1], root="nuclei"), tables())
        many = cell_features_blocked(
            build_cells_blocked(a_hierarchy(), [1, 2, 3, 4], root="nuclei"), tables()
        )
        assert list(one.columns) == list(many.columns)


class TestMembership:
    def test_it_refuses_a_table_that_is_not_one(self):
        with pytest.raises(ValueError, match="membership table needs"):
            CellMembership(pd.DataFrame({"cell_id": [1]}))

    def test_an_empty_association_set_is_no_cells_rather_than_an_error(self):
        membership = build_cells_blocked(AssociationSet(), None, root="nuclei")
        assert len(membership) == 0
        assert membership.roles() == ()


class TestThroughThePipeline:
    def test_a_blocked_protocol_builds_and_measures_cells(self):
        from vtea_core.blocked import BlockedPipeline, MemoryBudget, plan_tiles
        from vtea_core.workflow import Pipeline, Step

        labels = np.zeros((8, 16, 16), dtype=np.int32)
        labels[2:6, 2:6, 2:6] = 1
        labels[2:6, 9:13, 9:13] = 2
        protocol = Pipeline(
            [
                Step.for_function(
                    "cells",
                    "build_cells",
                    params={"root": "nuclei"},
                    input_keys={"associations": "associations", "root_labels": "labels"},
                    output_key="cells",
                ),
                Step.for_function(
                    "cells",
                    "cell_features",
                    input_keys={"cells": "cells", "measurement_tables": "measurement_tables"},
                    output_key="cell_table",
                ),
            ]
        )
        links = AssociationSet([link("cytoplasm", 1, "nuclei", 1)])
        plan = plan_tiles(labels.shape, budget=MemoryBudget(8192), bytes_per_voxel=8)
        with BlockedPipeline(protocol, plan=plan) as blocked:
            context = blocked.run(
                {
                    "labels": labels,
                    "associations": links,
                    "measurement_tables": {
                        "nuclei": pd.DataFrame({"object_id": [1, 2], "mean": [3.0, 4.0]}),
                        "cytoplasm": pd.DataFrame({"object_id": [1], "mean": [9.0]}),
                    },
                }
            )
        table = context["cell_table"]
        assert table["cell_id"].tolist() == [1, 2]
        assert table["nuclei.mean"].tolist() == [3.0, 4.0]
        assert pd.isna(table.loc[table["cell_id"] == 2, "cytoplasm.mean"].item())
