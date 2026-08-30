"""Composing associations into cells, and measuring them as cells.

The interesting cases are the incomplete ones. A cell missing a part, an
object in no cell at all, a chain that loops - each of these is what real
data looks like, and each is a way for a per-cell table to quietly become
wrong rather than visibly short.
"""

import numpy as np
import pandas as pd
import pytest
from vtea_core.objects import (
    DERIVED,
    Association,
    AssociationSet,
    Cell,
    CellSet,
    ObjectRef,
    build_cells,
    cell_features,
    load_cells,
    merge_associations,
    save_cells,
)


def link(child_segmentation, child_id, parent_segmentation, parent_id, mode="one_to_one"):
    return Association(
        child=ObjectRef(child_segmentation, child_id),
        parent=ObjectRef(parent_segmentation, parent_id),
        method="test",
        params={"mode": mode},
    )


def one_cell_each():
    """Three nuclei, a cytoplasm assigned one-to-one to each."""
    return AssociationSet([link("cytoplasm", index, "nuclei", index) for index in (1, 2, 3)])


def a_cell_with_organelles():
    """Nucleus 1 <- cytoplasm 1 <- three lysosomes."""
    return AssociationSet(
        [
            link("cytoplasm", 1, "nuclei", 1),
            link("lysosome", 10, "cytoplasm", 1, mode="many_to_one"),
            link("lysosome", 11, "cytoplasm", 1, mode="many_to_one"),
            link("lysosome", 12, "cytoplasm", 1, mode="many_to_one"),
        ]
    )


class TestBuildCells:
    def test_each_root_object_is_a_cell(self):
        cells = build_cells(one_cell_each(), None, root="nuclei")
        assert len(cells) == 3

    def test_the_cell_id_is_the_root_object_s_id(self):
        """Not a counter: a gate drawn on cell 412 has to still mean the same
        cell after a re-run that changed how many cells there are."""
        associations = AssociationSet([link("cytoplasm", 5, "nuclei", 412)])
        assert [cell.cell_id for cell in build_cells(associations, None, root="nuclei")] == [412]

    def test_a_cell_holds_its_parts_by_role(self):
        cells = build_cells(one_cell_each(), None, root="nuclei")
        assert cells.cell(2).object("cytoplasm") == ObjectRef("cytoplasm", 2)

    def test_the_hierarchy_is_followed_through_the_chain(self):
        """A lysosome belongs to a cytoplasm, which belongs to a nucleus, so
        the lysosome is part of that nucleus's cell."""
        cells = build_cells(a_cell_with_organelles(), None, root="nuclei")
        assert [ref.object_id for ref in cells.cell(1).objects("lysosome")] == [10, 11, 12]

    def test_a_root_object_with_nothing_attached_still_exists(self):
        """Given its ids - otherwise a nucleus nothing was assigned to would
        not be a cell, and the count would silently exclude it."""
        cells = build_cells(one_cell_each(), [1, 2, 3, 4], root="nuclei")
        assert len(cells) == 4
        assert cells.cell(4).parts == {}

    def test_the_root_label_image_says_which_objects_are_cells(self):
        """Which is how a protocol passes it: the segmentation itself, not a
        list of ids the caller had to extract first."""
        labels = np.zeros((6, 12), dtype=np.int32)
        labels[1:3, 1:3] = 1
        labels[1:3, 5:7] = 2
        labels[1:3, 9:11] = 4  # ids need not be contiguous
        cells = build_cells(one_cell_each(), labels, root="nuclei")
        assert [cell.cell_id for cell in cells] == [1, 2, 3, 4]

    def test_the_root_segmentation_has_to_be_named(self):
        with pytest.raises(ValueError, match="name of the segmentation"):
            build_cells(one_cell_each(), None)

    def test_a_cell_missing_a_part_is_reported_as_such(self):
        cells = build_cells(one_cell_each(), [1, 2, 3, 4], root="nuclei")
        assert [cell.cell_id for cell in cells.missing("cytoplasm")] == [4]

    def test_the_complete_cells_can_be_singled_out(self):
        cells = build_cells(one_cell_each(), [1, 2, 3, 4], root="nuclei")
        assert [cell.cell_id for cell in cells.complete(["cytoplasm"])] == [1, 2, 3]

    def test_an_object_whose_chain_never_reaches_a_root_is_unclaimed(self):
        """A lysosome in a cytoplasm that was never assigned a nucleus. An
        analysis that drops it looks exactly like one that does not."""
        associations = AssociationSet(
            [
                link("cytoplasm", 1, "nuclei", 1),
                link("lysosome", 10, "cytoplasm", 1, mode="many_to_one"),
                link("lysosome", 20, "cytoplasm", 9, mode="many_to_one"),  # orphan cytoplasm
            ]
        )
        cells = build_cells(associations, None, root="nuclei")
        assert cells.unclaimed == [ObjectRef("lysosome", 20)]

    def test_a_cycle_is_refused_rather_than_followed(self):
        associations = AssociationSet(
            [link("cytoplasm", 1, "nuclei", 1), link("nuclei", 1, "cytoplasm", 1)]
        )
        with pytest.raises(ValueError, match="cycle"):
            build_cells(associations, None, root="nuclei")

    def test_a_root_that_is_only_ever_a_child_is_refused_clearly(self):
        """Following links out from it would reach nothing, and an empty
        answer here reads as "no cells" rather than "wrong way round"."""
        associations = AssociationSet([link("nuclei", 1, "cytoplasm", 1)])
        with pytest.raises(ValueError, match="only ever a child"):
            build_cells(associations, None, root="nuclei")

    def test_the_roles_present_are_listed(self):
        """The root among them: a cell's own nucleus is one of the things
        there is a measurement of."""
        cells = build_cells(a_cell_with_organelles(), None, root="nuclei")
        assert cells.roles() == ("cytoplasm", "lysosome", "nuclei")
        assert cells.part_roles() == ("cytoplasm", "lysosome")

    def test_the_summary_says_how_complete_the_cells_are(self):
        cells = build_cells(one_cell_each(), [1, 2, 3, 4], root="nuclei")
        summary = cells.summary()
        assert "4 cells" in summary
        assert "cytoplasm: 3/4" in summary


class TestWholeCellFirst:
    """Nothing makes nuclei the root. A whole-cell segmentation parenting
    everything is the same model read the other way - and it is also how a
    multinucleate cell is expressed."""

    def test_a_cytoplasm_can_be_the_root(self):
        associations = AssociationSet(
            [
                link("nuclei", 1, "cytoplasm", 1, mode="many_to_one"),
                link("nuclei", 2, "cytoplasm", 1, mode="many_to_one"),
                link("nuclei", 3, "cytoplasm", 2, mode="many_to_one"),
            ]
        )
        cells = build_cells(associations, None, root="cytoplasm")
        assert len(cells) == 2

    def test_a_multinucleate_cell_keeps_both_its_nuclei(self):
        associations = AssociationSet(
            [
                link("nuclei", 1, "cytoplasm", 1, mode="many_to_one"),
                link("nuclei", 2, "cytoplasm", 1, mode="many_to_one"),
            ]
        )
        cells = build_cells(associations, None, root="cytoplasm")
        assert [ref.object_id for ref in cells.cell(1).objects("nuclei")] == [1, 2]


class TestSingleRoles:
    """Whether a role is aggregated comes from how the association was made,
    not from whether this field happens to contain a cell with two of
    something - or the same protocol would give differently shaped tables on
    different fields, and they could not be pooled."""

    def test_a_one_to_one_assignment_is_a_single_role(self):
        assert "cytoplasm" in build_cells(one_cell_each(), None, root="nuclei").single_roles

    def test_a_many_to_one_assignment_is_not(self):
        cells = build_cells(a_cell_with_organelles(), None, root="nuclei")
        assert "lysosome" not in cells.single_roles

    def test_a_derived_part_is_single_by_construction(self):
        associations = AssociationSet(
            [
                Association(
                    child=ObjectRef("envelope", 1),
                    parent=ObjectRef("nuclei", 1),
                    relationship=DERIVED,
                    method="identity",
                )
            ]
        )
        # The root is always single - it is what identifies the cell.
        assert build_cells(associations, None, root="nuclei").single_roles == frozenset(
            {"envelope", "nuclei"}
        )

    def test_one_many_to_one_link_makes_the_whole_role_aggregated(self):
        """Otherwise the columns would depend on which links happened to be
        in this particular field."""
        associations = AssociationSet(
            [
                link("cytoplasm", 1, "nuclei", 1),
                link("cytoplasm", 2, "nuclei", 2, mode="many_to_one"),
            ]
        )
        assert "cytoplasm" not in build_cells(associations, None, root="nuclei").single_roles


class TestMergeAssociations:
    def test_two_steps_become_one_hierarchy(self):
        """A protocol associates cytoplasm to nucleus in one step and
        lysosomes to cytoplasm in another; a cell spans both."""
        first = AssociationSet([link("cytoplasm", 1, "nuclei", 1)])
        second = AssociationSet([link("lysosome", 10, "cytoplasm", 1, mode="many_to_one")])
        cells = build_cells(merge_associations(first, second), None, root="nuclei")
        assert cells.cell(1).objects("lysosome") == (ObjectRef("lysosome", 10),)

    def test_the_later_set_corrects_the_earlier_one(self):
        first = AssociationSet([link("cytoplasm", 1, "nuclei", 1)])
        second = AssociationSet([link("cytoplasm", 1, "nuclei", 2)])
        merged = merge_associations(first, second)
        assert len(merged) == 1
        assert merged.parent_of(ObjectRef("cytoplasm", 1)) == ObjectRef("nuclei", 2)

    def test_the_unassigned_children_of_both_are_kept(self):
        first = AssociationSet(unassigned=[ObjectRef("cytoplasm", 7)])
        second = AssociationSet([link("cytoplasm", 1, "nuclei", 1)])
        assert merge_associations(first, second).unassigned == [ObjectRef("cytoplasm", 7)]

    def test_a_child_assigned_by_the_later_set_is_no_longer_unassigned(self):
        first = AssociationSet(unassigned=[ObjectRef("cytoplasm", 1)])
        second = AssociationSet([link("cytoplasm", 1, "nuclei", 1)])
        assert merge_associations(first, second).unassigned == []


def measured(ids, **columns):
    return pd.DataFrame({"object_id": list(ids), **columns})


class TestCellFeatures:
    def test_one_row_per_cell(self):
        cells = build_cells(one_cell_each(), [1, 2, 3], root="nuclei")
        table = cell_features(
            cells,
            {"nuclei": measured([1, 2, 3], mean_ch0=[10.0, 20.0, 30.0])},
        )
        assert list(table["cell_id"]) == [1, 2, 3]

    def test_columns_are_namespaced_by_role(self):
        """A nucleus's brightness and its cytoplasm's are two different
        things, and calling both `mean_ch0` loses which is which."""
        cells = build_cells(one_cell_each(), [1, 2, 3], root="nuclei")
        table = cell_features(
            cells,
            {
                "nuclei": measured([1, 2, 3], mean_ch0=[10.0, 20.0, 30.0]),
                "cytoplasm": measured([1, 2, 3], mean_ch0=[1.0, 2.0, 3.0]),
            },
        )
        assert table.loc[0, "cytoplasm.mean_ch0"] == 1.0
        # Including the root's own measurements, which are the features most
        # protocols start from.
        assert table.loc[0, "nuclei.mean_ch0"] == 10.0

    def test_a_single_role_keeps_its_own_column_names(self):
        cells = build_cells(one_cell_each(), [1, 2, 3], root="nuclei")
        table = cell_features(cells, {"cytoplasm": measured([1, 2, 3], count=[5, 6, 7])})
        assert list(table["cytoplasm.count"]) == [5, 6, 7]

    def test_a_one_to_many_role_is_counted(self):
        """"How many endolysosomes does this cell have" is the question the
        whole hierarchy exists to answer."""
        cells = build_cells(a_cell_with_organelles(), None, root="nuclei")
        table = cell_features(cells, {"lysosome": measured([10, 11, 12], count=[2, 3, 4])})
        assert table.loc[0, "lysosome.n"] == 3

    def test_a_one_to_many_role_is_aggregated(self):
        cells = build_cells(a_cell_with_organelles(), None, root="nuclei")
        table = cell_features(
            cells,
            {"lysosome": measured([10, 11, 12], count=[2, 3, 4])},
            aggregations=("n", "mean", "sum"),
        )
        assert table.loc[0, "lysosome.mean_count"] == pytest.approx(3.0)
        assert table.loc[0, "lysosome.sum_count"] == pytest.approx(9.0)

    def test_a_cell_with_none_of_a_role_counts_zero_rather_than_vanishing(self):
        associations = AssociationSet(
            [
                link("cytoplasm", 1, "nuclei", 1),
                link("cytoplasm", 2, "nuclei", 2),
                link("lysosome", 10, "cytoplasm", 1, mode="many_to_one"),
            ]
        )
        cells = build_cells(associations, None, root="nuclei")
        table = cell_features(cells, {"lysosome": measured([10], count=[2])})
        assert list(table["lysosome.n"]) == [1, 0]

    def test_a_cell_missing_a_part_gets_nan_not_a_dropped_row(self):
        cells = build_cells(one_cell_each(), [1, 2, 3, 4], root="nuclei")
        table = cell_features(cells, {"cytoplasm": measured([1, 2, 3], count=[5, 6, 7])})
        assert len(table) == 4
        assert np.isnan(table.loc[3, "cytoplasm.count"])

    def test_rows_for_objects_in_no_cell_are_left_out(self):
        cells = build_cells(one_cell_each(), None, root="nuclei")
        table = cell_features(cells, {"cytoplasm": measured([1, 2, 3, 99], count=[5, 6, 7, 8])})
        assert len(table) == 3

    def test_a_role_with_no_table_is_simply_absent(self):
        cells = build_cells(a_cell_with_organelles(), None, root="nuclei")
        table = cell_features(cells, {"cytoplasm": measured([1], count=[5])})
        assert "cytoplasm.count" in table.columns
        assert not [column for column in table.columns if column.startswith("lysosome")]

    def test_a_table_without_an_id_column_is_refused_clearly(self):
        cells = build_cells(one_cell_each(), None, root="nuclei")
        with pytest.raises(ValueError, match="no 'object_id' column"):
            cell_features(cells, {"cytoplasm": pd.DataFrame({"count": [1, 2, 3]})})

    def test_an_unknown_aggregation_is_refused(self):
        cells = build_cells(a_cell_with_organelles(), None, root="nuclei")
        with pytest.raises(ValueError, match="unknown aggregation"):
            cell_features(cells, {"lysosome": measured([10], count=[2])}, aggregations=("vibes",))

    def test_non_numeric_columns_are_left_out(self):
        cells = build_cells(one_cell_each(), None, root="nuclei")
        table = cell_features(
            cells, {"cytoplasm": measured([1, 2, 3], count=[5, 6, 7], note=["a", "b", "c"])}
        )
        assert "cytoplasm.note" not in table.columns

    def test_the_shape_does_not_depend_on_how_many_children_this_field_had(self):
        """The same protocol on two fields has to give the same columns, or
        the tables cannot be pooled."""
        crowded = build_cells(a_cell_with_organelles(), None, root="nuclei")
        sparse = build_cells(
            AssociationSet(
                [
                    link("cytoplasm", 1, "nuclei", 1),
                    link("lysosome", 10, "cytoplasm", 1, mode="many_to_one"),
                ]
            ),
            None,
            root="nuclei",
        )
        one = cell_features(crowded, {"lysosome": measured([10, 11, 12], count=[2, 3, 4])})
        two = cell_features(sparse, {"lysosome": measured([10], count=[2])})
        assert list(one.columns) == list(two.columns)


class TestPersistence:
    def test_cells_round_trip(self):
        cells = build_cells(a_cell_with_organelles(), None, root="nuclei")
        restored = CellSet.from_dict(cells.to_dict())
        assert len(restored) == 1
        assert restored.cell(1).objects("lysosome") == cells.cell(1).objects("lysosome")

    def test_which_roles_are_single_survives(self):
        """Or a reloaded cell set would build a differently shaped table."""
        cells = build_cells(a_cell_with_organelles(), None, root="nuclei")
        assert CellSet.from_dict(cells.to_dict()).single_roles == cells.single_roles

    def test_the_unclaimed_objects_survive(self):
        cells = CellSet([Cell(1, ObjectRef("nuclei", 1))], [ObjectRef("lysosome", 20)])
        assert CellSet.from_dict(cells.to_dict()).unclaimed == [ObjectRef("lysosome", 20)]

    def test_a_file_round_trips(self, tmp_path):
        cells = build_cells(a_cell_with_organelles(), None, root="nuclei")
        assert len(load_cells(save_cells(cells, tmp_path / "cells.json"))) == 1

    def test_a_newer_version_is_refused_clearly(self):
        with pytest.raises(ValueError, match="newer than this VTEA"):
            CellSet.from_dict({"vtea_cell_version": 99, "cells": []})
