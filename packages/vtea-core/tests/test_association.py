"""The link between two segmentations, and the exact case of a derived one."""

import json

import numpy as np
import pytest
from vtea_core.objects import (
    ASSIGNED,
    ASSOCIATION_FORMAT_VERSION,
    DERIVED,
    Association,
    AssociationSet,
    ObjectRef,
    associate_by_identity,
    load_associations,
    save_associations,
)
from vtea_core.segmentation import label_ring, label_shell


def link(child_id, parent_id, probability=1.0, alternatives=()):
    return Association(
        child=ObjectRef("cytoplasm_1", child_id),
        parent=ObjectRef("nuclei_1", parent_id),
        probability=probability,
        method="test",
        alternatives=[(ObjectRef("nuclei_1", other), score) for other, score in alternatives],
    )


class TestObjectRef:
    def test_it_names_the_segmentation_as_well_as_the_id(self):
        """An id alone is ambiguous the moment a protocol has two
        segmentations."""
        assert str(ObjectRef("watershed_split_1", 7)) == "watershed_split_1#7"

    def test_it_is_hashable_and_comparable(self):
        assert ObjectRef("a", 1) == ObjectRef("a", 1)
        assert ObjectRef("a", 1) != ObjectRef("b", 1)
        assert len({ObjectRef("a", 1), ObjectRef("a", 1)}) == 1

    def test_it_sorts_by_segmentation_then_id(self):
        refs = [ObjectRef("b", 1), ObjectRef("a", 2), ObjectRef("a", 1)]
        assert sorted(refs) == [ObjectRef("a", 1), ObjectRef("a", 2), ObjectRef("b", 1)]


class TestAssociationSet:
    def test_a_child_has_one_parent(self):
        associations = AssociationSet([link(1, 7)])
        assert associations.parent_of(ObjectRef("cytoplasm_1", 1)) == ObjectRef("nuclei_1", 7)

    def test_a_parent_may_have_many_children(self):
        associations = AssociationSet([link(1, 7), link(2, 7), link(3, 8)])
        assert associations.children_of(ObjectRef("nuclei_1", 7)) == [
            ObjectRef("cytoplasm_1", 1),
            ObjectRef("cytoplasm_1", 2),
        ]

    def test_children_can_be_narrowed_to_one_segmentation(self):
        """"The lysosomes of this cell" rather than everything attached."""
        associations = AssociationSet(
            [
                link(1, 7),
                Association(ObjectRef("lysosome_1", 5), ObjectRef("nuclei_1", 7)),
            ]
        )
        parent = ObjectRef("nuclei_1", 7)
        assert associations.children_of(parent, "lysosome_1") == [ObjectRef("lysosome_1", 5)]
        assert len(associations.children_of(parent)) == 2

    def test_re_associating_a_child_replaces_its_link(self):
        """Re-running should correct its own earlier answer, not leave two
        contradictory links behind."""
        associations = AssociationSet([link(1, 7)])
        associations.add(link(1, 9))
        assert len(associations) == 1
        assert associations.parent_of(ObjectRef("cytoplasm_1", 1)) == ObjectRef("nuclei_1", 9)
        assert associations.children_of(ObjectRef("nuclei_1", 7)) == []

    def test_removing_a_child_unlinks_it_from_its_parent_too(self):
        associations = AssociationSet([link(1, 7)])
        associations.remove_child(ObjectRef("cytoplasm_1", 1))
        assert len(associations) == 0
        assert associations.children_of(ObjectRef("nuclei_1", 7)) == []

    def test_an_unlinked_child_has_no_parent_rather_than_raising(self):
        assert AssociationSet().parent_of(ObjectRef("cytoplasm_1", 1)) is None

    def test_it_reports_which_segmentations_it_links(self):
        children, parents = AssociationSet([link(1, 7)]).segmentations()
        assert children == {"cytoplasm_1"}
        assert parents == {"nuclei_1"}


class TestUncertainty:
    def test_a_certain_link_has_no_alternatives(self):
        assert link(1, 7).is_certain is True

    def test_a_contested_link_is_not_certain(self):
        assert link(1, 7, 0.55, [(9, 0.44)]).is_certain is False

    def test_the_margin_is_the_gap_to_the_runner_up(self):
        assert link(1, 7, 0.55, [(9, 0.44)]).margin == pytest.approx(0.11)

    def test_a_link_with_no_runner_up_has_its_probability_as_the_margin(self):
        assert link(1, 7, 0.8).margin == pytest.approx(0.8)

    def test_the_contested_links_can_be_listed_worst_first(self):
        """The whole reason the alternatives are kept: surfacing the few
        percent of cells worth a human's attention."""
        associations = AssociationSet(
            [
                link(1, 7, 0.99),
                link(2, 7, 0.51, [(8, 0.49)]),
                link(3, 8, 0.70, [(7, 0.30)]),
            ]
        )
        contested = associations.uncertain(threshold=0.5)
        assert [a.child.object_id for a in contested] == [2, 3]

    def test_nothing_is_contested_at_a_permissive_threshold(self):
        assert AssociationSet([link(1, 7, 0.99)]).uncertain(threshold=0.5) == []


class TestPersistence:
    def make_set(self):
        return AssociationSet(
            [
                Association(
                    child=ObjectRef("cytosol_1", 1),
                    parent=ObjectRef("nuclei_1", 1),
                    relationship=DERIVED,
                    method="identity",
                ),
                link(2, 7, 0.55, [(9, 0.44)]),
            ]
        )

    def test_it_round_trips(self):
        restored = AssociationSet.from_dict(self.make_set().to_dict())
        assert len(restored) == 2
        assert restored.parent_of(ObjectRef("cytosol_1", 1)) == ObjectRef("nuclei_1", 1)

    def test_the_posterior_survives(self):
        restored = AssociationSet.from_dict(self.make_set().to_dict())
        contested = restored.link_for(ObjectRef("cytoplasm_1", 2))
        assert contested.probability == pytest.approx(0.55)
        assert contested.alternatives == [(ObjectRef("nuclei_1", 9), pytest.approx(0.44))]

    def test_how_the_link_was_made_survives(self):
        restored = AssociationSet.from_dict(self.make_set().to_dict())
        derived = restored.link_for(ObjectRef("cytosol_1", 1))
        assert derived.relationship == DERIVED
        assert derived.method == "identity"

    def test_a_file_round_trips(self, tmp_path):
        path = save_associations(self.make_set(), tmp_path / "associations.json")
        assert len(load_associations(path)) == 2

    def test_the_file_is_plain_versioned_json(self, tmp_path):
        path = save_associations(self.make_set(), tmp_path / "associations.json")
        data = json.loads(path.read_text())
        assert data["vtea_association_version"] == ASSOCIATION_FORMAT_VERSION
        assert data["associations"][0]["relationship"] == DERIVED

    def test_a_newer_version_is_refused_clearly(self):
        with pytest.raises(ValueError, match="newer than this VTEA"):
            AssociationSet.from_dict(
                {"vtea_association_version": ASSOCIATION_FORMAT_VERSION + 1, "associations": []}
            )

    def test_an_empty_set_round_trips(self):
        assert len(AssociationSet.from_dict(AssociationSet().to_dict())) == 0

    def test_the_default_relationship_is_assigned(self):
        assert Association(ObjectRef("a", 1), ObjectRef("b", 1)).relationship == ASSIGNED


class TestIdentityAssociation:
    """A derived segmentation keeps its parent's ids, so the link is exact."""

    @staticmethod
    def nuclei():
        labels = np.zeros((11, 11), dtype=np.int32)
        labels[2:4, 2:4] = 1
        labels[7:9, 7:9] = 2
        return labels

    def test_a_cytosol_ring_links_to_its_nucleus(self):
        nuclei = self.nuclei()
        associations = associate_by_identity(
            label_ring(nuclei, 1), nuclei, child_name="cytosol_1", parent_name="nuclei_1"
        )
        assert len(associations) == 2
        assert associations.parent_of(ObjectRef("cytosol_1", 2)) == ObjectRef("nuclei_1", 2)

    def test_an_envelope_links_to_its_nucleus(self):
        nuclei = self.nuclei()
        associations = associate_by_identity(
            label_shell(nuclei, inward=1, outward=1),
            nuclei,
            child_name="envelope_1",
            parent_name="nuclei_1",
        )
        assert len(associations) == 2

    def test_every_link_is_certain_and_marked_derived(self):
        nuclei = self.nuclei()
        associations = associate_by_identity(label_ring(nuclei, 1), nuclei)
        assert all(a.is_certain for a in associations)
        assert all(a.relationship == DERIVED for a in associations)
        assert all(a.method == "identity" for a in associations)

    def test_a_child_with_no_matching_parent_is_an_error_not_a_dropped_object(self):
        """Two independently segmented channels do not share ids, and
        pretending they do would silently mislabel every cell."""
        child = np.zeros((5, 5), dtype=np.int32)
        child[1, 1] = 9
        with pytest.raises(ValueError, match="no object of the same id"):
            associate_by_identity(child, self.nuclei())

    def test_the_error_names_the_two_segmentations(self):
        child = np.zeros((5, 5), dtype=np.int32)
        child[1, 1] = 9
        with pytest.raises(ValueError, match="'rings'.*'nuclei'"):
            associate_by_identity(child, self.nuclei(), child_name="rings", parent_name="nuclei")

    def test_orphans_can_be_tolerated_explicitly(self):
        child = np.zeros((11, 11), dtype=np.int32)
        child[2, 2] = 1
        child[5, 5] = 9  # no parent
        associations = associate_by_identity(child, self.nuclei(), require_parent=False)
        assert len(associations) == 1

    def test_a_parent_with_no_child_is_simply_unlinked(self):
        """An object that produced no derived counterpart - a nucleus too
        close to the edge to have a ring - is not an error."""
        nuclei = self.nuclei()
        child = np.zeros_like(nuclei)
        child[2, 2] = 1
        associations = associate_by_identity(child, nuclei)
        assert len(associations) == 1
        assert associations.children_of(ObjectRef("parent", 2)) == []

    def test_an_empty_child_gives_an_empty_set(self):
        assert len(associate_by_identity(np.zeros((5, 5), dtype=np.int32), self.nuclei())) == 0


class TestManualOverride:
    """Every automated assignment is wrong somewhere, and an analysis nobody
    can correct is one nobody can publish. What matters as much as the
    correction is that it stays visible as one."""

    def test_a_child_can_be_reassigned_by_hand(self):
        associations = AssociationSet([link(1, 7, 0.55, [(9, 0.44)])])
        associations.set_parent(ObjectRef("cytoplasm_1", 1), ObjectRef("nuclei_1", 9))
        assert associations.parent_of(ObjectRef("cytoplasm_1", 1)) == ObjectRef("nuclei_1", 9)

    def test_a_hand_made_link_says_so(self):
        associations = AssociationSet([link(1, 7, 0.55, [(9, 0.44)])])
        edited = associations.set_parent(ObjectRef("cytoplasm_1", 1), ObjectRef("nuclei_1", 9))
        assert edited.method == "manual"
        assert edited.probability == 1.0

    def test_the_answer_it_replaced_is_recorded(self):
        """So a reviewer can see afterwards what the algorithm had said, and
        how confident it had been about it."""
        associations = AssociationSet([link(1, 7, 0.55, [(9, 0.44)])])
        edited = associations.set_parent(ObjectRef("cytoplasm_1", 1), ObjectRef("nuclei_1", 9))
        assert edited.params["previous_parent"] == "nuclei_1#7"
        assert edited.params["previous_probability"] == pytest.approx(0.55)
        assert edited.params["previous_method"] == "test"

    def test_the_alternatives_the_algorithm_offered_are_kept(self):
        associations = AssociationSet([link(1, 7, 0.55, [(9, 0.44)])])
        edited = associations.set_parent(ObjectRef("cytoplasm_1", 1), ObjectRef("nuclei_1", 9))
        assert edited.alternatives == [(ObjectRef("nuclei_1", 9), pytest.approx(0.44))]

    def test_a_link_can_be_broken_by_hand(self):
        associations = AssociationSet([link(1, 7)])
        associations.unassign(ObjectRef("cytoplasm_1", 1))
        assert associations.parent_of(ObjectRef("cytoplasm_1", 1)) is None
        assert associations.unassigned == [ObjectRef("cytoplasm_1", 1)]

    def test_a_broken_link_is_still_marked_as_reviewed(self):
        """"Somebody looked at this and decided it has no parent" is a
        different statement from "the evidence ran out"."""
        associations = AssociationSet([link(1, 7)])
        associations.unassign(ObjectRef("cytoplasm_1", 1))
        assert associations.was_edited(ObjectRef("cytoplasm_1", 1))

    def test_an_untouched_link_is_not_marked(self):
        associations = AssociationSet([link(1, 7), link(2, 8)])
        associations.set_parent(ObjectRef("cytoplasm_1", 1), ObjectRef("nuclei_1", 9))
        assert associations.edited_by_hand == [ObjectRef("cytoplasm_1", 1)]

    def test_the_edits_survive_a_round_trip(self):
        associations = AssociationSet([link(1, 7, 0.55, [(9, 0.44)])])
        associations.set_parent(ObjectRef("cytoplasm_1", 1), ObjectRef("nuclei_1", 9))
        restored = AssociationSet.from_dict(associations.to_dict())
        assert restored.was_edited(ObjectRef("cytoplasm_1", 1))
        assert restored.link_for(ObjectRef("cytoplasm_1", 1)).method == "manual"

    def test_a_broken_link_survives_a_round_trip_as_reviewed(self):
        associations = AssociationSet([link(1, 7)])
        associations.unassign(ObjectRef("cytoplasm_1", 1))
        restored = AssociationSet.from_dict(associations.to_dict())
        assert restored.edited_by_hand == [ObjectRef("cytoplasm_1", 1)]

    def test_the_summary_counts_the_hand_edits(self):
        associations = AssociationSet([link(1, 7)])
        associations.set_parent(ObjectRef("cytoplasm_1", 1), ObjectRef("nuclei_1", 9))
        assert "1 set by hand" in associations.summary()

    def test_a_reviewed_link_stops_appearing_in_the_review_list(self):
        """A person's decision is not a posterior. A review list that keeps
        handing back the cases already reviewed is one nobody finishes - and
        the runners-up stay on the link, so the margin alone would not have
        dropped it."""
        associations = AssociationSet([link(1, 7, 0.51, [(9, 0.49)])])
        assert len(associations.uncertain(0.9)) == 1

        associations.set_parent(ObjectRef("cytoplasm_1", 1), ObjectRef("nuclei_1", 9))
        edited = associations.link_for(ObjectRef("cytoplasm_1", 1))
        assert edited.margin < 0.9  # still a close call by the numbers
        assert associations.uncertain(0.9) == []  # but not by anybody's question
