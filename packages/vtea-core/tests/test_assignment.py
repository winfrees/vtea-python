"""Turning candidate scores into one parent per child.

The tests that matter here are the ones where the obvious implementation is
wrong: a child that should have no parent at all, and a pair of children
whose per-child best answers contradict each other. Taking each child's
argmax gets both of those wrong while looking entirely reasonable.
"""

import numpy as np
import pytest
from vtea_core.objects import (
    MANY_TO_ONE,
    ONE_TO_ONE,
    CandidateScores,
    ObjectRef,
    assign,
    associate_objects,
    containment,
    posterior,
)
from vtea_core.objects.assignment import _blocks


def scores(mapping, *, parents=None):
    """CandidateScores straight from a {child: {parent: affinity}} dict."""
    parent_ids = parents or sorted({p for row in mapping.values() for p in row})
    return CandidateScores(
        scores={child: dict(row) for child, row in mapping.items()},
        child_ids=tuple(sorted(mapping)),
        parent_ids=tuple(parent_ids),
        method="test",
    )


class TestPosterior:
    def test_a_single_strong_candidate_is_nearly_certain(self):
        result = posterior(scores({1: {7: 1.0}}))
        assert result.for_child(1)[7] == pytest.approx(1 / 1.05)

    def test_two_candidates_split_in_proportion(self):
        result = posterior(scores({1: {7: 0.6, 9: 0.3}}), orphan_score=0.1)
        assert result.for_child(1)[7] == pytest.approx(0.6)
        assert result.for_child(1)[9] == pytest.approx(0.3)

    def test_the_probabilities_and_the_orphan_sum_to_one(self):
        result = posterior(scores({1: {7: 0.6, 9: 0.3}}), orphan_score=0.1)
        assert sum(result.for_child(1).values()) + result.orphan_probability(1) == pytest.approx(1.0)

    def test_a_weak_candidate_leaves_most_of_the_mass_on_orphan(self):
        """"This cytoplasm probably has no nucleus" has to be sayable, or
        every object is confidently assigned to something."""
        result = posterior(scores({1: {7: 0.01}}), orphan_score=0.05)
        assert result.orphan_probability(1) > result.for_child(1)[7]

    def test_a_child_with_no_candidate_is_certainly_an_orphan(self):
        candidates = CandidateScores(scores={}, child_ids=(1,), parent_ids=(7,))
        assert posterior(candidates).orphan_probability(1) == pytest.approx(1.0)

    def test_a_zero_orphan_score_makes_any_candidate_certain(self):
        """Occasionally what you want, usually not - which is why it isn't
        the default."""
        result = posterior(scores({1: {7: 0.01}}), orphan_score=0.0)
        assert result.for_child(1)[7] == pytest.approx(1.0)

    def test_the_orphan_score_is_recorded(self):
        assert posterior(scores({1: {7: 1.0}}), orphan_score=0.2).params["orphan_score"] == 0.2

    def test_a_negative_orphan_score_is_refused(self):
        with pytest.raises(ValueError, match="must not be negative"):
            posterior(scores({1: {7: 1.0}}), orphan_score=-0.1)


class TestManyToOne:
    def test_the_best_candidate_wins(self):
        matches = assign(posterior(scores({1: {7: 0.9, 9: 0.2}})), mode=MANY_TO_ONE)
        assert matches[0].parent_id == 7

    def test_one_parent_may_take_several_children(self):
        """A cytoplasm holds many lysosomes; nothing is competing."""
        matches = assign(posterior(scores({1: {7: 0.9}, 2: {7: 0.8}, 3: {7: 0.7}})))
        assert [match.parent_id for match in matches] == [7, 7, 7]

    def test_a_child_losing_to_the_orphan_option_gets_no_parent(self):
        matches = assign(posterior(scores({1: {7: 0.01}}), orphan_score=0.05))
        assert matches[0].parent_id is None

    def test_a_child_with_no_candidate_gets_no_parent(self):
        candidates = CandidateScores(scores={1: {7: 0.9}}, child_ids=(1, 2), parent_ids=(7,))
        matches = assign(posterior(candidates))
        assert [match.parent_id for match in matches] == [7, None]

    def test_the_threshold_refuses_a_link_the_evidence_does_not_support(self):
        confident = posterior(scores({1: {7: 0.5, 9: 0.45}}), orphan_score=0.05)
        assert assign(confident, min_probability=0.0)[0].parent_id == 7
        assert assign(confident, min_probability=0.9)[0].parent_id is None

    def test_the_runners_up_are_kept(self):
        matches = assign(posterior(scores({1: {7: 0.6, 9: 0.3, 11: 0.1}})))
        assert [parent for parent, _p in matches[0].alternatives] == [9, 11]

    def test_the_runners_up_are_ordered_best_first(self):
        matches = assign(posterior(scores({1: {7: 0.6, 9: 0.1, 11: 0.3}})))
        assert [parent for parent, _p in matches[0].alternatives] == [11, 9]

    def test_every_child_gets_a_match_in_id_order(self):
        candidates = CandidateScores(scores={3: {7: 0.9}}, child_ids=(1, 2, 3), parent_ids=(7,))
        assert [match.child_id for match in assign(posterior(candidates))] == [1, 2, 3]

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="unknown assignment mode"):
            assign(posterior(scores({1: {7: 0.9}})), mode="whatever")


class TestOneToOne:
    """"One and only one" is a global constraint, not a per-child one."""

    def test_two_children_cannot_share_a_parent(self):
        contested = posterior(scores({1: {7: 0.9, 9: 0.5}, 2: {7: 0.8, 9: 0.4}}))
        parents = [match.parent_id for match in assign(contested, mode=ONE_TO_ONE)]
        assert sorted(parents) == [7, 9]

    def test_per_child_argmax_would_have_given_both_the_same_parent(self):
        """The reason this mode exists: the greedy answer is not merely
        worse, it violates the constraint."""
        contested = posterior(scores({1: {7: 0.9, 9: 0.5}, 2: {7: 0.8, 9: 0.4}}))
        assert [match.parent_id for match in assign(contested, mode=MANY_TO_ONE)] == [7, 7]

    def test_the_global_optimum_can_give_a_child_its_second_choice(self):
        """Child 2 has nowhere else to go, so child 1 yields - which is what
        "global" means and why the alternatives are worth keeping."""
        contested = posterior(scores({1: {7: 0.9, 9: 0.6}, 2: {7: 0.8}}))
        matches = {match.child_id: match.parent_id for match in assign(contested, mode=ONE_TO_ONE)}
        assert matches == {1: 9, 2: 7}

    def test_a_child_left_over_has_no_parent(self):
        matches = assign(posterior(scores({1: {7: 0.9}, 2: {7: 0.8}})), mode=ONE_TO_ONE)
        assert sorted(match.parent_id is None for match in matches) == [False, True]

    def test_a_child_is_never_given_a_parent_it_was_not_a_candidate_for(self):
        """The infeasible entries of the cost matrix have to be expensive
        enough that the optimiser never buys one to complete a matching."""
        matches = assign(
            posterior(scores({1: {7: 0.9}, 2: {9: 0.9}, 3: {9: 0.1}}, parents=[7, 9])),
            mode=ONE_TO_ONE,
        )
        assert {match.child_id: match.parent_id for match in matches}[3] is None

    def test_independent_neighbourhoods_are_solved_separately(self):
        """Which is what makes the O(n^3) solve affordable: two clusters that
        cannot compete are two small problems, not one large one."""
        far_apart = {1: {7: 0.9, 9: 0.5}, 2: {7: 0.8}, 3: {21: 0.9}, 4: {21: 0.4, 23: 0.7}}
        assert len(_blocks(far_apart)) == 2

    def test_the_blocks_cover_every_child_exactly_once(self):
        far_apart = {1: {7: 0.9, 9: 0.5}, 2: {7: 0.8}, 3: {21: 0.9}, 4: {21: 0.4, 23: 0.7}}
        covered = [child for children, _parents in _blocks(far_apart) for child in children]
        assert sorted(covered) == [1, 2, 3, 4]

    def test_a_chain_of_shared_candidates_stays_one_block(self):
        """1 and 3 never compete directly, but both compete with 2."""
        chained = {1: {7: 0.9}, 2: {7: 0.5, 9: 0.5}, 3: {9: 0.9}}
        assert len(_blocks(chained)) == 1

    def test_blocking_does_not_change_the_answer(self):
        two_clusters = posterior(
            scores({1: {7: 0.9, 9: 0.5}, 2: {7: 0.8}, 3: {21: 0.9, 23: 0.5}, 4: {21: 0.8}})
        )
        matches = {m.child_id: m.parent_id for m in assign(two_clusters, mode=ONE_TO_ONE)}
        assert matches == {1: 9, 2: 7, 3: 23, 4: 21}


def nuclei_and_cytoplasms(offset=0):
    """Four cells in a row: a nucleus inside each cytoplasm, with known
    truth - cytoplasm k belongs to nucleus k."""
    cytoplasms = np.zeros((20, 80), dtype=np.int32)
    nuclei = np.zeros((20, 80), dtype=np.int32)
    for index in range(4):
        left = index * 20 + 2
        cytoplasms[3:17, left : left + 16] = index + 1
        centre = left + 8 + offset
        nuclei[8:12, centre - 2 : centre + 2] = index + 1
    return nuclei, cytoplasms


class TestAgainstKnownTruth:
    def test_every_cytoplasm_finds_its_own_nucleus(self):
        nuclei, cytoplasms = nuclei_and_cytoplasms()
        links = associate_objects(
            cytoplasms,
            nuclei,
            child_name="cytoplasm",
            parent_name="nucleus",
            method="containment",
            mode="one_to_one",
        )
        assert len(links) == 4
        assert all(link.child.object_id == link.parent.object_id for link in links)

    def test_a_cytoplasm_with_no_nucleus_is_left_unassigned(self):
        nuclei, cytoplasms = nuclei_and_cytoplasms()
        nuclei[nuclei == 3] = 0  # this cell lost its nucleus

        links = associate_objects(
            cytoplasms, nuclei, method="containment", mode="one_to_one", child_name="cytoplasm"
        )
        assert len(links) == 3
        assert [ref.object_id for ref in links.unassigned] == [3]

    def test_an_extra_nucleus_simply_goes_unused(self):
        """A parent with no child is not an error - the constraint is on
        children, and one-to-one is at most one, not exactly one."""
        nuclei, cytoplasms = nuclei_and_cytoplasms()
        cytoplasms[cytoplasms == 4] = 0
        links = associate_objects(cytoplasms, nuclei, method="containment", mode="one_to_one")
        assert len(links) == 3

    def test_boundary_distance_finds_the_same_answer_without_overlap(self):
        """Nuclei shrunk away from their cytoplasms - no containment to work
        from, so the geometry has to carry it."""
        nuclei, cytoplasms = nuclei_and_cytoplasms()
        cytoplasms[nuclei != 0] = 0  # cytosol only: the two no longer overlap

        links = associate_objects(
            cytoplasms,
            nuclei,
            method="boundary_distance",
            mode="one_to_one",
            max_distance=6,
        )
        assert len(links) == 4
        assert all(link.child.object_id == link.parent.object_id for link in links)

    def test_two_nuclei_in_one_cytoplasm_leaves_one_nucleus_over(self):
        """The multinucleate case with the flag off: exactly one nucleus wins
        the cytoplasm, and the other is visible as unassigned rather than
        silently sharing it."""
        nuclei, cytoplasms = nuclei_and_cytoplasms()
        nuclei[8:12, 26:30] = 5  # a second nucleus inside cytoplasm 2

        links = associate_objects(
            nuclei, cytoplasms, method="containment", mode="one_to_one", child_name="nucleus"
        )
        assert len(links) == 4
        assert [ref.object_id for ref in links.unassigned] == [5]

    def test_with_many_to_one_both_nuclei_keep_the_same_cytoplasm(self):
        """And this is the multinucleate case with the flag on: the same
        data, the same scores, a different constraint."""
        nuclei, cytoplasms = nuclei_and_cytoplasms()
        nuclei[8:12, 26:30] = 5

        links = associate_objects(
            nuclei,
            cytoplasms,
            method="containment",
            mode="many_to_one",
            child_name="nucleus",
            parent_name="cytoplasm",
        )
        assert len(links) == 5
        assert links.unassigned == []
        shared = links.parent_of(ObjectRef("nucleus", 2))
        assert links.parent_of(ObjectRef("nucleus", 5)) == shared
        assert links.children_of(shared) == [ObjectRef("nucleus", 2), ObjectRef("nucleus", 5)]

    def test_a_contested_child_keeps_the_parent_it_lost_to(self):
        nuclei = np.zeros((10, 20), dtype=np.int32)
        nuclei[4:6, 4:6] = 1
        nuclei[4:6, 14:16] = 2
        punctum = np.zeros((10, 20), dtype=np.int32)
        punctum[4:6, 9:11] = 1  # exactly between them

        links = associate_objects(punctum, nuclei, method="centroid_distance", max_distance=20)
        link = next(iter(links))
        assert link.alternatives
        assert link.margin < 0.2

    def test_the_method_and_its_parameters_are_on_every_link(self):
        nuclei, cytoplasms = nuclei_and_cytoplasms()
        links = associate_objects(
            cytoplasms, nuclei, method="boundary_distance", max_distance=6, mode="one_to_one"
        )
        link = next(iter(links))
        assert link.method == "boundary_distance"
        assert link.params["max_distance"] == 6.0
        assert link.params["mode"] == "one_to_one"

    def test_the_segmentations_are_named_on_every_reference(self):
        nuclei, cytoplasms = nuclei_and_cytoplasms()
        links = associate_objects(
            cytoplasms, nuclei, child_name="cytoplasm_1", parent_name="nuclei_1"
        )
        link = next(iter(links))
        assert link.child.segmentation == "cytoplasm_1"
        assert link.parent.segmentation == "nuclei_1"

    def test_containment_is_recorded_as_a_containment_relationship(self):
        nuclei, cytoplasms = nuclei_and_cytoplasms()
        links = associate_objects(nuclei, cytoplasms, method="containment")
        assert all(link.relationship == "contained" for link in links)

    def test_a_distance_based_link_is_recorded_as_assigned(self):
        nuclei, cytoplasms = nuclei_and_cytoplasms()
        links = associate_objects(nuclei, cytoplasms, method="centroid_distance", max_distance=10)
        assert all(link.relationship == "assigned" for link in links)


class TestSummary:
    def test_it_reports_what_was_linked_and_what_was_not(self):
        nuclei, cytoplasms = nuclei_and_cytoplasms()
        nuclei[nuclei == 3] = 0
        links = associate_objects(
            cytoplasms,
            nuclei,
            child_name="cytoplasm_1",
            parent_name="nuclei_1",
            mode="one_to_one",
        )
        summary = links.summary()
        assert "cytoplasm_1 -> nuclei_1" in summary
        assert "3 linked" in summary
        assert "1 unassigned" in summary

    def test_a_clean_run_says_nothing_about_unassigned_objects(self):
        nuclei, cytoplasms = nuclei_and_cytoplasms()
        links = associate_objects(cytoplasms, nuclei, mode="one_to_one")
        assert "unassigned" not in links.summary()


class TestUnassignedPersistence:
    def test_an_unassigned_child_survives_a_round_trip(self):
        """Losing them on save would turn "17 cytoplasms had no nucleus" back
        into an invisible absence."""
        from vtea_core.objects import AssociationSet

        nuclei, cytoplasms = nuclei_and_cytoplasms()
        nuclei[nuclei == 3] = 0
        links = associate_objects(cytoplasms, nuclei, mode="one_to_one")

        restored = AssociationSet.from_dict(links.to_dict())
        assert [ref.object_id for ref in restored.unassigned] == [3]

    def test_assigning_a_child_clears_it_from_the_unassigned_list(self):
        from vtea_core.objects import Association, AssociationSet, ObjectRef

        links = AssociationSet()
        child = ObjectRef("cytoplasm_1", 3)
        links.add_unassigned(child)
        links.add(Association(child, ObjectRef("nuclei_1", 3)))
        assert links.unassigned == []

    def test_containment_scores_are_available_on_their_own(self):
        """The scoring step is usable without the assignment - which is what
        makes a different assignment strategy a change of one call."""
        nuclei, cytoplasms = nuclei_and_cytoplasms()
        assert containment(nuclei, cytoplasms).n_candidates == 4
