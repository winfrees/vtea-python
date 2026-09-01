"""The candidate scores as three arrays rather than a dict of dicts.

The representation changed; the answers must not. These tests pin both
halves of that: the accessors behave exactly as they did, and the thing
that motivated the change - what a candidate pair weighs - is actually
smaller. A `dict[int, dict[int, float]]` is roughly 200 bytes per pair once
the boxed integers and the inner dicts are counted, and at ten million
children that is the ceiling rather than the image.
"""

import numpy as np
import pytest
from vtea_core.objects import (
    MANY_TO_ONE,
    ONE_TO_ONE,
    CandidateScores,
    assign,
    containment,
    posterior,
)
from vtea_core.objects.assignment import Posterior, _blocks


def scores(mapping, *, parents=None):
    parent_ids = parents or sorted({p for row in mapping.values() for p in row})
    return CandidateScores(
        scores={child: dict(row) for child, row in mapping.items()},
        child_ids=tuple(sorted(mapping)),
        parent_ids=tuple(parent_ids),
        method="test",
    )


class TestTheRepresentation:
    def test_a_mapping_round_trips(self):
        mapping = {1: {7: 0.9, 9: 0.5}, 2: {7: 0.8}}
        assert scores(mapping).scores == mapping

    def test_the_arrays_and_the_mapping_agree(self):
        """Both constructors exist because both callers exist - the scoring
        functions build arrays, a caller with three pairs writes a dict."""
        from_arrays = CandidateScores(
            child_ids=(1, 2),
            parent_ids=(7, 9),
            child_index=[0, 0, 1],
            parent_index=[0, 1, 0],
            affinity=[0.9, 0.5, 0.8],
        )
        assert from_arrays.scores == {1: {7: 0.9, 9: 0.5}, 2: {7: 0.8}}

    def test_rows_are_found_however_they_were_given(self):
        """Out-of-order input is a caller's business, not a silent wrong
        answer: every lookup here is a binary search into sorted ids."""
        shuffled = CandidateScores(
            child_ids=(5, 1, 3),
            parent_ids=(9, 7),
            child_index=[2, 0, 1],
            parent_index=[0, 1, 0],
            affinity=[0.3, 0.9, 0.5],
        )
        assert shuffled.for_child(5) == {7: 0.9}

    def test_a_child_with_no_candidates_reads_as_empty(self):
        assert scores({1: {7: 0.9}}, parents=[7]).for_child(2) == {}

    def test_candidates_for_gives_arrays(self):
        parents, values = scores({1: {7: 0.9, 9: 0.5}}).candidates_for(1)
        assert sorted(parents.tolist()) == [7, 9]
        assert values.dtype == np.float64

    def test_a_pair_naming_an_unknown_child_is_refused(self):
        """Rather than dropped: a score against an object that is not in the
        segmentation means the two came from different runs."""
        with pytest.raises(KeyError, match="child 4"):
            CandidateScores(scores={4: {7: 0.9}}, child_ids=(1, 2), parent_ids=(7,))

    def test_a_pair_weighs_sixteen_bytes(self):
        """The claim the change was made on. 10^7 children with three
        candidates each is half a gigabyte here and about six in dicts."""
        many = CandidateScores(
            child_ids=np.arange(1, 10_001),
            parent_ids=np.arange(1, 10_001),
            child_index=np.repeat(np.arange(10_000), 3),
            parent_index=np.tile([0, 1, 2], 10_000),
            affinity=np.full(30_000, 0.5),
        )
        assert many.n_candidates == 30_000
        per_pair = (many.nbytes - many.child_ids.nbytes - many.parent_ids.nbytes) / 30_000
        assert per_pair == 16

    def test_it_says_what_it_holds(self):
        assert "2 children" in repr(scores({1: {7: 0.9}, 2: {7: 0.5}}))


class TestTheAnswersAreUnchanged:
    """Against a reference computed from the mapping in plain Python, so the
    test is not the implementation restated."""

    mapping = {1: {7: 0.9, 9: 0.5}, 2: {7: 0.8}, 3: {21: 0.9}, 4: {21: 0.4, 23: 0.7}}

    def test_the_posterior_is_the_normalised_affinity(self):
        result = posterior(scores(self.mapping), orphan_score=0.05)
        for child, row in self.mapping.items():
            total = sum(row.values()) + 0.05
            for parent, affinity in row.items():
                assert result.for_child(child)[parent] == pytest.approx(affinity / total)
            assert result.orphan_probability(child) == pytest.approx(0.05 / total)

    def test_many_to_one_takes_each_childs_best(self):
        matches = assign(posterior(scores(self.mapping)), mode=MANY_TO_ONE)
        chosen = {match.child_id: match.parent_id for match in matches}
        assert chosen == {1: 7, 2: 7, 3: 21, 4: 23}

    def test_one_to_one_still_gives_a_child_its_second_choice(self):
        """The whole point of the global solve, and the thing a per-child
        argmax gets wrong: 1 yields 7 to 2, which has nothing else."""
        matches = assign(posterior(scores(self.mapping)), mode=ONE_TO_ONE)
        chosen = {match.child_id: match.parent_id for match in matches}
        assert chosen == {1: 9, 2: 7, 3: 21, 4: 23}

    def test_alternatives_are_the_rest_of_the_row_worst_last(self):
        matches = assign(posterior(scores(self.mapping)), mode=ONE_TO_ONE)
        first = next(match for match in matches if match.child_id == 1)
        assert [parent for parent, _p in first.alternatives] == [7]

    def test_containment_from_an_image_agrees_with_the_mapping_form(self):
        children = np.zeros((10, 10), dtype=np.int32)
        parents = np.zeros((10, 10), dtype=np.int32)
        children[2:4, 2:4] = 1
        parents[0:6, 0:6] = 5
        parents[2:3, 2:4] = 7
        result = containment(children, parents)
        assert result.for_child(1) == pytest.approx({5: 0.5, 7: 0.5})


class TestBlocksByUnionFind:
    def test_a_chain_is_one_block_and_a_gap_is_two(self):
        chained = posterior(scores({1: {7: 0.9}, 2: {7: 0.5, 9: 0.5}, 3: {9: 0.9}}))
        apart = posterior(scores({1: {7: 0.9}, 2: {9: 0.5}}))
        assert len(_blocks(chained)) == 1
        assert len(_blocks(apart)) == 2

    def test_a_child_with_no_candidate_is_in_no_block(self):
        """It needs no solving - it has already lost to the orphan option -
        and putting it in a block would make an empty row of the cost
        matrix."""
        lonely = posterior(scores({1: {7: 0.9}}, parents=[7]))
        lonely = Posterior(
            child_ids=np.array([1, 2]),
            parent_ids=lonely.parent_ids,
            child_index=lonely.child_index,
            parent_index=lonely.parent_index,
            probability=lonely.probability,
            orphan=[lonely.orphan[0], 1.0],
        )
        assert [children for children, _parents in _blocks(lonely)] == [[1]]

    def test_nothing_at_all_is_no_blocks(self):
        assert _blocks(posterior(CandidateScores(child_ids=(1, 2), parent_ids=(7,)))) == []

    def test_a_long_chain_does_not_recurse(self):
        """Path compression, not recursion: a thousand cytoplasms in a row,
        each sharing a nucleus with the next, is one block and no stack."""
        chain = {child: {child: 0.9, child + 1: 0.5} for child in range(1, 1001)}
        assert len(_blocks(posterior(scores(chain)))) == 1
