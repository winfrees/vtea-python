"""Scoring which parent an independently segmented child might belong to.

Two segmentations made in different channels share no ids, so every one of
these numbers is the evidence standing in for a relationship nothing in the
data states. What matters is that the evidence is sparse (only nearby
parents are proposed, which is what keeps the global assignment tractable),
that it is on a comparable scale, and that distances are physical wherever
the voxel size is known.
"""

import numpy as np
import pytest
from vtea_core.data import Spacing
from vtea_core.objects import (
    BOUNDARY_DISTANCE,
    CENTROID_DISTANCE,
    CONTAINMENT,
    boundary_distance,
    centroid_distance,
    containment,
    score_candidates,
)

ISOTROPIC = Spacing((1.0, 1.0, 1.0), source="user")
ANISOTROPIC = Spacing((4.0, 1.0, 1.0), source="user")


def two_cells():
    """Two parents side by side, each holding one child entirely."""
    parents = np.zeros((12, 24), dtype=np.int32)
    parents[2:10, 2:10] = 1
    parents[2:10, 14:22] = 2

    children = np.zeros((12, 24), dtype=np.int32)
    children[4:8, 4:8] = 1
    children[4:8, 16:20] = 2
    return children, parents


class TestContainment:
    def test_a_child_wholly_inside_one_parent_scores_one(self):
        children, parents = two_cells()
        scores = containment(children, parents)
        assert scores.for_child(1) == {1: pytest.approx(1.0)}

    def test_a_child_straddling_two_parents_scores_against_both(self):
        """Which is the ambiguity the posterior is supposed to carry, not
        something to resolve here."""
        parents = np.zeros((6, 8), dtype=np.int32)
        parents[:, :4] = 1
        parents[:, 4:] = 2
        children = np.zeros((6, 8), dtype=np.int32)
        children[2:4, 2:6] = 1  # half in each

        scores = containment(children, parents).for_child(1)
        assert scores == {1: pytest.approx(0.5), 2: pytest.approx(0.5)}

    def test_the_fraction_is_of_the_child_not_the_parent(self):
        """A small child in a huge parent is fully contained; scoring by the
        parent's volume would call it a poor match."""
        parents = np.zeros((10, 10), dtype=np.int32)
        parents[1:9, 1:9] = 1
        children = np.zeros((10, 10), dtype=np.int32)
        children[4, 4] = 1
        assert containment(children, parents).for_child(1) == {1: pytest.approx(1.0)}

    def test_a_child_outside_every_parent_has_no_candidate(self):
        """Absent, not zero - "not a candidate" is a stronger statement than
        "a bad one", and it is what keeps the matrix sparse."""
        children, parents = two_cells()
        children[0, 12] = 3
        scores = containment(children, parents)
        assert scores.for_child(3) == {}
        assert 3 in scores.child_ids  # still a child that needs an answer

    def test_partial_overlap_scores_the_overlapping_fraction(self):
        parents = np.zeros((6, 8), dtype=np.int32)
        parents[:, :4] = 1
        children = np.zeros((6, 8), dtype=np.int32)
        children[2:4, 2:6] = 1  # half inside, half in background
        assert containment(children, parents).for_child(1) == {1: pytest.approx(0.5)}

    def test_it_needs_no_distance_parameter(self):
        children, parents = two_cells()
        assert containment(children, parents).params == {}

    def test_it_records_its_own_method(self):
        children, parents = two_cells()
        assert containment(children, parents).method == CONTAINMENT

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="shapes differ"):
            containment(np.zeros((4, 4), dtype=np.int32), np.zeros((5, 5), dtype=np.int32))

    def test_a_float_image_is_refused(self):
        with pytest.raises(TypeError, match="label image"):
            containment(np.zeros((4, 4)), np.zeros((4, 4), dtype=np.int32))


class TestCentroidDistance:
    def test_the_nearer_parent_scores_higher(self):
        children, parents = two_cells()
        scores = centroid_distance(children, parents, max_distance=30).for_child(1)
        assert scores[1] > scores[2]

    def test_a_parent_beyond_the_reach_is_not_a_candidate(self):
        """max_distance is the neighbourhood restriction as well as the
        falloff - the two are the same number on purpose."""
        children, parents = two_cells()
        scores = centroid_distance(children, parents, max_distance=6).for_child(1)
        assert set(scores) == {1}

    def test_coincident_centres_score_one(self):
        parents = np.zeros((6, 6), dtype=np.int32)
        parents[1:5, 1:5] = 1
        children = np.zeros((6, 6), dtype=np.int32)
        children[2:4, 2:4] = 1
        assert centroid_distance(children, parents, max_distance=5).for_child(1)[1] == pytest.approx(
            1.0
        )

    def test_a_tall_z_step_puts_a_parent_out_of_reach(self):
        """The reason spacing is threaded through: two slices apart is 2
        units in an isotropic stack and 8 in this one."""
        children = np.zeros((5, 6, 6), dtype=np.int32)
        children[0, 2:4, 2:4] = 1
        parents = np.zeros((5, 6, 6), dtype=np.int32)
        parents[2, 2:4, 2:4] = 1

        assert centroid_distance(
            children, parents, spacing=ISOTROPIC, max_distance=3
        ).for_child(1) == {1: pytest.approx(1 / 3)}
        assert (
            centroid_distance(children, parents, spacing=ANISOTROPIC, max_distance=3).for_child(1)
            == {}
        )

    def test_an_unknown_spacing_measures_in_voxels(self):
        children, parents = two_cells()
        assert centroid_distance(children, parents, max_distance=30).scores == centroid_distance(
            children, parents, spacing=Spacing.unknown(2), max_distance=30
        ).scores

    def test_it_records_the_reach_it_used(self):
        children, parents = two_cells()
        scores = centroid_distance(children, parents, max_distance=7)
        assert scores.method == CENTROID_DISTANCE
        assert scores.params == {"max_distance": 7.0}

    def test_a_non_positive_reach_is_refused(self):
        children, parents = two_cells()
        with pytest.raises(ValueError, match="must be positive"):
            centroid_distance(children, parents, max_distance=0)


class TestBoundaryDistance:
    def test_overlapping_objects_score_one(self):
        parents = np.zeros((6, 8), dtype=np.int32)
        parents[:, :4] = 1
        children = np.zeros((6, 8), dtype=np.int32)
        children[2:4, 2:6] = 1  # half inside the parent
        assert boundary_distance(children, parents, max_distance=4).for_child(1)[1] == pytest.approx(
            1.0
        )

    def test_touching_objects_are_one_voxel_step_apart(self):
        """The gap is between nearest voxels, so adjacency is one step rather
        than zero - worth pinning down, since "touching" reads as zero."""
        parents = np.zeros((6, 8), dtype=np.int32)
        parents[:, :4] = 1
        children = np.zeros((6, 8), dtype=np.int32)
        children[2:4, 4:6] = 1  # right up against the parent, no overlap
        assert boundary_distance(children, parents, max_distance=4).for_child(1)[1] == pytest.approx(
            1.0 - 1.0 / 4.0
        )

    def test_a_gap_lowers_the_score(self):
        parents = np.zeros((6, 12), dtype=np.int32)
        parents[:, :4] = 1
        near = np.zeros((6, 12), dtype=np.int32)
        near[2:4, 5:7] = 1
        far = np.zeros((6, 12), dtype=np.int32)
        far[2:4, 8:10] = 1

        assert (
            boundary_distance(near, parents, max_distance=8).for_child(1)[1]
            > boundary_distance(far, parents, max_distance=8).for_child(1)[1]
        )

    def test_it_is_not_fooled_by_a_large_parent(self):
        """A punctum against the rim of a big parent is close to it, though
        their centroids are far apart - which is the case centroid distance
        gets wrong."""
        parents = np.zeros((24, 24), dtype=np.int32)
        parents[2:22, 2:22] = 1
        children = np.zeros((24, 24), dtype=np.int32)
        children[3, 3] = 1

        assert boundary_distance(children, parents, max_distance=5).for_child(1) == {
            1: pytest.approx(1.0)
        }
        assert centroid_distance(children, parents, max_distance=5).for_child(1) == {}

    def test_a_parent_beyond_the_reach_is_not_a_candidate(self):
        children, parents = two_cells()
        assert set(boundary_distance(children, parents, max_distance=3).for_child(1)) == {1}

    def test_both_parents_are_candidates_when_both_are_within_reach(self):
        children, parents = two_cells()
        assert set(boundary_distance(children, parents, max_distance=12).for_child(1)) == {1, 2}

    def test_a_tall_z_step_puts_a_parent_out_of_reach(self):
        children = np.zeros((5, 6, 6), dtype=np.int32)
        children[0, 2:4, 2:4] = 1
        parents = np.zeros((5, 6, 6), dtype=np.int32)
        parents[2, 2:4, 2:4] = 1

        assert boundary_distance(children, parents, spacing=ISOTROPIC, max_distance=3).for_child(1)
        assert (
            boundary_distance(children, parents, spacing=ANISOTROPIC, max_distance=3).for_child(1)
            == {}
        )

    def test_every_child_gets_its_own_nearest_distance(self):
        """Not the nearest voxel of whichever child came first."""
        parents = np.zeros((8, 12), dtype=np.int32)
        parents[:, :4] = 1
        children = np.zeros((8, 12), dtype=np.int32)
        children[1:3, 4:6] = 1  # touching
        children[5:7, 8:10] = 2  # four voxels away

        scores = boundary_distance(children, parents, max_distance=8)
        assert scores.for_child(1)[1] > scores.for_child(2)[1]

    def test_it_records_its_own_method_and_reach(self):
        children, parents = two_cells()
        scores = boundary_distance(children, parents, max_distance=5)
        assert scores.method == BOUNDARY_DISTANCE
        assert scores.params == {"max_distance": 5.0}


class TestDispatch:
    def test_each_method_is_reachable_by_name(self):
        children, parents = two_cells()
        for method in (CONTAINMENT, CENTROID_DISTANCE, BOUNDARY_DISTANCE):
            assert score_candidates(children, parents, method=method, max_distance=30).method == (
                method
            )

    def test_an_unknown_method_is_refused_with_the_options(self):
        children, parents = two_cells()
        with pytest.raises(ValueError, match="unknown scoring method"):
            score_candidates(children, parents, method="vibes")


class TestSparsity:
    def test_only_nearby_parents_are_proposed(self):
        """The property the global assignment depends on: with 20 parents in
        a row, a child is a candidate for a handful, not for all of them."""
        parents = np.zeros((6, 200), dtype=np.int32)
        for index in range(20):
            parents[2:4, index * 10 + 2 : index * 10 + 6] = index + 1
        children = np.zeros((6, 200), dtype=np.int32)
        children[2:4, 3:5] = 1

        scores = boundary_distance(children, parents, max_distance=12)
        assert 0 < len(scores.for_child(1)) < 5
        assert len(scores.parent_ids) == 20

    def test_the_candidate_count_is_reported(self):
        children, parents = two_cells()
        assert containment(children, parents).n_candidates == 2

    def test_an_empty_child_image_scores_nothing(self):
        _children, parents = two_cells()
        scores = containment(np.zeros_like(parents), parents)
        assert len(scores.child_ids) == 0
        assert scores.n_candidates == 0
