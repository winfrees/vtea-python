"""Which cell owns a voxel, when the honest answer is "probably that one".

The tests that matter are the ones about *doubt*: that a voxel halfway
between two nuclei comes out as a coin toss rather than as a confident
answer, that the doubt survives into the measurements, and that a person can
overrule it and have that fact recorded.
"""

import numpy as np
import pytest
from vtea_core.data import Spacing
from vtea_core.measurements import (
    extract_measurements,
    weighted_measurements,
    weighted_measurements_by_channel,
)
from vtea_core.objects import (
    DISTANCE,
    Ownership,
    distance_ownership,
    load_ownership,
)
from vtea_core.segmentation import watershed_ownership

ISOTROPIC = Spacing((1.0, 1.0, 1.0), source="user")
ANISOTROPIC = Spacing((4.0, 1.0, 1.0), source="user")


def two_markers():
    """Two nuclei at either end of one shared region."""
    markers = np.zeros((10, 30), dtype=np.int32)
    markers[4:6, 4:6] = 1
    markers[4:6, 24:26] = 2
    region = np.zeros((10, 30), dtype=bool)
    region[2:8, 2:28] = True
    return markers, region


class TestDistanceOwnership:
    def test_every_voxel_within_reach_gets_an_owner(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        assert (ownership.hard()[region] != 0).all()

    def test_nothing_outside_the_region_is_claimed(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        assert (ownership.hard()[~region] == 0).all()

    def test_a_voxel_inside_a_marker_is_certain(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=4.0)
        assert ownership.confidence()[5, 5] == pytest.approx(1.0)

    def test_a_voxel_between_two_markers_is_a_coin_toss(self):
        """The whole point: the same voxel a watershed hands to one cell
        without comment."""
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        assert ownership.confidence()[5, 15] == pytest.approx(0.5, abs=0.05)
        assert ownership.margin()[5, 15] < 0.1

    def test_the_nearer_marker_still_wins(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        assert ownership.hard()[5, 6] == 1
        assert ownership.hard()[5, 23] == 2

    def test_a_small_falloff_makes_the_split_nearly_hard(self):
        """`falloff` is the width of the zone of doubt, so it is the control
        for how much of a boundary is treated as genuinely ambiguous."""
        markers, region = two_markers()
        sharp = distance_ownership(markers, region, falloff=0.5, reach=40)
        broad = distance_ownership(markers, region, falloff=8.0, reach=40)
        assert sharp.contested(0.9).sum() < broad.contested(0.9).sum()

    def test_the_probabilities_at_a_voxel_sum_to_at_most_one(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        assert ownership.probabilities.sum(axis=0).max() <= 1.0 + 1e-9

    def test_with_two_markers_they_sum_to_exactly_one(self):
        """Nothing is lost below the top k when there is nothing below it."""
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0, top_k=2)
        total = ownership.probabilities.sum(axis=0)
        assert total[region].min() == pytest.approx(1.0)

    def test_a_third_owner_is_kept_when_there_is_room_for_it(self):
        markers, region = two_markers()
        markers[4:6, 14:16] = 3
        ownership = distance_ownership(markers, region, falloff=8.0, top_k=3)
        assert (ownership.owners[2] != 0).any()

    def test_only_the_best_two_are_kept_by_default(self):
        markers, region = two_markers()
        markers[4:6, 14:16] = 3
        assert distance_ownership(markers, region, falloff=8.0).top_k == 2

    def test_a_marker_beyond_the_reach_does_not_claim(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=1.0, reach=4.0)
        assert ownership.hard()[5, 15] == 0  # too far from either

    def test_the_falloff_is_physical_where_the_spacing_is_known(self):
        """A claim reaching two slices in an isotropic stack reaches none in
        one whose z-step is four times the pixel size."""
        markers = np.zeros((7, 9, 9), dtype=np.int32)
        markers[3, 4, 4] = 1
        region = np.ones((7, 9, 9), dtype=bool)

        flat = distance_ownership(markers, region, falloff=1.0, reach=2.0, spacing=ISOTROPIC)
        tall = distance_ownership(markers, region, falloff=1.0, reach=2.0, spacing=ANISOTROPIC)
        assert flat.hard()[1, 4, 4] == 1
        assert tall.hard()[1, 4, 4] == 0

    def test_it_records_how_it_was_made(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=3.0, segmentation="nuclei_1")
        assert ownership.method == DISTANCE
        assert ownership.params["falloff"] == 3.0
        assert ownership.segmentation == "nuclei_1"

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="shapes differ"):
            distance_ownership(np.zeros((4, 4), dtype=np.int32), np.zeros((5, 5), dtype=bool))

    def test_a_non_positive_falloff_is_refused(self):
        markers, region = two_markers()
        with pytest.raises(ValueError, match="falloff must be positive"):
            distance_ownership(markers, region, falloff=0)

    def test_it_agrees_with_the_watershed_where_the_region_is_convex(self):
        """Not a claim that the two are interchangeable - they are not, and
        the watershed follows the region's shape where this does not - but a
        symmetric blob should not have them disagreeing about which half is
        whose."""
        markers, region = two_markers()
        soft = distance_ownership(markers, region, falloff=6.0).hard()
        hard = watershed_ownership(markers, region)
        agreement = (soft == hard)[region].mean()
        assert agreement > 0.95


class TestConfidenceAndContest:
    def test_the_confidence_map_is_the_winning_probability(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        np.testing.assert_allclose(ownership.confidence(), ownership.probabilities[0])

    def test_contested_voxels_are_the_ones_below_the_threshold(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        contested = ownership.contested(0.8)
        assert contested[5, 15]
        assert not contested[5, 5]

    def test_unowned_voxels_are_not_contested(self):
        """Background is not a close call; it is not a call at all."""
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        assert not ownership.contested(0.99)[0, 0]

    def test_a_low_probability_is_not_the_same_as_a_close_call(self):
        """One cell weakly claiming a far voxel is confident about who owns
        it; two cells claiming it equally are not. `margin` is what tells
        them apart."""
        markers = np.zeros((10, 30), dtype=np.int32)
        markers[4:6, 4:6] = 1
        region = np.zeros((10, 30), dtype=bool)
        region[2:8, 2:28] = True
        ownership = distance_ownership(markers, region, falloff=2.0, reach=30)
        assert ownership.confidence()[5, 20] == pytest.approx(1.0)
        assert ownership.margin()[5, 20] == pytest.approx(1.0)

    def test_the_summary_reports_how_much_was_contested(self):
        markers, region = two_markers()
        summary = distance_ownership(markers, region, falloff=6.0).summary(0.8)
        assert "2 owners" in summary
        assert "contested" in summary


class TestOverride:
    def test_a_region_can_be_given_to_one_owner(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        patch = np.zeros_like(region)
        patch[5, 15] = True

        ownership.override(patch, 2)
        assert ownership.hard()[5, 15] == 2
        assert ownership.confidence()[5, 15] == pytest.approx(1.0)

    def test_an_override_clears_the_runners_up(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        patch = np.zeros_like(region)
        patch[5, 15] = True
        ownership.override(patch, 2)
        assert ownership.probabilities[1][5, 15] == 0.0

    def test_an_overridden_voxel_is_marked_as_set_by_hand(self):
        """A correction that becomes indistinguishable from an inference is
        worse than no correction: nobody can tell later what was reviewed."""
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        patch = np.zeros_like(region)
        patch[5, 15] = True
        ownership.override(patch, 2)
        assert ownership.manual[5, 15]
        assert not ownership.manual[5, 5]
        assert "set by hand" in ownership.summary()

    def test_a_mismatched_region_is_refused(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        with pytest.raises(ValueError, match="shapes differ"):
            ownership.override(np.zeros((3, 3), dtype=bool), 1)


class TestFromLabels:
    def test_a_hard_label_image_becomes_a_certain_ownership(self):
        """So a watershed result and a probabilistic one are measured by the
        same code, and the difference shows up in the numbers."""
        markers, region = two_markers()
        ownership = Ownership.from_labels(watershed_ownership(markers, region))
        assert ownership.confidence()[region].min() == pytest.approx(1.0)
        assert ownership.contested(0.9).sum() == 0

    def test_it_keeps_the_label_ids(self):
        markers, region = two_markers()
        ownership = Ownership.from_labels(watershed_ownership(markers, region))
        assert ownership.object_ids() == [1, 2]


class TestPersistence:
    def test_it_round_trips(self, tmp_path):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0, segmentation="nuclei_1")
        restored = load_ownership(ownership.save(tmp_path / "ownership.npz"))
        np.testing.assert_array_equal(restored.owners, ownership.owners)
        np.testing.assert_allclose(restored.probabilities, ownership.probabilities)

    def test_the_provenance_survives(self, tmp_path):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0, segmentation="nuclei_1")
        restored = load_ownership(ownership.save(tmp_path / "ownership.npz"))
        assert restored.method == DISTANCE
        assert restored.params["falloff"] == 6.0
        assert restored.segmentation == "nuclei_1"

    def test_the_hand_edits_survive(self, tmp_path):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        patch = np.zeros_like(region)
        patch[5, 15] = True
        ownership.override(patch, 2)
        restored = load_ownership(ownership.save(tmp_path / "ownership.npz"))
        assert restored.manual[5, 15]


class TestWeightedMeasurements:
    def test_a_certain_ownership_measures_exactly_like_a_hard_one(self):
        """The weighted reducer has to be a generalisation of the ordinary
        one, not a different statistic that happens to look similar."""
        markers, region = two_markers()
        labels = watershed_ownership(markers, region)
        intensity = np.linspace(0, 1, labels.size).reshape(labels.shape)

        hard = extract_measurements(labels, intensity)
        soft = weighted_measurements(Ownership.from_labels(labels), intensity)

        for column in ("count", "mean", "sum", "min", "max", "stddev"):
            np.testing.assert_allclose(soft[column], hard[column], rtol=1e-9, atol=1e-9)

    def test_the_centroids_match_a_hard_measurement_too(self):
        markers, region = two_markers()
        labels = watershed_ownership(markers, region)
        intensity = np.ones_like(labels, dtype=float)
        hard = extract_measurements(labels, intensity)
        soft = weighted_measurements(Ownership.from_labels(labels), intensity)
        np.testing.assert_allclose(soft["centroid-0"], hard["centroid-0"], atol=1e-9)
        np.testing.assert_allclose(soft["centroid-1"], hard["centroid-1"], atol=1e-9)

    def test_a_count_is_the_expected_number_of_voxels(self):
        """A cell that half-owns twenty contested voxels counts ten of
        them - which is what makes a volume comparable between cells whose
        boundaries were resolved to different degrees."""
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        counts = weighted_measurements(ownership, np.ones(region.shape))["count"]
        assert counts.sum() == pytest.approx(region.sum())
        assert counts.iloc[0] == pytest.approx(counts.iloc[1], rel=0.05)

    def test_a_contested_voxel_contributes_fractionally_to_the_mean(self):
        markers = np.zeros((5, 9), dtype=np.int32)
        markers[2, 1] = 1
        markers[2, 7] = 2
        region = np.zeros((5, 9), dtype=bool)
        region[2, 1:8] = True
        intensity = np.zeros((5, 9))
        intensity[2, 4] = 100.0  # only the contested middle voxel is bright

        ownership = distance_ownership(markers, region, falloff=4.0)
        table = weighted_measurements(ownership, intensity).set_index("object_id")
        # Both cells see some of that voxel, neither sees all of it.
        assert 0 < table.loc[1, "sum"] < 100.0
        assert table.loc[1, "sum"] + table.loc[2, "sum"] == pytest.approx(100.0)

    def test_the_extremes_are_not_scaled_by_a_probability(self):
        """An extreme is an extreme; scaling it would report a value that
        occurs nowhere in the image."""
        markers = np.zeros((5, 9), dtype=np.int32)
        markers[2, 1] = 1
        markers[2, 7] = 2
        region = np.zeros((5, 9), dtype=bool)
        region[2, 1:8] = True
        intensity = np.zeros((5, 9))
        intensity[2, 4] = 100.0

        ownership = distance_ownership(markers, region, falloff=4.0)
        table = weighted_measurements(ownership, intensity).set_index("object_id")
        assert table.loc[1, "max"] == 100.0

    def test_a_physical_volume_appears_only_with_a_known_spacing(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        assert "volume" not in weighted_measurements(ownership, np.ones(region.shape)).columns
        with_spacing = weighted_measurements(
            ownership, np.ones(region.shape), spacing=Spacing((2.0, 2.0), source="user")
        )
        assert with_spacing["volume"].iloc[0] == pytest.approx(with_spacing["count"].iloc[0] * 4)

    def test_an_empty_ownership_gives_an_empty_table_with_columns(self):
        empty = Ownership.from_labels(np.zeros((4, 4), dtype=np.int32))
        table = weighted_measurements(empty, np.zeros((4, 4)))
        assert table.empty
        assert "mean" in table.columns

    def test_mismatched_shapes_are_refused(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        with pytest.raises(ValueError, match="shapes differ"):
            weighted_measurements(ownership, np.zeros((3, 3)))


class TestWeightedByChannel:
    def test_intensity_columns_are_suffixed_by_channel(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        intensity = np.stack([np.ones(region.shape), np.ones(region.shape) * 5])

        table = weighted_measurements_by_channel(ownership, intensity, channel_axis=0)
        assert table["mean_ch0"].iloc[0] == pytest.approx(1.0)
        assert table["mean_ch1"].iloc[0] == pytest.approx(5.0)

    def test_geometry_appears_once(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        intensity = np.stack([np.ones(region.shape), np.ones(region.shape)])
        table = weighted_measurements_by_channel(ownership, intensity, channel_axis=0)
        assert "count" in table.columns
        assert "count_ch0" not in table.columns

    def test_one_channel_can_be_singled_out(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        intensity = np.stack([np.ones(region.shape), np.ones(region.shape) * 5])
        table = weighted_measurements_by_channel(ownership, intensity, channel_axis=0, channel=1)
        assert "mean_ch1" in table.columns
        assert "mean_ch0" not in table.columns

    def test_a_channel_out_of_range_is_refused(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        intensity = np.stack([np.ones(region.shape)])
        with pytest.raises(ValueError, match="out of range"):
            weighted_measurements_by_channel(ownership, intensity, channel_axis=0, channel=4)

    def test_an_image_with_no_channel_axis_is_measured_as_one(self):
        markers, region = two_markers()
        ownership = distance_ownership(markers, region, falloff=6.0)
        table = weighted_measurements_by_channel(ownership, np.ones(region.shape))
        assert "mean" in table.columns
