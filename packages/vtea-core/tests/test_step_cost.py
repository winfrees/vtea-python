import pytest

from vtea_core.workflow import (
    Calibration,
    Step,
    cost_for,
    estimate_seconds,
    format_duration,
)


def a_step(category, function_name, **kwargs):
    return Step.for_function(category, function_name, **kwargs)


class TestCostFor:
    def test_a_timed_step_has_its_own_entry(self):
        assert cost_for("segmentation", "watershed_split").per_voxel_ns > 0

    def test_the_heaviest_segmentation_step_costs_more_than_a_threshold(self):
        watershed = cost_for("segmentation", "watershed_split").per_voxel_ns
        threshold = cost_for("segmentation", "threshold_mask").per_voxel_ns
        assert watershed > 10 * threshold

    def test_an_unknown_image_step_falls_back_to_a_per_voxel_guess(self):
        cost = cost_for("segmentation", "some_third_party_step")
        assert cost.per_voxel_ns > 0
        assert cost.per_object_ns == 0

    def test_an_unknown_table_step_is_not_costed_per_voxel(self):
        """A step's declared block mode decides which default it gets, so a
        third-party clustering step is not estimated as if it walked every
        voxel of the image."""
        cost = cost_for("clustering", "some_third_party_clustering")
        assert cost.per_voxel_ns == 0
        assert cost.per_object_ns > 0


class TestEstimateSeconds:
    def test_a_bigger_tile_takes_proportionally_longer(self):
        step = a_step("segmentation", "threshold_mask")
        small = estimate_seconds(step, voxels=1_000_000)
        large = estimate_seconds(step, voxels=4_000_000)
        assert large == pytest.approx(4 * small)

    def test_more_tiles_multiply_the_estimate(self):
        step = a_step("segmentation", "threshold_mask")
        one = estimate_seconds(step, voxels=1_000_000, tiles=1)
        many = estimate_seconds(step, voxels=1_000_000, tiles=10)
        assert many == pytest.approx(10 * one)

    def test_a_table_step_is_estimated_from_objects_not_voxels(self):
        step = a_step("clustering", "kmeans")
        assert estimate_seconds(step, voxels=10**9) is None
        assert estimate_seconds(step, n_objects=10_000, n_features=8) > 0

    def test_more_features_cost_more(self):
        step = a_step("clustering", "kmeans")
        few = estimate_seconds(step, n_objects=10_000, n_features=2)
        many = estimate_seconds(step, n_objects=10_000, n_features=40)
        assert many > few

    def test_a_superlinear_step_has_no_estimate(self):
        """t-SNE's runtime depends on how its optimisation converges, so any
        fraction shown for it would be invented - the caller is meant to show
        a continuous bar instead."""
        assert estimate_seconds(a_step("reduction", "tsne"), n_objects=10_000) is None
        assert estimate_seconds(a_step("reduction", "umap"), n_objects=10_000) is None
        assert estimate_seconds(a_step("clustering", "leiden"), n_objects=10_000) is None
        assert estimate_seconds(a_step("clustering", "louvain"), n_objects=10_000) is None

    def test_no_size_at_all_means_no_estimate(self):
        assert estimate_seconds(a_step("segmentation", "threshold_mask")) is None


class TestFormatDuration:
    def test_none_is_blank_rather_than_zero(self):
        assert format_duration(None) == ""

    def test_sub_second_is_not_reported_as_zero_seconds(self):
        assert format_duration(0.4) == "under 1 s"

    def test_seconds_minutes_and_hours(self):
        assert format_duration(3) == "about 3 s"
        assert format_duration(150) == "about 2 min"
        assert format_duration(9000) == "about 2.5 h"

    def test_something_instant_says_so(self):
        assert format_duration(0.001) == "instant"


class TestCalibration:
    def test_an_uncalibrated_step_is_left_at_the_table_value(self):
        calibration = Calibration()
        assert calibration.scale_for("segmentation", "watershed_split") == 1.0

    def test_a_run_slower_than_predicted_raises_later_estimates(self):
        calibration = Calibration()
        step = a_step("segmentation", "watershed_split")
        for _ in range(20):
            calibration.observe(step, seconds=40.0, predicted=10.0)
        assert calibration.scale_for(step.category, step.function_name) == pytest.approx(4.0, rel=0.1)

    def test_it_moves_the_estimate_it_is_given(self):
        calibration = Calibration()
        step = a_step("segmentation", "watershed_split")
        for _ in range(20):
            calibration.observe(step, seconds=20.0, predicted=10.0)
        plain = estimate_seconds(step, voxels=10**6)
        calibrated = estimate_seconds(step, voxels=10**6, calibration=calibration)
        assert calibrated == pytest.approx(2 * plain, rel=0.1)

    def test_a_single_outlier_does_not_replace_the_estimate(self):
        calibration = Calibration()
        step = a_step("segmentation", "watershed_split")
        calibration.observe(step, seconds=100.0, predicted=1.0)
        assert calibration.scale_for(step.category, step.function_name) < 40

    def test_an_absurd_ratio_is_clamped(self):
        calibration = Calibration()
        step = a_step("segmentation", "watershed_split")
        for _ in range(50):
            calibration.observe(step, seconds=10_000.0, predicted=0.1)
        assert calibration.scale_for(step.category, step.function_name) <= Calibration.MAX_SCALE

    def test_a_step_too_quick_to_time_is_ignored(self):
        calibration = Calibration()
        step = a_step("segmentation", "threshold_mask")
        calibration.observe(step, seconds=0.001, predicted=0.0005)
        assert calibration.as_dict() == {}
