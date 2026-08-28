import numpy as np
import pandas as pd
import pytest

from vtea_core.measurements import (
    extract_measurements,
    extract_measurements_by_channel,
    feature_matrix,
    threshold_mean,
)


class TestThresholdMean:
    def test_matches_java_semantics_on_known_values(self):
        # values 0..19; range 0-19, cutoff = 19 - 19/4 = 14.25 -> top values 15..19
        values = np.arange(20, dtype=float)
        mask = np.ones_like(values, dtype=bool)
        result = threshold_mean(mask, values)
        assert result == pytest.approx(np.mean([15, 16, 17, 18, 19]))

    def test_constant_values(self):
        values = np.full(5, 7.0)
        mask = np.ones_like(values, dtype=bool)
        assert threshold_mean(mask, values) == pytest.approx(7.0)

    def test_empty_returns_nan(self):
        assert np.isnan(threshold_mean(np.array([], dtype=bool), np.array([])))

    def test_only_masked_values_are_considered(self):
        values = np.array([1.0, 1000.0])  # 1000 would dominate if unmasked
        mask = np.array([True, False])
        assert threshold_mean(mask, values) == pytest.approx(1.0)


class TestExtractMeasurements:
    def make_two_object_volume(self):
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[0:2, 0:2] = 1  # 4 px, intensity 10
        labels[3:5, 3:5] = 2  # 4 px, intensity 20
        intensity = np.zeros((5, 5), dtype=float)
        intensity[labels == 1] = 10.0
        intensity[labels == 2] = 20.0
        return labels, intensity

    def test_returns_one_row_per_object(self):
        labels, intensity = self.make_two_object_volume()
        table = extract_measurements(labels, intensity)
        assert sorted(table["object_id"]) == [1, 2]

    def test_basic_stats_correct(self):
        labels, intensity = self.make_two_object_volume()
        table = extract_measurements(labels, intensity).set_index("object_id")
        assert table.loc[1, "count"] == 4
        assert table.loc[1, "mean"] == pytest.approx(10.0)
        assert table.loc[1, "sum"] == pytest.approx(40.0)
        assert table.loc[1, "min"] == pytest.approx(10.0)
        assert table.loc[1, "max"] == pytest.approx(10.0)
        assert table.loc[1, "stddev"] == pytest.approx(0.0)
        assert table.loc[2, "mean"] == pytest.approx(20.0)

    def test_includes_threshold_mean_column(self):
        labels, intensity = self.make_two_object_volume()
        table = extract_measurements(labels, intensity)
        assert "threshold_mean" in table.columns

    def test_includes_centroid_columns(self):
        labels, intensity = self.make_two_object_volume()
        table = extract_measurements(labels, intensity).set_index("object_id")
        # object 1 occupies rows/cols 0:2, 0:2 -> centroid (0.5, 0.5)
        assert table.loc[1, "centroid-0"] == pytest.approx(0.5)
        assert table.loc[1, "centroid-1"] == pytest.approx(0.5)

    def test_shape_mismatch_raises(self):
        labels = np.zeros((5, 5), dtype=np.int32)
        intensity = np.zeros((3, 3))
        with pytest.raises(ValueError, match="shape"):
            extract_measurements(labels, intensity)

    def test_3d_volume(self):
        labels = np.zeros((2, 4, 4), dtype=np.int32)
        labels[0, 0:2, 0:2] = 1
        intensity = np.full((2, 4, 4), 5.0)
        table = extract_measurements(labels, intensity)
        assert list(table["object_id"]) == [1]
        assert table.loc[0, "count"] == 4


class TestExtractMeasurementsByChannel:
    """One segmentation, every channel, one flat table - the shape the
    napari plot's X/Y menus read features from."""

    @staticmethod
    def make_multichannel():
        # (z, c, y, x): two objects, three channels of known brightness.
        labels = np.zeros((2, 4, 4), dtype=np.int32)
        labels[:, 0:2, 0:2] = 1
        labels[:, 2:4, 2:4] = 2
        intensity = np.zeros((2, 3, 4, 4), dtype=float)
        for channel in range(3):
            intensity[:, channel] = 10.0 * (channel + 1)
        return labels, intensity

    def test_one_column_set_per_channel_tagged_with_the_channel(self):
        labels, intensity = self.make_multichannel()
        table = extract_measurements_by_channel(labels, intensity, channel_axis=1)
        assert {"mean_ch0", "mean_ch1", "mean_ch2"} <= set(table.columns)
        assert table.loc[0, "mean_ch0"] == pytest.approx(10.0)
        assert table.loc[0, "mean_ch1"] == pytest.approx(20.0)
        assert table.loc[0, "mean_ch2"] == pytest.approx(30.0)

    def test_geometry_columns_appear_once_and_unsuffixed(self):
        labels, intensity = self.make_multichannel()
        table = extract_measurements_by_channel(labels, intensity, channel_axis=1)
        assert list(table.columns).count("object_id") == 1
        assert list(table.columns).count("count") == 1
        assert "count_ch0" not in table.columns
        assert {"centroid-0", "centroid-1", "centroid-2"} <= set(table.columns)

    def test_one_row_per_object_not_per_object_and_channel(self):
        labels, intensity = self.make_multichannel()
        table = extract_measurements_by_channel(labels, intensity, channel_axis=1)
        assert len(table) == 2
        assert list(table["object_id"]) == [1, 2]

    def test_selecting_one_channel_still_tags_the_name(self):
        labels, intensity = self.make_multichannel()
        table = extract_measurements_by_channel(labels, intensity, channel_axis=1, channel=2)
        assert "mean_ch2" in table.columns
        assert "mean_ch0" not in table.columns

    def test_single_channel_image_measures_as_before(self):
        labels, intensity = self.make_multichannel()
        table = extract_measurements_by_channel(labels, intensity[:, 0], channel_axis=1)
        assert "mean" in table.columns
        assert "mean_ch0" not in table.columns

    def test_channel_out_of_range_raises(self):
        labels, intensity = self.make_multichannel()
        with pytest.raises(ValueError, match="channel 5"):
            extract_measurements_by_channel(labels, intensity, channel_axis=1, channel=5)

    def test_channel_axis_out_of_range_raises(self):
        labels, intensity = self.make_multichannel()
        with pytest.raises(ValueError, match="channel axis"):
            extract_measurements_by_channel(labels, intensity, channel_axis=9)


class TestFeatureMatrix:
    def test_drops_identifiers_and_centroids(self):
        frame = pd.DataFrame(
            {
                "object_id": [1, 2],
                "centroid-0": [0.5, 2.5],
                "centroid-1": [0.5, 2.5],
                "count": [4, 4],
                "mean_ch0": [10.0, 20.0],
            }
        )
        matrix, columns = feature_matrix(frame)
        assert columns == ["count", "mean_ch0"]
        assert matrix.shape == (2, 2)

    def test_skips_non_numeric_columns(self):
        frame = pd.DataFrame({"mean": [1.0, 2.0], "note": ["a", "b"]})
        _, columns = feature_matrix(frame)
        assert columns == ["mean"]

    def test_nans_become_zero_so_scikit_learn_can_fit(self):
        frame = pd.DataFrame({"mean": [1.0, np.nan]})
        matrix, _ = feature_matrix(frame)
        assert matrix[1, 0] == 0.0

    def test_explicit_columns_are_used_as_given(self):
        frame = pd.DataFrame({"object_id": [1, 2], "mean": [1.0, 2.0]})
        matrix, columns = feature_matrix(frame, ["object_id"])
        assert columns == ["object_id"]
        assert matrix.shape == (2, 1)

    def test_a_table_with_no_features_gives_an_empty_matrix(self):
        frame = pd.DataFrame({"object_id": [1, 2]})
        matrix, columns = feature_matrix(frame)
        assert columns == []
        assert matrix.shape == (2, 0)
