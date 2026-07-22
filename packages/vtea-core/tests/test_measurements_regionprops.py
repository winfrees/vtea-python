import numpy as np
import pytest

from vtea_core.measurements import extract_measurements, threshold_mean


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
