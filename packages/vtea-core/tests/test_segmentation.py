import numpy as np
import pytest

from vtea_core.segmentation import (
    filter_by_size,
    import_labels,
    label_components,
    labels_from_points,
    threshold_mask,
    watershed_split,
)


class TestThresholdMask:
    def test_fixed_threshold(self):
        volume = np.array([0, 5, 10, 15, 20])
        mask = threshold_mask(volume, method="fixed", value=10)
        np.testing.assert_array_equal(mask, [False, False, True, True, True])

    def test_fixed_requires_value(self):
        with pytest.raises(ValueError, match="value"):
            threshold_mask(np.zeros(3), method="fixed")

    def test_otsu_separates_bimodal_data(self):
        rng = np.random.default_rng(0)
        low = rng.normal(20, 2, size=1000)
        high = rng.normal(200, 2, size=1000)
        volume = np.concatenate([low, high])
        mask = threshold_mask(volume, method="otsu")
        assert mask[:1000].mean() < 0.05
        assert mask[1000:].mean() > 0.95

    def test_percentile(self):
        volume = np.arange(100)
        mask = threshold_mask(volume, method="percentile", percentile=90)
        assert mask.sum() == pytest.approx(10, abs=1)

    def test_percentile_requires_percentile(self):
        with pytest.raises(ValueError, match="percentile"):
            threshold_mask(np.zeros(3), method="percentile")

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            threshold_mask(np.zeros(3), method="bogus")


class TestLabelComponents:
    def test_two_separated_blobs_get_different_labels(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        mask[6:8, 6:8, 6:8] = True
        labels = label_components(mask)
        assert labels.max() == 2
        assert len(np.unique(labels)) == 3  # background + 2 objects

    def test_empty_mask_has_no_objects(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        labels = label_components(mask)
        assert labels.max() == 0

    def test_connectivity_1_vs_full_diagonal_only_touch(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        mask[2, 2] = True  # diagonal neighbor only
        face_labels = label_components(mask, connectivity=1)
        full_labels = label_components(mask, connectivity=2)
        assert face_labels.max() == 2  # not face-connected -> stay separate
        assert full_labels.max() == 1  # corner-connected -> merged


class TestWatershedSplit:
    def test_splits_two_blobs_joined_by_a_narrow_bridge(self):
        mask = np.zeros((20, 40), dtype=bool)
        mask[5:15, 2:18] = True  # left blob
        mask[5:15, 22:38] = True  # right blob
        mask[9:11, 18:22] = True  # narrow connecting bridge
        intensity = mask.astype(float)

        labels = watershed_split(intensity, mask, min_distance=5)

        assert labels.max() >= 2
        left_label = labels[10, 10]
        right_label = labels[10, 30]
        assert left_label != 0 and right_label != 0
        assert left_label != right_label


class TestFilterBySize:
    def test_removes_small_and_large_objects(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[0, 0] = 1  # size 1 (too small)
        labels[2:5, 2:5] = 2  # size 9 (kept)
        labels[5:9, 5:9] = 3  # size 16 (too large)

        filtered = filter_by_size(labels, min_size=5, max_size=15)

        assert set(np.unique(filtered).tolist()) == {0, 1}
        assert (filtered == 1).sum() == 9

    def test_no_bounds_returns_input_unchanged(self):
        labels = np.array([[0, 1], [1, 2]])
        result = filter_by_size(labels)
        np.testing.assert_array_equal(result, labels)

    def test_relabels_sequentially(self):
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[0, 0] = 5
        labels[4, 4] = 10
        filtered = filter_by_size(labels, min_size=1)
        assert set(np.unique(filtered).tolist()) == {0, 1, 2}

    def test_min_size_only(self):
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[0, 0] = 1
        labels[1:4, 1:4] = 2  # size 9
        filtered = filter_by_size(labels, min_size=5)
        assert set(np.unique(filtered).tolist()) == {0, 1}

    def test_everything_filtered_out(self):
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[0, 0] = 1
        filtered = filter_by_size(labels, min_size=100)
        assert filtered.max() == 0


class TestLabelsFromPoints:
    def test_creates_one_label_per_point(self):
        points = np.array([[2, 2], [7, 7]])
        labels = labels_from_points(points, shape=(10, 10), radius=1.5)
        assert set(np.unique(labels).tolist()) == {0, 1, 2}
        assert labels[2, 2] == 1
        assert labels[7, 7] == 2

    def test_radius_controls_size(self):
        points = np.array([[5, 5]])
        small = labels_from_points(points, shape=(10, 10), radius=1)
        large = labels_from_points(points, shape=(10, 10), radius=3)
        assert (large > 0).sum() > (small > 0).sum()

    def test_no_points_gives_empty_labels(self):
        labels = labels_from_points(np.empty((0, 2)), shape=(5, 5), radius=1)
        assert labels.max() == 0


class TestImportLabels:
    def test_passthrough(self):
        array = np.array([[0, 1], [2, 0]])
        result = import_labels(array)
        np.testing.assert_array_equal(result, array)
