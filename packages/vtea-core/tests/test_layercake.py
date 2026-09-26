"""LayerCake3D: 2D regions per slice, linked across z by bounding-box centre.

The Java default segmentation. What matters is that it links what the Java
links and splits what the Java splits - the tests below pin each rule in
the module docstring on a volume small enough to reason about by hand.
"""

import numpy as np
import pytest
from skimage.draw import ellipsoid
from vtea_core.segmentation import label_components, layercake_3d
from vtea_core.segmentation.layercake import _bound_centres, watershed_slice
from vtea_core.workflow import Pipeline, Step


def sphere_volume():
    volume = np.zeros((12, 40, 40), dtype=np.uint16)
    shape = ellipsoid(4, 6, 6).astype(bool)
    volume[0:11, 2:17, 2:17][shape] = 200
    volume[0:11, 20:35, 5:20][shape] = 200
    return volume


class TestBoundCentre:
    def test_is_the_bounding_box_centre_not_the_centroid(self):
        """An L-shaped region: its centroid leans into the long arm, its
        bounding-box centre does not - and the Java links by the latter."""
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[1:8, 1] = 1
        labels[7, 1:4] = 1
        ((label, centre),) = _bound_centres(labels)
        assert label == 1
        # rows 1..7 -> 1 + 6 // 2 = 4; columns 1..3 -> 1 + 2 // 2 = 2
        assert centre == (4, 2)


class TestLinking:
    def test_two_separate_spheres_are_two_objects(self):
        labels = layercake_3d(sphere_volume(), low_threshold=100, min_size=1, max_size=10**6)
        assert labels.max() == 2
        assert set(np.unique(labels[:, :20, :20])) == {0, 1}

    def test_a_column_of_offset_squares_links_within_the_offset(self):
        """Each slice's square is shifted two pixels: within an offset of 5
        they are one object, and with an offset of 1 every slice is its own.
        (Within 3 they are not: the running midpoint lags a steady drift, and
        by the fourth slice it is 3.5 behind.)"""
        volume = np.zeros((5, 30, 30), dtype=np.uint8)
        for z in range(5):
            volume[z, 5:10, 5 + 2 * z : 10 + 2 * z] = 255
        linked = layercake_3d(volume, low_threshold=1, centroid_offset=5, min_size=1, watershed=False)
        split = layercake_3d(volume, low_threshold=1, centroid_offset=1, min_size=1, watershed=False)
        assert linked.max() == 1
        assert split.max() == 5

    def test_the_running_position_is_the_midpoint(self):
        """A drift of 4 pixels a slice outruns an offset of 5 by the third
        slice only because the position moves halfway, not all the way."""
        volume = np.zeros((3, 40, 40), dtype=np.uint8)
        # centres at x = 7, 11, 15 (bounding-box centres of 5-wide squares)
        for z, start in enumerate((5, 9, 13)):
            volume[z, 5:10, start : start + 5] = 255
        labels = layercake_3d(volume, low_threshold=1, centroid_offset=5, min_size=1, watershed=False)
        # slice 1 is 4 from slice 0 -> linked; midpoint 9; slice 2 is 6 from 9 -> not linked
        assert labels.max() == 2
        assert labels[0].max() == labels[1].max() == 1
        assert labels[2].max() == 2

    def test_links_only_to_the_next_slice(self):
        volume = np.zeros((3, 20, 20), dtype=np.uint8)
        volume[0, 5:10, 5:10] = 255
        volume[2, 5:10, 5:10] = 255  # a gap at z = 1
        labels = layercake_3d(volume, low_threshold=1, min_size=1, watershed=False)
        assert labels.max() == 2

    def test_an_object_can_branch(self):
        """Two regions in the slice below, both within the offset of the one
        above, both join it - the Java recursion continues its scan after
        following the first."""
        volume = np.zeros((2, 30, 30), dtype=np.uint8)
        volume[0, 10:15, 10:15] = 255
        volume[1, 10:15, 7:10] = 255
        volume[1, 10:15, 15:18] = 255
        labels = layercake_3d(volume, low_threshold=1, centroid_offset=5, min_size=1, watershed=False)
        assert labels.max() == 1

    def test_a_single_slice_links_within_the_slice(self):
        image = np.zeros((20, 20), dtype=np.uint8)
        image[5:8, 5:8] = 255
        image[5:8, 10:13] = 255  # centres 5 apart
        assert layercake_3d(image, low_threshold=1, centroid_offset=6, min_size=1, watershed=False).max() == 1
        assert layercake_3d(image, low_threshold=1, centroid_offset=4, min_size=1, watershed=False).max() == 2

    def test_an_offset_of_zero_links_nothing(self):
        volume = np.zeros((3, 10, 10), dtype=np.uint8)
        volume[:, 2:6, 2:6] = 255
        assert layercake_3d(volume, low_threshold=1, centroid_offset=0, min_size=1).max() == 3


class TestWatershed:
    def test_splits_two_touching_discs_in_a_slice(self):
        mask = np.zeros((30, 50), dtype=bool)
        yy, xx = np.mgrid[:30, :50]
        mask |= (yy - 15) ** 2 + (xx - 15) ** 2 < 100
        mask |= (yy - 15) ** 2 + (xx - 32) ** 2 < 100
        assert watershed_slice(mask).max() == 2

    def test_splits_touching_spheres_that_3d_connectivity_merges(self):
        """The reason LayerCake3D has to be ported rather than replaced by
        connected components: touching nuclei come apart."""
        volume = np.zeros((11, 30, 50), dtype=np.uint16)
        shape = ellipsoid(4, 8, 8).astype(bool)  # (11, 19, 19)
        volume[:, 2:21, 5:24][shape] = 200
        volume[:, 2:21, 20:39][shape] = 200
        assert label_components(volume > 100).max() == 1
        assert layercake_3d(volume, low_threshold=100, min_size=1, max_size=10**6).max() == 2

    def test_can_be_turned_off(self):
        volume = np.zeros((1, 30, 50), dtype=np.uint16)
        yy, xx = np.mgrid[:30, :50]
        volume[0][((yy - 15) ** 2 + (xx - 15) ** 2 < 100) | ((yy - 15) ** 2 + (xx - 32) ** 2 < 100)] = 9
        assert layercake_3d(volume, low_threshold=1, min_size=1, watershed=False).max() == 1


class TestThresholdAndSize:
    def test_threshold_is_inclusive(self):
        volume = np.zeros((1, 10, 10), dtype=np.uint8)
        volume[0, 2:5, 2:5] = 100
        assert layercake_3d(volume, low_threshold=100, min_size=1).max() == 1
        assert layercake_3d(volume, low_threshold=101, min_size=1).max() == 0

    def test_size_filter_renumbers_the_survivors(self):
        volume = np.zeros((1, 30, 30), dtype=np.uint8)
        volume[0, 2:4, 2:4] = 255  # 4 voxels
        volume[0, 10:20, 10:20] = 255  # 100 voxels
        labels = layercake_3d(volume, low_threshold=1, min_size=10, max_size=1000, watershed=False)
        assert labels.max() == 1
        assert (labels == 1).sum() == 100

    def test_max_below_min_is_refused(self):
        with pytest.raises(ValueError, match="max_size"):
            layercake_3d(np.zeros((2, 4, 4)), min_size=10, max_size=5)

    def test_refuses_a_multichannel_array(self):
        with pytest.raises(ValueError, match="pick a channel"):
            layercake_3d(np.zeros((2, 3, 4, 4)))

    def test_keeps_the_input_shape_and_dtype(self):
        labels = layercake_3d(sphere_volume(), low_threshold=100, min_size=1, max_size=10**6)
        assert labels.shape == sphere_volume().shape
        assert labels.dtype == np.int32


class TestAsAStep:
    def test_runs_in_a_pipeline_and_can_be_measured(self):
        pipeline = Pipeline()
        pipeline.add_step(
            Step.for_function(
                "segmentation",
                "layercake_3d",
                params={"low_threshold": 100, "min_size": 1, "max_size": 10**6},
            )
        )
        pipeline.add_step(
            Step.for_function(
                "measurements", "extract_measurements", available={"labels", "intensity"}
            )
        )
        volume = sphere_volume()
        context = pipeline.run({"volume": volume, "intensity": volume})
        assert len(context["measurements"]) == 2
