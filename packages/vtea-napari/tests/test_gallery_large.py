"""The gallery on data that does not fit.

A crop is a bounding-box read, which is the cheapest thing a chunked store
does - so this is the one view in VTEA that should get *faster* on large
data rather than slower. What these tests check is that nothing accidentally
touches the whole array on the way there.
"""

import numpy as np
import pandas as pd
import pytest

import dask.array as da
from vtea_napari.widgets.gallery import (
    GalleryWidget,
    choose_level,
    crop_around,
    level_scale,
    pyramid_levels,
)


class RefusesToBeRead:
    """Slices like an array, and objects to being materialized whole.

    Stands in for a Zarr array larger than memory: `np.asarray` on it is
    the mistake, and `np.asarray` on a slice of it is the point.
    """

    def __init__(self, data):
        self._data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.reads: list = []

    def __getitem__(self, index):
        self.reads.append(index)
        return self._data[index]

    def __array__(self, dtype=None, copy=None):
        raise AssertionError("the whole volume was materialized")


def read_voxels(source: RefusesToBeRead) -> int:
    return sum(int(np.asarray(source._data[index]).size) for index in source.reads)


@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    return rng.integers(0, 4000, (64, 256, 256)).astype(np.uint16)


@pytest.fixture
def table():
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3],
            "centroid-0": [10.0, 32.0, 50.0],
            "centroid-1": [40.0, 128.0, 200.0],
            "centroid-2": [40.0, 128.0, 200.0],
        }
    )


class TestNothingIsMaterialized:
    def test_a_stored_volume_is_never_read_whole(self, qtbot, volume, table):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        source = RefusesToBeRead(volume)
        widget.show_objects(source, table, [1, 2, 3], crop_radius=20, z_radius=8)
        assert len(widget._thumbnails) == 3
        # Three crops of at most 16 x 40 x 40, against a volume of four
        # million voxels.
        assert read_voxels(source) < volume.size // 50

    def test_a_dask_array_works_the_same_way(self, qtbot, volume, table):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        lazy = da.from_array(volume, chunks=(16, 64, 64))
        widget.show_objects(lazy, table, [1, 2, 3], crop_radius=20, z_radius=8)
        assert len(widget._thumbnails) == 3

    def test_a_plain_numpy_volume_still_works(self, qtbot, volume, table):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.show_objects(volume, table, [1, 2, 3], crop_radius=20)
        assert len(widget._thumbnails) == 3


class TestDepthIsCropped:
    def test_projecting_every_slice_is_no_longer_the_only_option(self, volume):
        # The bug this fixes: a full projection of a thousand-slice stack is
        # a reduction over the volume wearing a thumbnail's clothes.
        whole = crop_around(volume, 128, 128, radius=20)
        windowed = crop_around(volume, 128, 128, radius=20, depth=32, z_radius=8)
        assert whole.shape[0] == volume.shape[0]
        assert windowed.shape[0] == 16
        assert windowed.shape[1:] == whole.shape[1:]

    def test_the_window_is_clipped_at_the_top_of_the_stack(self, volume):
        crop = crop_around(volume, 128, 128, radius=20, depth=2, z_radius=8)
        assert crop.shape[0] == 10  # 0..10, not -6..10

    def test_a_two_dimensional_volume_has_no_depth_to_crop(self):
        plane = np.zeros((64, 64), np.uint16)
        crop = crop_around(plane, 32, 32, radius=8, depth=4, z_radius=2)
        assert crop.shape == (16, 16)

    def test_the_explorer_asks_for_a_window(self):
        from vtea_napari.widgets.explorer import GALLERY_Z_RADIUS

        assert GALLERY_Z_RADIUS > 0


class TestPyramidLevels:
    def pyramid(self, base=(32, 512, 512), levels=4):
        return [
            np.zeros((base[0], base[1] // 2**level, base[2] // 2**level), np.uint16)
            for level in range(levels)
        ]

    def test_a_plain_array_is_a_pyramid_of_one(self, volume):
        assert pyramid_levels(volume) == [volume]
        assert len(pyramid_levels([volume, volume])) == 2

    def test_the_scale_is_measured_rather_than_assumed(self):
        # A store written by another tool may downsample by three, and a
        # centroid divided by the wrong number crops the wrong place.
        levels = [np.zeros((4, 90, 90)), np.zeros((4, 30, 30))]
        assert level_scale(levels, 1) == pytest.approx(3.0)
        assert level_scale(levels, 0) == 1.0

    def test_a_small_crop_is_read_at_full_resolution(self):
        assert choose_level(self.pyramid(), crop_radius=20, thumbnail_size=64) == 0

    def test_a_large_crop_is_read_from_a_coarser_level(self):
        # 400 pixels shown at 64: level 0 is sixteen times the I/O for a
        # picture nobody can tell apart.
        assert choose_level(self.pyramid(), crop_radius=200, thumbnail_size=64) >= 2

    def test_it_stops_before_the_crop_is_smaller_than_the_thumbnail(self):
        level = choose_level(self.pyramid(levels=8), crop_radius=200, thumbnail_size=64)
        scale = level_scale(self.pyramid(levels=8), level)
        assert (2 * 200) / scale >= 64

    def test_a_single_level_is_always_level_zero(self, volume):
        assert choose_level([volume], crop_radius=10_000, thumbnail_size=64) == 0

    def test_the_coarse_level_is_the_one_actually_read(self, qtbot, table):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        levels = [
            RefusesToBeRead(np.zeros((16, 512, 512), np.uint16)),
            RefusesToBeRead(np.zeros((16, 256, 256), np.uint16)),
            RefusesToBeRead(np.zeros((16, 128, 128), np.uint16)),
        ]
        widget.show_objects(levels, table, [1, 2, 3], crop_radius=200, thumbnail_size=64)
        assert not levels[0].reads, "the finest level was read for a coarse thumbnail"
        assert levels[2].reads or levels[1].reads

    def test_centroids_are_scaled_into_the_level_they_are_read_from(self, qtbot):
        # A centroid is in level-0 voxels. Read at level 1 without dividing
        # it, the crop lands at twice the right place - which looks like a
        # gallery of the wrong objects rather than like an error.
        marked = np.zeros((4, 256, 256), np.uint16)
        marked[:, 100:110, 100:110] = 4000
        coarse = marked[:, ::2, ::2]
        frame = pd.DataFrame(
            {"object_id": [1], "centroid-0": [2.0], "centroid-1": [105.0], "centroid-2": [105.0]}
        )
        crop = crop_around(coarse, round(105 / 2), round(105 / 2), radius=8)
        assert crop.max() == 4000, "the scaled crop missed the object"


class TestUnchangedBehaviour:
    def test_a_table_without_centroids_shows_nothing(self, qtbot, volume):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.show_objects(volume, pd.DataFrame({"object_id": [1]}), [1])
        assert widget._thumbnails == []

    def test_ids_that_are_not_in_the_table_are_skipped(self, qtbot, volume, table):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.show_objects(volume, table, [1, 99])
        assert len(widget._thumbnails) == 1
