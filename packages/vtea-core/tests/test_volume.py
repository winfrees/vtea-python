import dask.array as da
import numpy as np
import pytest

from vtea_core.data import ChunkedVolumeDataset, InMemoryVolumeDataset, VolumeDataset


def make_array(c=2, z=3, y=4, x=5):
    return np.arange(c * z * y * x, dtype=np.float32).reshape(c, z, y, x)


class TestInMemoryVolumeDataset:
    def test_shape_and_dims(self):
        ds = InMemoryVolumeDataset(make_array(2, 3, 4, 5))
        assert ds.shape == (2, 3, 4, 5)
        assert (ds.n_channels, ds.depth, ds.height, ds.width) == (2, 3, 4, 5)

    def test_rejects_non_4d(self):
        with pytest.raises(ValueError, match="4D"):
            InMemoryVolumeDataset(np.zeros((3, 4, 5)))

    def test_is_chunked_false(self):
        ds = InMemoryVolumeDataset(make_array())
        assert ds.is_chunked is False

    def test_channel_returns_correct_slice(self):
        array = make_array()
        ds = InMemoryVolumeDataset(array)
        np.testing.assert_array_equal(ds.channel(1), array[1])

    def test_voxel(self):
        array = make_array()
        ds = InMemoryVolumeDataset(array)
        assert ds.voxel(1, 2, 3, 4) == array[1, 2, 3, 4]

    def test_subvolume(self):
        array = make_array(2, 6, 8, 10)
        ds = InMemoryVolumeDataset(array)
        sub = ds.subvolume(c=0, z0=1, y0=2, x0=3, depth=2, height=3, width=4)
        np.testing.assert_array_equal(sub, array[0, 1:3, 2:5, 3:7])

    def test_fits_in_memory_respects_budget(self):
        array = make_array()
        ds = InMemoryVolumeDataset(array)
        assert ds.fits_in_memory(max_bytes=array.nbytes + 1) is True
        assert ds.fits_in_memory(max_bytes=array.nbytes - 1) is False

    def test_to_numpy_returns_same_data(self):
        array = make_array()
        ds = InMemoryVolumeDataset(array)
        np.testing.assert_array_equal(ds.to_numpy(), array)


class TestChunkedVolumeDataset:
    def make_ds(self, c=2, z=6, y=8, x=10, chunks=(1, 2, 4, 5)):
        array = make_array(c, z, y, x)
        return ChunkedVolumeDataset(da.from_array(array, chunks=chunks)), array

    def test_rejects_non_4d(self):
        with pytest.raises(ValueError, match="4D"):
            ChunkedVolumeDataset(da.zeros((3, 4, 5)))

    def test_is_chunked_true(self):
        ds, _ = self.make_ds()
        assert ds.is_chunked is True

    def test_channel_returns_dask_array(self):
        ds, array = self.make_ds()
        chan = ds.channel(1)
        assert isinstance(chan, da.Array)
        np.testing.assert_array_equal(chan.compute(), array[1])

    def test_voxel(self):
        ds, array = self.make_ds()
        assert ds.voxel(1, 2, 3, 4) == array[1, 2, 3, 4]

    def test_subvolume_materializes_numpy(self):
        ds, array = self.make_ds()
        sub = ds.subvolume(c=0, z0=1, y0=2, x0=3, depth=2, height=3, width=4)
        assert isinstance(sub, np.ndarray)
        np.testing.assert_array_equal(sub, array[0, 1:3, 2:5, 3:7])

    def test_fits_in_memory(self):
        ds, array = self.make_ds()
        assert ds.fits_in_memory(max_bytes=array.nbytes + 1) is True
        assert ds.fits_in_memory(max_bytes=array.nbytes - 1) is False

    def test_to_numpy_matches_source(self):
        ds, array = self.make_ds()
        np.testing.assert_array_equal(ds.to_numpy(), array)


def test_volume_dataset_is_abstract():
    with pytest.raises(TypeError):
        VolumeDataset()
