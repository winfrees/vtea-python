import numpy as np

from vtea_core.data import ChunkedVolumeDataset, InMemoryVolumeDataset
from vtea_core.io import open_volume, read_zarr, write_zarr


def test_write_and_read_in_memory_dataset_round_trip(tmp_path):
    array = np.random.default_rng(0).random((2, 3, 8, 10)).astype(np.float32)
    ds = InMemoryVolumeDataset(array)
    path = tmp_path / "volume.zarr"

    write_zarr(ds, path)
    loaded = read_zarr(path)

    assert isinstance(loaded, ChunkedVolumeDataset)
    assert loaded.shape == array.shape
    np.testing.assert_array_equal(loaded.to_numpy(), array)


def test_write_and_read_chunked_dataset_round_trip(tmp_path):
    import dask.array as da

    array = np.arange(2 * 4 * 6 * 6, dtype=np.int32).reshape(2, 4, 6, 6)
    ds = ChunkedVolumeDataset(da.from_array(array, chunks=(1, 2, 3, 3)))
    path = tmp_path / "chunked.zarr"

    write_zarr(ds, path)
    loaded = read_zarr(path)

    np.testing.assert_array_equal(loaded.to_numpy(), array)


def test_open_volume_dispatches_zarr_by_extension(tmp_path):
    array = np.zeros((1, 1, 4, 5), dtype=np.uint8)
    path = tmp_path / "volume.zarr"
    write_zarr(InMemoryVolumeDataset(array), path)

    loaded = open_volume(path)
    assert loaded.shape == (1, 1, 4, 5)


def test_open_volume_dispatches_zarr_by_marker_file_without_extension(tmp_path):
    array = np.zeros((1, 1, 4, 5), dtype=np.uint8)
    path = tmp_path / "some_directory"
    write_zarr(InMemoryVolumeDataset(array), path)
    assert (path / ".zarray").exists() or (path / ".zgroup").exists() or (path / "zarr.json").exists()

    loaded = open_volume(path)
    assert loaded.shape == (1, 1, 4, 5)


def test_zarr_rejects_non_4d_array(tmp_path):
    import dask.array as da

    path = tmp_path / "bad.zarr"
    da.zeros((3, 4, 5), chunks=(1, 2, 2)).to_zarr(str(path))

    import pytest

    with pytest.raises(ValueError, match="4D"):
        read_zarr(path)
