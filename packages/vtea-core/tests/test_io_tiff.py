import numpy as np
import pytest
import tifffile

from vtea_core.io import open_volume, read_tiff, write_tiff
from vtea_core.io.tiff import _to_czyx


class TestAxesNormalization:
    def test_yx_only_gets_c_and_z_inserted(self):
        array = np.zeros((4, 5))
        out = _to_czyx(array, "YX")
        assert out.shape == (1, 1, 4, 5)

    def test_zyx_gets_c_inserted(self):
        array = np.zeros((3, 4, 5))
        out = _to_czyx(array, "ZYX")
        assert out.shape == (1, 3, 4, 5)

    def test_cyx_gets_z_inserted(self):
        array = np.zeros((2, 4, 5))
        out = _to_czyx(array, "CYX")
        assert out.shape == (2, 1, 4, 5)

    def test_zcyx_reordered_to_czyx(self):
        array = np.arange(3 * 2 * 4 * 5).reshape(3, 2, 4, 5)  # Z,C,Y,X
        out = _to_czyx(array, "ZCYX")
        assert out.shape == (2, 3, 4, 5)
        np.testing.assert_array_equal(out, np.transpose(array, (1, 0, 2, 3)))

    def test_singleton_time_axis_is_dropped(self):
        array = np.zeros((1, 3, 4, 5))  # T,Z,Y,X
        out = _to_czyx(array, "TZYX")
        assert out.shape == (1, 3, 4, 5)  # C inserted, T dropped

    def test_nonsingleton_time_axis_raises(self):
        array = np.zeros((2, 3, 4, 5))
        with pytest.raises(NotImplementedError, match="time series"):
            _to_czyx(array, "TZYX")

    def test_sample_interleaved_axes_raise(self):
        array = np.zeros((4, 5, 3))
        with pytest.raises(NotImplementedError, match="interleaved"):
            _to_czyx(array, "YXS")


class TestTiffRoundTrip:
    def test_round_trip_preserves_data(self, tmp_path):
        from vtea_core.data import InMemoryVolumeDataset

        array = np.random.default_rng(0).integers(0, 65535, size=(2, 3, 8, 10), dtype=np.uint16)
        ds = InMemoryVolumeDataset(array)
        path = tmp_path / "volume.tif"
        write_tiff(ds, path)

        loaded = read_tiff(path)
        assert loaded.shape == ds.shape
        np.testing.assert_array_equal(loaded.to_numpy(), array)

    def test_read_single_channel_2d_tiff(self, tmp_path):
        array = np.random.default_rng(1).integers(0, 255, size=(4, 5), dtype=np.uint8)
        path = tmp_path / "plain.tif"
        tifffile.imwrite(path, array)

        loaded = read_tiff(path)
        assert loaded.shape == (1, 1, 4, 5)
        np.testing.assert_array_equal(loaded.to_numpy()[0, 0], array)

    def test_open_volume_dispatches_tiff(self, tmp_path):
        array = np.zeros((3, 4), dtype=np.uint8)
        path = tmp_path / "plain.tif"
        tifffile.imwrite(path, array)

        loaded = open_volume(path)
        assert loaded.shape == (1, 1, 3, 4)

    def test_open_volume_unknown_extension_raises(self, tmp_path):
        path = tmp_path / "data.bin"
        path.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="unrecognized"):
            open_volume(path)
