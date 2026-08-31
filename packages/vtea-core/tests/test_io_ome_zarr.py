"""OME-NGFF round trips, and the interoperability check that makes the
claim mean something."""

import json

import numpy as np
import pytest
import tifffile

from vtea_core.data.axes import TimeSeriesNotSupported
from vtea_core.data.spacing import FROM_METADATA, UNKNOWN, Spacing
from vtea_core.data.volume import InMemoryVolumeDataset
from vtea_core.io import (
    ingest,
    is_ome_zarr,
    open_volume,
    read_info,
    read_ome_zarr,
    read_zarr,
    write_ome_zarr,
    write_zarr,
)
from vtea_core.io import store
from vtea_core.io.ome_zarr import MEAN, NEAREST, pyramid_levels

SPACING = Spacing((2.0, 0.2, 0.2))


@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    return InMemoryVolumeDataset(rng.integers(0, 4000, size=(2, 8, 64, 64)).astype(np.uint16))


class TestRoundTrip:
    def test_pixels_survive(self, tmp_path, volume):
        path = write_ome_zarr(volume, tmp_path / "v.zarr")
        back = read_ome_zarr(path)
        np.testing.assert_array_equal(back.to_numpy(), volume.to_numpy())

    def test_the_result_is_lazy(self, tmp_path, volume):
        back = read_ome_zarr(write_ome_zarr(volume, tmp_path / "v.zarr"))
        assert back.is_chunked

    def test_the_dtype_is_not_promoted(self, tmp_path, volume):
        back = read_ome_zarr(write_ome_zarr(volume, tmp_path / "v.zarr"))
        assert back.dtype == np.uint16

    def test_a_known_spacing_comes_back(self, tmp_path, volume):
        path = write_ome_zarr(volume, tmp_path / "v.zarr", spacing=SPACING)
        info = read_info(path)
        assert info.spacing is not None
        assert info.spacing.values == pytest.approx((2.0, 0.2, 0.2))
        assert info.spacing.unit == "µm"
        assert info.spacing.source == FROM_METADATA

    def test_an_unknown_spacing_comes_back_as_unknown_not_as_ones(self, tmp_path, volume):
        # An all-ones scale is NGFF's spelling of "nobody said", exactly as
        # an all-ones napari layer.scale is. Reporting it as a confident one
        # micron isotropic is the mistake Spacing exists to prevent.
        unknown = Spacing((1.0, 1.0, 1.0), source=UNKNOWN)
        path = write_ome_zarr(volume, tmp_path / "v.zarr", spacing=unknown)
        assert read_info(path).spacing is None

    def test_no_spacing_at_all_is_also_unknown(self, tmp_path, volume):
        assert read_info(write_ome_zarr(volume, tmp_path / "v.zarr")).spacing is None


class TestFiveAxes:
    def test_the_store_has_a_time_axis_even_though_the_volume_does_not(
        self, tmp_path, volume
    ):
        path = write_ome_zarr(volume, tmp_path / "v.zarr")
        info = read_info(path)
        assert info.axes.order == "TCZYX"
        assert info.shape[0] == 1
        assert info.n_timepoints == 1

    def test_reading_squeezes_the_time_axis_back_out(self, tmp_path, volume):
        back = read_ome_zarr(write_ome_zarr(volume, tmp_path / "v.zarr"))
        assert back.array.ndim == 4

    def test_a_store_with_real_timepoints_is_refused_rather_than_truncated(
        self, tmp_path, volume
    ):
        path = write_ome_zarr(volume, tmp_path / "v.zarr")
        # Forge a four-timepoint store by rewriting level 0 and its shape,
        # which is what another tool's output would look like.
        group = store.open_group(path, mode="a")
        del group["0"]
        store.create_array(
            group, "0", shape=(4, 2, 8, 64, 64), dtype=np.uint16, chunks=(1, 1, 8, 32, 32)
        )
        with pytest.raises(TimeSeriesNotSupported, match="4 timepoints"):
            read_ome_zarr(path)


class TestPyramid:
    def test_a_small_volume_gets_one_level(self, volume, tmp_path):
        path = write_ome_zarr(volume, tmp_path / "v.zarr")
        assert read_info(path).n_levels == 1

    def test_a_large_volume_gets_several(self):
        from vtea_core.data.axes import Axes

        assert pyramid_levels((1, 1, 1, 2048, 2048), Axes("TCZYX")) > 1

    def test_each_level_halves_the_large_axes(self, tmp_path):
        data = np.zeros((1, 4, 1024, 1024), dtype=np.uint16)
        path = write_ome_zarr(InMemoryVolumeDataset(data), tmp_path / "v.zarr")
        info = read_info(path)
        assert info.n_levels >= 3
        shapes = [read_ome_zarr(path, level=n).shape for n in range(info.n_levels)]
        assert shapes[0][-1] == 1024
        assert shapes[1][-1] == 512
        # A four-slice z axis is not halved into nothing while xy comes down.
        assert all(shape[1] >= 1 for shape in shapes)

    def test_a_coarse_level_is_physically_coarser(self, tmp_path):
        data = np.zeros((1, 1, 1024, 1024), dtype=np.uint16)
        path = write_ome_zarr(InMemoryVolumeDataset(data), tmp_path / "v.zarr", spacing=SPACING)
        info = read_info(path)
        # Level 1's voxels are twice the size in the axes that were halved,
        # which is how a viewer knows the levels are one specimen.
        assert info.scales[1][-1] == pytest.approx(2 * info.scales[0][-1])

    def test_averaging_is_the_default_for_intensity(self, tmp_path):
        data = np.tile(np.array([[0, 100]], dtype=np.uint16), (1, 1, 512, 256))
        path = write_ome_zarr(InMemoryVolumeDataset(data), tmp_path / "v.zarr")
        coarse = read_ome_zarr(path, level=1).to_numpy()
        assert coarse.max() == 50  # the mean of 0 and 100, not one or the other

    def test_labels_are_subsampled_rather_than_averaged(self, tmp_path):
        # The average of label 4 and label 6 is an object that does not
        # exist. Every value in a downsampled label image must be a real id.
        rng = np.random.default_rng(0)
        data = rng.choice([0, 4, 6], size=(1, 1, 512, 512)).astype(np.int32)
        path = write_ome_zarr(
            InMemoryVolumeDataset(data), tmp_path / "l.zarr", reduction=NEAREST
        )
        coarse = read_ome_zarr(path, level=1).to_numpy()
        assert set(np.unique(coarse)) <= {0, 4, 6}

    def test_an_explicit_level_count_is_honoured(self, tmp_path):
        data = np.zeros((1, 1, 1024, 1024), dtype=np.uint16)
        path = write_ome_zarr(InMemoryVolumeDataset(data), tmp_path / "v.zarr", levels=2)
        assert read_info(path).n_levels == 2

    def test_asking_for_a_level_that_is_not_there_says_so(self, tmp_path, volume):
        path = write_ome_zarr(volume, tmp_path / "v.zarr")
        with pytest.raises(IndexError, match="out of range"):
            read_ome_zarr(path, level=9)

    def test_an_unknown_reduction_is_refused(self, tmp_path, volume):
        with pytest.raises(ValueError, match="unknown reduction"):
            write_ome_zarr(volume, tmp_path / "v.zarr", reduction="cubic")


class TestMetadata:
    def test_the_store_stamps_what_wrote_it(self, tmp_path, volume):
        path = write_ome_zarr(volume, tmp_path / "v.zarr")
        attrs = store.group_attrs(store.open_group(path))
        assert attrs["vtea"]["format_version"] == store.VTEA_FORMAT_VERSION
        assert attrs["vtea"]["zarr_format"] == 2

    def test_the_ngff_version_written_is_the_constant(self, tmp_path, volume):
        path = write_ome_zarr(volume, tmp_path / "v.zarr")
        assert read_info(path).ngff_version == store.NGFF_VERSION == "0.4"

    def test_units_are_written_in_the_spelling_ngff_requires(self, tmp_path, volume):
        # "µm" is not a UDUNITS-2 name, and a reader is entitled to ignore
        # a scale whose unit it does not recognise.
        path = write_ome_zarr(volume, tmp_path / "v.zarr", spacing=SPACING)
        attrs = json.loads((path / ".zattrs").read_text())
        units = {
            axis.get("unit") for axis in attrs["multiscales"][0]["axes"] if axis["type"] == "space"
        }
        assert units == {"micrometer"}

    def test_a_newer_store_version_is_refused_with_advice(self, tmp_path, volume):
        path = write_ome_zarr(volume, tmp_path / "v.zarr")
        group = store.open_group(path, mode="a")
        multiscales = list(group.attrs["multiscales"])
        multiscales[0]["version"] = "0.9"
        group.attrs["multiscales"] = multiscales
        with pytest.raises(store.UnsupportedStoreVersion, match="0.9"):
            read_info(path)

    def test_a_future_version_this_reader_already_accepts(self, tmp_path, volume):
        # Reading is where a version mismatch actually costs a user
        # something, so the reader is wider than the writer.
        path = write_ome_zarr(volume, tmp_path / "v.zarr")
        group = store.open_group(path, mode="a")
        multiscales = list(group.attrs["multiscales"])
        multiscales[0]["version"] = "0.5"
        group.attrs["multiscales"] = multiscales
        assert read_info(path).ngff_version == "0.5"

    def test_a_plain_zarr_array_is_not_mistaken_for_an_image(self, tmp_path, volume):
        write_zarr(volume, tmp_path / "plain.zarr")
        assert not is_ome_zarr(tmp_path / "plain.zarr")
        with pytest.raises(ValueError, match="not an OME-NGFF"):
            read_info(tmp_path / "plain.zarr")

    def test_an_ngff_store_is_recognised(self, tmp_path, volume):
        assert is_ome_zarr(write_ome_zarr(volume, tmp_path / "v.zarr"))


class TestIngest:
    def test_a_tiff_becomes_a_pyramidal_store(self, tmp_path):
        source = tmp_path / "stack.tif"
        data = np.random.default_rng(0).integers(0, 4000, (8, 2, 64, 64)).astype(np.uint16)
        tifffile.imwrite(source, data, imagej=True, metadata={"axes": "ZCYX"})

        path = ingest(source, tmp_path / "out.zarr", spacing=SPACING)
        info = read_info(path)
        assert info.axes.order == "TCZYX"
        assert info.spacing.values == pytest.approx((2.0, 0.2, 0.2))
        np.testing.assert_array_equal(
            read_ome_zarr(path).to_numpy(), np.transpose(data, (1, 0, 2, 3))
        )

    def test_the_store_is_chunked_for_random_access(self, tmp_path):
        source = tmp_path / "stack.tif"
        tifffile.imwrite(
            source, np.zeros((8, 512, 512), np.uint16), imagej=True, metadata={"axes": "ZYX"}
        )
        path = ingest(source, tmp_path / "out.zarr")
        level0 = store.open_group(path)["0"]
        # A chunk should be a cube-ish block, not a whole plane - the point
        # of converting is that pulling a cube out of the middle is cheap.
        assert level0.chunks[-1] < 512


class TestLazyTiff:
    def test_a_tiff_can_be_mapped_without_reading_it(self, tmp_path):
        source = tmp_path / "stack.tif"
        data = np.random.default_rng(0).integers(0, 255, (8, 2, 32, 32)).astype(np.uint16)
        tifffile.imwrite(source, data, imagej=True, metadata={"axes": "ZCYX"})

        lazy = open_volume(source, lazy=True)
        assert lazy.is_chunked
        assert lazy.shape == (2, 8, 32, 32)
        np.testing.assert_array_equal(lazy.to_numpy(), np.transpose(data, (1, 0, 2, 3)))

    def test_lazy_and_eager_agree(self, tmp_path):
        source = tmp_path / "stack.tif"
        data = np.random.default_rng(1).integers(0, 255, (4, 16, 16)).astype(np.uint16)
        tifffile.imwrite(source, data, imagej=True, metadata={"axes": "ZYX"})
        np.testing.assert_array_equal(
            open_volume(source, lazy=True).to_numpy(), open_volume(source).to_numpy()
        )

    def test_a_subvolume_can_be_read_without_the_whole_file(self, tmp_path):
        source = tmp_path / "stack.tif"
        data = np.arange(8 * 32 * 32, dtype=np.uint16).reshape(8, 32, 32)
        tifffile.imwrite(source, data, imagej=True, metadata={"axes": "ZYX"})
        lazy = open_volume(source, lazy=True)
        np.testing.assert_array_equal(
            lazy.subvolume(0, 2, 4, 4, 2, 8, 8), data[2:4, 4:12, 4:12]
        )


class TestDispatch:
    def test_open_volume_reads_an_ngff_store(self, tmp_path, volume):
        path = write_ome_zarr(volume, tmp_path / "v.zarr")
        np.testing.assert_array_equal(open_volume(path).to_numpy(), volume.to_numpy())

    def test_open_volume_still_reads_a_plain_zarr_array(self, tmp_path, volume):
        write_zarr(volume, tmp_path / "plain.zarr")
        np.testing.assert_array_equal(
            open_volume(tmp_path / "plain.zarr").to_numpy(), volume.to_numpy()
        )
        assert read_zarr(tmp_path / "plain.zarr").is_chunked

    def test_an_unknown_format_says_so(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello")
        with pytest.raises(ValueError, match="unrecognized"):
            open_volume(tmp_path / "notes.txt")


class TestInteroperability:
    """Compatibility claimed and never checked against another
    implementation is not compatibility."""

    def test_ome_zarr_py_reads_what_we_write(self, tmp_path, volume):
        reader = pytest.importorskip("ome_zarr.reader")
        io = pytest.importorskip("ome_zarr.io")

        path = write_ome_zarr(volume, tmp_path / "v.zarr", spacing=SPACING)
        location = io.parse_url(str(path))
        assert location is not None, "ome-zarr-py did not recognise the store"
        node = next(iter(reader.Reader(location)()))

        axes = [axis["name"] for axis in node.metadata["axes"]]
        assert axes == ["t", "c", "z", "y", "x"]
        data = np.asarray(node.data[0])
        assert data.shape == (1, *volume.shape)
        np.testing.assert_array_equal(data[0], volume.to_numpy())
