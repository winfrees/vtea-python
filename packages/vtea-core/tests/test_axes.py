"""The axis model, and the line between reading a time axis and analysing
one."""

import dask.array as da
import numpy as np
import pytest

from vtea_core.data.axes import (
    CANONICAL,
    VOLUME,
    Axes,
    TimeSeriesNotSupported,
    canonical_for,
    n_timepoints,
    to_canonical,
)


class TestAxes:
    def test_describes_its_own_layout(self):
        axes = Axes("CZYX")
        assert axes.ndim == 4
        assert axes.has_channel and not axes.has_time
        assert axes.spatial == "ZYX"
        assert axes.index_of("Z") == 1
        assert "C" in axes

    def test_a_missing_axis_is_minus_one_not_an_error(self):
        # "Is there a channel axis, and where?" is one question.
        assert Axes("ZYX").index_of("C") == -1

    def test_a_files_own_order_can_be_described(self):
        # An ImageJ hyperstack really is ZCYX on disk. A reader has to be
        # able to say so before it can transpose it.
        axes = Axes("ZCYX")
        assert not axes.is_canonical
        assert Axes("CZYX").is_canonical

    def test_only_spatial_axes_are_tileable(self):
        assert Axes("TCZYX").spatial_indices == (2, 3, 4)
        assert Axes("CZYX").spatial_indices == (1, 2, 3)
        assert Axes("YX").spatial_indices == (0, 1)

    def test_axis_types_use_the_ngff_vocabulary(self):
        assert Axes("TCZYX").types() == ["time", "channel", "space", "space", "space"]

    def test_lowercase_is_accepted(self):
        assert Axes("czyx").order == "CZYX"

    @pytest.mark.parametrize("order", ["", "CZYQ", "CCZY", "CZYX!"])
    def test_nonsense_orders_are_refused(self, order):
        with pytest.raises(ValueError):
            Axes(order)

    def test_transpose_to_gives_the_reordering(self):
        assert Axes("ZCYX").transpose_to("CZYX") == (1, 0, 2, 3)

    def test_transpose_between_different_axes_is_refused(self):
        with pytest.raises(ValueError, match="different axes"):
            Axes("ZYX").transpose_to("CZYX")

    def test_canonical_for_takes_the_trailing_axes(self):
        # An unlabelled 3D array is a z-stack far more often than it is
        # three channels of a plane.
        assert canonical_for(2).order == "YX"
        assert canonical_for(3).order == "ZYX"
        assert canonical_for(5).order == CANONICAL


class TestToCanonical:
    def test_a_plane_becomes_a_one_slice_one_channel_volume(self):
        assert to_canonical(np.zeros((5, 7)), "YX").shape == (1, 1, 5, 7)

    def test_a_stack_gains_a_channel_axis(self):
        assert to_canonical(np.zeros((3, 5, 7)), "ZYX").shape == (1, 3, 5, 7)

    def test_a_hyperstack_is_reordered_not_reshaped(self):
        array = np.arange(3 * 2 * 4 * 5).reshape(3, 2, 4, 5)  # ZCYX
        result = to_canonical(array, "ZCYX")
        assert result.shape == (2, 3, 4, 5)
        np.testing.assert_array_equal(result, np.transpose(array, (1, 0, 2, 3)))

    def test_values_survive_the_round_trip(self):
        array = np.random.default_rng(0).random((3, 2, 4, 5))
        there = to_canonical(array, "ZCYX", target="CZYX")
        back = to_canonical(there, "CZYX", target="ZCYX")
        np.testing.assert_array_equal(back, array)

    def test_a_single_timepoint_is_squeezed_away(self):
        assert to_canonical(np.zeros((1, 2, 3, 5, 7)), "TCZYX").shape == (2, 3, 5, 7)

    def test_a_real_time_series_is_refused_by_name(self):
        with pytest.raises(TimeSeriesNotSupported) as excinfo:
            to_canonical(np.zeros((4, 2, 3, 5, 7)), "TCZYX")
        message = str(excinfo.value)
        assert "4 timepoints" in message
        assert "time series" in message

    def test_the_refusal_is_still_a_notimplementederror(self):
        # Callers written against the old tiff reader keep working.
        with pytest.raises(NotImplementedError):
            to_canonical(np.zeros((4, 5, 7)), "TYX")

    def test_a_time_axis_is_added_on_the_way_out(self):
        # Which is what makes a store a valid time-series store from the
        # first one written, rather than one to convert later.
        assert to_canonical(np.zeros((2, 3, 5, 7)), VOLUME, target=CANONICAL).shape == (
            1,
            2,
            3,
            5,
            7,
        )

    def test_a_mismatched_shape_is_caught_early(self):
        with pytest.raises(ValueError, match="does not have"):
            to_canonical(np.zeros((5, 7)), "CZYX")

    def test_an_axis_the_target_has_no_room_for_is_refused(self):
        with pytest.raises(ValueError, match="not in the target"):
            to_canonical(np.zeros((2, 5, 7)), "CYX", target="ZYX")

    def test_it_works_on_a_dask_array_without_computing(self):
        array = da.zeros((3, 2, 4, 5), chunks=(1, 1, 2, 2))
        result = to_canonical(array, "ZCYX")
        assert isinstance(result, da.Array)
        assert result.shape == (2, 3, 4, 5)


class TestTimepointCount:
    def test_counted_even_when_there_is_only_one(self):
        # Recorded rather than assumed, so a store's metadata says what it
        # is and not what this version happens to support.
        assert n_timepoints((1, 2, 3, 4, 5), "TCZYX") == 1
        assert n_timepoints((7, 2, 3, 4, 5), "TCZYX") == 7

    def test_no_time_axis_means_one_timepoint(self):
        assert n_timepoints((2, 3, 4, 5), "CZYX") == 1
