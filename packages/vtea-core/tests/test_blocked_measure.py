"""Measuring objects out of core, and the claim that it changes nothing.

The whole value of the accumulator design is that the numbers come out the
same, so nearly every test here is a comparison against
`extract_measurements` on the same data held whole. `threshold_mean` gets
its own attention: it is the feature that does not compose, and the one the
plan originally proposed solving with random reads.
"""

import numpy as np
import pandas as pd
import pytest

from vtea_core.blocked import MemoryBudget, plan_tiles
from vtea_core.blocked.measure import (
    ObjectStats,
    accumulate,
    measure_blocked,
    measure_blocked_by_channel,
    threshold_means,
    with_seam_columns,
)
from vtea_core.data import Spacing
from vtea_core.measurements import (
    MeasurementStore,
    extract_measurements,
    extract_measurements_by_channel,
    read_measurements,
    write_measurements,
)

SPACING = Spacing((2.0, 0.5, 0.5))
# stddev comes from a sum of squares rather than a second pass, so it
# cancels to about ten significant figures instead of being bit-exact.
STDDEV_TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def labelled():
    """Boxes on a grid, so ids and sizes are known and seams cut plenty."""
    labels = np.zeros((20, 60, 60), np.int32)
    identifier = 0
    for z in range(2, 18, 6):
        for y in range(4, 56, 12):
            for x in range(4, 56, 12):
                identifier += 1
                labels[z : z + 4, y : y + 8, x : x + 8] = identifier
    return labels, identifier


@pytest.fixture(scope="module")
def intensity(labelled):
    labels, _count = labelled
    rng = np.random.default_rng(0)
    return (rng.random(labels.shape) * 1000 + labels * 3).astype(np.uint16)


def plan_for(shape, tiles_wanted, halo=0):
    return plan_tiles(
        shape,
        budget=MemoryBudget(int(np.prod(shape) * 8 / max(tiles_wanted, 1) / 0.6) + 8192),
        bytes_per_voxel=8,
        halo=halo,
    )


def assert_tables_match(got, expected, *, tolerance=0.0):
    assert list(got.columns) == list(expected.columns)
    for column in expected.columns:
        limit = STDDEV_TOLERANCE if column == "stddev" else tolerance
        np.testing.assert_allclose(
            got[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            rtol=limit,
            atol=limit,
            err_msg=f"column {column!r}",
        )


class TestExactness:
    @pytest.mark.parametrize("tiles_wanted", [1, 8, 27])
    def test_it_matches_the_whole_image_table(self, labelled, intensity, tiles_wanted):
        labels, count = labelled
        expected = extract_measurements(labels, intensity, spacing=SPACING)
        got = measure_blocked(
            labels,
            intensity,
            plan=plan_for(labels.shape, tiles_wanted),
            n_objects=count,
            spacing=SPACING,
        )
        assert_tables_match(got, expected)

    def test_the_columns_are_the_same_columns_in_the_same_order(self, labelled, intensity):
        labels, count = labelled
        expected = extract_measurements(labels, intensity, spacing=SPACING)
        got = measure_blocked(
            labels, intensity, plan=plan_for(labels.shape, 8), n_objects=count, spacing=SPACING
        )
        assert list(got.columns) == list(expected.columns)

    def test_no_spacing_means_no_volume_column(self, labelled, intensity):
        labels, count = labelled
        got = measure_blocked(
            labels, intensity, plan=plan_for(labels.shape, 8), n_objects=count
        )
        assert "volume" not in got.columns
        assert "count" in got.columns

    def test_an_unknown_spacing_is_not_a_calibration(self, labelled, intensity):
        from vtea_core.data.spacing import UNKNOWN

        labels, count = labelled
        got = measure_blocked(
            labels,
            intensity,
            plan=plan_for(labels.shape, 8),
            n_objects=count,
            spacing=Spacing((1.0, 1.0, 1.0), source=UNKNOWN),
        )
        assert "volume" not in got.columns

    def test_float_intensities_agree_to_floating_point(self, labelled):
        labels, count = labelled
        values = np.random.default_rng(1).normal(500, 100, labels.shape)
        expected = extract_measurements(labels, values)
        got = measure_blocked(
            labels, values, plan=plan_for(labels.shape, 27), n_objects=count
        )
        assert_tables_match(got, expected, tolerance=1e-9)

    def test_ids_with_gaps_are_not_invented(self, intensity):
        # An id the array never used must not appear as an object measuring
        # zero of everything.
        labels = np.zeros((8, 16, 16), np.int32)
        labels[1:4, 2:6, 2:6] = 3
        labels[4:7, 9:14, 9:14] = 7
        values = intensity[:8, :16, :16]
        expected = extract_measurements(labels, values)
        got = measure_blocked(labels, values, plan=plan_for(labels.shape, 4), n_objects=7)
        assert list(got["object_id"]) == [3, 7]
        assert_tables_match(got, expected)

    def test_an_empty_label_array_measures_nothing(self):
        labels = np.zeros((4, 8, 8), np.int32)
        got = measure_blocked(
            labels, labels, plan=plan_for(labels.shape, 1), n_objects=0
        )
        assert len(got) == 0


class TestThresholdMean:
    """The feature that does not compose."""

    @pytest.mark.parametrize("tiles_wanted", [1, 8, 27])
    def test_it_is_exact_at_any_tile_size(self, labelled, intensity, tiles_wanted):
        labels, count = labelled
        expected = extract_measurements(labels, intensity)["threshold_mean"].to_numpy()
        plan = plan_for(labels.shape, tiles_wanted)
        stats = accumulate(labels, intensity, plan=plan, n_objects=count)
        got = threshold_means(labels, intensity, plan=plan, stats=stats)
        np.testing.assert_allclose(got[stats.object_ids], expected)

    def test_it_needs_the_first_pass_to_have_finished(self, labelled, intensity):
        # The point of the two passes: a cutoff derived from one tile's
        # extremes is not the object's cutoff, and using it silently gives a
        # different number.
        labels, count = labelled
        plan = plan_for(labels.shape, 27)
        full = accumulate(labels, intensity, plan=plan, n_objects=count)

        one_tile = ObjectStats.empty(count, labels.ndim)
        first = next(iter(plan.tiles()))
        one_tile.add_tile(
            np.asarray(labels[first.core]),
            np.asarray(intensity[first.core]),
            [part.start for part in first.core],
        )
        shared = np.intersect1d(full.object_ids, one_tile.object_ids)
        assert shared.size
        assert not np.allclose(full.cutoffs()[shared], one_tile.cutoffs()[shared])

    def test_an_object_of_one_value_is_still_measurable(self):
        labels = np.zeros((4, 8, 8), np.int32)
        labels[1:3, 2:5, 2:5] = 1
        flat = np.full(labels.shape, 700, np.uint16)
        expected = extract_measurements(labels, flat)
        got = measure_blocked(labels, flat, plan=plan_for(labels.shape, 1), n_objects=1)
        assert_tables_match(got, expected)


class TestChannels:
    @pytest.fixture
    def multichannel(self, labelled):
        labels, count = labelled
        rng = np.random.default_rng(2)
        stack = np.stack(
            [
                (rng.random(labels.shape) * 500 + labels * 2).astype(np.uint16),
                (rng.random(labels.shape) * 900).astype(np.uint16),
            ]
        )
        return labels, count, stack

    def test_it_matches_the_whole_image_by_channel_table(self, multichannel):
        labels, count, stack = multichannel
        expected = extract_measurements_by_channel(
            labels, stack, channel_axis=0, spacing=SPACING
        )
        got = measure_blocked_by_channel(
            labels,
            stack,
            plan=plan_for(labels.shape, 8),
            n_objects=count,
            channel_axis=0,
            spacing=SPACING,
        )
        assert_tables_match(got, expected)

    def test_geometry_appears_once_and_intensities_carry_their_channel(self, multichannel):
        labels, count, stack = multichannel
        got = measure_blocked_by_channel(
            labels, stack, plan=plan_for(labels.shape, 8), n_objects=count, channel_axis=0
        )
        assert list(got.columns).count("count") == 1
        assert "mean_ch0" in got.columns and "mean_ch1" in got.columns
        assert "mean" not in got.columns

    def test_one_channel_can_be_singled_out(self, multichannel):
        labels, count, stack = multichannel
        got = measure_blocked_by_channel(
            labels,
            stack,
            plan=plan_for(labels.shape, 4),
            n_objects=count,
            channel_axis=0,
            channel=1,
        )
        assert "mean_ch1" in got.columns
        assert "mean_ch0" not in got.columns

    def test_a_channel_that_is_not_there_says_so(self, multichannel):
        labels, count, stack = multichannel
        with pytest.raises(ValueError, match="out of range"):
            measure_blocked_by_channel(
                labels,
                stack,
                plan=plan_for(labels.shape, 1),
                n_objects=count,
                channel_axis=0,
                channel=5,
            )

    def test_no_channel_axis_falls_back_to_a_single_measurement(self, labelled, intensity):
        labels, count = labelled
        expected = extract_measurements(labels, intensity)
        got = measure_blocked_by_channel(
            labels, intensity, plan=plan_for(labels.shape, 4), n_objects=count
        )
        assert_tables_match(got, expected)

    def test_measuring_one_channel_does_not_read_the_others(self, multichannel):
        # The reason _ChannelView exists: intensity[channel] on a stored
        # array would pull a whole channel into memory per tile.
        labels, count, stack = multichannel
        reads = []

        class Counting:
            shape = stack.shape
            dtype = stack.dtype
            ndim = stack.ndim

            def __getitem__(self, index):
                reads.append(index)
                return stack[index]

        measure_blocked_by_channel(
            labels,
            Counting(),
            plan=plan_for(labels.shape, 8),
            n_objects=count,
            channel_axis=0,
            channel=1,
        )
        assert reads
        assert all(index[0] == 1 for index in reads)


class TestSeamColumns:
    def test_the_ledger_joins_onto_the_table(self, labelled, intensity):
        from vtea_core.blocked import LabelLedger
        from vtea_core.blocked.reconcile import Fragment

        labels, count = labelled
        frame = measure_blocked(
            labels, intensity, plan=plan_for(labels.shape, 4), n_objects=count
        )
        ledger = LabelLedger()
        for object_id in frame["object_id"]:
            ledger.add(
                int(object_id),
                [
                    Fragment(
                        tile=(0,),
                        local_id=int(object_id),
                        provisional_id=int(object_id),
                        core_voxels=1,
                        block_voxels=1,
                        centroid=(0.0, 0.0, 0.0),
                        bbox=((0, 1), (0, 1), (0, 1)),
                    )
                ],
            )
        joined = with_seam_columns(frame, ledger)
        assert "seam_confidence" in joined.columns
        assert len(joined) == len(frame)
        assert (joined["seam_confidence"] == 1.0).all()

    def test_no_ledger_leaves_the_table_alone(self, labelled, intensity):
        labels, count = labelled
        frame = measure_blocked(
            labels, intensity, plan=plan_for(labels.shape, 1), n_objects=count
        )
        assert with_seam_columns(frame, None) is frame


class TestParquetStore:
    def test_a_table_survives_a_round_trip(self, tmp_path):
        frame = pd.DataFrame({"object_id": [1, 2, 3], "mean": [1.5, 2.5, 3.5]})
        path = write_measurements(frame, tmp_path / "m.parquet")
        pd.testing.assert_frame_equal(read_measurements(path), frame)

    def test_duckdb_queries_the_file_rather_than_loading_it(self, tmp_path):
        frame = pd.DataFrame({"object_id": [1, 2, 3], "mean": [1.5, 2.5, 3.5]})
        path = write_measurements(frame, tmp_path / "m.parquet")
        store = MeasurementStore()
        store.register_parquet("OBJECTS", path)
        result = store.query("SELECT count(*) AS n, avg(mean) AS m FROM OBJECTS")
        assert result.loc[0, "n"] == 3
        assert result.loc[0, "m"] == pytest.approx(2.5)

    def test_rewriting_the_file_updates_the_table(self, tmp_path):
        # Registered as a view, so a re-run does not need re-registering.
        path = write_measurements(pd.DataFrame({"a": [1, 2, 3]}), tmp_path / "m.parquet")
        store = MeasurementStore()
        store.register_parquet("T", path)
        assert store.query("SELECT count(*) AS n FROM T").loc[0, "n"] == 3
        write_measurements(pd.DataFrame({"a": [1]}), path)
        assert store.query("SELECT count(*) AS n FROM T").loc[0, "n"] == 1

    def test_awkward_names_and_paths_are_quoted_not_trusted(self, tmp_path):
        odd = tmp_path / "it's odd"
        path = write_measurements(pd.DataFrame({"a": [1, 2]}), odd / "m.parquet")
        store = MeasurementStore()
        store.register_parquet('watershed "1"', path)
        assert store.query('SELECT count(*) AS n FROM "watershed ""1"""').loc[0, "n"] == 2

    def test_a_directory_is_created_for_the_file(self, tmp_path):
        path = write_measurements(
            pd.DataFrame({"a": [1]}), tmp_path / "deep" / "deeper" / "m.parquet"
        )
        assert path.exists()


@pytest.fixture(scope="module")
def protocol():
    from vtea_core.workflow import Pipeline, Step

    return Pipeline(
        [
            Step.for_function("imageprocessing", "gaussian_blur", params={"sigma": 1.0}),
            Step.for_function(
                "segmentation",
                "threshold_mask",
                params={"method": "otsu"},
                available={"volume"},
            ),
            Step.for_function("segmentation", "label_components", available={"mask"}),
            Step.for_function(
                "measurements",
                "extract_measurements",
                available={"labels", "intensity", "spacing"},
            ),
        ]
    )


@pytest.fixture(scope="module")
def volume():
    rng = np.random.default_rng(0)
    data = rng.normal(800, 200, (24, 80, 80)).clip(0, 4000).astype(np.uint16)
    grid = np.ogrid[-5:6, -5:6, -5:6]
    ball = sum(axis**2 for axis in grid) <= 25
    for centre in [(6, 15, 15), (6, 15, 55), (16, 55, 15), (16, 55, 55), (11, 40, 40)]:
        data[tuple(slice(v - 5, v + 6) for v in centre)][ball] = 3500
    return data


class TestPipelineIntegration:
    """The whole way through: pixels in, a measurement table out, none of it
    ever held whole."""

    def run_blocked(self, protocol, volume, budget=700_000, **kwargs):
        from vtea_core.blocked import BlockedPipeline

        plan = plan_tiles(
            volume.shape, budget=MemoryBudget(budget), bytes_per_voxel=8, halo=12
        )
        with BlockedPipeline(protocol, plan=plan, spacing=SPACING, **kwargs) as blocked:
            context = blocked.run(
                {"volume": volume, "intensity": volume, "spacing": SPACING}
            )
            return plan, blocked, context["measurements"]

    def test_the_table_matches_the_in_memory_run(self, protocol, volume):
        expected = protocol.run(
            {"volume": volume, "intensity": volume, "spacing": SPACING}
        )["measurements"]
        plan, _blocked, got = self.run_blocked(protocol, volume)
        assert plan.n_tiles > 1
        assert_tables_match(got[list(expected.columns)], expected)

    def test_the_seam_columns_come_along(self, protocol, volume):
        _plan, _blocked, got = self.run_blocked(protocol, volume)
        for column in ("n_fragments", "seam_rule", "seam_confidence"):
            assert column in got.columns
        # Objects the tiling cut are the ones a review would want.
        assert got["touches_seam"].any()

    def test_measurements_land_in_the_context_as_a_table(self, protocol, volume):
        _plan, blocked, got = self.run_blocked(protocol, volume)
        assert isinstance(got, pd.DataFrame)
        card = blocked.results["extract_measurements_1"]
        assert card.table is not None
        assert card.array is None

    def test_it_can_be_written_where_it_outlives_the_run(self, protocol, volume, tmp_path):
        _plan, _blocked, got = self.run_blocked(protocol, volume)
        path = write_measurements(got, tmp_path / "objects.parquet")
        store = MeasurementStore()
        store.register_parquet("OBJECTS", path)
        counted = store.query("SELECT count(*) AS n FROM OBJECTS").loc[0, "n"]
        assert counted == len(got)

    def test_a_weighted_measurement_says_which_phase_it_waits_on(self, volume):
        from vtea_core.blocked import BlockedPipeline, NotBlockableYet
        from vtea_core.workflow import Pipeline, Step

        step = Step.for_function("measurements", "weighted_measurements_by_channel")
        plan = plan_tiles(volume.shape, budget=MemoryBudget(10**9), bytes_per_voxel=8)
        with BlockedPipeline(Pipeline([step]), plan=plan) as blocked:
            with pytest.raises(NotBlockableYet, match="L6"):
                blocked.run({"ownership": volume, "intensity": volume})
