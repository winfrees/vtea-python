"""Phase L6: ownership that costs the same order as the image it describes.

The dense top-k form is the largest thing a protocol produces - over the
volume the plan works from it is 201 GB, six times the image. It is dense
over a field that is mostly background, and ownership is only defined inside
the mask, so the fix is to keep it where it means something.
"""

import numpy as np
import pytest

from vtea_core.blocked import MemoryBudget, plan_tiles
from vtea_core.blocked.measure import (
    weighted_measure_blocked,
    weighted_measure_blocked_by_channel,
)
from vtea_core.blocked.ownership import (
    HaloTooSmallForReach,
    SparseOwnership,
    load_sparse_ownership,
    ownership_blocked,
    required_reach,
)
from vtea_core.data import Spacing
from vtea_core.measurements import weighted_measurements, weighted_measurements_by_channel
from vtea_core.objects import distance_ownership

# Probabilities are float32 rather than float64 - a posterior meaningful to
# seven decimal places is not a posterior anybody has, and it halves the
# largest array. The cost shows up here, and only here.
WEIGHTED_TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def field():
    labels = np.zeros((16, 64, 64), np.int32)
    for identifier, (z, y, x) in enumerate(
        [(6, 14, 14), (6, 14, 46), (9, 46, 14), (9, 46, 46)], start=1
    ):
        labels[z : z + 3, y : y + 4, x : x + 4] = identifier
    mask = np.zeros(labels.shape, bool)
    mask[3:13, 6:58, 6:58] = True
    intensity = (np.random.default_rng(0).random(labels.shape) * 1000).astype(np.uint16)
    return labels, mask, intensity


@pytest.fixture(scope="module")
def dense(field):
    labels, mask, _intensity = field
    return distance_ownership(labels, mask, falloff=2.0, top_k=2)


def plan_for(shape, tiles_wanted, halo):
    core = float(np.prod(shape)) / max(tiles_wanted, 1)
    edge = core ** (1 / len(shape))
    factor = ((edge + 2 * halo) / edge) ** len(shape)
    return plan_tiles(
        shape,
        budget=MemoryBudget(int(core * factor * 8 / 0.6) + 8192),
        bytes_per_voxel=8,
        halo=halo,
    )


class TestSparseForm:
    def test_it_says_the_same_thing_as_the_dense_form(self, dense):
        sparse = SparseOwnership.from_dense(dense)
        np.testing.assert_array_equal(sparse.hard(), dense.hard())
        np.testing.assert_allclose(sparse.confidence(), dense.confidence(), atol=1e-6)
        assert sparse.object_ids() == dense.object_ids()

    def test_it_is_smaller_by_the_share_of_background(self, dense):
        sparse = SparseOwnership.from_dense(dense)
        assert sparse.nbytes < sparse.dense_nbytes
        # The saving is the point, so it is reported rather than assumed.
        assert "smaller than the dense form" in sparse.summary()

    def test_the_saving_grows_as_the_field_empties(self):
        # The realistic case is a mask that is a few percent of the volume,
        # where the dense form is almost entirely zeros.
        labels = np.zeros((16, 64, 64), np.int32)
        labels[8:10, 30:34, 30:34] = 1
        sparse = SparseOwnership.from_dense(
            distance_ownership(labels, labels > 0, falloff=1.0, top_k=2)
        )
        assert sparse.density < 0.01
        assert sparse.dense_nbytes / sparse.nbytes > 50

    def test_it_round_trips_through_dense(self, dense):
        back = SparseOwnership.from_dense(dense).to_dense()
        np.testing.assert_array_equal(back.hard(), dense.hard())
        np.testing.assert_allclose(back.confidence(), dense.confidence(), atol=1e-6)

    def test_hard_can_be_written_into_an_array_it_did_not_allocate(self, dense):
        # How a label image larger than memory comes out of this: hand it a
        # stored array and nothing dense is ever held.
        sparse = SparseOwnership.from_dense(dense)
        out = np.zeros(sparse.shape, dtype=np.int32)
        assert sparse.hard(out=out) is out
        np.testing.assert_array_equal(out, dense.hard())

    def test_margin_distinguishes_a_coin_toss_from_a_weak_claim(self, dense):
        sparse = SparseOwnership.from_dense(dense)
        margin = sparse.margin()
        assert margin.shape == (sparse.n_voxels,)
        assert (margin >= 0).all()

    def test_contested_returns_positions_not_a_dense_mask(self, dense):
        sparse = SparseOwnership.from_dense(dense)
        contested = sparse.contested(0.99)
        assert contested.ndim == 1
        assert (sparse.probabilities[0][contested] < 0.99).all()

    def test_a_correction_stays_distinguishable_from_an_inference(self, dense):
        sparse = SparseOwnership.from_dense(dense)
        where = np.arange(5)
        sparse.override(where, 7)
        assert (sparse.owners[0][where] == 7).all()
        assert (sparse.probabilities[0][where] == 1.0).all()
        assert sparse.manual[where].all()
        assert not sparse.manual[10:].any()

    def test_weights_for_an_owner_are_its_claim_everywhere(self, dense):
        sparse = SparseOwnership.from_dense(dense)
        where, weights = sparse.weights_for(1)
        assert where.size
        assert (weights > 0).all()
        np.testing.assert_allclose(weights, dense.weights(1)[sparse.coordinates(where)])

    def test_it_survives_a_round_trip_to_disk(self, dense, tmp_path):
        sparse = SparseOwnership.from_dense(dense)
        path = sparse.save(tmp_path / "own.npz")
        back = load_sparse_ownership(path)
        np.testing.assert_array_equal(back.hard(), sparse.hard())
        assert back.object_ids() == sparse.object_ids()

    def test_mismatched_arrays_are_refused(self):
        with pytest.raises(ValueError, match="same shape"):
            SparseOwnership(
                shape=(4, 4),
                indices=np.arange(3),
                owners=np.zeros((2, 3), np.int32),
                probabilities=np.zeros((1, 3), np.float32),
            )
        with pytest.raises(ValueError, match="owner entries"):
            SparseOwnership(
                shape=(4, 4),
                indices=np.arange(3),
                owners=np.zeros((2, 5), np.int32),
                probabilities=np.zeros((2, 5), np.float32),
            )


class TestBlockedOwnership:
    @pytest.mark.parametrize("tiles_wanted", [1, 8, 27])
    def test_it_matches_the_whole_image_answer(self, field, dense, tiles_wanted):
        labels, mask, _intensity = field
        plan = plan_for(labels.shape, tiles_wanted, 10)
        blocked = ownership_blocked(labels, mask, plan=plan, falloff=2.0, top_k=2)
        np.testing.assert_array_equal(blocked.hard(), dense.hard())
        np.testing.assert_allclose(blocked.confidence(), dense.confidence(), atol=1e-6)

    def test_a_halo_shorter_than_the_reach_is_refused(self, field):
        # Not a rounding error: a tile that cannot see the marker across the
        # seam gives the voxel to the nearest one it *can* see, which is a
        # visible line down the result.
        labels, mask, _intensity = field
        with pytest.raises(HaloTooSmallForReach, match="further than the tiles overlap"):
            ownership_blocked(
                labels, mask, plan=plan_for(labels.shape, 27, 2), falloff=2.0
            )

    def test_the_reach_check_is_anisotropic(self, field):
        # A reach of 8 microns is 4 voxels along a 2 micron z-step and 16 in
        # x at 0.5. A scalar check would pass on z and be wrong in x.
        labels, mask, _intensity = field
        spacing = Spacing((2.0, 0.5, 0.5))
        with pytest.raises(HaloTooSmallForReach, match="axis 1"):
            ownership_blocked(
                labels,
                mask,
                plan=plan_for(labels.shape, 8, 10),
                spacing=spacing,
                falloff=2.0,
            )

    def test_the_default_reach_is_four_falloffs(self):
        assert required_reach(2.0, None) == 8.0
        assert required_reach(2.0, 3.0) == 3.0

    def test_a_single_tile_needs_no_halo_at_all(self, field, dense):
        labels, mask, _intensity = field
        plan = plan_tiles(labels.shape, budget=MemoryBudget(10**9), bytes_per_voxel=8)
        assert plan.is_single_tile
        blocked = ownership_blocked(labels, mask, plan=plan, falloff=2.0, top_k=2)
        np.testing.assert_array_equal(blocked.hard(), dense.hard())

    def test_entries_are_grouped_by_tile(self, field):
        labels, mask, _intensity = field
        plan = plan_for(labels.shape, 8, 10)
        blocked = ownership_blocked(labels, mask, plan=plan, falloff=2.0)
        assert blocked.offsets is not None
        assert len(blocked.offsets) == plan.n_tiles + 1
        assert blocked.offsets[-1] == blocked.n_voxels
        # Every entry in a tile's slice really is inside that tile's core.
        for index, tile in enumerate(plan.tiles()):
            coordinates = blocked.coordinates(blocked.tile_slice(index))
            for axis, part in zip(coordinates, tile.core):
                assert ((axis >= part.start) & (axis < part.stop)).all()

    def test_no_voxel_is_counted_twice(self, field):
        labels, mask, _intensity = field
        blocked = ownership_blocked(
            labels, mask, plan=plan_for(labels.shape, 8, 10), falloff=2.0
        )
        assert len(np.unique(blocked.indices)) == blocked.n_voxels

    def test_an_empty_mask_produces_an_empty_ownership(self, field):
        labels, _mask, _intensity = field
        empty = np.zeros(labels.shape, bool)
        blocked = ownership_blocked(
            labels, empty, plan=plan_for(labels.shape, 8, 10), falloff=2.0
        )
        assert blocked.n_voxels == 0
        assert blocked.object_ids() == []


class TestWeightedMeasurement:
    @pytest.mark.parametrize("tiles_wanted", [1, 8])
    def test_it_matches_the_whole_image_table(self, field, dense, tiles_wanted):
        labels, mask, intensity = field
        spacing = Spacing((2.0, 0.5, 0.5))
        expected = weighted_measurements(dense, intensity, spacing=spacing)

        plan = plan_for(labels.shape, tiles_wanted, 10)
        blocked = ownership_blocked(labels, mask, plan=plan, falloff=2.0, top_k=2)
        got = weighted_measure_blocked(blocked, intensity, plan=plan, spacing=spacing)

        assert list(got.columns) == list(expected.columns)
        for column in expected.columns:
            np.testing.assert_allclose(
                got[column].to_numpy(dtype=float),
                expected[column].to_numpy(dtype=float),
                rtol=WEIGHTED_TOLERANCE,
                atol=WEIGHTED_TOLERANCE,
                err_msg=f"column {column!r}",
            )

    def test_it_needs_no_second_pass(self, field):
        # Unlike threshold_mean, every weighted quantity is additive, so one
        # pass over the tiles is the whole calculation.
        labels, mask, intensity = field
        plan = plan_for(labels.shape, 8, 10)
        blocked = ownership_blocked(labels, mask, plan=plan, falloff=2.0)
        reads = {"n": 0}

        class Counting:
            shape = intensity.shape
            dtype = intensity.dtype
            ndim = intensity.ndim

            def __getitem__(self, index):
                reads["n"] += 1
                return intensity[index]

        weighted_measure_blocked(blocked, Counting(), plan=plan)
        assert reads["n"] <= plan.n_tiles

    def test_by_channel_matches_and_names_its_channels(self, field, dense):
        labels, mask, _intensity = field
        rng = np.random.default_rng(3)
        stack = np.stack(
            [
                (rng.random(labels.shape) * 500).astype(np.uint16),
                (rng.random(labels.shape) * 900).astype(np.uint16),
            ]
        )
        expected = weighted_measurements_by_channel(dense, stack, channel_axis=0)

        plan = plan_for(labels.shape, 8, 10)
        blocked = ownership_blocked(labels, mask, plan=plan, falloff=2.0, top_k=2)
        got = weighted_measure_blocked_by_channel(
            blocked, stack, plan=plan, channel_axis=0
        )
        assert list(got.columns) == list(expected.columns)
        assert "mean_ch0" in got.columns and "mean_ch1" in got.columns
        for column in expected.columns:
            np.testing.assert_allclose(
                got[column].to_numpy(dtype=float),
                expected[column].to_numpy(dtype=float),
                rtol=WEIGHTED_TOLERANCE,
                atol=WEIGHTED_TOLERANCE,
                err_msg=f"column {column!r}",
            )

    def test_an_empty_ownership_gives_an_empty_table(self, field):
        labels, _mask, intensity = field
        plan = plan_for(labels.shape, 8, 10)
        blocked = ownership_blocked(
            labels, np.zeros(labels.shape, bool), plan=plan, falloff=2.0
        )
        assert len(weighted_measure_blocked(blocked, intensity, plan=plan)) == 0


class TestPipelineIntegration:
    def test_ownership_and_its_measurements_run_out_of_core(self, field):
        from vtea_core.blocked import BlockedPipeline
        from vtea_core.workflow import Pipeline, Step

        labels, mask, intensity = field
        spacing = Spacing((2.0, 0.5, 0.5))
        protocol = Pipeline(
            [
                Step.for_function(
                    "ownership",
                    "distance_ownership",
                    params={"falloff": 2.0, "top_k": 2},
                    available={"labels", "mask", "spacing"},
                ),
                Step.for_function(
                    "measurements",
                    "weighted_measurements_by_channel",
                    available={"ownership", "intensity", "channel_axis", "spacing"},
                ),
            ]
        )
        seed = {
            "labels": labels,
            "mask": mask,
            "intensity": intensity,
            "spacing": spacing,
            "channel_axis": None,
        }
        expected = protocol.run(seed)["measurements"]

        plan = plan_for(labels.shape, 8, 18)
        with BlockedPipeline(protocol, plan=plan, spacing=spacing) as blocked:
            context = blocked.run(seed)
            got = context["measurements"]
            assert isinstance(context["ownership"], SparseOwnership)

        assert list(got.columns) == list(expected.columns)
        for column in expected.columns:
            np.testing.assert_allclose(
                got[column].to_numpy(dtype=float),
                expected[column].to_numpy(dtype=float),
                rtol=WEIGHTED_TOLERANCE,
                atol=WEIGHTED_TOLERANCE,
                err_msg=f"column {column!r}",
            )
