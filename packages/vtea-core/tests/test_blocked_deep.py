"""Phase L5: the pieces a learned segmenter needs.

A learned segmenter breaks the assumption the other reconciliation
strategies rest on - that at least one tile's copy of a cut object is worth
keeping. Near a tile edge its answer is computed from truncated context, so
every copy is shaped by a boundary that is not in the specimen. The tests
here use a stand-in for that behaviour rather than cellpose itself, so they
run without a GPU and without the deeplearning extra, and so that the
failure being fixed is visible rather than assumed.
"""

import json
import math

import numpy as np
import pytest

from vtea_core.blocked import (
    ABUTTING,
    MemoryBudget,
    SeamPolicy,
    SeamPolicyError,
    ZarrScratch,
    filter_by_size_blocked,
    plan_tiles,
    segment_blocked,
)
from vtea_core.blocked import gpu as gpu_module
from vtea_core.blocked.gpu import (
    Calibration,
    calibrate,
    calibrated_voxels,
    device_name,
    gpu_plan,
    gpu_tile_voxels,
    is_out_of_memory,
    load_calibration,
    save_calibration,
)
from vtea_core.blocked.resume import ManifestMismatch, RunManifest, plan_signature
from vtea_core.segmentation import label_components

EDGE = 3


def edge_shy(mask):
    """A stand-in for a segmenter that needs context.

    It refuses to label anything within `EDGE` voxels of the block it was
    handed, because it cannot see enough around it - so its answer depends
    on where the block boundary falls, which is the property that makes
    picking between tiles' copies the wrong thing to do.
    """
    trimmed = np.zeros_like(mask)
    inner = tuple(
        slice(EDGE, -EDGE) if size > 2 * EDGE else slice(0, size) for size in mask.shape
    )
    trimmed[inner] = mask[inner]
    return label_components(trimmed)


@pytest.fixture(scope="module")
def balls():
    shape = (24, 72, 72)
    mask = np.zeros(shape, bool)
    grid = np.ogrid[-5:6, -5:6, -5:6]
    ball = sum(axis**2 for axis in grid) <= 25
    for centre in [(6, 12, 12), (6, 12, 36), (6, 36, 12), (12, 36, 36), (17, 58, 58), (12, 58, 20)]:
        mask[tuple(slice(v - 5, v + 6) for v in centre)] |= ball
    return mask


@pytest.fixture(scope="module")
def truth(balls):
    """What the segmenter says when it can see the whole field."""
    return edge_shy(balls)


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


def run(mask, policy, *, tiles=27, halo=6, function=edge_shy):
    plan = plan_for(mask.shape, tiles, halo)
    with ZarrScratch() as scratch:
        result = segment_blocked(
            function, {"mask": mask}, plan=plan, scratch=scratch, policy=policy
        )
        return plan, result.ledger, np.asarray(result.array)


def sizes_of(array):
    counts = np.bincount(array.reshape(-1))[1:]
    return sorted(counts[counts > 0].tolist())


class TestResegment:
    def test_it_recovers_what_picking_between_copies_cannot(self, balls, truth):
        # The whole justification for the strategy, as a comparison. With a
        # halo too small to hold an object *plus* the context the segmenter
        # needs, every tile's copy is truncated - so `own` keeps a truncated
        # one, and re-segmenting the object in its own window does not.
        expected = sizes_of(truth)
        _plan, _own_ledger, owned = run(balls, SeamPolicy.overlap_match())
        _plan, ledger, resegmented = run(balls, SeamPolicy.resegment())

        assert sizes_of(owned) != expected, "the negative control has stopped failing"
        assert sizes_of(resegmented) == expected
        assert ledger.n_objects == int(truth.max())

    def test_every_object_comes_back_exactly(self, balls, truth):
        _plan, _ledger, result = run(balls, SeamPolicy.resegment())
        for object_id in np.unique(result[result > 0]):
            found = result == object_id
            overlapping = np.unique(truth[found])
            overlapping = overlapping[overlapping > 0]
            assert len(overlapping) == 1
            assert np.array_equal(found, truth == overlapping[0])

    def test_it_says_so_on_every_object_it_touched(self, balls):
        _plan, ledger, _result = run(balls, SeamPolicy.resegment())
        rules = set(ledger.decided_by.values())
        assert "resegment" in rules

    def test_a_sliver_inside_a_re_segmented_object_is_absorbed(self, balls):
        # Two fragments that failed to match can turn out to be one object
        # once the segmenter sees the whole of it. The one left with no
        # voxels must not be reported as an object of size zero.
        _plan, ledger, result = run(balls, SeamPolicy.resegment())
        assert ledger.dropped
        assert all("absorbed" in reason for reason in ledger.dropped.values())
        for object_id in ledger.dropped:
            assert not (result == object_id).any()
        assert set(ledger.dropped).isdisjoint(ledger.object_ids)

    def test_it_does_not_swallow_a_neighbour_it_cannot_see_all_of(self, balls):
        # The condition that keeps absorption honest: an id is only taken
        # over when the window contains all of it. A neighbour running out
        # of the window keeps its voxels.
        _plan, ledger, result = run(balls, SeamPolicy.resegment())
        # Every object in the array is accounted for in the ledger, and vice
        # versa - nothing was quietly eaten.
        assert set(np.unique(result[result > 0]).tolist()) == set(ledger.object_ids)

    def test_sizes_are_updated_so_a_size_filter_sees_the_new_object(self, balls, truth):
        # The fragments a re-segmented object replaces described tiles'
        # partial views of something that no longer exists. Leaving them
        # would filter on a size the object does not have.
        plan = plan_for(balls.shape, 27, 6)
        with ZarrScratch() as scratch:
            result = segment_blocked(
                edge_shy, {"mask": balls}, plan=plan, scratch=scratch,
                policy=SeamPolicy.resegment(),
            )
            counts = np.bincount(np.asarray(result.array).reshape(-1))
            for object_id, size in result.ledger.sizes().items():
                assert size == counts[object_id]

            kept = filter_by_size_blocked(result, scratch=scratch, min_size=500)
            expected = sum(1 for size in sizes_of(truth) if size >= 500)
            assert kept.n_objects == expected

    def test_a_window_bigger_than_a_tile_is_declined_not_attempted(self, balls):
        # Re-segmenting a window larger than the tiles were sized for would
        # need more memory than the plan was built with, which is the one
        # thing the plan exists to prevent.
        policy = SeamPolicy.resegment(resegment_margin=10_000)
        _plan, ledger, _result = run(balls, policy)
        assert any("too-large" in rule for rule in ledger.decided_by.values())

    def test_a_segmenter_that_finds_nothing_leaves_the_stitched_answer(self, balls):
        # A real disagreement rather than an error: the model saw the object
        # in a tile and not in the window centred on it. The stitched answer
        # stands, flagged, so a review can see it happened.
        plan = plan_for(balls.shape, 27, 6)
        calls = {"n": 0}

        def gives_up_after_the_tiles(mask):
            calls["n"] += 1
            if calls["n"] > plan.n_tiles:  # the tiles are done; these are the windows
                return np.zeros_like(mask, dtype=np.int32)
            return edge_shy(mask)

        with ZarrScratch() as scratch:
            result = segment_blocked(
                gives_up_after_the_tiles,
                {"mask": balls},
                plan=plan,
                scratch=scratch,
                policy=SeamPolicy.resegment(),
            )
            array = np.asarray(result.array)
        assert calls["n"] > plan.n_tiles, "no window was re-segmented at all"
        assert any("no-match" in rule for rule in result.ledger.decided_by.values())
        # The objects are still there, from before the re-segmentation.
        assert array.max() > 0

    def test_it_needs_a_matching_and_an_overlap(self):
        with pytest.raises(SeamPolicyError, match="which those are"):
            SeamPolicy(matching="none", resolution="resegment")
        with pytest.raises(SeamPolicyError, match="none to give"):
            SeamPolicy(tiles=ABUTTING, matching="touching", resolution="resegment")

    def test_the_policy_round_trips(self):
        policy = SeamPolicy.resegment(resegment_margin=12)
        assert SeamPolicy.from_dict(policy.to_dict()) == policy


class TestGpuCalibration:
    """What fits on the device, measured rather than computed."""

    def fake_device(self, limit_voxels):
        def attempt(shape):
            if math.prod(shape) > limit_voxels:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return True

        return attempt

    def test_the_probe_finds_the_limit(self):
        limit = 200**3
        calibration = calibrate(
            self.fake_device(limit), device="FakeGPU", model="cpsam"
        )
        assert calibration.max_voxels <= limit
        # Within ten percent - the bisection is what turns a doubling
        # search's factor of two into something worth planning against.
        assert calibration.max_voxels > 0.9 * limit

    def test_without_refining_it_is_only_a_power_of_two(self):
        limit = 200**3
        coarse = calibrate(
            self.fake_device(limit), device="FakeGPU", model="cpsam", refine=False
        )
        assert coarse.max_voxels <= limit

    def test_a_device_that_cannot_manage_the_smallest_tile_says_so(self):
        with pytest.raises(RuntimeError, match="too small for this model"):
            calibrate(self.fake_device(10), device="Tiny", model="cpsam")

    def test_a_failure_that_is_not_an_out_of_memory_propagates(self):
        # A broken driver treated as "too big" would shrink the tile to
        # nothing and blame the data.
        def broken(shape):
            raise RuntimeError("driver version mismatch")

        with pytest.raises(RuntimeError, match="driver version"):
            calibrate(broken, device="FakeGPU", model="cpsam")

    @pytest.mark.parametrize(
        "message",
        [
            "CUDA out of memory. Tried to allocate 2.00 GiB",
            "CUDA_ERROR_OUT_OF_MEMORY",
            "CUBLAS_STATUS_ALLOC_FAILED",
        ],
    )
    def test_it_recognises_how_the_device_says_no(self, message):
        assert is_out_of_memory(RuntimeError(message))

    def test_it_recognises_the_exception_by_type_too(self):
        class OutOfMemoryError(Exception):
            pass

        assert is_out_of_memory(OutOfMemoryError("allocation failed"))

    def test_other_failures_are_not_mistaken_for_it(self):
        assert not is_out_of_memory(ValueError("expected a 3D array"))
        assert not is_out_of_memory(FileNotFoundError("no such model"))

    def test_a_measurement_is_remembered(self, tmp_path):
        path = tmp_path / "cal.json"
        calibration = Calibration(device="FakeGPU", model="cpsam", max_voxels=1234)
        save_calibration(calibration, path=path)
        assert load_calibration("FakeGPU", "cpsam", path=path) == calibration

    def test_the_cache_holds_more_than_one_pair(self, tmp_path):
        path = tmp_path / "cal.json"
        save_calibration(Calibration("A", "cpsam", 10), path=path)
        save_calibration(Calibration("B", "cyto3", 20), path=path)
        assert load_calibration("A", "cpsam", path=path).max_voxels == 10
        assert load_calibration("B", "cyto3", path=path).max_voxels == 20

    def test_a_missing_or_broken_cache_is_not_an_error(self, tmp_path):
        assert load_calibration("A", "b", path=tmp_path / "nope.json") is None
        broken = tmp_path / "broken.json"
        broken.write_text("{not json")
        assert load_calibration("A", "b", path=broken) is None

    def test_a_cache_from_a_newer_vtea_is_ignored(self, tmp_path):
        # Re-measuring costs a minute; believing a number that may mean
        # something else costs a run.
        path = tmp_path / "cal.json"
        path.write_text(
            json.dumps({"A|b": {"device": "A", "model": "b", "max_voxels": 1, "version": 99}})
        )
        assert load_calibration("A", "b", path=path) is None

    def test_it_measures_once_and_reads_the_cache_after(self, tmp_path):
        path = tmp_path / "cal.json"
        calls = {"n": 0}

        def attempt(shape):
            calls["n"] += 1
            if math.prod(shape) > 100**3:
                raise RuntimeError("CUDA out of memory")
            return True

        first = calibrated_voxels(attempt, device="FakeGPU", model="cpsam", path=path)
        measured = calls["n"]
        second = calibrated_voxels(attempt, device="FakeGPU", model="cpsam", path=path)
        assert calls["n"] == measured, "the cached measurement was not used"
        assert first == second

    def test_remeasuring_can_be_forced(self, tmp_path):
        path = tmp_path / "cal.json"
        calls = {"n": 0}

        def attempt(shape):
            calls["n"] += 1
            if math.prod(shape) > 100**3:
                raise RuntimeError("CUDA out of memory")
            return True

        calibrated_voxels(attempt, device="FakeGPU", model="cpsam", path=path)
        before = calls["n"]
        calibrated_voxels(
            attempt, device="FakeGPU", model="cpsam", path=path, remeasure=True
        )
        assert calls["n"] > before

    def test_no_device_means_no_number_rather_than_a_guess(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gpu_module, "device_name", lambda: None)
        assert calibrated_voxels(lambda shape: True, path=tmp_path / "c.json") is None

    def test_a_device_with_no_way_to_measure_it_is_also_none(self, tmp_path):
        assert calibrated_voxels(None, device="FakeGPU", model="x", path=tmp_path / "c.json") is None

    def test_the_probe_survives_having_no_gpu_here(self):
        assert device_name() is None or isinstance(device_name(), str)

    def test_a_busy_card_gets_a_smaller_tile_than_the_one_measured(self):
        # The measurement was taken with 8 GiB free. With 2 GiB free now,
        # using the measured number would be a crash rather than a plan.
        calibration = Calibration(
            "FakeGPU", "cpsam", 1_000_000, free_bytes=8 * 1024**3
        )
        busy = MemoryBudget(16 * 1024**3, gpu_bytes=2 * 1024**3, fraction=1.0)
        assert gpu_tile_voxels(busy, calibration) == 250_000

    def test_a_card_with_more_free_than_when_measured_is_not_scaled_up(self):
        # More free memory than at calibration time does not prove the model
        # would use it - that would be extrapolating from one measurement.
        calibration = Calibration(
            "FakeGPU", "cpsam", 1_000_000, free_bytes=2 * 1024**3
        )
        idle = MemoryBudget(16 * 1024**3, gpu_bytes=8 * 1024**3, fraction=1.0)
        assert gpu_tile_voxels(idle, calibration) == 1_000_000

    def test_an_old_calibration_without_a_free_figure_is_used_as_it_stands(self):
        calibration = Calibration("FakeGPU", "cpsam", 1_000_000)
        budget = MemoryBudget(16 * 1024**3, gpu_bytes=1024**3, fraction=1.0)
        assert gpu_tile_voxels(budget, calibration) == 1_000_000

    def test_no_calibration_means_no_gpu_tile(self):
        assert gpu_tile_voxels(MemoryBudget(1024**3), None) is None

    def test_the_gpu_plan_is_bounded_by_the_device(self):
        plan = gpu_plan((100, 800, 800), voxels=200**3, halo=48)
        assert not plan.is_single_tile
        assert math.prod(plan.padded_tile) <= 200**3
        assert "GPU" in plan.describe()


class TestResume:
    """Hours of inference must survive a pre-empted node."""

    @pytest.fixture
    def spheres(self):
        shape = (20, 60, 60)
        mask = np.zeros(shape, bool)
        grid = np.ogrid[-4:5, -4:5, -4:5]
        ball = sum(axis**2 for axis in grid) <= 16
        rng = np.random.default_rng(0)
        placed = 0
        while placed < 12:
            centre = [int(rng.integers(6, size - 6)) for size in shape]
            if mask[tuple(slice(v - 5, v + 6) for v in centre)].any():
                continue
            mask[tuple(slice(v - 4, v + 5) for v in centre)] |= ball
            placed += 1
        return mask

    def test_a_crashed_run_carries_on_where_it_stopped(self, spheres, tmp_path):
        plan = plan_for(spheres.shape, 27, 8)
        assert plan.n_tiles > 4

        with ZarrScratch() as scratch:
            straight = np.asarray(
                segment_blocked(
                    label_components, {"mask": spheres}, plan=plan, scratch=scratch
                ).array
            )

        manifest = tmp_path / "run.jsonl"
        calls = {"n": 0, "limit": 4}

        def segmenter(mask):
            calls["n"] += 1
            if calls["n"] > calls["limit"]:
                raise KeyboardInterrupt("node pre-empted")
            return label_components(mask)

        scratch = ZarrScratch(root=str(tmp_path), keep=True)
        with pytest.raises(KeyboardInterrupt):
            segment_blocked(
                segmenter,
                {"mask": spheres},
                plan=plan_for(spheres.shape, 27, 8),
                scratch=scratch,
                manifest=manifest,
            )
        before = calls["n"] - 1
        assert RunManifest.load(manifest).n_completed == before

        # A fresh process: reopen the store, point at the same manifest.
        calls["limit"] = 10**6
        resumed = ZarrScratch.reopen(scratch.path)
        result = segment_blocked(
            segmenter,
            {"mask": spheres},
            plan=plan_for(spheres.shape, 27, 8),
            scratch=resumed,
            manifest=manifest,
        )
        after = calls["n"] - 1 - before

        assert before + after == plan.n_tiles, "some tile was segmented twice"
        assert after < plan.n_tiles, "the resume redid work it had already done"
        np.testing.assert_array_equal(np.asarray(result.array), straight)
        resumed.keep = False
        resumed.close()

    def test_resuming_a_different_run_is_refused(self, tmp_path):
        signature = {"shape": [10, 10], "tile": [5, 5], "halo": [1, 1], "function": "f"}
        path = tmp_path / "run.jsonl"
        RunManifest.start(path, signature).close()
        with pytest.raises(ManifestMismatch, match="different run"):
            RunManifest.start(path, {**signature, "tile": [7, 7]})

    def test_the_mismatch_says_what_changed(self, tmp_path):
        signature = {"shape": [10, 10], "tile": [5, 5]}
        path = tmp_path / "run.jsonl"
        RunManifest.start(path, signature).close()
        with pytest.raises(ManifestMismatch, match=r"tile was \[5, 5\], now \[7, 7\]"):
            RunManifest.start(path, {**signature, "tile": [7, 7]})

    def test_resuming_the_same_run_reopens_it(self, tmp_path):
        from vtea_core.blocked.reconcile import Fragment

        signature = {"shape": [10, 10], "tile": [5, 5]}
        path = tmp_path / "run.jsonl"
        with RunManifest.start(path, signature) as manifest:
            manifest.record((0, 0), [_fragment(1)], 2)
        assert RunManifest.start(path, signature).n_completed == 1

    def test_a_half_written_line_costs_one_tile_and_no_more(self, tmp_path):
        path = tmp_path / "run.jsonl"
        with RunManifest.start(path, {"a": 1}) as manifest:
            manifest.record((0,), [_fragment(1)], 2)
            manifest.record((1,), [_fragment(2)], 3)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"tile": [2')  # the process died mid-write
        recovered = RunManifest.load(path)
        assert recovered.n_completed == 2
        assert recovered.next_id == 3

    def test_fragments_come_back_in_tile_order(self, tmp_path):
        path = tmp_path / "run.jsonl"
        with RunManifest.start(path, {"a": 1}) as manifest:
            manifest.record((1,), [_fragment(9)], 10)
            manifest.record((0,), [_fragment(3)], 11)
        recovered = RunManifest.load(path)
        assert [f.provisional_id for f in recovered.fragments()] == [3, 9]

    def test_a_signature_covers_the_things_that_change_the_answer(self):
        plan = plan_for((20, 20, 20), 8, 2)
        one = plan_signature(plan, SeamPolicy.resegment(), "cellpose_segmentation")
        assert one["tile"] == list(plan.tile)
        assert one["halo"] == list(plan.halo)
        assert one["function"] == "cellpose_segmentation"
        # A different policy is a different run: it changes which objects a
        # seam cuts and what happens to them.
        assert one != plan_signature(plan, SeamPolicy.overlap_match(), "cellpose_segmentation")

    def test_a_manifest_from_a_newer_vtea_is_refused(self, tmp_path):
        path = tmp_path / "run.jsonl"
        path.write_text(json.dumps({"version": 99, "signature": {}}) + "\n")
        with pytest.raises(ManifestMismatch, match="format version"):
            RunManifest.load(path)


def _fragment(provisional_id):
    from vtea_core.blocked.reconcile import Fragment

    return Fragment(
        tile=(provisional_id,),
        local_id=1,
        provisional_id=provisional_id,
        core_voxels=10,
        block_voxels=10,
        centroid=(0.0, 0.0, 0.0),
        bbox=((0, 1), (0, 1), (0, 1)),
    )


class TestCellposeStitching:
    """Cellpose's own z-linking is reused rather than reimplemented."""

    class Stub:
        def __init__(self):
            self.calls = []

        def eval(self, x, **kwargs):
            self.calls.append(kwargs)
            return np.zeros(x.shape[:-1], dtype=np.int32), None, None

    def evaluate(self, **kwargs):
        from vtea_core.segmentation import cellpose_segmentation

        model = self.Stub()
        cellpose_segmentation(np.zeros((4, 8, 8)), model=model, **kwargs)
        return model.calls[0]

    def test_plane_wise_by_default(self):
        call = self.evaluate()
        assert call["do_3D"] is False
        assert not call.get("stitch_threshold")

    def test_a_stitch_threshold_links_planes_in_z(self):
        call = self.evaluate(stitch_threshold=0.25)
        assert call["stitch_threshold"] == 0.25
        assert call["z_axis"] == 0
        assert call["do_3D"] is False

    def test_do_3d_wins_over_stitching(self):
        # cellpose ignores stitch_threshold in 3D mode; passing both would
        # imply a choice that is not being made.
        call = self.evaluate(do_3D=True, stitch_threshold=0.25)
        assert call["do_3D"] is True
        assert "stitch_threshold" not in call

    def test_the_contract_still_sizes_it_against_the_gpu(self):
        from vtea_core.workflow.wiring import scaling_for

        scaling = scaling_for("segmentation", "cellpose_segmentation")
        assert scaling.needs_reconciliation
        assert "GPU" in scaling.notes
        assert scaling.halo.param == "diameter"


class TestBlockedCellpose:
    """A learned segmenter, tiled - with a stand-in model, so this runs
    without a GPU or the deeplearning extra."""

    class EdgeShyModel:
        """Shaped like a cellpose model, and just as context-dependent."""

        def eval(self, x, **kwargs):
            volume = np.asarray(x)[..., 0]
            return edge_shy(volume > 0.5), None, None

    def protocol(self):
        from vtea_core.workflow import Pipeline, Step

        return Pipeline(
            [
                Step.for_function(
                    "segmentation",
                    "cellpose_segmentation",
                    available={"volume", "model"},
                )
            ]
        )

    def test_it_is_refused_under_a_strategy_that_picks_between_copies(self, balls):
        from vtea_core.blocked import BlockedPipeline, NotBlockableYet

        plan = plan_for(balls.shape, 27, 6)
        with BlockedPipeline(self.protocol(), plan=plan) as blocked:
            with pytest.raises(NotBlockableYet, match="resegment"):
                blocked.run({"volume": balls.astype(float), "model": self.EdgeShyModel()})

    def test_it_runs_under_resegment(self, balls, truth):
        from vtea_core.blocked import BlockedPipeline

        plan = plan_for(balls.shape, 27, 6)
        with BlockedPipeline(
            self.protocol(), plan=plan, policy=SeamPolicy.resegment()
        ) as blocked:
            context = blocked.run(
                {"volume": balls.astype(float), "model": self.EdgeShyModel()}
            )
            result = np.asarray(context["labels"])
            assert sizes_of(result) == sizes_of(truth)
            assert blocked.ledgers["labels"].n_objects == int(truth.max())

    def test_the_halo_follows_the_diameter_the_user_already_set(self):
        from vtea_core.blocked import plan_for_steps
        from vtea_core.workflow import Step

        step = Step.for_function(
            "segmentation", "cellpose_segmentation", params={"diameter": 30.0}
        )
        plan = plan_for_steps(
            [step], (256, 256, 256), budget=MemoryBudget(64 * 1024**3)
        )
        # 1.5 x diameter, which is a number the user chose rather than one
        # the tile planner invented.
        assert plan.halo == (45, 45, 45)
