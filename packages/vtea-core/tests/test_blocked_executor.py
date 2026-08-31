"""The two properties that make a blocked run trustworthy:

1. **One tile equals the whole image.** With a budget large enough to hold
   everything, the blocked path must give bit-identical results to calling
   the function directly. This catches padding, trimming and off-by-one
   errors for nothing.
2. **Tiling is invariant.** The same data at several tile sizes - including
   sizes chosen so seams fall in awkward places - gives the same answer.

Both are run against the real step functions rather than against stand-ins,
because the thing being tested is whether *those* survive being tiled.
"""

import numpy as np
import pytest

from vtea_core.blocked import (
    BlockedPipeline,
    MemoryBudget,
    NotBlockableYet,
    ZarrScratch,
    apply_blocked,
    numpy_pad_mode,
    plan_for_steps,
    plan_tiles,
    run_step_blocked,
)
from vtea_core.imageprocessing import enhance_contrast, gaussian_blur, median_filter
from vtea_core.segmentation import threshold_mask
from vtea_core.workflow import Pipeline, Step

GB = 1024**3


@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    data = rng.normal(1000, 300, (24, 96, 96)).clip(0, 4000).astype(np.uint16)
    data[4:12, 20:50, 20:50] += 2000
    return data


def tiled_plan(shape, tiles_wanted, *, halo=0, bytes_per_voxel=8):
    """A plan with roughly `tiles_wanted` tiles, by squeezing the budget.

    The halo has to be paid for in the budget, or the planner (rightly)
    refuses: a core smaller than its own halo is not a plan.
    """
    core = int(np.prod(shape)) / max(tiles_wanted, 1)
    edge = core ** (1 / len(shape))
    padding_factor = ((edge + 2 * halo) / edge) ** len(shape)
    total = core * padding_factor * bytes_per_voxel / 0.6
    return plan_tiles(
        shape,
        budget=MemoryBudget(int(total) + 4096),
        bytes_per_voxel=bytes_per_voxel,
        halo=halo,
    )


class TestPadModes:
    def test_scipys_names_are_translated_to_numpys(self):
        # The trap: both libraries use "reflect", for different things.
        assert numpy_pad_mode("reflect") == "symmetric"
        assert numpy_pad_mode("mirror") == "reflect"
        assert numpy_pad_mode("nearest") == "edge"

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="unknown boundary mode"):
            numpy_pad_mode("bounce")


class TestOneTileEqualsWholeImage:
    def test_for_a_gaussian_blur(self, volume):
        plan = plan_tiles(volume.shape, budget=MemoryBudget(GB), bytes_per_voxel=8, halo=8)
        assert plan.is_single_tile
        blocked = apply_blocked(
            gaussian_blur, {"volume": volume}, plan=plan, params={"sigma": 2.0}
        )
        np.testing.assert_array_equal(blocked, gaussian_blur(volume, sigma=2.0))

    def test_for_a_threshold(self, volume):
        plan = plan_tiles(volume.shape, budget=MemoryBudget(GB), bytes_per_voxel=8)
        blocked = apply_blocked(
            threshold_mask, {"volume": volume}, plan=plan, params={"method": "fixed", "value": 1500}
        )
        np.testing.assert_array_equal(
            blocked, threshold_mask(volume, method="fixed", value=1500)
        )


class TestTilingIsInvariant:
    @pytest.mark.parametrize("tiles_wanted", [1, 4, 12, 27])
    def test_a_gaussian_blur_is_exact_at_any_tile_size(self, volume, tiles_wanted):
        # sigma=2 with scipy's truncate=4 reaches 8 voxels, which is exactly
        # the halo the scaling contract asks for. If the halo arithmetic is
        # off by one, this is where it shows.
        expected = gaussian_blur(volume, sigma=2.0)
        plan = tiled_plan(volume.shape, tiles_wanted, halo=8)
        result = apply_blocked(
            gaussian_blur, {"volume": volume}, plan=plan, params={"sigma": 2.0}
        )
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("tiles_wanted", [4, 27])
    def test_a_median_filter_is_exact_at_any_tile_size(self, volume, tiles_wanted):
        expected = median_filter(volume, radius=2)
        plan = tiled_plan(volume.shape, tiles_wanted, halo=2)
        result = apply_blocked(median_filter, {"volume": volume}, plan=plan, params={"radius": 2})
        np.testing.assert_array_equal(result, expected)

    def test_too_small_a_halo_is_wrong_which_is_why_the_contract_declares_one(self, volume):
        # The negative control. Without it, the tests above could be passing
        # because the tiling never mattered.
        expected = gaussian_blur(volume, sigma=2.0)
        starved = apply_blocked(
            gaussian_blur,
            {"volume": volume},
            plan=tiled_plan(volume.shape, 27, halo=1),
            params={"sigma": 2.0},
        )
        assert not np.array_equal(starved, expected)

    def test_an_elementwise_step_needs_no_halo_at_all(self, volume):
        expected = threshold_mask(volume, method="fixed", value=1500)
        for tiles_wanted in (1, 8, 60):
            result = apply_blocked(
                threshold_mask,
                {"volume": volume},
                plan=tiled_plan(volume.shape, tiles_wanted),
                params={"method": "fixed", "value": 1500},
            )
            np.testing.assert_array_equal(result, expected)

    def test_a_seam_falling_inside_the_bright_region(self, volume):
        # Seams in the easy places prove less than seams in the hard ones.
        expected = gaussian_blur(volume, sigma=1.5)
        plan = plan_tiles(volume.shape, budget=MemoryBudget(GB), bytes_per_voxel=8, halo=6)
        awkward = plan_tiles(
            volume.shape,
            budget=MemoryBudget(int(np.prod(volume.shape) * 8 / 0.6 / 9)),
            bytes_per_voxel=8,
            halo=6,
        )
        assert not awkward.is_single_tile
        assert awkward.tile[1] < 50  # a seam runs through the bright block
        for candidate in (plan, awkward):
            result = apply_blocked(
                gaussian_blur, {"volume": volume}, plan=candidate, params={"sigma": 1.5}
            )
            np.testing.assert_array_equal(result, expected)


class TestGlobalStatistics:
    def test_a_tiled_otsu_is_the_same_threshold_as_a_whole_image_one(self, volume):
        expected = threshold_mask(volume, method="otsu")
        step = Step.for_function("segmentation", "threshold_mask", params={"method": "otsu"})
        for tiles_wanted in (1, 6, 40):
            plan = tiled_plan(volume.shape, tiles_wanted)
            result = run_step_blocked(step, {"volume": volume}, plan=plan)
            np.testing.assert_array_equal(result.array, expected)

    def test_the_threshold_it_used_is_reported_rather_than_left_implicit(self, volume):
        step = Step.for_function("segmentation", "threshold_mask", params={"method": "otsu"})
        result = run_step_blocked(step, {"volume": volume}, plan=tiled_plan(volume.shape, 8))
        assert result.resolved_params["method"] == "fixed"
        assert result.resolved_params["value"] > 0
        assert result.stats is not None
        assert result.stats.exact
        assert "otsu" not in result.describe()  # it describes what it resolved to

    def test_a_tiled_percentile_matches_too(self, volume):
        expected = threshold_mask(volume, method="percentile", percentile=99.0)
        step = Step.for_function(
            "segmentation", "threshold_mask", params={"method": "percentile", "percentile": 99.0}
        )
        result = run_step_blocked(step, {"volume": volume}, plan=tiled_plan(volume.shape, 12))
        np.testing.assert_array_equal(result.array, expected)

    def test_rescaling_uses_the_volumes_range_not_each_tiles(self, volume):
        # Per-tile rescaling would stretch every tile to its own extremes,
        # which is visible as a checkerboard and is the reason this step is
        # marked GLOBAL_STAT.
        expected = enhance_contrast(volume, method="normalize")
        step = Step.for_function(
            "imageprocessing", "enhance_contrast", params={"method": "normalize"}
        )
        result = run_step_blocked(step, {"volume": volume}, plan=tiled_plan(volume.shape, 20))
        np.testing.assert_array_equal(result.array, expected)

    def test_without_the_global_pass_it_would_be_wrong(self, volume):
        # The negative control for the machinery above.
        naive = apply_blocked(
            enhance_contrast,
            {"volume": volume},
            plan=tiled_plan(volume.shape, 20),
            params={"method": "normalize"},
        )
        assert not np.array_equal(naive, enhance_contrast(volume, method="normalize"))


class TestRefusals:
    @pytest.mark.parametrize(
        "category,function_name",
        [
            ("segmentation", "label_components"),
            ("segmentation", "watershed_split"),
            ("segmentation", "filter_by_size"),
        ],
    )
    def test_a_step_that_assigns_object_ids_is_refused(self, volume, category, function_name):
        # The dangerous case: these are shape-preserving neighbourhood or
        # elementwise steps that the executor could otherwise schedule
        # happily, and every tile would number its objects from 1.
        labels = (volume > 2500).astype(np.int32)
        sources = {
            "mask": labels > 0,
            "labels": labels,
            "intensity": volume,
        }
        step = Step.for_function(category, function_name)
        with pytest.raises(NotBlockableYet, match="object identities"):
            run_step_blocked(
                step,
                {name: sources[name] for name in step.input_keys},
                plan=tiled_plan(volume.shape, 4),
            )

    def test_a_measurement_step_says_which_phase_it_waits_on(self, volume):
        step = Step.for_function("measurements", "extract_measurements")
        with pytest.raises(NotBlockableYet, match="L4"):
            run_step_blocked(
                step, {"labels": volume, "intensity": volume}, plan=tiled_plan(volume.shape, 4)
            )

    def test_inputs_of_different_shapes_are_refused(self, volume):
        with pytest.raises(ValueError, match="plan's shape"):
            apply_blocked(
                gaussian_blur,
                {"volume": volume[:, :10]},
                plan=tiled_plan(volume.shape, 4),
                params={"sigma": 1.0},
            )

    def test_a_step_that_changes_shape_is_refused_clearly(self, volume):
        def halve(volume):
            return volume[::2]

        with pytest.raises(ValueError, match="shape-preserving"):
            apply_blocked(halve, {"volume": volume}, plan=tiled_plan(volume.shape, 4))

    def test_no_inputs_at_all_is_refused(self, volume):
        with pytest.raises(ValueError, match="at least one array"):
            apply_blocked(gaussian_blur, {}, plan=tiled_plan(volume.shape, 1))


class TestBlockedPipeline:
    def steps(self):
        blur = Step.for_function("imageprocessing", "gaussian_blur", params={"sigma": 1.5})
        threshold = Step.for_function(
            "segmentation", "threshold_mask", params={"method": "otsu"}, available={"volume"}
        )
        return Pipeline([blur, threshold])

    def test_it_matches_the_in_memory_pipeline(self, volume):
        pipeline = self.steps()
        expected = pipeline.run({"volume": volume})

        plan = plan_for_steps(pipeline.steps, volume.shape, budget=MemoryBudget(4 * 1024**2))
        assert not plan.is_single_tile
        with BlockedPipeline(pipeline, plan=plan) as blocked:
            result = blocked.run({"volume": volume})
            np.testing.assert_array_equal(np.asarray(result["mask"]), expected["mask"])
            np.testing.assert_array_equal(np.asarray(result["volume"]), expected["volume"])

    def test_intermediates_live_in_scratch_rather_than_in_the_context(self, volume):
        pipeline = self.steps()
        plan = plan_for_steps(pipeline.steps, volume.shape, budget=MemoryBudget(4 * 1024**2))
        with ZarrScratch() as scratch:
            with BlockedPipeline(pipeline, plan=plan, scratch=scratch) as blocked:
                result = blocked.run({"volume": volume})
            # Both steps' outputs are on disk, not NumPy arrays in a dict.
            assert len(scratch.names()) == 2
            assert not isinstance(result["mask"], np.ndarray)

    def test_each_steps_result_is_kept_under_its_own_name(self, volume):
        pipeline = self.steps()
        plan = plan_for_steps(pipeline.steps, volume.shape, budget=MemoryBudget(4 * 1024**2))
        with BlockedPipeline(pipeline, plan=plan) as blocked:
            result = blocked.run({"volume": volume})
        assert set(blocked.results) == {step.name for step in pipeline.steps}
        for step in pipeline.steps:
            assert step.name in result

    def test_a_dtype_change_between_steps_is_stored_correctly(self, volume):
        # uint16 in, bool out. Guessing the output dtype from the input
        # would store a threshold as uint16 and quietly change nothing
        # visible until something counted the ones.
        pipeline = self.steps()
        plan = plan_for_steps(pipeline.steps, volume.shape, budget=MemoryBudget(4 * 1024**2))
        with BlockedPipeline(pipeline, plan=plan) as blocked:
            result = blocked.run({"volume": volume})
            assert np.asarray(result["mask"]).dtype == np.bool_
            assert np.asarray(result["volume"]).dtype == np.uint16

    def test_progress_is_reported_per_step(self, volume):
        pipeline = self.steps()
        plan = plan_for_steps(pipeline.steps, volume.shape, budget=MemoryBudget(4 * 1024**2))
        seen = []
        with BlockedPipeline(pipeline, plan=plan) as blocked:
            blocked.run({"volume": volume}, progress=lambda name, done, total: seen.append(name))
        assert set(seen) == {step.name for step in pipeline.steps}

    def test_a_missing_input_names_what_is_available(self, volume):
        step = Step.for_function("segmentation", "threshold_mask")
        step.input_keys = {"volume": "nonexistent"}
        pipeline = Pipeline([step])
        plan = plan_tiles(volume.shape, budget=MemoryBudget(GB), bytes_per_voxel=8)
        with BlockedPipeline(pipeline, plan=plan) as blocked:
            with pytest.raises(KeyError, match="nonexistent"):
                blocked.run({"volume": volume})

    def test_non_array_context_values_are_passed_as_parameters(self, volume):
        # `expand_labels` takes a Spacing as a wired input. It travels in
        # the context beside the arrays and is emphatically not something to
        # slice into tiles.
        from vtea_core.data import Spacing
        from vtea_core.segmentation import expand_labels

        labels = (volume > 2500).astype(np.int32)
        spacing = Spacing((2.0, 0.5, 0.5))
        step = Step.for_function(
            "segmentation", "expand_labels", params={"distance": 2.0}, available={"spacing"}
        )
        plan = plan_tiles(labels.shape, budget=MemoryBudget(GB), bytes_per_voxel=40, halo=4)
        with BlockedPipeline(Pipeline([step]), plan=plan) as blocked:
            result = blocked.run({"labels": labels, "spacing": spacing})
            # Inside the context: the scratch store still exists. See
            # test_results_are_only_valid_inside_the_context for why that
            # matters more than it looks.
            np.testing.assert_array_equal(
                np.asarray(result["labels"]), expand_labels(labels, 2.0, spacing=spacing)
            )

    def test_a_spacing_actually_reaches_the_step(self, volume):
        # Not a tautology: an earlier version collected the non-array inputs
        # and then rebuilt the parameters from the step, dropping them. The
        # result was a dilation in voxels where the caller asked for one in
        # microns, which looks entirely plausible.
        from vtea_core.data import Spacing
        from vtea_core.segmentation import expand_labels

        labels = (volume > 2500).astype(np.int32)
        spacing = Spacing((2.0, 0.5, 0.5))
        assert not np.array_equal(
            expand_labels(labels, 2.0, spacing=spacing), expand_labels(labels, 2.0)
        )
        step = Step.for_function(
            "segmentation", "expand_labels", params={"distance": 2.0}, available={"spacing"}
        )
        plan = plan_tiles(labels.shape, budget=MemoryBudget(GB), bytes_per_voxel=40, halo=4)
        with BlockedPipeline(Pipeline([step]), plan=plan) as blocked:
            result = blocked.run({"labels": labels, "spacing": spacing})
            np.testing.assert_array_equal(
                np.asarray(result["labels"]), expand_labels(labels, 2.0, spacing=spacing)
            )

    def test_results_are_only_valid_inside_the_context(self, volume):
        # A trap worth pinning down rather than discovering. Scratch arrays
        # live in a directory that the context manager deletes, and a Zarr
        # array whose chunk files have gone does not raise - it returns its
        # fill value. So a result read after the context has closed is
        # silently zeros, not an error.
        pipeline = self.steps()
        plan = plan_for_steps(pipeline.steps, volume.shape, budget=MemoryBudget(4 * 1024**2))
        with BlockedPipeline(pipeline, plan=plan) as blocked:
            result = blocked.run({"volume": volume})
            assert np.asarray(result["mask"]).any()
        assert not np.asarray(result["mask"]).any()

    def test_a_step_that_only_carries_ids_forward_can_be_tiled(self, volume):
        # expand_labels assigns no new identities - every grown region keeps
        # its parent's id - so it needs no reconciliation and tiles today.
        from vtea_core.segmentation import expand_labels

        labels = (volume > 2500).astype(np.int32)
        expected = expand_labels(labels, 2.0)
        for tiles_wanted in (1, 8):
            result = apply_blocked(
                expand_labels,
                {"labels": labels},
                plan=tiled_plan(labels.shape, tiles_wanted, halo=3, bytes_per_voxel=40),
                params={"distance": 2.0},
            )
            np.testing.assert_array_equal(result, expected)

    def test_it_needs_a_scratch_store(self, volume):
        plan = plan_tiles(volume.shape, budget=MemoryBudget(GB), bytes_per_voxel=8)
        blocked = BlockedPipeline(self.steps(), plan=plan)
        with pytest.raises(RuntimeError, match="context manager"):
            blocked.run({"volume": volume})


class TestCancellation:
    """A run measured in hours has to be stoppable, and the place to stop
    is a tile boundary - the smallest unit that leaves the output
    consistent."""

    def test_it_stops_between_tiles(self, volume):
        from vtea_core.blocked import Cancelled

        plan = tiled_plan(volume.shape, 27)
        seen = {"n": 0}

        def stop_after_three():
            seen["n"] += 1
            return seen["n"] > 3

        with pytest.raises(Cancelled):
            apply_blocked(
                threshold_mask,
                {"volume": volume},
                plan=plan,
                params={"method": "fixed", "value": 1500},
                should_stop=stop_after_three,
            )
        assert seen["n"] < plan.n_tiles, "it ran every tile before noticing"

    def test_a_run_nobody_stops_is_unaffected(self, volume):
        expected = threshold_mask(volume, method="fixed", value=1500)
        result = apply_blocked(
            threshold_mask,
            {"volume": volume},
            plan=tiled_plan(volume.shape, 8),
            params={"method": "fixed", "value": 1500},
            should_stop=lambda: False,
        )
        np.testing.assert_array_equal(result, expected)

    def test_a_pipeline_stops_between_steps_too(self, volume):
        from vtea_core.blocked import Cancelled

        pipeline = Pipeline(
            [
                Step.for_function("imageprocessing", "gaussian_blur", params={"sigma": 1.0}),
                Step.for_function(
                    "segmentation",
                    "threshold_mask",
                    params={"method": "otsu"},
                    available={"volume"},
                ),
            ]
        )
        plan = plan_for_steps(pipeline.steps, volume.shape, budget=MemoryBudget(4 * 1024**2))
        with BlockedPipeline(pipeline, plan=plan) as blocked:
            with pytest.raises(Cancelled):
                blocked.run({"volume": volume}, should_stop=lambda: True)

    def test_a_segmentation_stops_between_tiles(self, volume):
        from vtea_core.blocked import Cancelled, ZarrScratch, segment_blocked
        from vtea_core.segmentation import label_components

        mask = volume > 2500
        plan = tiled_plan(volume.shape, 27, halo=4, bytes_per_voxel=10)
        seen = {"n": 0}

        def stop_after_two():
            seen["n"] += 1
            return seen["n"] > 2

        with ZarrScratch() as scratch:
            with pytest.raises(Cancelled):
                segment_blocked(
                    label_components,
                    {"mask": mask},
                    plan=plan,
                    scratch=scratch,
                    should_stop=stop_after_two,
                )

    def test_cancelling_leaves_the_finished_tiles_for_a_resume(self, volume, tmp_path):
        # The synergy with L5's manifest: a cancelled run has written every
        # tile it finished, so the same manifest resumes rather than
        # restarts. Cancel becomes "stop for now", not "throw away an hour".
        from vtea_core.blocked import Cancelled, ZarrScratch, segment_blocked
        from vtea_core.blocked.resume import RunManifest
        from vtea_core.segmentation import label_components

        mask = volume > 2500
        plan = tiled_plan(volume.shape, 27, halo=4, bytes_per_voxel=10)
        manifest = tmp_path / "run.jsonl"
        seen = {"n": 0}

        scratch = ZarrScratch(root=str(tmp_path), keep=True)
        with pytest.raises(Cancelled):
            segment_blocked(
                label_components,
                {"mask": mask},
                plan=plan,
                scratch=scratch,
                manifest=manifest,
                should_stop=lambda: seen.__setitem__("n", seen["n"] + 1) or seen["n"] > 3,
            )
        done = RunManifest.load(manifest).n_completed
        assert 0 < done < plan.n_tiles

        resumed = ZarrScratch.reopen(scratch.path)
        result = segment_blocked(
            label_components,
            {"mask": mask},
            plan=tiled_plan(volume.shape, 27, halo=4, bytes_per_voxel=10),
            scratch=resumed,
            manifest=manifest,
        )
        assert RunManifest.load(manifest).n_completed == plan.n_tiles
        assert result.n_objects > 0
        resumed.keep = False
        resumed.close()
