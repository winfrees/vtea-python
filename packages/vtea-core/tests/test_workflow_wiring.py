import numpy as np
import pytest

from vtea_core.workflow import STEP_IO, STEP_REGISTRY, Pipeline, Step, default_wiring, step_io


class TestStepIOTable:
    def test_covers_every_registered_step(self):
        """Every function reachable from the GUI's Add Step menu needs I/O
        metadata, or adding it produces a step that cannot run."""
        registered = {
            (category, name) for category, funcs in STEP_REGISTRY.items() for name in funcs
        }
        assert registered - set(STEP_IO) == set()

    def test_unknown_step_raises(self):
        with pytest.raises(KeyError, match="no I/O description"):
            step_io("segmentation", "not_a_real_function")


class TestDefaultWiring:
    def test_required_data_input_is_wired(self):
        input_keys, output_key = default_wiring("segmentation", "threshold_mask")
        assert input_keys == {"volume": "volume"}
        assert output_key == "mask"

    def test_optional_data_input_is_skipped_when_unavailable(self):
        # cellpose's `model` defaults to None; nothing upstream produces it,
        # so it must not be wired or the step fails with a missing key.
        input_keys, _ = default_wiring("segmentation", "cellpose_segmentation")
        assert input_keys == {"volume": "volume"}

    def test_optional_data_input_is_wired_when_available(self):
        input_keys, _ = default_wiring(
            "segmentation", "cellpose_segmentation", available={"volume", "model"}
        )
        assert input_keys == {"volume": "volume", "model": "model"}

    def test_required_input_wired_even_if_nothing_produces_it(self):
        # Better a clear "needs context key(s) ['mask']" at run time than a
        # bare TypeError from inside the function.
        input_keys, _ = default_wiring("segmentation", "label_components")
        assert input_keys == {"mask": "mask"}

    def test_distinct_steps_get_distinct_output_keys(self):
        _, threshold_out = default_wiring("segmentation", "threshold_mask")
        _, label_out = default_wiring("segmentation", "label_components")
        assert threshold_out != label_out


class TestStepForFunction:
    def test_builds_a_runnable_step_without_manual_wiring(self):
        step = Step.for_function("segmentation", "threshold_mask", params={"method": "otsu"})
        volume = np.zeros((4, 4))
        volume[1:3, 1:3] = 100.0
        assert step.run({"volume": volume}).sum() == 4

    def test_chains_by_name(self):
        pipeline = Pipeline()
        pipeline.add_step(
            Step.for_function(
                "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
            )
        )
        pipeline.add_step(Step.for_function("segmentation", "label_components"))

        volume = np.zeros((10, 10))
        volume[1:3, 1:3] = 100.0
        volume[6:9, 6:9] = 100.0
        result = pipeline.run({"volume": volume})

        assert result["mask"].sum() == 13
        assert result["labels"].max() == 2


class TestChannelSelection:
    def make_multichannel(self):
        # (Z, C, Y, X) - the shape an ImageJ hyperstack loads as, and what
        # prompted per-step channel selection.
        volume = np.zeros((3, 4, 8, 8))
        volume[:, 2, 1:4, 1:4] = 100.0  # signal only in channel 2
        return volume

    def test_selects_the_requested_channel(self):
        step = Step.for_function(
            "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
        )
        step.channel = 2
        pipeline = Pipeline([step], channel_axis=1)
        result = pipeline.run({"volume": self.make_multichannel()})
        assert result["mask"].shape == (3, 8, 8)
        assert result["mask"].sum() == 3 * 9

    def test_other_channels_are_empty(self):
        step = Step.for_function(
            "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
        )
        step.channel = 0
        pipeline = Pipeline([step], channel_axis=1)
        result = pipeline.run({"volume": self.make_multichannel()})
        assert result["mask"].sum() == 0

    def test_channel_none_passes_the_whole_array_through(self):
        step = Step.for_function(
            "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
        )
        pipeline = Pipeline([step], channel_axis=1)
        result = pipeline.run({"volume": self.make_multichannel()})
        assert result["mask"].shape == (3, 4, 8, 8)

    def test_derived_arrays_are_not_sliced_again(self):
        """A second channel-selecting step must not slice a spatial axis of
        an array a previous step already reduced to one channel."""
        threshold = Step.for_function(
            "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
        )
        threshold.channel = 2
        label = Step.for_function("segmentation", "label_components")
        label.channel = 2  # same choice, but "mask" no longer has a channel axis

        pipeline = Pipeline([threshold, label], channel_axis=1)
        result = pipeline.run({"volume": self.make_multichannel()})

        assert result["mask"].shape == (3, 8, 8)
        assert result["labels"].shape == (3, 8, 8)
        assert result["labels"].max() == 1

    def test_out_of_range_channel_reports_what_is_available(self):
        step = Step.for_function(
            "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
        )
        step.channel = 9
        pipeline = Pipeline([step], channel_axis=1)
        with pytest.raises(IndexError, match="has 4 channel"):
            pipeline.run({"volume": self.make_multichannel()})

    def test_channel_ignored_without_a_channel_axis(self):
        step = Step.for_function(
            "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
        )
        step.channel = 2
        pipeline = Pipeline([step], channel_axis=None)
        result = pipeline.run({"volume": self.make_multichannel()})
        assert result["mask"].shape == (3, 4, 8, 8)

    def test_run_argument_overrides_the_pipeline_setting(self):
        step = Step.for_function(
            "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
        )
        step.channel = 2
        pipeline = Pipeline([step], channel_axis=None)
        result = pipeline.run({"volume": self.make_multichannel()}, channel_axis=1)
        assert result["mask"].shape == (3, 8, 8)


class TestAvailableKeys:
    def test_includes_seeds_and_step_outputs(self):
        pipeline = Pipeline()
        pipeline.add_step(Step.for_function("segmentation", "threshold_mask"))
        assert pipeline.available_keys({"volume", "intensity"}) == {
            "volume",
            "intensity",
            "mask",
            # A step's result is reachable by its own name too, so a later
            # step can name the one it wants.
            "threshold_mask_1",
        }
