"""Each step's result is published under a unique name as well as its
shared output key.

Without this a protocol with two segmentations has no way to say which one
a measurement step should measure: every segmentation writes "labels", so
the measurement step silently takes whichever ran last.
"""

import numpy as np
import pytest

from vtea_core.workflow import Pipeline, Step, unique_step_name


class TestUniqueStepName:
    def test_first_use_is_numbered_one(self):
        assert unique_step_name("watershed_split", []) == "watershed_split_1"

    def test_skips_names_already_taken(self):
        taken = ["watershed_split_1", "watershed_split_2"]
        assert unique_step_name("watershed_split", taken) == "watershed_split_3"

    def test_ignores_unrelated_names(self):
        assert unique_step_name("kmeans", ["pca_1", "tsne_1"]) == "kmeans_1"

    def test_fills_a_gap_left_by_a_deleted_step(self):
        assert unique_step_name("pca", ["pca_2"]) == "pca_1"


class TestStepNaming:
    def test_for_function_names_the_step_after_its_function(self):
        step = Step.for_function("segmentation", "watershed_split")
        assert step.name == "watershed_split_1"

    def test_two_of_the_same_function_get_distinct_names(self):
        first = Step.for_function("segmentation", "watershed_split")
        second = Step.for_function("segmentation", "watershed_split", taken_names=[first.name])
        assert first.name != second.name
        assert second.name == "watershed_split_2"

    def test_an_explicit_name_is_kept(self):
        step = Step.for_function("segmentation", "watershed_split", name="nuclei")
        assert step.name == "nuclei"

    def test_result_key_falls_back_to_the_output_key_when_unnamed(self):
        step = Step("segmentation", "label_components", output_key="labels")
        assert step.name == ""
        assert step.result_key == "labels"

    def test_result_key_is_the_name_when_named(self):
        step = Step.for_function("segmentation", "label_components")
        assert step.result_key == "label_components_1"


class TestNamedResultsInTheContext:
    @staticmethod
    def make_volume():
        volume = np.zeros((4, 8, 8), dtype=np.uint8)
        volume[:, 1:4, 1:4] = 200
        volume[:, 5:7, 5:7] = 200
        return volume

    def test_result_is_published_under_both_keys(self):
        pipeline = Pipeline()
        step = Step.for_function("segmentation", "threshold_mask", params={"method": "fixed", "value": 100})
        pipeline.add_step(step)
        context = pipeline.run({"volume": self.make_volume()})
        assert "mask" in context
        assert step.name in context
        assert context[step.name] is context["mask"]

    def test_two_segmentations_stay_separately_addressable(self):
        """The shared "labels" key holds the last one; the names hold both."""
        pipeline = Pipeline()
        pipeline.add_step(Step.for_function("segmentation", "threshold_mask", params={"method": "fixed", "value": 100}))
        first = Step.for_function(
            "segmentation", "label_components", available={"mask"}, taken_names=pipeline.step_names()
        )
        pipeline.add_step(first)
        second = Step.for_function(
            "segmentation",
            "filter_by_size",
            available={"labels"},
            params={"min_size": 100000},
            taken_names=pipeline.step_names(),
        )
        pipeline.add_step(second)

        context = pipeline.run({"volume": self.make_volume()})
        assert context[first.name].max() > 0
        # filter_by_size with an impossible minimum removes everything.
        assert context[second.name].max() == 0
        assert context["labels"] is context[second.name]

    def test_a_step_can_be_wired_to_a_named_result(self):
        pipeline = Pipeline()
        pipeline.add_step(Step.for_function("segmentation", "threshold_mask", params={"method": "fixed", "value": 100}))
        first = Step.for_function(
            "segmentation", "label_components", available={"mask"}, taken_names=pipeline.step_names()
        )
        pipeline.add_step(first)
        cleared = Step.for_function(
            "segmentation",
            "filter_by_size",
            available={"labels"},
            params={"min_size": 100000},
            taken_names=pipeline.step_names(),
        )
        pipeline.add_step(cleared)

        # Measure the *first* segmentation, not the emptied one that ran last.
        measure = Step.for_function(
            "measurements",
            "extract_measurements",
            available=pipeline.available_keys({"volume", "intensity"}),
            taken_names=pipeline.step_names(),
        )
        measure.input_keys["labels"] = first.name
        pipeline.add_step(measure)

        volume = self.make_volume()
        context = pipeline.run({"volume": volume, "intensity": volume})
        assert len(context["measurements"]) == 2

    def test_names_are_reachable_as_wiring_targets(self):
        pipeline = Pipeline()
        step = Step.for_function("segmentation", "threshold_mask", params={"method": "fixed", "value": 100})
        pipeline.add_step(step)
        assert step.name in pipeline.available_keys({"volume"})

    def test_names_producing_lists_the_matching_steps(self):
        pipeline = Pipeline()
        threshold = Step.for_function("segmentation", "threshold_mask")
        labeler = Step.for_function("segmentation", "label_components")
        pipeline.add_step(threshold)
        pipeline.add_step(labeler)
        assert pipeline.names_producing("labels") == [labeler.name]
        assert pipeline.names_producing("mask") == [threshold.name]


class TestChannelAwareSteps:
    """A channel-aware function sees every channel at once, so it can tag
    each feature with the channel it came from; Step.run must not slice the
    array down to one channel before calling it."""

    @staticmethod
    def make_multichannel():
        labels = np.zeros((2, 4, 4), dtype=np.int32)
        labels[:, 0:2, 0:2] = 1
        intensity = np.zeros((2, 3, 4, 4), dtype=float)
        for channel in range(3):
            intensity[:, channel] = 10.0 * (channel + 1)
        return labels, intensity

    def test_by_channel_measurement_is_channel_aware(self):
        step = Step.for_function("measurements", "extract_measurements_by_channel")
        assert step.channel_aware is True

    def test_plain_measurement_is_not(self):
        step = Step.for_function("measurements", "extract_measurements")
        assert step.channel_aware is False

    def test_gets_the_whole_multichannel_array(self):
        labels, intensity = self.make_multichannel()
        step = Step.for_function(
            "measurements",
            "extract_measurements_by_channel",
            available={"labels", "intensity", "channel_axis"},
        )
        table = step.run(
            {"labels": labels, "intensity": intensity, "channel_axis": 1},
            channel_axis=1,
            full_ndim=4,
        )
        assert {"mean_ch0", "mean_ch1", "mean_ch2"} <= set(table.columns)

    def test_a_channel_choice_is_passed_through_rather_than_pre_sliced(self):
        labels, intensity = self.make_multichannel()
        step = Step.for_function(
            "measurements",
            "extract_measurements_by_channel",
            available={"labels", "intensity", "channel_axis"},
            channel=1,
        )
        table = step.run(
            {"labels": labels, "intensity": intensity, "channel_axis": 1},
            channel_axis=1,
            full_ndim=4,
        )
        # Still tagged - a pre-sliced array would have produced a bare "mean".
        assert "mean_ch1" in table.columns
        assert table.loc[0, "mean_ch1"] == pytest.approx(20.0)
