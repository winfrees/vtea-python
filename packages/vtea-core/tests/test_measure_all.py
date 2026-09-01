from vtea_core.workflow import (
    Pipeline,
    Step,
    measurement_name_for,
    rename_segmentation,
    segmentation_names,
    sync_measurement_steps,
)


def a_protocol():
    """A nucleus segmentation and a ring derived from it - the case the
    whole feature is for."""
    processing = Pipeline()
    processing.add_step(
        Step.for_function("segmentation", "threshold_mask", name="threshold_mask_1")
    )
    processing.add_step(
        Step.for_function("segmentation", "label_components", available={"mask"}, name="nuclei")
    )
    processing.add_step(
        Step.for_function(
            "segmentation", "label_ring", available={"labels", "spacing"}, name="ring"
        )
    )
    return processing, Pipeline()


class TestSegmentationNames:
    def test_only_the_steps_that_produce_labels_count(self):
        processing, _ = a_protocol()
        assert segmentation_names(processing) == ["nuclei", "ring"]


class TestSyncMeasurementSteps:
    def test_every_segmentation_gets_a_measurement_step(self):
        processing, analysis = a_protocol()
        added, removed = sync_measurement_steps(processing, analysis)
        assert [step.name for step in added] == ["measure_nuclei", "measure_ring"]
        assert removed == []

    def test_each_one_is_pointed_at_its_own_segmentation(self):
        processing, analysis = a_protocol()
        sync_measurement_steps(processing, analysis)
        wiring = {step.name: step.input_keys["labels"] for step in analysis.steps}
        assert wiring == {"measure_nuclei": "nuclei", "measure_ring": "ring"}

    def test_it_is_idempotent(self):
        processing, analysis = a_protocol()
        sync_measurement_steps(processing, analysis)
        added, removed = sync_measurement_steps(processing, analysis)
        assert (added, removed) == ([], [])
        assert len(analysis) == 2

    def test_a_hand_added_measurement_step_is_not_duplicated(self):
        """Somebody who has already measured the nuclei has answered the
        question; raising a second step would double every row of the
        table."""
        processing, analysis = a_protocol()
        mine = Step.for_function(
            "measurements", "extract_measurements", available={"labels", "intensity"}, name="mine"
        )
        mine.input_keys["labels"] = "nuclei"
        analysis.add_step(mine)

        added, _removed = sync_measurement_steps(processing, analysis)
        assert [step.auto_for for step in added] == ["ring"]

    def test_a_deleted_segmentation_takes_its_measurement_step_with_it(self):
        processing, analysis = a_protocol()
        sync_measurement_steps(processing, analysis)
        processing.remove_step(2)  # the ring

        added, removed = sync_measurement_steps(processing, analysis)
        assert added == []
        assert [step.name for step in removed] == ["measure_ring"]
        assert [step.name for step in analysis.steps] == ["measure_nuclei"]

    def test_a_hand_added_step_is_never_retired(self):
        processing, analysis = a_protocol()
        mine = Step.for_function(
            "measurements", "extract_measurements", available={"labels", "intensity"}, name="mine"
        )
        mine.input_keys["labels"] = "ring"
        analysis.add_step(mine)
        processing.remove_step(2)

        _added, removed = sync_measurement_steps(processing, analysis)
        assert removed == []
        assert "mine" in [step.name for step in analysis.steps]

    def test_the_raised_step_measures_every_channel(self):
        processing, analysis = a_protocol()
        added, _ = sync_measurement_steps(processing, analysis)
        assert added[0].function_name == "extract_measurements_by_channel"

    def test_names_do_not_collide_with_an_existing_step(self):
        processing, analysis = a_protocol()
        analysis.add_step(
            Step.for_function("measurements", "extract_measurements", name="measure_nuclei")
        )
        added, _ = sync_measurement_steps(processing, analysis)
        names = [step.name for step in analysis.steps]
        assert len(names) == len(set(names))
        assert added[0].name != "measure_nuclei"


class TestRename:
    def test_the_measurement_step_follows_the_segmentation_s_new_name(self):
        processing, analysis = a_protocol()
        sync_measurement_steps(processing, analysis)

        updated = rename_segmentation(analysis, "nuclei", "podocytes")
        assert [(step.name, previous) for step, previous in updated] == [
            ("measure_podocytes", "measure_nuclei")
        ]
        assert analysis.steps[0].auto_for == "podocytes"

    def test_a_step_the_user_has_renamed_keeps_its_own_name(self):
        """They have said what they want this table called; a rename
        somewhere else in the protocol is not a reason to overrule them."""
        processing, analysis = a_protocol()
        sync_measurement_steps(processing, analysis)
        analysis.steps[0].name = "my_nuclear_features"

        rename_segmentation(analysis, "nuclei", "podocytes")
        assert analysis.steps[0].name == "my_nuclear_features"
        assert analysis.steps[0].auto_for == "podocytes"

    def test_a_renamed_segmentation_is_not_measured_twice(self):
        processing, analysis = a_protocol()
        sync_measurement_steps(processing, analysis)
        processing.steps[1].name = "podocytes"
        rename_segmentation(analysis, "nuclei", "podocytes")
        analysis.steps[0].input_keys["labels"] = "podocytes"

        added, removed = sync_measurement_steps(processing, analysis)
        assert (added, removed) == ([], [])


class TestMeasurementNameFor:
    def test_it_is_the_segmentation_s_own_name(self):
        assert measurement_name_for("ring") == "measure_ring"

    def test_a_taken_name_gets_a_number(self):
        assert measurement_name_for("ring", {"measure_ring"}) == "measure_ring_1"
