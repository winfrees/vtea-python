import numpy as np
import pytest

from vtea_core.workflow import Pipeline, Step, available_steps, get_step_function


class TestRegistry:
    def test_get_step_function_returns_callable(self):
        func = get_step_function("segmentation", "threshold_mask")
        assert callable(func)

    def test_get_step_function_unknown_category_raises(self):
        with pytest.raises(KeyError, match="unknown step"):
            get_step_function("bogus", "threshold_mask")

    def test_get_step_function_unknown_name_raises(self):
        with pytest.raises(KeyError, match="unknown step"):
            get_step_function("segmentation", "bogus")

    def test_available_steps_all_categories(self):
        registry = available_steps()
        assert "segmentation" in registry
        assert "clustering" in registry

    def test_available_steps_one_category(self):
        steps = available_steps("clustering")
        assert "kmeans" in steps

    def test_available_steps_unknown_category_raises(self):
        with pytest.raises(KeyError, match="unknown category"):
            available_steps("bogus")


class TestStep:
    def test_run_calls_function_with_resolved_inputs_and_params(self):
        step = Step(
            category="segmentation",
            function_name="threshold_mask",
            params={"method": "fixed", "value": 10},
            input_keys={"volume": "raw"},
        )
        context = {"raw": np.array([0, 5, 15, 20])}
        result = step.run(context)
        np.testing.assert_array_equal(result, [False, False, True, True])

    def test_run_missing_context_key_raises(self):
        step = Step(category="segmentation", function_name="threshold_mask", input_keys={"volume": "missing_key"})
        with pytest.raises(KeyError, match="missing_key"):
            step.run({})


class TestPipeline:
    def make_synthetic_volume(self):
        volume = np.zeros((6, 6), dtype=float)
        volume[1:3, 1:3] = 100.0  # object 1
        volume[3:5, 3:5] = 100.0  # object 2
        return volume

    def test_empty_pipeline_returns_input_context_unchanged(self):
        pipeline = Pipeline()
        result = pipeline.run({"x": 1})
        assert result == {"x": 1}

    def test_add_insert_remove_move_step(self):
        pipeline = Pipeline()
        s1 = pipeline.add_step(Step(category="segmentation", function_name="threshold_mask", output_key="mask"))
        s2 = pipeline.add_step(Step(category="segmentation", function_name="label_components", output_key="labels"))
        assert list(pipeline) == [s1, s2]

        s0 = Step(category="imageprocessing", function_name="gaussian_blur", output_key="blurred")
        pipeline.insert_step(0, s0)
        assert list(pipeline) == [s0, s1, s2]

        removed = pipeline.remove_step(0)
        assert removed is s0
        assert list(pipeline) == [s1, s2]

        pipeline.move_step(1, 0)
        assert list(pipeline) == [s2, s1]
        assert len(pipeline) == 2

    def test_end_to_end_segmentation_and_measurement_pipeline(self):
        volume = self.make_synthetic_volume()
        pipeline = Pipeline(
            [
                Step(
                    category="segmentation",
                    function_name="threshold_mask",
                    params={"method": "fixed", "value": 50},
                    input_keys={"volume": "volume"},
                    output_key="mask",
                ),
                Step(
                    category="segmentation",
                    function_name="label_components",
                    input_keys={"mask": "mask"},
                    output_key="labels",
                ),
                Step(
                    category="measurements",
                    function_name="extract_measurements",
                    input_keys={"labels": "labels", "intensity": "volume"},
                    output_key="measurements",
                ),
            ]
        )

        result = pipeline.run({"volume": volume})

        assert result["mask"].sum() == 8  # two 2x2 objects
        assert result["labels"].max() == 2
        assert sorted(result["measurements"]["object_id"]) == [1, 2]
        assert (result["measurements"]["mean"] == 100.0).all()

    def test_end_to_end_pipeline_distinguishes_object_intensities(self):
        volume = np.zeros((30, 30), dtype=float)
        # two well-separated blobs of distinct intensity, far apart in space
        volume[2:6, 2:6] = 50.0
        volume[20:26, 20:26] = 200.0

        pipeline = Pipeline(
            [
                Step(
                    category="segmentation",
                    function_name="threshold_mask",
                    params={"method": "fixed", "value": 10},
                    input_keys={"volume": "volume"},
                    output_key="mask",
                ),
                Step(
                    category="segmentation",
                    function_name="label_components",
                    input_keys={"mask": "mask"},
                    output_key="labels",
                ),
                Step(
                    category="measurements",
                    function_name="extract_measurements",
                    input_keys={"labels": "labels", "intensity": "volume"},
                    output_key="measurements",
                ),
            ]
        )
        result = pipeline.run({"volume": volume})
        means = result["measurements"]["mean"].to_numpy()
        assert sorted(means) == [50.0, 200.0]
