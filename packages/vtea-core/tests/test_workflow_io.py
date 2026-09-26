"""A protocol round-trips through JSON and re-runs to the same answer.

The protocol is the file people will actually share: it has to come back as
the same steps, refuse what it cannot represent rather than half-save it,
and say clearly when a file is from a newer VTEA.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from vtea_core.data import Spacing
from vtea_core.gates import Gate, GateSet, rectangle_vertices
from vtea_core.workflow import (
    PROTOCOL_FORMAT_VERSION,
    Pipeline,
    Protocol,
    ProtocolError,
    Step,
    capture_environment,
    describe_source,
    load_protocol,
    protocol_from_dict,
    protocol_to_dict,
    save_protocol,
)
from vtea_core.workflow.io import decode_value, encode_value


def make_volume():
    rng = np.random.default_rng(0)
    volume = rng.normal(5.0, 1.0, (40, 40))
    for index, (y, x) in enumerate([(5, 5), (5, 30), (30, 5), (30, 30), (18, 18)]):
        volume[y : y + 5, x : x + 5] += 40.0 + 20.0 * index
    return volume


def make_protocol() -> Protocol:
    taken: list[str] = []

    def step(category, function, **kwargs):
        made = Step.for_function(category, function, taken_names=taken, **kwargs)
        taken.append(made.name)
        return made

    processing = Pipeline(
        [
            step("segmentation", "threshold_mask", params={"method": "fixed", "value": 20.0}),
            step("segmentation", "label_components"),
        ]
    )
    analysis = Pipeline(
        [
            step(
                "measurements",
                "extract_measurements",
                input_keys={"labels": "label_components_1", "intensity": "volume"},
                output_key="measurements",
            ),
        ]
    )
    gates = GateSet()
    gates.add(
        Gate(
            name="bright",
            x_axis="mean",
            y_axis="count",
            vertices=rectangle_vertices(50, 0, 200, 100),
        )
    )
    return Protocol(
        processing=processing,
        analysis=analysis,
        z_axis=None,
        spacing=Spacing((0.5, 0.25, 0.25), unit="µm"),
        gates={"Objects": gates},
        source={"name": "sample"},
    )


def run(protocol: Protocol, volume) -> dict:
    context = protocol.processing.run({"volume": volume})
    return protocol.analysis.run(context)


class TestRoundTrip:
    def test_steps_come_back_as_they_were(self, tmp_path):
        original = make_protocol()
        loaded = load_protocol(save_protocol(original, tmp_path / "p.vtea.json"))
        assert [s.name for s in loaded.steps] == [s.name for s in original.steps]
        for before, after in zip(original.steps, loaded.steps):
            assert after.settings_signature == before.settings_signature
            assert after.output_key == before.output_key

    def test_a_loaded_protocol_re_runs_to_the_same_table(self, tmp_path):
        volume = make_volume()
        original = make_protocol()
        loaded = load_protocol(save_protocol(original, tmp_path / "p.vtea.json"))
        expected = run(original, volume)["measurements"]
        actual = run(loaded, volume)["measurements"]
        assert len(expected) == 5
        pd.testing.assert_frame_equal(actual, expected)

    def test_axes_spacing_and_switches_survive(self, tmp_path):
        original = make_protocol()
        original.channel_axis = 1
        original.z_axis = 0
        original.measure_every_segmentation = False
        loaded = load_protocol(save_protocol(original, tmp_path / "p.vtea.json"))
        assert (loaded.channel_axis, loaded.z_axis) == (1, 0)
        assert loaded.processing.channel_axis == 1
        assert loaded.analysis.channel_axis == 1
        assert loaded.spacing == original.spacing
        assert loaded.measure_every_segmentation is False

    def test_gates_survive_per_table(self, tmp_path):
        loaded = load_protocol(save_protocol(make_protocol(), tmp_path / "p.vtea.json"))
        (gate,) = list(loaded.gates["Objects"])
        assert gate.name == "bright"
        assert (gate.x_axis, gate.y_axis) == ("mean", "count")

    def test_features_and_auto_for_are_kept_verbatim(self):
        step = Step.for_function("clustering", "kmeans", params={"n_clusters": 3})
        step.features = ["mean", "count"]
        step.auto_for = "label_components_1"
        protocol = Protocol(analysis=Pipeline([step]))
        (loaded,) = protocol_from_dict(protocol_to_dict(protocol)).steps
        assert loaded.features == ["mean", "count"]
        assert loaded.auto_for == "label_components_1"

    def test_an_empty_feature_selection_stays_empty(self):
        step = Step.for_function("clustering", "kmeans", params={"n_clusters": 3})
        data = protocol_to_dict(Protocol(analysis=Pipeline([step])))
        assert data["analysis"][0]["features"] == []

    def test_the_file_is_plain_json_with_a_version(self, tmp_path):
        path = save_protocol(make_protocol(), tmp_path / "p.vtea.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["vtea_protocol_version"] == PROTOCOL_FORMAT_VERSION
        assert data["processing"][0]["function"] == "threshold_mask"
        assert "numpy" in data["environment"]["packages"]


class TestParameters:
    @pytest.mark.parametrize(
        "value",
        [None, True, 3, 2.5, "otsu", [1, 2], (1, 2), {"a": (1.0, 2.0)}],
    )
    def test_plain_values_round_trip_exactly(self, value):
        decoded = decode_value(json.loads(json.dumps(encode_value(value))))
        assert decoded == value
        assert type(decoded) is type(value)

    def test_arrays_keep_their_dtype(self):
        value = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        decoded = decode_value(json.loads(json.dumps(encode_value(value))))
        assert decoded.dtype == np.uint16
        np.testing.assert_array_equal(decoded, value)

    def test_numpy_scalars_become_python_numbers(self):
        assert encode_value(np.float32(1.5)) == 1.5
        assert isinstance(encode_value(np.int64(3)), int)

    def test_spacing_round_trips(self):
        spacing = Spacing((1.0, 0.5, 0.5), unit="µm")
        assert decode_value(json.loads(json.dumps(encode_value(spacing)))) == spacing

    def test_an_arbitrary_object_is_refused_naming_the_step_and_parameter(self, tmp_path):
        step = Step.for_function("segmentation", "threshold_mask", params={"value": object()})
        path = tmp_path / "p.vtea.json"
        with pytest.raises(ProtocolError, match="threshold_mask_1.*'value'"):
            save_protocol(Protocol(processing=Pipeline([step])), path)
        assert not path.exists()


class TestRefusals:
    def test_a_newer_version_is_refused(self):
        data = protocol_to_dict(make_protocol())
        data["vtea_protocol_version"] = PROTOCOL_FORMAT_VERSION + 1
        with pytest.raises(ProtocolError, match="newer"):
            protocol_from_dict(data)

    def test_an_unknown_step_is_refused_by_name(self):
        data = protocol_to_dict(make_protocol())
        data["processing"][0]["function"] = "layer_cake_3d"
        with pytest.raises(ProtocolError, match="threshold_mask_1.*layer_cake_3d"):
            protocol_from_dict(data)

    def test_a_file_that_is_not_a_protocol_is_refused(self, tmp_path):
        path = tmp_path / "gates.json"
        path.write_text(json.dumps({"vtea_gates_version": 1, "gates": []}))
        with pytest.raises(ProtocolError, match="not a VTEA protocol"):
            load_protocol(path)

    def test_invalid_json_is_a_protocol_error(self, tmp_path):
        path = tmp_path / "broken.vtea.json"
        path.write_text("{")
        with pytest.raises(ProtocolError, match="not valid JSON"):
            load_protocol(path)


class TestProvenance:
    def test_environment_records_python_and_installed_packages(self):
        environment = capture_environment()
        assert environment["python"]
        assert "numpy" in environment["packages"]
        assert "not-a-real-package" not in capture_environment(("not-a-real-package",))["packages"]

    def test_source_is_hashed_when_small(self, tmp_path):
        path = tmp_path / "image.tif"
        path.write_bytes(b"pixels")
        source = describe_source(path, name="image", shape=(2, 3), dtype=np.uint16)
        assert source["sha256"] == hashlib.sha256(b"pixels").hexdigest()
        assert source["shape"] == [2, 3]
        assert source["dtype"] == "uint16"
        assert source["bytes"] == 6

    def test_a_large_source_records_that_the_hash_was_skipped(self, tmp_path):
        path = tmp_path / "image.tif"
        path.write_bytes(b"pixels")
        source = describe_source(path, hash_limit=2)
        assert source["sha256"] is None
        assert "larger than 2 bytes" in source["sha256_skipped"]
