"""Tests cellpose_segmentation()'s glue logic against a stub model, so this
file runs without the `deeplearning` extra (torch/cellpose) installed.
"""

import numpy as np
import pytest

from vtea_core.segmentation.deep import cellpose_segmentation


class _StubCellposeModel:
    """Minimal stand-in for cellpose.models.CellposeModel: records how it
    was called and returns a fixed (masks, flows, styles) tuple, so tests
    can verify our glue logic (channel replication, argument passing,
    output shape) without needing real model weights."""

    def __init__(self):
        self.eval_calls = []

    def eval(self, x, **kwargs):
        self.eval_calls.append((x, kwargs))
        masks = np.zeros(x.shape[:-1], dtype=np.int32)  # drop the channel axis
        masks[tuple(np.array(masks.shape) // 2)] = 1
        return masks, None, None


class TestChannelReplication:
    def test_2d_volume_replicated_to_three_channels(self):
        volume = np.random.default_rng(0).random((5, 6))
        stub = _StubCellposeModel()
        cellpose_segmentation(volume, model=stub)

        x, kwargs = stub.eval_calls[0]
        assert x.shape == (5, 6, 3)
        np.testing.assert_array_equal(x[..., 0], volume)
        np.testing.assert_array_equal(x[..., 0], x[..., 1])
        np.testing.assert_array_equal(x[..., 0], x[..., 2])
        assert kwargs["channel_axis"] == -1

    def test_3d_volume_replicated_to_three_channels(self):
        volume = np.random.default_rng(0).random((4, 5, 6))
        stub = _StubCellposeModel()
        cellpose_segmentation(volume, model=stub, do_3D=True)

        x, kwargs = stub.eval_calls[0]
        assert x.shape == (4, 5, 6, 3)
        assert kwargs["z_axis"] == 0
        assert kwargs["do_3D"] is True


class TestParameterPassing:
    def test_defaults(self):
        stub = _StubCellposeModel()
        cellpose_segmentation(np.zeros((4, 4)), model=stub)
        _, kwargs = stub.eval_calls[0]
        assert kwargs["do_3D"] is False
        assert kwargs["diameter"] is None
        assert kwargs["flow_threshold"] == pytest.approx(0.4)
        assert kwargs["cellprob_threshold"] == pytest.approx(0.0)

    def test_custom_values_passed_through(self):
        stub = _StubCellposeModel()
        cellpose_segmentation(
            np.zeros((4, 4)),
            model=stub,
            diameter=30.0,
            flow_threshold=0.6,
            cellprob_threshold=0.2,
        )
        _, kwargs = stub.eval_calls[0]
        assert kwargs["diameter"] == pytest.approx(30.0)
        assert kwargs["flow_threshold"] == pytest.approx(0.6)
        assert kwargs["cellprob_threshold"] == pytest.approx(0.2)

    def test_z_axis_only_set_for_3d(self):
        stub = _StubCellposeModel()
        cellpose_segmentation(np.zeros((5, 5)), model=stub, do_3D=False)
        _, kwargs = stub.eval_calls[0]
        assert "z_axis" not in kwargs


class TestReturnValue:
    def test_returns_numpy_label_array_matching_input_shape(self):
        volume = np.zeros((5, 6))
        stub = _StubCellposeModel()
        result = cellpose_segmentation(volume, model=stub)
        assert isinstance(result, np.ndarray)
        assert result.shape == volume.shape

    def test_returns_the_models_masks(self):
        volume = np.zeros((4, 4))
        stub = _StubCellposeModel()
        result = cellpose_segmentation(volume, model=stub)
        assert result[2, 2] == 1  # matches _StubCellposeModel's fixed marker
