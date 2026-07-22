"""Tests for the torch-based classifier. Skipped entirely if torch (the
`deeplearning` extra) isn't installed.
"""

import importlib.util

import numpy as np
import pytest

torch_available = importlib.util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(not torch_available, reason="torch (deeplearning extra) not installed")

if torch_available:
    from vtea_core.classification import Cell3DClassifier, predict, train_classifier


def make_two_class_crops(n_per_class=20, size=8, seed=0):
    rng = np.random.default_rng(seed)
    # class 0: low-intensity noise; class 1: a bright cube in the center
    class0 = rng.normal(0.1, 0.05, size=(n_per_class, size, size, size))
    class1 = rng.normal(0.1, 0.05, size=(n_per_class, size, size, size))
    center = slice(size // 2 - 2, size // 2 + 2)
    class1[:, center, center, center] += 1.0

    crops = np.concatenate([class0, class1]).astype(np.float32)
    labels = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)
    return crops, labels


class TestCell3DClassifier:
    def test_forward_output_shape(self):
        model = Cell3DClassifier(n_classes=3)
        x = torch_zeros_input(batch=4, size=8)
        logits = model(x)
        assert tuple(logits.shape) == (4, 3)


class TestTrainAndPredict:
    def test_learns_a_clearly_separable_two_class_problem(self):
        crops, labels = make_two_class_crops()
        model = Cell3DClassifier(n_classes=2)
        train_classifier(model, crops, labels, epochs=30, lr=1e-2)

        predictions = predict(model, crops)
        accuracy = (predictions == labels).mean()
        assert accuracy > 0.9

    def test_predict_returns_correct_shape_and_dtype(self):
        crops, labels = make_two_class_crops(n_per_class=3)
        model = Cell3DClassifier(n_classes=2)
        train_classifier(model, crops, labels, epochs=2)

        predictions = predict(model, crops)
        assert predictions.shape == (6,)
        assert np.issubdtype(predictions.dtype, np.integer)


def torch_zeros_input(*, batch: int, size: int):
    import torch

    return torch.zeros(batch, 1, size, size, size)
