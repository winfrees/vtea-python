"""Unit tests for the comparison utilities themselves (no fixtures needed)."""

import numpy as np
import pandas as pd
import pytest

from compare import cluster_assignment_ari, feature_table_diff, segmentation_iou


def test_segmentation_iou_identical_masks():
    mask = np.array([[0, 1, 1], [0, 1, 0]])
    assert segmentation_iou(mask, mask) == pytest.approx(1.0)


def test_segmentation_iou_disjoint_masks():
    a = np.array([[1, 0], [0, 0]])
    b = np.array([[0, 1], [0, 0]])
    assert segmentation_iou(a, b) == pytest.approx(0.0)


def test_segmentation_iou_both_empty():
    a = np.zeros((3, 3))
    b = np.zeros((3, 3))
    assert segmentation_iou(a, b) == pytest.approx(1.0)


def test_segmentation_iou_partial_overlap():
    a = np.array([1, 1, 1, 0, 0])
    b = np.array([1, 1, 0, 0, 1])
    # intersection = 2, union = 4
    assert segmentation_iou(a, b) == pytest.approx(0.5)


def test_feature_table_diff_within_tolerance():
    expected = pd.DataFrame({"id": [1, 2, 3], "mean": [1.0, 2.0, 3.0]})
    actual = pd.DataFrame({"id": [1, 2, 3], "mean": [1.0000001, 2.0, 3.0]})
    diff = feature_table_diff(actual, expected, key="id")
    assert diff.empty


def test_feature_table_diff_out_of_tolerance():
    expected = pd.DataFrame({"id": [1, 2, 3], "mean": [1.0, 2.0, 3.0]})
    actual = pd.DataFrame({"id": [1, 2, 3], "mean": [1.0, 2.0, 30.0]})
    diff = feature_table_diff(actual, expected, key="id")
    assert not diff.empty
    assert diff.iloc[0]["column"] == "mean"
    assert diff.iloc[0]["status"] == "out_of_tolerance"


def test_feature_table_diff_row_key_mismatch_raises():
    expected = pd.DataFrame({"id": [1, 2, 3], "mean": [1.0, 2.0, 3.0]})
    actual = pd.DataFrame({"id": [1, 2, 4], "mean": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="Row key mismatch"):
        feature_table_diff(actual, expected, key="id")


def test_cluster_assignment_ari_identical_labels():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert cluster_assignment_ari(labels, labels) == pytest.approx(1.0)


def test_cluster_assignment_ari_relabeled_but_equivalent_grouping():
    expected = np.array([0, 0, 1, 1, 2, 2])
    actual = np.array([2, 2, 0, 0, 1, 1])  # same grouping, different label ids
    assert cluster_assignment_ari(actual, expected) == pytest.approx(1.0)


def test_cluster_assignment_ari_unrelated_labels():
    rng = np.random.default_rng(0)
    expected = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2] * 5)
    actual = rng.integers(0, 3, size=len(expected))
    ari = cluster_assignment_ari(actual, expected)
    assert ari < 0.5
