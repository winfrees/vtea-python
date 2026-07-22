import numpy as np
import pytest

from vtea_core.classification import class_map


def test_maps_object_ids_to_class_labels():
    labels = np.array([[0, 1, 1], [2, 2, 0]])
    result = class_map(labels, object_ids=[1, 2], class_labels=[5, 9])
    expected = np.array([[0, 5, 5], [9, 9, 0]])
    np.testing.assert_array_equal(result, expected)


def test_unlisted_objects_become_background():
    labels = np.array([1, 2, 3])
    result = class_map(labels, object_ids=[1, 3], class_labels=[7, 8])
    np.testing.assert_array_equal(result, [7, 0, 8])


def test_background_stays_zero():
    labels = np.zeros((3, 3), dtype=np.int32)
    result = class_map(labels, object_ids=[], class_labels=[])
    np.testing.assert_array_equal(result, labels)


def test_mismatched_lengths_raise():
    labels = np.array([1, 2])
    with pytest.raises(ValueError, match="shape"):
        class_map(labels, object_ids=[1, 2], class_labels=[5])
