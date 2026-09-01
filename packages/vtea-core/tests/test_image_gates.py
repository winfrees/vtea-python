import numpy as np
import pandas as pd
import pytest

from vtea_core.gates import (
    CENTROID,
    MAJORITY,
    centroids_from_frame,
    column_name,
    image_gate,
    objects_in_rois,
)


def a_segmentation():
    """Three objects in a 10x10 plane: two on the left, one on the right."""
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[6:8, 1:3] = 2
    labels[1:3, 7:9] = 3
    return labels


def two_rois():
    """Two painted regions: region 1 over the left half, 2 over the right."""
    rois = np.zeros((10, 10), dtype=np.int32)
    rois[:, :5] = 1
    rois[:, 6:] = 2
    return rois


class TestObjectsInRois:
    def test_each_object_gets_the_region_it_is_in(self):
        table = objects_in_rois(a_segmentation(), two_rois())
        assert list(table["object_id"]) == [1, 2, 3]
        assert list(table["roi"]) == [1, 1, 2]

    def test_an_object_outside_every_region_gets_zero(self):
        labels = a_segmentation()
        rois = np.zeros_like(labels)
        rois[:, :5] = 1
        assert list(objects_in_rois(labels, rois)["roi"]) == [1, 1, 0]

    def test_the_answer_is_the_region_id_not_a_boolean(self):
        """Three tubules are three populations, and the id is what joins the
        objects to the region's own colour in napari."""
        assert set(objects_in_rois(a_segmentation(), two_rois())["roi"]) == {1, 2}

    def test_a_subset_of_objects_can_be_asked_about(self):
        table = objects_in_rois(a_segmentation(), two_rois(), object_ids=[3])
        assert list(table["object_id"]) == [3]
        assert list(table["roi"]) == [2]

    def test_centroids_already_measured_are_used_rather_than_recomputed(self):
        labels = a_segmentation()
        # Deliberately wrong centroids: if they were ignored the answer
        # would be the real one, so this proves they are read.
        table = objects_in_rois(
            labels,
            two_rois(),
            object_ids=[1, 2, 3],
            centroids=np.array([[1.0, 8.0], [7.0, 8.0], [2.0, 1.0]]),
        )
        assert list(table["roi"]) == [2, 2, 1]

    def test_a_3d_volume_works_the_same_way(self):
        labels = np.zeros((4, 10, 10), dtype=np.int32)
        labels[1:3, 1:3, 1:3] = 1
        labels[1:3, 1:3, 7:9] = 2
        rois = np.zeros_like(labels)
        rois[:, :, :5] = 1
        rois[:, :, 6:] = 2
        assert list(objects_in_rois(labels, rois)["roi"]) == [1, 2]

    def test_a_singleton_axis_on_the_roi_layer_is_squeezed_rather_than_refused(self):
        """A layer painted over a multi-channel image carries an axis the
        label image does not; that is not a disagreement about the data."""
        labels = a_segmentation()
        rois = two_rois()[np.newaxis, ...]
        assert list(objects_in_rois(labels, rois)["roi"]) == [1, 1, 2]

    def test_a_real_shape_mismatch_says_so(self):
        with pytest.raises(ValueError, match="same image"):
            objects_in_rois(a_segmentation(), np.zeros((4, 4), dtype=int))

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="unknown mode"):
            objects_in_rois(a_segmentation(), two_rois(), mode="nearby")


class TestMajorityMode:
    def test_an_object_belongs_to_the_region_most_of_it_is_in(self):
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[1:5, 1:5] = 1
        rois = np.zeros((6, 6), dtype=np.int32)
        rois[:, :2] = 1  # a quarter of the object
        rois[:, 2:] = 2  # three quarters
        table = objects_in_rois(labels, rois, mode=MAJORITY)
        assert list(table["roi"]) == [2]
        assert table["fraction"].iloc[0] == pytest.approx(0.75)

    def test_an_object_mostly_outside_can_be_excluded_by_fraction(self):
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[1:5, 1:5] = 1
        rois = np.zeros((6, 6), dtype=np.int32)
        rois[:, 4:] = 1  # only a quarter of the object
        strict = objects_in_rois(labels, rois, mode=MAJORITY, minimum_fraction=0.5)
        assert list(strict["roi"]) == [0]
        lenient = objects_in_rois(labels, rois, mode=MAJORITY, minimum_fraction=0.1)
        assert list(lenient["roi"]) == [1]

    def test_an_object_straddling_a_boundary_goes_to_the_region_not_the_background(self):
        labels = np.zeros((4, 4), dtype=np.int32)
        labels[1:3, 1:3] = 1
        rois = np.zeros((4, 4), dtype=np.int32)
        rois[1:3, 2:3] = 5  # exactly half the object
        assert list(objects_in_rois(labels, rois, mode=MAJORITY)["roi"]) == [5]

    def test_an_empty_segmentation_is_not_an_error(self):
        table = objects_in_rois(
            np.zeros((4, 4), dtype=np.int32), np.ones((4, 4), dtype=np.int32),
            object_ids=[1, 2], mode=MAJORITY,
        )
        assert list(table["roi"]) == [0, 0]


class TestImageGateColumn:
    def test_it_is_one_value_per_object_in_order(self):
        column = image_gate(a_segmentation(), two_rois(), object_ids=[3, 1, 2])
        assert list(column) == [2, 1, 1]

    def test_the_column_name_is_readable_in_a_class_definition(self):
        assert column_name("tubules") == "roi_tubules"
        assert column_name("Tubules (hand drawn)") == "roi_Tubules_hand_drawn"
        assert column_name("") == "roi_layer"


class TestCentroidsFromFrame:
    def test_it_reads_the_columns_a_measurement_step_produced(self):
        frame = pd.DataFrame({"centroid-0": [1.0, 2.0], "centroid-1": [3.0, 4.0]})
        assert centroids_from_frame(frame, 2).tolist() == [[1.0, 3.0], [2.0, 4.0]]

    def test_a_table_without_centroids_gives_none(self):
        assert centroids_from_frame(pd.DataFrame({"mean": [1.0]}), 2) is None

    def test_a_per_cell_table_namespaces_its_centroids(self):
        frame = pd.DataFrame({"nuclei.centroid-0": [1.0], "nuclei.centroid-1": [2.0]})
        assert centroids_from_frame(frame, 2, prefix="nuclei.").tolist() == [[1.0, 2.0]]
