import numpy as np
import pandas as pd

from vtea_napari.widgets.gallery import GalleryWidget
from vtea_napari.widgets.thumbnail import array_to_pixmap, max_projection


class TestMaxProjection:
    def test_collapses_3d_to_2d(self):
        volume = np.zeros((3, 4, 4))
        volume[1] = 5.0
        projected = max_projection(volume)
        assert projected.shape == (4, 4)
        assert (projected == 5.0).all()

    def test_2d_is_unchanged(self):
        array = np.arange(9).reshape(3, 3)
        np.testing.assert_array_equal(max_projection(array), array)


class TestArrayToPixmap:
    def test_returns_a_pixmap_of_requested_size(self, qtbot):
        array = np.arange(100, dtype=float).reshape(10, 10)
        pixmap = array_to_pixmap(array, size=64)
        assert not pixmap.isNull()
        assert pixmap.width() <= 64
        assert pixmap.height() <= 64

    def test_constant_array_does_not_crash(self, qtbot):
        array = np.full((5, 5), 3.0)
        pixmap = array_to_pixmap(array)
        assert not pixmap.isNull()

    def test_rejects_non_2d(self, qtbot):
        import pytest

        with pytest.raises(ValueError, match="2D"):
            array_to_pixmap(np.zeros((2, 3, 3)))


def make_frame():
    return pd.DataFrame(
        {
            "object_id": [1, 2],
            "centroid-0": [5.0, 15.0],
            "centroid-1": [5.0, 15.0],
        }
    )


class TestGalleryWidget:
    def test_show_objects_creates_one_thumbnail_per_object(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        volume = np.random.default_rng(0).random((20, 20))

        widget.show_objects(volume, make_frame(), [1, 2], crop_radius=4)

        assert widget._grid.count() == 2

    def test_clicking_a_thumbnail_emits_its_object_id(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        volume = np.random.default_rng(0).random((20, 20))
        widget.show_objects(volume, make_frame(), [1, 2], crop_radius=4)

        received = []
        widget.object_selected.connect(received.append)
        first_cell = widget._grid.itemAt(0).widget()
        first_cell.clicked.emit()

        assert received == [1]

    def test_unknown_object_id_is_skipped_not_crashed(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        volume = np.zeros((20, 20))
        widget.show_objects(volume, make_frame(), [1, 999], crop_radius=4)
        assert widget._grid.count() == 1

    def test_re_calling_show_objects_clears_previous_thumbnails(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        volume = np.zeros((20, 20))
        widget.show_objects(volume, make_frame(), [1, 2], crop_radius=4)
        widget.show_objects(volume, make_frame(), [1], crop_radius=4)
        assert widget._grid.count() == 1
