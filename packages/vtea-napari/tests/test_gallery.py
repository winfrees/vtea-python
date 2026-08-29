import numpy as np
import pandas as pd

from vtea_napari.widgets.gallery import SELECTED_STYLE, UNSELECTED_STYLE, GalleryWidget
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


class TestSelectionOutline:
    """The selected crop is outlined in yellow, so which cell you clicked is
    visible against the crops themselves."""

    @staticmethod
    def make_gallery(qtbot):
        gallery = GalleryWidget()
        qtbot.addWidget(gallery)
        volume = np.arange(400, dtype=float).reshape(20, 20)
        frame = pd.DataFrame(
            {
                "object_id": [1, 2, 3],
                "centroid-0": [5.0, 10.0, 15.0],
                "centroid-1": [5.0, 10.0, 15.0],
            }
        )
        gallery.show_objects(volume, frame, [1, 2, 3], crop_radius=4, thumbnail_size=32)
        return gallery

    def test_nothing_is_outlined_to_start_with(self, qtbot):
        gallery = self.make_gallery(qtbot)
        assert gallery.selected_object_id is None
        assert all(
            SELECTED_STYLE not in thumbnail.styleSheet()
            for thumbnail in gallery._thumbnails
        )

    def test_selecting_outlines_that_crop_in_yellow(self, qtbot):
        gallery = self.make_gallery(qtbot)
        gallery.select(2)

        assert gallery.selected_object_id == 2
        outlined = [t.object_id for t in gallery._thumbnails if SELECTED_STYLE in t.styleSheet()]
        assert outlined == [2]
        assert "#ffd400" in SELECTED_STYLE

    def test_only_one_crop_is_outlined_at_a_time(self, qtbot):
        gallery = self.make_gallery(qtbot)
        gallery.select(1)
        gallery.select(3)

        outlined = [t.object_id for t in gallery._thumbnails if SELECTED_STYLE in t.styleSheet()]
        assert outlined == [3]

    def test_clicking_a_crop_outlines_it_and_reports_it(self, qtbot):
        gallery = self.make_gallery(qtbot)
        reported = []
        gallery.object_selected.connect(reported.append)

        gallery._thumbnails[1].clicked.emit()

        assert reported == [2]
        assert gallery.selected_object_id == 2

    def test_the_unselected_border_reserves_the_same_space(self, qtbot):
        """Otherwise selecting a cell reflows the whole grid."""
        assert "3px" in SELECTED_STYLE
        assert "3px" in UNSELECTED_STYLE
        assert "transparent" in UNSELECTED_STYLE

    def test_the_outline_survives_a_refresh_that_still_shows_it(self, qtbot):
        gallery = self.make_gallery(qtbot)
        gallery.select(2)

        volume = np.arange(400, dtype=float).reshape(20, 20)
        frame = pd.DataFrame(
            {
                "object_id": [1, 2, 3],
                "centroid-0": [5.0, 10.0, 15.0],
                "centroid-1": [5.0, 10.0, 15.0],
            }
        )
        gallery.show_objects(volume, frame, [1, 2, 3], crop_radius=4, thumbnail_size=32)

        assert gallery.selected_object_id == 2

    def test_a_refresh_that_drops_the_object_clears_the_outline(self, qtbot):
        gallery = self.make_gallery(qtbot)
        gallery.select(2)

        volume = np.arange(400, dtype=float).reshape(20, 20)
        frame = pd.DataFrame(
            {"object_id": [1], "centroid-0": [5.0], "centroid-1": [5.0]}
        )
        gallery.show_objects(volume, frame, [1], crop_radius=4, thumbnail_size=32)

        assert gallery.selected_object_id is None

    def test_selecting_an_object_that_is_not_shown_just_clears(self, qtbot):
        gallery = self.make_gallery(qtbot)
        gallery.select(99)
        assert gallery.selected_object_id is None
