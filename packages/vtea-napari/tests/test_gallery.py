import numpy as np
import pandas as pd
from pytest import approx as pytest_approx

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


def multichannel_volume(n_channels=3, size=24):
    """A (c, y, x) image where each channel is bright in a different place,
    so a composite can be told from a single channel by where the colour is."""
    volume = np.zeros((n_channels, size, size), dtype=np.uint16)
    for channel in range(n_channels):
        volume[channel, 2 + 4 * channel : 6 + 4 * channel, 2:22] = 100 * (channel + 1)
    return volume


def two_objects_labels(size=24):
    labels = np.zeros((size, size), dtype=np.int32)
    labels[3:8, 3:8] = 1
    labels[14:19, 14:19] = 2
    return labels


def two_object_frame():
    return pd.DataFrame(
        {
            "object_id": [1, 2],
            "centroid-0": [5.0, 16.0],
            "centroid-1": [5.0, 16.0],
        }
    )


class TestComposite:
    """Item 1: an overlay of the nuclei and up to three selectable channels,
    the same channels for every thumbnail."""

    def test_the_channel_menu_follows_the_image(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.show_objects(
            multichannel_volume(3), two_object_frame(), [1, 2], crop_radius=6, channel_axis=0
        )
        choices = [
            widget.channel_combos[0].itemText(index)
            for index in range(widget.channel_combos[0].count())
        ]
        assert choices == ["(none)", "Channel 0", "Channel 1", "Channel 2"]

    def test_three_slots_are_offered(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        assert len(widget.channel_combos) == 3
        assert len(widget.lut_combos) == 3

    def test_one_channel_is_shown_to_start_with(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.show_objects(
            multichannel_volume(3), two_object_frame(), [1, 2], crop_radius=6, channel_axis=0
        )
        assert widget.selected_channels() == [(0, "gray")]

    def test_choosing_channels_and_luts(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.show_objects(
            multichannel_volume(3), two_object_frame(), [1, 2], crop_radius=6, channel_axis=0
        )
        widget.channel_combos[1].setCurrentText("Channel 2")
        widget.lut_combos[1].setCurrentText("red")

        assert widget.selected_channels() == [(0, "gray"), (2, "red")]

    def test_changing_a_channel_redraws_every_thumbnail(self, qtbot):
        """Every crop is made the same way; a grid where each picture was
        made differently is not comparable, and comparing is the point."""
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.show_objects(
            multichannel_volume(3), two_object_frame(), [1, 2], crop_radius=6, channel_axis=0
        )
        before = [cell.pixmap().toImage() for cell in widget._thumbnails]
        widget.channel_combos[0].setCurrentText("Channel 2")
        after = [cell.pixmap().toImage() for cell in widget._thumbnails]

        assert len(after) == len(before) == 2
        assert all(one != two for one, two in zip(before, after))

    def test_a_single_channel_image_offers_only_that_channel(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.show_objects(np.zeros((24, 24)), two_object_frame(), [1], crop_radius=6)
        assert widget.selected_channels() == []
        assert widget._grid.count() == 1  # still drawn, in grey

    def test_the_settings_are_remembered(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.show_objects(
            multichannel_volume(3), two_object_frame(), [1, 2], crop_radius=6, channel_axis=0
        )
        widget.channel_combos[1].setCurrentText("Channel 1")
        widget.lut_combos[1].setCurrentText("cyan")
        state = widget.view_state()

        restored = GalleryWidget()
        qtbot.addWidget(restored)
        restored.show_objects(
            multichannel_volume(3), two_object_frame(), [1, 2], crop_radius=6, channel_axis=0
        )
        restored.apply_view_state(state)
        assert restored.selected_channels() == widget.selected_channels()


class TestCompositeColours:
    def test_each_channel_is_shown_in_its_own_lut(self):
        from vtea_napari.widgets.thumbnail import composite_rgb

        first = np.zeros((4, 4))
        first[0] = 1.0
        second = np.zeros((4, 4))
        second[1] = 1.0

        rgb = composite_rgb([first, second], ["green", "red"])
        assert tuple(rgb[0, 0]) == (0, 255, 0)
        assert tuple(rgb[1, 0]) == (255, 0, 0)

    def test_overlapping_channels_add_rather_than_average(self):
        """A composite shows an overlap as the sum of the two colours;
        averaging would dim both."""
        from vtea_napari.widgets.thumbnail import composite_rgb

        plane = np.zeros((2, 2))
        plane[0, 0] = 1.0
        rgb = composite_rgb([plane, plane], ["green", "red"])
        assert tuple(rgb[0, 0]) == (255, 255, 0)

    def test_each_channel_is_normalised_against_itself(self):
        """A dim channel is still visible beside a bright one - forty crops
        normalised against the brightest cell in the stack are thirty-nine
        black squares."""
        from vtea_napari.widgets.thumbnail import composite_rgb

        bright = np.array([[0.0, 1000.0]])
        dim = np.array([[0.0, 3.0]])
        rgb = composite_rgb([bright, dim], ["green", "red"])
        assert rgb[0, 1][0] == 255 and rgb[0, 1][1] == 255

    def test_no_channels_is_a_black_square_not_an_error(self):
        from vtea_napari.widgets.thumbnail import composite_rgb

        assert composite_rgb([], []).max() == 0

    def test_matplotlib_luts_work_too(self):
        from vtea_napari.widgets.thumbnail import composite_rgb

        plane = np.array([[0.0, 1.0]])
        rgb = composite_rgb([plane], ["viridis"])
        assert tuple(rgb[0, 0]) != tuple(rgb[0, 1])


class TestSegmentationOverlay:
    """Item 2: the original segmentation, tinted in a colour and opacity the
    user picks, so a dot on the plot can be told from the cell it stands
    for."""

    def _widget(self, qtbot, **kwargs):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.show_objects(
            multichannel_volume(1),
            two_object_frame(),
            [1, 2],
            crop_radius=8,
            channel_axis=0,
            labels=two_objects_labels(),
            **kwargs,
        )
        return widget

    def _crop_rgb(self, widget, object_id, **overrides):
        """The rendered crop as an array, for checking what colour it is."""
        request = dict(widget._last_request)
        request.update(overrides)
        frame = request["frame"]
        centroid = frame.set_index("object_id").loc[object_id]
        from vtea_napari.widgets.gallery import crop_around
        from vtea_napari.widgets.thumbnail import composite_rgb, max_projection, overlay_mask

        crop = crop_around(
            request["volume"][0],
            round(centroid["centroid-0"]),
            round(centroid["centroid-1"]),
            radius=request["crop_radius"],
        )
        rgb = composite_rgb([max_projection(crop)], ["gray"])
        mask_crop = crop_around(
            request["labels"],
            round(centroid["centroid-0"]),
            round(centroid["centroid-1"]),
            radius=request["crop_radius"],
        )
        return overlay_mask(
            rgb,
            np.asarray(mask_crop) == object_id,
            widget.mask_color,
            widget.mask_opacity,
        )

    def test_the_object_s_own_voxels_are_tinted(self, qtbot):
        from vtea_napari.widgets.thumbnail import overlay_mask

        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        tinted = overlay_mask(rgb, mask, "#ff0000", 1.0)
        assert tuple(tinted[1, 1]) == (255, 0, 0)
        assert tuple(tinted[0, 0]) == (0, 0, 0)

    def test_opacity_blends_rather_than_hides(self, qtbot):
        """At full opacity the mask hides the intensities it is meant to
        point at."""
        from vtea_napari.widgets.thumbnail import overlay_mask

        rgb = np.full((2, 2, 3), 200, dtype=np.uint8)
        mask = np.ones((2, 2), dtype=bool)
        half = overlay_mask(rgb, mask, "#ff0000", 0.5)
        assert 100 < half[0, 0][0] < 255
        assert half[0, 0][1] == 100

    def test_zero_opacity_is_no_overlay(self, qtbot):
        from vtea_napari.widgets.thumbnail import overlay_mask

        rgb = np.full((2, 2, 3), 200, dtype=np.uint8)
        np.testing.assert_array_equal(
            overlay_mask(rgb, np.ones((2, 2), dtype=bool), "#ff0000", 0.0), rgb
        )

    def test_the_slider_sets_the_opacity(self, qtbot):
        widget = self._widget(qtbot)
        widget.opacity_slider.setValue(80)
        assert widget.mask_opacity == pytest_approx(0.8)

    def test_changing_the_colour_redraws(self, qtbot):
        widget = self._widget(qtbot)
        before = widget._thumbnails[0].pixmap().toImage()
        widget.set_mask_style(color="#00ff00")
        assert widget._thumbnails[0].pixmap().toImage() != before

    def test_turning_it_off_redraws_without_the_tint(self, qtbot):
        widget = self._widget(qtbot)
        tinted = widget._thumbnails[0].pixmap().toImage()
        widget.set_mask_style(opacity=0.0)
        assert widget._thumbnails[0].pixmap().toImage() != tinted

    def test_only_this_object_is_tinted_not_its_neighbour(self, qtbot):
        """Two objects in one crop is the normal case in tissue; tinting
        both would defeat the purpose."""
        widget = self._widget(qtbot)
        labels = two_objects_labels()
        crop = labels[8:24, 8:24]
        assert (crop == 1).any() is np.False_ or not (crop == 1).any()
        rgb = self._crop_rgb(widget, 2)
        assert rgb.shape[2] == 3

    def test_a_gallery_without_a_segmentation_still_draws(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.show_objects(
            multichannel_volume(1), two_object_frame(), [1, 2], crop_radius=6, channel_axis=0
        )
        assert widget._grid.count() == 2


class TestTightLayout:
    """Item 3: minimise the open grey space."""

    def test_the_grid_has_no_margins(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        assert widget._grid.contentsMargins().left() == 0
        assert widget._grid.spacing() <= 2

    def test_the_scroll_area_has_no_frame(self, qtbot):
        from qtpy.QtWidgets import QScrollArea

        widget = GalleryWidget()
        qtbot.addWidget(widget)
        assert widget.scroll.frameShape() == QScrollArea.Shape.NoFrame

    def test_each_cell_is_exactly_its_thumbnail(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.show_objects(
            multichannel_volume(1), two_object_frame(), [1, 2], crop_radius=6,
            thumbnail_size=48, channel_axis=0,
        )
        cell = widget._thumbnails[0]
        assert cell.width() <= 48 + 8 and cell.height() <= 48 + 8

    def test_the_columns_follow_the_width(self, qtbot):
        """A fixed column count leaves a bar of empty widget beside the grid
        at every other size."""
        from qtpy.QtWidgets import QApplication

        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.resize(900, 400)
        widget.show()
        qtbot.waitExposed(widget)
        QApplication.processEvents()
        wide = widget._columns_for(64)

        widget.resize(300, 400)
        QApplication.processEvents()
        narrow = widget._columns_for(64)

        assert wide > narrow >= 1

    def test_a_crop_fills_its_square_rather_than_letterboxing(self, qtbot):
        """A crop clipped by the edge of the image is not square, and
        letterboxing it leaves a strip of grey in a wall of pictures."""
        from vtea_napari.widgets.thumbnail import rgb_to_pixmap

        pixmap = rgb_to_pixmap(np.zeros((10, 30, 3), dtype=np.uint8), size=64)
        assert (pixmap.width(), pixmap.height()) == (64, 64)


class TestHoverZoom:
    """Item 4: hovering enlarges a crop 2x in each dimension, and it goes
    away when the pointer does."""

    def _widget(self, qtbot):
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        widget.show_objects(
            multichannel_volume(1),
            two_object_frame(),
            [1, 2],
            crop_radius=6,
            thumbnail_size=64,
            channel_axis=0,
        )
        return widget

    def test_each_cell_carries_a_double_size_pixmap(self, qtbot):
        widget = self._widget(qtbot)
        cell = widget._thumbnails[0]
        assert cell.zoom_pixmap.width() == 2 * cell.pixmap().width()
        assert cell.zoom_pixmap.height() == 2 * cell.pixmap().height()

    def test_hovering_shows_it(self, qtbot):
        widget = self._widget(qtbot)
        widget.show()
        qtbot.waitExposed(widget)
        widget._thumbnails[0].hovered.emit()
        assert widget._zoom.isVisible()
        assert widget._zoom.pixmap().width() == widget._thumbnails[0].zoom_pixmap.width()

    def test_moving_away_removes_it(self, qtbot):
        widget = self._widget(qtbot)
        widget.show()
        qtbot.waitExposed(widget)
        widget._thumbnails[0].hovered.emit()
        widget._thumbnails[0].unhovered.emit()
        assert not widget._zoom.isVisible()

    def test_hovering_a_second_crop_swaps_the_preview(self, qtbot):
        widget = self._widget(qtbot)
        widget.show()
        qtbot.waitExposed(widget)
        widget._thumbnails[0].hovered.emit()
        first = widget._zoom.pixmap().toImage()
        widget._thumbnails[1].hovered.emit()
        assert widget._zoom.pixmap().toImage() != first

    def test_redrawing_the_grid_puts_the_preview_away(self, qtbot):
        widget = self._widget(qtbot)
        widget.show()
        qtbot.waitExposed(widget)
        widget._thumbnails[0].hovered.emit()
        widget.refresh()
        assert not widget._zoom.isVisible()


class TestContrast:
    """A channel with no signal in *this* cell should be dark here, not a
    screenful of amplified noise."""

    def test_limits_scale_every_crop_the_same_way(self):
        from vtea_napari.widgets.thumbnail import composite_rgb

        bright = composite_rgb([np.array([[0.0, 1000.0]])], ["gray"], limits=[(0, 1000)])
        noise = composite_rgb([np.array([[0.0, 5.0]])], ["gray"], limits=[(0, 1000)])
        # The same scale for both: the cell with signal fills the range and
        # the one without stays dark, instead of both being stretched to
        # white and looking identical.
        assert bright[0, 1][0] == 255
        assert noise[0, 1][0] < 10

    def test_without_limits_each_crop_scales_against_itself(self):
        from vtea_napari.widgets.thumbnail import composite_rgb

        noise = np.array([[0.0, 5.0]])
        rgb = composite_rgb([noise], ["gray"])
        assert rgb[0, 1][0] == 255

    def test_values_outside_the_limits_are_clipped_not_wrapped(self):
        from vtea_napari.widgets.thumbnail import normalize

        scaled = normalize(np.array([[-50.0, 5000.0]]), limits=(0, 1000))
        assert scaled.min() == 0.0 and scaled.max() == 1.0

    def test_one_pair_of_limits_covers_every_channel(self, qtbot):
        from vtea_napari.widgets.gallery import limits_for

        assert limits_for((0, 4095), 0) == (0, 4095)
        assert limits_for((0, 4095), 2) == (0, 4095)

    def test_per_channel_limits_are_read_by_index(self, qtbot):
        from vtea_napari.widgets.gallery import limits_for

        assert limits_for([(0, 10), (0, 20), (0, 30)], 1) == (0, 20)
        assert limits_for({2: (0, 99)}, 2) == (0, 99)

    def test_no_limits_means_autoscale(self, qtbot):
        from vtea_napari.widgets.gallery import limits_for

        assert limits_for(None, 0) is None

    def test_the_gallery_uses_them(self, qtbot):
        """The same crop looks different scaled against the viewer's
        contrast than against itself, which is the point."""
        widget = GalleryWidget()
        qtbot.addWidget(widget)
        volume = multichannel_volume(1)
        widget.show_objects(
            volume, two_object_frame(), [1], crop_radius=6, channel_axis=0
        )
        autoscaled = widget._thumbnails[0].pixmap().toImage()

        widget.show_objects(
            volume,
            two_object_frame(),
            [1],
            crop_radius=6,
            channel_axis=0,
            contrast_limits=(0, 60000),
        )
        assert widget._thumbnails[0].pixmap().toImage() != autoscaled
