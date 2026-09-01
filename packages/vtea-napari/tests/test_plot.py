import numpy as np
import pandas as pd
import pytest

from vtea_napari.widgets.plot import ScatterPlotWidget, scale_sizes


class FakeMouseEvent:
    """Duck-types matplotlib's MouseEvent: xdata/ydata/button/dblclick are
    all the ScatterPlotWidget click handler reads."""

    def __init__(self, xdata, ydata, button=1, dblclick=False):
        self.xdata = xdata
        self.ydata = ydata
        self.button = button
        self.dblclick = dblclick


def make_frame():
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3],
            "mean": [1.0, 2.0, 3.0],
            "count": [10.0, 20.0, 30.0],
            "sum": [100.0, 200.0, 300.0],
        }
    )


class TestSetData:
    def test_populates_axis_combos_with_numeric_columns(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())

        choices = {widget.x_combo.itemText(i) for i in range(widget.x_combo.count())}
        assert choices == {"object_id", "mean", "count", "sum"}

    def test_defaults_to_first_two_columns(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        assert widget.x_column == "object_id"
        assert widget.y_column == "mean"

    def test_explicit_axes_are_honored(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame(), x_column="count", y_column="sum")
        assert widget.x_column == "count"
        assert widget.y_column == "sum"


class TestGateDrawing:
    def test_three_clicks_and_a_double_click_emits_a_triangle(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())

        received = []
        widget.gate_drawn.connect(lambda vertices: received.append(vertices))

        widget._on_click(FakeMouseEvent(0, 0))
        widget._on_click(FakeMouseEvent(10, 0))
        widget._on_click(FakeMouseEvent(5, 10))
        widget._on_click(FakeMouseEvent(5, 10, dblclick=True))

        assert len(received) == 1
        np.testing.assert_array_equal(received[0], [[0, 0], [10, 0], [5, 10]])

    def test_pending_vertices_cleared_after_emit(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())

        for point in [(0, 0), (10, 0), (5, 10)]:
            widget._on_click(FakeMouseEvent(*point))
        widget._on_click(FakeMouseEvent(5, 10, dblclick=True))

        assert widget._pending_vertices == []

    def test_fewer_than_three_vertices_does_not_emit(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())

        received = []
        widget.gate_drawn.connect(lambda vertices: received.append(vertices))

        widget._on_click(FakeMouseEvent(0, 0))
        widget._on_click(FakeMouseEvent(10, 0, dblclick=True))

        assert received == []

    def test_right_click_cancels_pending_gate(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())

        widget._on_click(FakeMouseEvent(0, 0))
        widget._on_click(FakeMouseEvent(10, 0))
        widget._on_click(FakeMouseEvent(5, 10, button=3))

        assert widget._pending_vertices == []

    def test_click_outside_axes_is_ignored(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())

        widget._on_click(FakeMouseEvent(None, None))

        assert widget._pending_vertices == []


class TestColorBy:
    def test_none_is_default(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        assert widget.color_column is None

    def test_selecting_a_column_sets_color_by(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.color_combo.setCurrentText("sum")
        assert widget.color_column == "sum"


class TestColormap:
    def test_defaults_to_first_option(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        assert widget.colormap == widget.colormap_combo.itemText(0)

    def test_selecting_a_colormap_updates_it(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.colormap_combo.setCurrentText("plasma")
        assert widget.colormap == "plasma"

    def test_redraw_with_color_by_and_colormap_does_not_crash(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.color_combo.setCurrentText("sum")
        widget.colormap_combo.setCurrentText("magma")
        widget._redraw()  # should not raise


class TestScaleSizes:
    """Mapping a feature onto marker areas, and back again for the legend."""

    def test_the_smallest_and_largest_values_span_the_range(self):
        sizes, _ = scale_sizes(np.array([0.0, 5.0, 10.0]), 10.0, 110.0)
        np.testing.assert_allclose(sizes, [10.0, 60.0, 110.0])

    def test_the_inverse_recovers_the_feature_values(self):
        """What lets the size legend be labelled in the feature's own units
        rather than in matplotlib's points²."""
        values = np.array([2.0, 4.0, 9.0])
        sizes, to_value = scale_sizes(values, 8.0, 120.0)
        np.testing.assert_allclose(to_value(sizes), values)

    def test_a_constant_feature_gets_the_middle_of_the_range(self):
        """There is no spread to encode; a gradient over identical values
        would be a lie."""
        sizes, to_value = scale_sizes(np.array([7.0, 7.0, 7.0]), 10.0, 110.0)
        np.testing.assert_allclose(sizes, [60.0, 60.0, 60.0])
        np.testing.assert_allclose(to_value(sizes), [7.0, 7.0, 7.0])

    def test_nans_do_not_break_the_scale(self):
        sizes, _ = scale_sizes(np.array([1.0, np.nan, 3.0]), 10.0, 110.0)
        assert np.isfinite(sizes).all()
        assert sizes.min() >= 10.0 and sizes.max() <= 110.0

    def test_an_all_nan_feature_falls_back_to_one_size(self):
        sizes, _ = scale_sizes(np.array([np.nan, np.nan]), 10.0, 110.0)
        np.testing.assert_allclose(sizes, [60.0, 60.0])


class TestSizeBy:
    def test_the_combo_offers_none_plus_every_numeric_feature(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        choices = [widget.size_combo.itemText(i) for i in range(widget.size_combo.count())]
        assert choices == ["(none)", "object_id", "mean", "count", "sum"]

    def test_nothing_is_size_encoded_by_default(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        assert widget.size_column is None
        sizes, to_value = widget._point_sizes()
        assert sizes == widget.point_size
        assert to_value is None

    def test_choosing_a_feature_scales_the_points(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.size_combo.setCurrentText("count")

        assert widget.size_column == "count"
        sizes, _ = widget._point_sizes()
        # count is 10/20/30, so the points span the whole size range.
        assert sizes[0] == widget.size_range[0]
        assert sizes[-1] == widget.size_range[1]

    def test_the_choice_survives_a_re_run(self, qtbot):
        """Size is as much part of "the plot I am looking at" as the axes."""
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.size_combo.setCurrentText("count")
        widget.color_combo.setCurrentText("mean")

        widget.set_data(make_frame())

        assert widget.size_column == "count"
        assert widget.color_column == "mean"

    def test_a_feature_that_vanished_is_dropped_rather_than_kept_stale(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.size_combo.setCurrentText("sum")

        widget.set_data(make_frame().drop(columns=["sum"]))

        assert widget.size_column is None


class TestScaleBars:
    """A colour gradient or a size gradient means nothing without the values
    it stands for."""

    def test_a_colour_feature_gets_a_labelled_colorbar(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.color_combo.setCurrentText("count")

        assert widget._colorbar is not None
        assert widget._colorbar.ax.get_ylabel() == "count"

    def test_no_colour_feature_means_no_colorbar(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        assert widget._colorbar is None

    def test_turning_colour_off_removes_the_bar(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.color_combo.setCurrentText("count")
        widget.color_combo.setCurrentText("(none)")

        assert widget._colorbar is None
        assert len(widget.figure.axes) == 1

    def test_redrawing_does_not_stack_up_colorbars(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.color_combo.setCurrentText("count")
        for _ in range(5):
            widget._redraw()
        # One for the scatter, one for the bar - not six.
        assert len(widget.figure.axes) == 2

    def test_a_size_feature_gets_a_legend_titled_after_it(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.size_combo.setCurrentText("count")

        legend = widget.ax.get_legend()
        assert legend is not None
        assert legend.get_title().get_text() == "count"

    def test_the_size_legend_is_labelled_in_the_features_own_units(self, qtbot):
        """Not in matplotlib's points², which would be a number about the
        drawing rather than about the data."""
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.size_combo.setCurrentText("count")

        labels = [text.get_text() for text in widget.ax.get_legend().get_texts()]
        numbers = [float(label.replace("$\\mathdefault{", "").rstrip("}$")) for label in labels]
        assert min(numbers) >= 10.0
        assert max(numbers) <= 30.0

    def test_no_size_feature_means_no_legend(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        assert widget.ax.get_legend() is None

    def test_both_bars_can_be_shown_at_once(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.color_combo.setCurrentText("mean")
        widget.size_combo.setCurrentText("count")

        assert widget._colorbar is not None
        assert widget.ax.get_legend() is not None

    def test_a_constant_size_feature_does_not_crash(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        frame = make_frame()
        frame["flat"] = 1.0
        widget.set_data(frame)
        widget.size_combo.setCurrentText("flat")  # must not raise


class TestPointStyle:
    def test_defaults(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        assert widget.alpha == 1.0
        assert widget.marker == "o"

    def test_set_point_style_applies_each_setting(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.set_point_style(point_size=42.0, alpha=0.25, marker="s", size_range=(5.0, 50.0))

        assert widget.point_size == 42.0
        assert widget.alpha == 0.25
        assert widget.marker == "s"
        assert widget.size_range == (5.0, 50.0)

    def test_omitted_settings_are_left_alone(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_point_style(alpha=0.5)
        assert widget.alpha == 0.5
        assert widget.marker == "o"

    def test_the_size_range_drives_the_encoded_sizes(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(make_frame())
        widget.size_combo.setCurrentText("count")
        widget.set_point_style(size_range=(20.0, 40.0))

        sizes, _ = widget._point_sizes()
        assert sizes.min() == 20.0
        assert sizes.max() == 40.0


class TestAspectRatio:
    """Item 2: the plot keeps 4:3 whatever the pane does. A plot whose
    proportions follow the furniture is a plot whose shape is an artefact of
    how somebody dragged a dock."""

    def test_a_tall_box_letterboxes_the_canvas(self, qtbot):
        from qtpy.QtWidgets import QWidget

        from vtea_napari.widgets.plot import AspectRatioBox

        box = AspectRatioBox(QWidget())
        qtbot.addWidget(box)
        _x, y, width, height = box.child_geometry(800, 800)
        assert (width, height) == (800, 600)
        assert width / height == pytest.approx(4 / 3)
        assert y > 0  # centred in the leftover height

    def test_a_wide_box_pillarboxes_the_canvas(self, qtbot):
        from qtpy.QtWidgets import QWidget

        from vtea_napari.widgets.plot import AspectRatioBox

        box = AspectRatioBox(QWidget())
        qtbot.addWidget(box)
        x, _y, width, height = box.child_geometry(1200, 600)
        assert (width, height) == (800, 600)
        assert width / height == pytest.approx(4 / 3)
        assert x > 0  # centred in the leftover width

    def test_resizing_keeps_the_ratio(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.resize(900, 700)
        widget.show()
        qtbot.waitExposed(widget)
        for size in ((900, 700), (600, 900), (1200, 400)):
            widget.canvas_box.resize(*size)
            ratio = widget.canvas.width() / max(widget.canvas.height(), 1)
            assert ratio == pytest.approx(4 / 3, abs=0.02)

    def test_the_canvas_is_inside_the_box(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        assert widget.canvas.parent() is widget.canvas_box


class TestDiscreteLut:
    """Item 3: a cluster id is not a quantity. A gradient over it puts
    cluster 2 "between" 1 and 3, which is a statement about a numbering."""

    def _categorical_frame(self):
        return pd.DataFrame(
            {
                "object_id": [1, 2, 3, 4],
                "mean": [10.5, 20.25, 30.75, 41.5],
                "kmeans_1": [0, 1, 2, 1],
                "in_gate": [True, False, True, False],
            }
        )

    def test_a_measurement_is_continuous(self, qtbot):
        from vtea_napari.widgets.plot import is_discrete

        assert is_discrete(self._categorical_frame()["mean"]) is False

    def test_a_cluster_id_is_discrete(self, qtbot):
        from vtea_napari.widgets.plot import is_discrete

        assert is_discrete(self._categorical_frame()["kmeans_1"]) is True

    def test_a_boolean_class_is_discrete(self, qtbot):
        from vtea_napari.widgets.plot import is_discrete

        assert is_discrete(self._categorical_frame()["in_gate"]) is True

    def test_a_whole_numbered_measurement_is_not_mistaken_for_categories(self, qtbot):
        """A voxel count is whole numbers too - what makes an id an id is
        that it starts at 0 (or -1) and stays small."""
        from vtea_napari.widgets.plot import is_discrete

        assert is_discrete(pd.Series([120.0, 340.0, 512.0])) is False

    def test_the_mode_overrides_the_heuristic_both_ways(self, qtbot):
        from vtea_napari.widgets.plot import LUT_CONTINUOUS, LUT_DISCRETE, is_discrete

        assert is_discrete(pd.Series([0, 1, 2]), LUT_CONTINUOUS) is False
        assert is_discrete(pd.Series([1.5, 2.5]), LUT_DISCRETE) is True

    def test_a_discrete_column_gets_a_legend_and_no_colorbar(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(self._categorical_frame())
        widget.color_combo.setCurrentText("kmeans_1")

        assert widget.color_is_discrete() is True
        assert widget._colorbar is None
        assert widget.ax.get_legend() is not None

    def test_a_continuous_column_still_gets_its_colorbar(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(self._categorical_frame())
        widget.color_combo.setCurrentText("mean")

        assert widget.color_is_discrete() is False
        assert widget._colorbar is not None

    def test_forcing_continuous_brings_the_colorbar_back(self, qtbot):
        from vtea_napari.widgets.plot import LUT_CONTINUOUS

        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(self._categorical_frame())
        widget.color_combo.setCurrentText("kmeans_1")
        widget.set_lut_mode(LUT_CONTINUOUS)

        assert widget._colorbar is not None

    def test_the_analysis_can_declare_a_column_categorical(self, qtbot):
        """The feature catalog knows a column came from a clustering step
        even when its values would not give it away."""
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        frame = self._categorical_frame()
        frame["odd_ids"] = [100, 200, 300, 200]
        widget.set_data(frame)
        widget.color_combo.setCurrentText("odd_ids")
        assert widget.color_is_discrete() is False

        widget.set_discrete_columns({"odd_ids"})
        assert widget.color_is_discrete() is True

    def test_one_colour_per_level(self, qtbot):
        from vtea_napari.widgets.plot import discrete_colors

        colors = discrete_colors([0, 1, 2], "tab10")
        assert len(colors) == 3
        assert len(set(map(tuple, [c[:3] for c in colors]))) == 3

    def test_a_gradient_is_sampled_across_its_whole_range(self, qtbot):
        """Ten clusters get ten different colours, not ten neighbouring
        shades of the same one."""
        from vtea_napari.widgets.plot import discrete_colors

        colors = discrete_colors(list(range(4)), "viridis")
        assert len({tuple(color[:3]) for color in colors}) == 4

    def test_the_lut_mode_is_remembered_with_the_view(self, qtbot):
        from vtea_napari.widgets.plot import LUT_DISCRETE

        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(self._categorical_frame())
        widget.set_lut_mode(LUT_DISCRETE)
        state = widget.view_state()

        restored = ScatterPlotWidget()
        qtbot.addWidget(restored)
        restored.set_data(self._categorical_frame())
        restored.apply_view_state(state)
        assert restored.lut_mode == LUT_DISCRETE


class TestRings:
    """Item 4's plot half: objects inside a region painted on the image are
    ringed, in that region's own colour."""

    def _frame(self):
        return pd.DataFrame(
            {
                "object_id": [1, 2, 3, 4],
                "mean": [1.0, 2.0, 3.0, 4.0],
                "count": [10.0, 20.0, 30.0, 40.0],
            }
        )

    def _ring_collections(self, widget):
        # A ring scatter is the one drawn without a face colour.
        return [
            collection
            for collection in widget.ax.collections
            if len(getattr(collection, "get_facecolors", lambda: [])()) == 0
        ]

    def test_no_rings_by_default(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(self._frame())
        assert self._ring_collections(widget) == []

    def test_one_ring_group_per_region(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(self._frame())
        widget.set_rings([0, 1, 2, 1], {1: "#ff0000", 2: "#00ff00"})
        assert len(self._ring_collections(widget)) == 2

    def test_objects_outside_every_region_are_not_ringed(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(self._frame())
        widget.set_rings([0, 0, 0, 1], {1: "#ff0000"})
        ringed = self._ring_collections(widget)
        assert len(ringed) == 1
        assert len(ringed[0].get_offsets()) == 1

    def test_a_single_colour_rings_everything_in_it(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(self._frame())
        widget.set_rings([1, 1, 0, 2], "#123456")
        assert len(self._ring_collections(widget)) == 2  # one group per region id

    def test_rings_are_drawn_larger_than_their_points(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(self._frame())
        widget.set_rings([1, 1, 1, 1], {1: "#ff0000"})
        ring = self._ring_collections(widget)[0]
        assert ring.get_sizes()[0] > widget.point_size

    def test_clearing_them(self, qtbot):
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(self._frame())
        widget.set_rings([1, 1, 1, 1], {1: "#ff0000"})
        widget.set_rings(None)
        assert self._ring_collections(widget) == []

    def test_rings_survive_a_colour_encoding(self, qtbot):
        """The LUT owns the fill and the image gate owns the outline, so
        both can be read at once."""
        widget = ScatterPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(self._frame())
        widget.set_rings([1, 0, 1, 0], {1: "#ff0000"})
        widget.color_combo.setCurrentText("mean")
        assert len(self._ring_collections(widget)) == 1
