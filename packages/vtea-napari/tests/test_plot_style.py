"""The plot's style helper pane: size, shape and transparency of the dots.

These are about being able to *see* the analysis - a few thousand
overlapping opaque circles hide the structure underneath them - so the
controls drive the plot directly rather than re-running anything.
"""

import numpy as np
import pandas as pd

from vtea_napari.widgets.plot import DEFAULT_SIZE_RANGE, MARKERS, ScatterPlotWidget
from vtea_napari.widgets.plot_style import PlotStylePanel


def make_frame():
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3],
            "mean": [1.0, 2.0, 3.0],
            "count": [10.0, 20.0, 30.0],
        }
    )


def make_panel(qtbot):
    plot = ScatterPlotWidget()
    qtbot.addWidget(plot)
    plot.set_data(make_frame())
    panel = PlotStylePanel(plot)
    qtbot.addWidget(panel)
    return panel


class TestInitialState:
    def test_it_starts_from_the_plot_s_current_style(self, qtbot):
        panel = make_panel(qtbot)
        assert panel.size_spin.value() == panel.plot.point_size
        assert panel.alpha_slider.value() == int(panel.plot.alpha * 100)
        assert panel.min_size_spin.value() == DEFAULT_SIZE_RANGE[0]
        assert panel.max_size_spin.value() == DEFAULT_SIZE_RANGE[1]

    def test_it_offers_marker_shapes_by_name(self, qtbot):
        panel = make_panel(qtbot)
        choices = [panel.marker_combo.itemText(i) for i in range(panel.marker_combo.count())]
        assert choices == list(MARKERS)
        assert "circle" in choices


class TestDrivingThePlot:
    def test_changing_the_size_reaches_the_plot(self, qtbot):
        panel = make_panel(qtbot)
        panel.size_spin.setValue(48.0)
        assert panel.plot.point_size == 48.0

    def test_changing_the_opacity_reaches_the_plot(self, qtbot):
        panel = make_panel(qtbot)
        panel.alpha_slider.setValue(30)
        assert panel.plot.alpha == 0.3
        assert panel.alpha_label.text() == "30%"

    def test_changing_the_shape_reaches_the_plot_as_a_matplotlib_marker(self, qtbot):
        panel = make_panel(qtbot)
        panel.marker_combo.setCurrentText("square")
        assert panel.plot.marker == "s"

    def test_changing_the_size_range_reaches_the_plot(self, qtbot):
        panel = make_panel(qtbot)
        panel.min_size_spin.setValue(20.0)
        panel.max_size_spin.setValue(60.0)
        assert panel.plot.size_range == (20.0, 60.0)

    def test_a_backwards_range_is_read_in_the_order_meant(self, qtbot):
        """Typing the larger number first would otherwise invert the size
        encoding silently."""
        panel = make_panel(qtbot)
        panel.min_size_spin.setValue(90.0)
        panel.max_size_spin.setValue(20.0)
        assert panel.plot.size_range == (20.0, 90.0)

    def test_opacity_cannot_be_turned_all_the_way_off(self, qtbot):
        """Fully transparent points are indistinguishable from no data."""
        panel = make_panel(qtbot)
        panel.alpha_slider.setValue(0)
        assert panel.plot.alpha > 0.0


class TestItAffectsWhatIsDrawn:
    def test_the_size_range_changes_the_encoded_sizes(self, qtbot):
        panel = make_panel(qtbot)
        panel.plot.size_combo.setCurrentText("count")
        panel.min_size_spin.setValue(15.0)
        panel.max_size_spin.setValue(45.0)

        sizes, _ = panel.plot._point_sizes()
        np.testing.assert_allclose([sizes.min(), sizes.max()], [15.0, 45.0])

    def test_the_style_survives_a_data_refresh(self, qtbot):
        panel = make_panel(qtbot)
        panel.alpha_slider.setValue(40)
        panel.marker_combo.setCurrentText("diamond")

        panel.plot.set_data(make_frame())

        assert panel.plot.alpha == 0.4
        assert panel.plot.marker == "D"
