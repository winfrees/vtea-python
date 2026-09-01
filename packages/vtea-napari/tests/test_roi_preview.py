"""Previewing the protocol over what is on screen.

Two claims worth pinning, because the obvious implementation gets both
wrong. The preview must agree with a full run inside the region it shows -
which means it is a tile of the protocol's own tiling, halo and all, not a
crop run in isolation. And panning must not queue a run per frame.
"""

import numpy as np
import pytest
from vtea_core.workflow import Pipeline, Step
from vtea_napari.widgets.protocol_builder import ProtocolBuilderWidget
from vtea_napari.widgets.roi_preview import (
    PREVIEW_PREFIX,
    PreviewControl,
    Region,
    displayed_level,
    level_scale,
    levels_of,
    visible_region,
)


class FakeLayer:
    """A layer with a view on it, without a GL context to put it in."""

    def __init__(self, data, corners, level=0, factors=None, multiscale=False):
        self.data = data
        self.corner_pixels = np.asarray(corners)
        self.data_level = level
        self.multiscale = multiscale
        self.downsample_factors = factors
        self.name = "fake"


def volume(shape=(8, 64, 64)):
    rng = np.random.default_rng(0)
    return rng.integers(0, 4000, shape).astype(np.uint16)


class TestVisibleRegion:
    def test_the_box_on_screen_becomes_a_region(self):
        layer = FakeLayer(volume(), [[2, 10, 20], [2, 29, 39]])
        region = visible_region(layer)
        assert region.core == (slice(2, 3), slice(10, 30), slice(20, 40))

    def test_an_axis_not_being_displayed_is_the_one_plane_on_screen(self):
        """napari reports the same index twice for it, which is one plane
        rather than none - and the halo around it is what keeps a 3D step's
        answer for that plane right."""
        layer = FakeLayer(volume(), [[5, 0, 0], [5, 63, 63]])
        assert visible_region(layer).core[0] == slice(5, 6)

    def test_the_channel_axis_is_taken_whole(self):
        """A protocol that measures every channel would compute something
        different, not merely somewhere else, on one channel."""
        layer = FakeLayer(volume((3, 8, 64, 64)), [[1, 2, 0, 0], [1, 2, 31, 31]])
        region = visible_region(layer, whole_axes=(0,))
        assert region.core[0] == slice(0, 3)
        assert region.core[1] == slice(2, 3)

    def test_a_multiscale_layer_reads_the_level_on_screen(self):
        """Reading full resolution for a view showing every fourth voxel is
        exactly the I/O the pyramid exists to avoid."""
        levels = [volume((8, 64, 64)), volume((8, 32, 32)), volume((8, 16, 16))]
        layer = FakeLayer(
            levels,
            [[0, 0, 0], [0, 15, 15]],
            level=2,
            factors=[[1, 1, 1], [1, 2, 2], [1, 4, 4]],
            multiscale=True,
        )
        region = visible_region(layer)
        assert region.level == 2
        assert region.core == (slice(0, 1), slice(0, 16), slice(0, 16))
        assert region.scale == (1.0, 4.0, 4.0)

    def test_the_region_places_itself_back_on_the_image(self):
        region = Region(1, (slice(0, 4), slice(8, 16), slice(8, 16)), (1.0, 2.0, 2.0))
        assert region.translate([0, 8, 8]) == [0.0, 16.0, 16.0]

    def test_the_scale_is_measured_rather_than_assumed(self):
        """A store written by another tool is entitled to downsample by
        three, or differently per axis."""
        levels = [volume((8, 64, 64)), volume((8, 21, 21))]
        layer = FakeLayer(
            levels, [[0, 0, 0], [0, 20, 20]], level=1,
            factors=[[1, 1, 1], [1, 3, 3]], multiscale=True,
        )
        assert visible_region(layer).scale == (1.0, 3.0, 3.0)

    def test_a_layer_with_no_view_yet_is_not_an_error(self):
        layer = FakeLayer(volume(), [[0, 0, 0], [0, 0, 0]])
        layer.corner_pixels = None
        assert visible_region(layer) is None

    def test_levels_and_level_of_a_plain_array(self):
        layer = FakeLayer(volume(), [[0, 0, 0], [0, 1, 1]])
        assert len(levels_of(layer)) == 1
        assert displayed_level(layer) == 0
        assert level_scale(layer, 0, 3) == (1.0, 1.0, 1.0)


class TestDebounce:
    def test_a_request_does_not_run_immediately(self, qtbot):
        """A pan emits a camera event per frame; running on each one is the
        difference between a preview and a frozen window."""
        control = PreviewControl(delay_ms=50)
        qtbot.addWidget(control)
        control.checkbox.setChecked(True)
        fired = []
        control.requested.connect(lambda: fired.append(1))
        control.request()
        assert fired == []

    def test_it_runs_once_the_view_is_still(self, qtbot):
        control = PreviewControl(delay_ms=20)
        qtbot.addWidget(control)
        control.checkbox.setChecked(True)
        fired = []
        control.requested.connect(lambda: fired.append(1))
        with qtbot.waitSignal(control.requested, timeout=1000):
            control.request()
        assert fired == [1]

    def test_moving_again_postpones_rather_than_queues(self, qtbot):
        control = PreviewControl(delay_ms=40)
        qtbot.addWidget(control)
        control.checkbox.setChecked(True)
        fired = []
        control.requested.connect(lambda: fired.append(1))
        for _ in range(10):
            control.request()
        with qtbot.waitSignal(control.requested, timeout=1000):
            pass
        assert fired == [1], "each view change queued a run instead of postponing one"

    def test_switching_it_off_stops_it(self, qtbot):
        control = PreviewControl(delay_ms=10)
        qtbot.addWidget(control)
        control.checkbox.setChecked(True)
        fired = []
        control.requested.connect(lambda: fired.append(1))
        control.request()
        control.checkbox.setChecked(False)
        qtbot.wait(60)
        assert fired == []


@pytest.fixture
def builder(qtbot):
    import napari

    viewer = napari.Viewer(show=False)
    qtbot.addWidget(viewer.window._qt_window)
    widget = ProtocolBuilderWidget(napari_viewer=viewer)
    qtbot.addWidget(widget)
    try:
        yield viewer, widget
    finally:
        widget._close_scratch()
        viewer.close()


def add_volume(viewer, data):
    """`add_labels` rather than `add_image`: this napari build tears an
    Image layer down through a vispy path that needs a GL context."""
    layer = viewer.add_labels(data.astype(np.int32), name="volume")
    viewer.layers.selection.active = layer
    return layer


class TestThePreviewItself:
    """`add_labels` throughout, and a protocol ending in a mask: this napari
    build tears an Image layer down through a vispy path that needs a GL
    context, as the rest of this package's viewer tests already note."""

    def protocol(self, widget):
        widget.pipeline.add_step(
            Step.for_function("imageprocessing", "gaussian_blur", params={"sigma": 2.0})
        )
        widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "threshold_mask",
                params={"method": "fixed", "value": 2000.0},
                available={"volume"},
            )
        )

    def test_it_agrees_with_a_full_run_inside_the_region(self, builder):
        """The claim the halo exists for: a crop run in isolation differs
        from the run at its edges, and the edges are what gets looked at."""
        viewer, widget = builder
        data = volume((8, 64, 64))
        layer = add_volume(viewer, data)
        self.protocol(widget)
        widget.refresh_sources()
        layer.corner_pixels = np.array([[4, 16, 16], [4, 47, 47]])

        preview = widget.run_preview()
        whole = widget.pipeline.run({"volume": data, "intensity": data})["mask"]
        np.testing.assert_array_equal(preview, whole[4:5, 16:48, 16:48])

    def test_a_crop_without_the_halo_would_not_have(self, builder):
        """The negative control: the same region run in isolation differs
        at its edges, which is what makes the test above worth having."""
        viewer, widget = builder
        data = volume((8, 64, 64))
        add_volume(viewer, data)
        self.protocol(widget)
        widget.refresh_sources()

        crop = data[4:5, 16:48, 16:48]
        naive = widget.pipeline.run({"volume": crop, "intensity": crop})["mask"]
        whole = widget.pipeline.run({"volume": data, "intensity": data})["mask"]
        assert not np.array_equal(naive, whole[4:5, 16:48, 16:48])

    def test_the_layer_is_named_as_a_preview(self, builder):
        """It sits in the layer list beside committed results, and one that
        looked like a result would be exported and reported as one."""
        viewer, widget = builder
        layer = add_volume(viewer, volume((8, 32, 32)))
        self.protocol(widget)
        widget.refresh_sources()
        layer.corner_pixels = np.array([[2, 0, 0], [2, 31, 31]])
        widget.run_preview()
        names = [existing.name for existing in viewer.layers]
        assert any(name.startswith(PREVIEW_PREFIX) for name in names)

    def test_previews_replace_rather_than_accumulate(self, builder):
        viewer, widget = builder
        layer = add_volume(viewer, volume((8, 32, 32)))
        self.protocol(widget)
        widget.refresh_sources()
        for row in (10, 20):
            layer.corner_pixels = np.array([[2, 0, 0], [2, row, row]])
            widget.run_preview()
        previews = [
            existing for existing in viewer.layers if existing.name.startswith(PREVIEW_PREFIX)
        ]
        assert len(previews) == 1

    def test_it_does_not_touch_the_run_context(self, builder):
        """A preview is not a result: the Show buttons, the plot and the
        tables go on reading whatever the last real run produced."""
        viewer, widget = builder
        layer = add_volume(viewer, volume((8, 32, 32)))
        self.protocol(widget)
        widget.refresh_sources()
        widget.last_context = {"volume": "the real run"}
        layer.corner_pixels = np.array([[2, 0, 0], [2, 31, 31]])
        widget.run_preview()
        assert widget.last_context == {"volume": "the real run"}

    def test_a_preview_layer_is_not_offered_as_a_source(self, builder):
        """Running a protocol on a preview would compound the approximation
        rather than show anything."""
        viewer, widget = builder
        layer = add_volume(viewer, volume((8, 32, 32)))
        self.protocol(widget)
        widget.refresh_sources()
        layer.corner_pixels = np.array([[2, 0, 0], [2, 31, 31]])
        widget.run_preview()
        widget.refresh_sources()
        offered = [
            widget.layer_combo.itemData(index)
            for index in range(widget.layer_combo.count())
        ]
        assert not any(str(name).startswith(PREVIEW_PREFIX) for name in offered)

    def test_a_view_too_large_for_the_budget_is_refused_with_a_reason(self, builder):
        from vtea_core.blocked import MemoryBudget

        viewer, widget = builder
        layer = add_volume(viewer, volume((8, 64, 64)))
        self.protocol(widget)
        widget.refresh_sources()
        widget.memory_control.set_budget(MemoryBudget(4096))
        layer.corner_pixels = np.array([[2, 0, 0], [2, 63, 63]])
        assert widget.run_preview() is None
        assert "zoom in" in widget.preview_control.status.text()

    def test_an_empty_protocol_previews_nothing(self, builder):
        viewer, widget = builder
        add_volume(viewer, volume((8, 32, 32)))
        widget.refresh_sources()
        assert widget.run_preview() is None
        assert "nothing" in widget.preview_control.status.text()

    def test_a_failing_step_is_reported_rather_than_raised(self, builder):
        viewer, widget = builder
        layer = add_volume(viewer, volume((8, 32, 32)))
        widget.pipeline.add_step(
            Step.for_function("segmentation", "label_components", available={"mask"})
        )
        widget.refresh_sources()
        layer.corner_pixels = np.array([[2, 0, 0], [2, 31, 31]])
        assert widget.run_preview() is None
        assert widget.preview_control.status.text()

    def test_the_status_says_what_it_previewed(self, builder):
        viewer, widget = builder
        layer = add_volume(viewer, volume((8, 32, 32)))
        self.protocol(widget)
        widget.refresh_sources()
        layer.corner_pixels = np.array([[2, 0, 0], [2, 31, 31]])
        widget.run_preview()
        assert "preview" in widget.preview_control.status.text()

    def test_a_global_statistic_says_it_was_measured_on_the_view(self, builder):
        """An Otsu threshold over the region on screen is often exactly what
        a user tuning it wants to see, and is never what the full run will
        do. The difference is invisible unless it is stated."""
        viewer, widget = builder
        layer = add_volume(viewer, volume((8, 32, 32)))
        widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "threshold_mask",
                params={"method": "otsu"},
                available={"volume"},
            )
        )
        widget.refresh_sources()
        layer.corner_pixels = np.array([[2, 0, 0], [2, 31, 31]])
        widget.run_preview()
        assert "on the view" in widget.preview_control.status.text()

    def test_a_fixed_threshold_makes_no_such_claim(self, builder):
        viewer, widget = builder
        layer = add_volume(viewer, volume((8, 32, 32)))
        self.protocol(widget)
        widget.refresh_sources()
        layer.corner_pixels = np.array([[2, 0, 0], [2, 31, 31]])
        widget.run_preview()
        assert "on the view" not in widget.preview_control.status.text()


class TestPlacement:
    """Where the preview layer lands. A preview that sits somewhere the user
    is not looking is worse than no preview, and an anisotropic stack -
    which is most of them - is where that goes wrong."""

    def test_the_layers_own_scale_is_kept(self, builder):
        viewer, widget = builder
        layer = add_volume(viewer, volume((8, 32, 32)))
        layer.scale = (2.0, 0.5, 0.5)
        widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "threshold_mask",
                params={"method": "fixed", "value": 2000.0},
                available={"volume"},
            )
        )
        widget.refresh_sources()
        layer.corner_pixels = np.array([[2, 8, 8], [2, 23, 23]])
        widget.run_preview()

        preview = next(
            existing for existing in viewer.layers if existing.name.startswith(PREVIEW_PREFIX)
        )
        assert tuple(preview.scale) == (2.0, 0.5, 0.5)
        assert tuple(preview.translate) == (4.0, 4.0, 4.0)
