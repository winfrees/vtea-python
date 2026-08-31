"""Phase L8: the GUI side of running data larger than memory.

Two things matter here and the first is easy to get wrong invisibly: the
builder must stop reading a lazily-loaded layer into memory the moment a
protocol runs, which would undo everything the out-of-core work is for. The
second is that the user can see, and change, how the data is being divided.
"""

import numpy as np
import pytest

import dask.array as da
from vtea_core.blocked import MemoryBudget, detect_memory_budget
from vtea_napari.widgets.memory_control import MemoryControl, MemoryDialog
from vtea_napari.widgets.protocol_builder import ProtocolBuilderWidget
from vtea_core.workflow import Step

GIB = 1024**3


class CountingArray:
    """A stand-in for a stored array that objects to being read whole."""

    def __init__(self, data):
        self._data = data
        self.reads = 0
        self.shape = data.shape
        self.dtype = data.dtype
        self.ndim = data.ndim

    def __getitem__(self, index):
        self.reads += 1
        return self._data[index]

    def __array__(self, dtype=None, copy=None):
        raise AssertionError("the layer was materialized when it should not have been")


@pytest.fixture
def builder(qtbot):
    """A builder attached to a real viewer, closed afterwards.

    `napari_viewer=` rather than positional: the first argument is the
    pipeline.
    """
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


def add_volume(viewer, shape=(8, 32, 32), *, lazy=False):
    """Put a volume in the viewer for the builder to find.

    `add_labels` rather than `add_image`: this napari build tears an Image
    layer down through a vispy path that needs a GL context, which a
    headless test runner has not got. The builder reads `layer.data` and
    does not care which kind of layer produced it, so nothing under test is
    being avoided - the rest of this package's viewer tests do the same.
    """
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, shape).astype(np.int32)
    if lazy:
        data = da.from_array(data, chunks=(4, 16, 16))
    layer = viewer.add_labels(data, name="volume")
    viewer.layers.selection.active = layer
    return layer


class TestLazySource:
    def test_source_data_does_not_materialize_the_layer(self, builder):
        viewer, widget = builder
        add_volume(viewer, lazy=True)
        widget.refresh_sources()
        data = widget.source_data()
        assert isinstance(data, da.Array), "a lazy layer came back as a NumPy array"

    def test_active_image_still_materializes_for_the_in_memory_path(self, builder):
        viewer, widget = builder
        add_volume(viewer, lazy=True)
        widget.refresh_sources()
        assert isinstance(widget.active_image(), np.ndarray)

    def test_nothing_selected_is_not_an_error(self, builder):
        _viewer, widget = builder
        assert widget.source_data() is None
        assert widget.active_image() is None
        assert widget.tile_plan() is None


class TestTilePlan:
    def test_data_that_fits_plans_one_tile(self, builder):
        viewer, widget = builder
        add_volume(viewer)
        widget.refresh_sources()
        widget.memory_control.set_budget(MemoryBudget(8 * GIB))
        plan = widget.tile_plan()
        assert plan is not None
        assert plan.is_single_tile

    def test_a_tight_budget_divides_it(self, builder):
        viewer, widget = builder
        add_volume(viewer, shape=(16, 128, 128))
        widget.refresh_sources()
        widget.pipeline.add_step(
            Step.for_function("imageprocessing", "gaussian_blur", params={"sigma": 1.0})
        )
        widget.memory_control.set_budget(MemoryBudget(200_000))
        plan = widget.tile_plan()
        assert plan is not None and not plan.is_single_tile

    def test_the_channel_axis_is_held_out_of_the_tiling(self, builder):
        viewer, widget = builder
        add_volume(viewer, shape=(3, 16, 64, 64))
        widget.refresh_sources()
        widget.pipeline.channel_axis = 0
        assert widget.spatial_axes(4) == (1, 2, 3)
        widget.memory_control.set_budget(MemoryBudget(400_000))
        plan = widget.tile_plan()
        assert plan.tile[0] == 3, "a tile must hold every channel of the voxels it holds"

    def test_the_plan_survives_a_protocol_it_cannot_size(self, builder):
        # An unplannable protocol is a thing to report, not a crash inside a
        # widget that repaints on every keystroke.
        viewer, widget = builder
        add_volume(viewer, shape=(16, 128, 128))
        widget.refresh_sources()
        widget.pipeline.add_step(Step.for_function("segmentation", "watershed_split"))
        widget.memory_control.set_budget(MemoryBudget(4096))
        assert widget.tile_plan() is None


class TestMemoryControl:
    def test_it_starts_from_what_was_detected(self, qtbot):
        control = MemoryControl()
        qtbot.addWidget(control)
        assert control.budget().total_bytes == detect_memory_budget().total_bytes
        assert "Memory:" in control.button.text()

    def test_it_reports_the_plan_rather_than_the_budget_once_there_is_one(
        self, builder, qtbot
    ):
        viewer, widget = builder
        add_volume(viewer, shape=(16, 128, 128))
        widget.refresh_sources()
        widget.pipeline.add_step(
            Step.for_function("imageprocessing", "gaussian_blur", params={"sigma": 1.0})
        )
        widget.memory_control.set_budget(MemoryBudget(200_000))
        plan = widget.refresh_plan()
        assert plan is not None and not plan.is_single_tile
        assert "tiles of" in widget.memory_control.button.text()
        # And the tooltip carries the whole explanation, including what
        # bounded the tile size.
        assert "bounded by" in widget.memory_control.button.toolTip()

    def test_it_says_when_the_data_fits(self, builder):
        viewer, widget = builder
        add_volume(viewer)
        widget.refresh_sources()
        widget.memory_control.set_budget(MemoryBudget(8 * GIB))
        widget.refresh_plan()
        assert "fits in one piece" in widget.memory_control.button.text()

    def test_changing_the_budget_announces_it(self, qtbot):
        control = MemoryControl()
        qtbot.addWidget(control)
        seen = []
        control.budget_changed.connect(seen.append)
        control.set_budget(MemoryBudget(2 * GIB))
        assert len(seen) == 1
        assert seen[0].total_bytes == 2 * GIB

    def test_the_dialog_can_set_a_budget_by_hand(self, qtbot):
        dialog = MemoryDialog(MemoryBudget(8 * GIB, fraction=0.6))
        qtbot.addWidget(dialog)
        dialog.use_detected.setChecked(False)
        dialog.amount.setValue(3.0)
        dialog.fraction.setValue(0.5)
        budget = dialog.budget()
        assert budget.total_bytes == 3 * GIB
        assert budget.fraction == 0.5
        assert budget.source == "user"

    def test_the_dialog_can_put_it_back(self, qtbot):
        dialog = MemoryDialog(MemoryBudget(1 * GIB, source="user"))
        qtbot.addWidget(dialog)
        dialog.use_detected.setChecked(True)
        assert dialog.budget().total_bytes == detect_memory_budget().total_bytes


class TestBlockedRun:
    def protocol(self, widget):
        widget.pipeline.add_step(
            Step.for_function("imageprocessing", "gaussian_blur", params={"sigma": 1.0})
        )
        widget.pipeline.add_step(
            Step.for_function(
                "segmentation",
                "threshold_mask",
                params={"method": "otsu"},
                available={"volume"},
            )
        )

    def test_a_tight_budget_runs_out_of_core_and_agrees_with_memory(self, builder):
        viewer, widget = builder
        layer = add_volume(viewer, shape=(16, 96, 96))
        self.protocol(widget)

        widget.memory_control.set_budget(MemoryBudget(8 * GIB))
        widget.refresh_sources()
        in_memory = widget.run_processing()
        expected = np.asarray(in_memory["mask"])

        widget.memory_control.set_budget(MemoryBudget(400_000))
        widget.last_context = {}
        blocked = widget.run_processing()
        assert widget._scratch is not None, "a tight budget did not take the blocked path"
        np.testing.assert_array_equal(np.asarray(blocked["mask"]), expected)
        widget._close_scratch()

    def test_it_says_how_many_tiles_when_it_finishes(self, builder):
        viewer, widget = builder
        add_volume(viewer, shape=(16, 96, 96))
        self.protocol(widget)
        widget.refresh_sources()
        widget.memory_control.set_budget(MemoryBudget(400_000))
        widget.run_processing()
        assert "tiles" in widget.status_label.text()
        widget._close_scratch()

    def test_progress_names_the_step_and_the_tile(self, builder):
        viewer, widget = builder
        add_volume(viewer, shape=(16, 96, 96))
        self.protocol(widget)
        widget.refresh_sources()
        widget.memory_control.set_budget(MemoryBudget(400_000))
        widget._on_block_progress("gaussian_blur_1", 3, 27)
        assert "gaussian_blur_1" in widget.status_label.text()
        assert "3" in widget.status_label.text()

    def test_a_failure_is_reported_rather_than_raised(self, builder):
        viewer, widget = builder
        add_volume(viewer, shape=(16, 96, 96))
        widget.pipeline.add_step(Step.for_function("segmentation", "label_components"))
        widget.refresh_sources()
        widget.memory_control.set_budget(MemoryBudget(400_000))
        widget.run_processing()
        # label_components needs a mask nothing produced; the widget says so
        # rather than taking napari down with it.
        assert widget.status_label.text()
        widget._close_scratch()

    def test_the_scratch_store_is_replaced_rather_than_accumulated(self, builder):
        viewer, widget = builder
        add_volume(viewer, shape=(16, 96, 96))
        self.protocol(widget)
        widget.refresh_sources()
        widget.memory_control.set_budget(MemoryBudget(400_000))
        widget.run_processing()
        first = widget._scratch.path
        widget.run_processing()
        assert widget._scratch.path != first
        assert not first.exists(), "the previous run's scratch was left behind"
        widget._close_scratch()
