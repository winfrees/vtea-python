"""The voxel-size control: read it from the image where the file recorded
one, ask for it where it didn't."""

import numpy as np
from vtea_core.data import FROM_METADATA, FROM_USER, Spacing

from vtea_napari.widgets.spacing_control import (
    UNKNOWN_TEXT,
    SpacingControl,
    SpacingDialog,
    axis_names,
)


def _stub_dialog(monkeypatch, spacing, accepted=True):
    """Replace the modal dialog with one that answers immediately - a real
    one just hangs a headless run."""
    from qtpy.QtWidgets import QDialog

    seen = {}

    class FakeDialog:
        def __init__(self, current, parent=None):
            seen["current"] = current

        def exec(self):
            return QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected

        def spacing(self):
            return spacing

    monkeypatch.setattr("vtea_napari.widgets.spacing_control.SpacingDialog", FakeDialog)
    return seen


class FakeLayer:
    def __init__(self, scale):
        self.scale = scale


class TestReadingFromTheImage:
    def test_it_starts_unknown(self, qtbot):
        control = SpacingControl()
        qtbot.addWidget(control)
        assert control.spacing() is None
        assert control.text() == UNKNOWN_TEXT

    def test_a_real_scale_is_taken_from_the_layer(self, qtbot):
        control = SpacingControl()
        qtbot.addWidget(control)
        control.read_from_layer(FakeLayer([1.0, 0.28, 0.28]))

        assert control.spacing().values == (1.0, 0.28, 0.28)
        assert control.spacing().source == FROM_METADATA
        assert "0.28" in control.text()

    def test_an_all_ones_scale_stays_unknown(self, qtbot):
        """napari's default when the file carries no scale - believing it
        would be how anisotropy goes unnoticed."""
        control = SpacingControl()
        qtbot.addWidget(control)
        control.read_from_layer(FakeLayer([1.0, 1.0, 1.0]))

        assert control.spacing().is_known is False
        assert control.text() == UNKNOWN_TEXT

    def test_a_value_typed_by_hand_survives_switching_layers(self, qtbot):
        control = SpacingControl()
        qtbot.addWidget(control)
        control.set_spacing(Spacing((1.0, 0.1, 0.1), source=FROM_USER))

        control.read_from_layer(FakeLayer([1.0, 0.28, 0.28]))

        assert control.spacing().values == (1.0, 0.1, 0.1)

    def test_a_metadata_value_is_replaced_by_another_layer_s(self, qtbot):
        control = SpacingControl()
        qtbot.addWidget(control)
        control.read_from_layer(FakeLayer([1.0, 0.28, 0.28]))
        control.read_from_layer(FakeLayer([2.0, 0.11, 0.11]))

        assert control.spacing().values == (2.0, 0.11, 0.11)

    def test_a_layer_without_a_scale_is_ignored(self, qtbot):
        control = SpacingControl()
        qtbot.addWidget(control)
        control.read_from_layer(object())  # no .scale
        control.read_from_layer(None)
        assert control.spacing() is None


class TestAsking:
    def test_editing_stores_what_was_entered(self, qtbot, monkeypatch):
        control = SpacingControl()
        qtbot.addWidget(control)
        _stub_dialog(monkeypatch, Spacing((1.5, 0.2, 0.2), source=FROM_USER))

        assert control.edit() is True
        assert control.spacing().values == (1.5, 0.2, 0.2)
        assert control.spacing().source == FROM_USER

    def test_cancelling_leaves_it_alone(self, qtbot, monkeypatch):
        control = SpacingControl()
        qtbot.addWidget(control)
        control.read_from_layer(FakeLayer([1.0, 0.28, 0.28]))
        _stub_dialog(monkeypatch, Spacing((9.0, 9.0, 9.0)), accepted=False)

        assert control.edit() is False
        assert control.spacing().values == (1.0, 0.28, 0.28)

    def test_ensure_known_asks_when_it_is_not_set(self, qtbot, monkeypatch):
        control = SpacingControl()
        qtbot.addWidget(control)
        _stub_dialog(monkeypatch, Spacing((1.0, 0.3, 0.3), source=FROM_USER))

        assert control.ensure_known() is True
        assert control.spacing().is_known

    def test_ensure_known_does_not_ask_when_it_is(self, qtbot, monkeypatch):
        control = SpacingControl()
        qtbot.addWidget(control)
        control.read_from_layer(FakeLayer([1.0, 0.28, 0.28]))
        seen = _stub_dialog(monkeypatch, Spacing((9.0, 9.0, 9.0)))

        assert control.ensure_known() is True
        assert seen == {}  # never opened

    def test_ensure_known_reports_failure_if_the_user_declines(self, qtbot, monkeypatch):
        control = SpacingControl()
        qtbot.addWidget(control)
        _stub_dialog(monkeypatch, Spacing((1.0, 1.0, 1.0)), accepted=False)

        assert control.ensure_known() is False

    def test_the_dialog_is_prefilled_with_the_current_value(self, qtbot, monkeypatch):
        control = SpacingControl()
        qtbot.addWidget(control)
        control.set_spacing(Spacing((1.0, 0.28, 0.28), source=FROM_METADATA))
        seen = _stub_dialog(monkeypatch, Spacing((1.0, 0.28, 0.28)))

        control.edit()
        assert seen["current"].values == (1.0, 0.28, 0.28)

    def test_a_change_is_announced(self, qtbot):
        control = SpacingControl()
        qtbot.addWidget(control)
        heard = []
        control.spacing_changed.connect(heard.append)
        control.set_spacing(Spacing((1.0, 0.3, 0.3)))
        assert heard and heard[0].values == (1.0, 0.3, 0.3)


class TestTheDialogItself:
    def test_one_spin_per_axis_named_for_it(self, qtbot):
        dialog = SpacingDialog(Spacing((1.0, 0.25, 0.25)))
        qtbot.addWidget(dialog)
        assert len(dialog.spins) == 3
        assert [spin.value() for spin in dialog.spins] == [1.0, 0.25, 0.25]

    def test_what_it_returns_is_marked_as_the_user_s(self, qtbot):
        """So it is never mistaken for something the file supplied."""
        dialog = SpacingDialog(Spacing.unknown(3))
        qtbot.addWidget(dialog)
        dialog.spins[0].setValue(2.0)
        spacing = dialog.spacing()

        assert spacing.values[0] == 2.0
        assert spacing.source == FROM_USER
        assert spacing.is_known is True

    def test_the_unit_is_editable(self, qtbot):
        dialog = SpacingDialog(Spacing((1.0, 1.0, 1.0), unit="µm"))
        qtbot.addWidget(dialog)
        dialog.unit_edit.setText("nm")
        assert dialog.spacing().unit == "nm"

    def test_axis_names_are_readable_for_the_usual_shapes(self):
        assert axis_names(3) == ("Z", "Y", "X")
        assert axis_names(2) == ("Y", "X")
        assert axis_names(7)[0] == "axis 0"


class TestInTheBuilder:
    def _builder(self, qtbot, scale=None):
        from napari.components import ViewerModel

        from vtea_napari.widgets.protocol_builder import ProtocolBuilderWidget

        viewer = ViewerModel()
        volume = np.zeros((4, 12, 12))
        volume[:, 1:4, 1:4] = 100.0
        kwargs = {"scale": scale} if scale is not None else {}
        viewer.add_image(volume, name="src", **kwargs)
        widget = ProtocolBuilderWidget(napari_viewer=viewer)
        qtbot.addWidget(widget)
        return widget

    def test_the_control_reads_the_layer_s_scale(self, qtbot):
        widget = self._builder(qtbot, scale=(1.0, 0.28, 0.28))
        assert widget.spacing_control.spacing().values == (1.0, 0.28, 0.28)

    def test_an_uncalibrated_image_leaves_it_unset(self, qtbot):
        widget = self._builder(qtbot)
        assert widget.spacing_control.spacing().is_known is False

    def test_it_reaches_the_shared_session(self, qtbot):
        widget = self._builder(qtbot, scale=(1.0, 0.28, 0.28))
        assert widget.session.spacing.values == (1.0, 0.28, 0.28)

    def test_it_is_seeded_into_the_run_context(self, qtbot):
        widget = self._builder(qtbot, scale=(1.0, 0.28, 0.28))
        assert widget.seed_context()["spacing"].values == (1.0, 0.28, 0.28)

    def test_a_measurement_step_picks_it_up_and_reports_a_volume(self, qtbot):
        from vtea_core.workflow import Step

        widget = self._builder(qtbot, scale=(2.0, 0.5, 0.5))
        widget.pipeline.add_step(
            Step.for_function(
                "segmentation", "threshold_mask", params={"method": "fixed", "value": 50.0}
            )
        )
        widget.pipeline.add_step(Step.for_function("segmentation", "label_components"))
        widget.run_processing()

        measure = widget.analysis_pipeline.add_step(
            Step.for_function(
                "measurements",
                "extract_measurements",
                available=set(widget.last_context) | {"spacing"},
            )
        )
        widget.run_single_step(measure)

        table = widget.results_table()
        assert "volume" in table.columns
        # 3x3 in x/y over 4 z-slices, at 2.0 x 0.5 x 0.5 per voxel.
        assert table.loc[0, "volume"] == table.loc[0, "count"] * 0.5
