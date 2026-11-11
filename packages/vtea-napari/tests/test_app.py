from unittest.mock import MagicMock, patch

from vtea_napari.app import main, self_test


def test_main_opens_viewer_with_protocol_builder_and_runs_event_loop():
    fake_napari = MagicMock()
    fake_viewer = fake_napari.Viewer.return_value

    with patch.dict("sys.modules", {"napari": fake_napari}):
        assert main([]) == 0

    fake_napari.Viewer.assert_called_once_with()
    fake_viewer.window.add_plugin_dock_widget.assert_called_once_with("vtea-napari", "Protocol Builder")
    fake_napari.run.assert_called_once_with()


def test_self_test_flag_runs_self_test_instead_of_launching_the_gui():
    fake_napari = MagicMock()

    with patch.dict("sys.modules", {"napari": fake_napari}):
        with patch("vtea_napari.app.self_test", return_value=0) as fake_self_test:
            assert main(["--self-test"]) == 0

    fake_self_test.assert_called_once_with()
    fake_napari.Viewer.assert_not_called()


def test_self_test_propagates_a_failing_exit_code():
    with patch("vtea_napari.app.self_test", return_value=1):
        assert main(["--self-test"]) == 1


def test_a_self_test_that_raises_reports_and_fails_instead_of_raising(capsys):
    """A windowed Windows build shows an unhandled exception in a message box
    and waits for a click; in CI that hung the job for six hours. The error is
    reported and the exit code is 1 instead."""
    boom = ModuleNotFoundError("No module named 'napari_resources.resources'")
    with patch("vtea_napari.app.self_test", side_effect=boom):
        assert main(["--self-test"]) == 1
    out = capsys.readouterr().out
    assert "self-test failed with an error" in out and "napari_resources.resources" in out


def test_self_test_passes_in_a_working_install(qtbot):
    """The check the packaged build runs. In a normal install this should
    always pass; it's the frozen bundle where a plugin command can fail to
    resolve or a dependency's metadata can go missing (see
    packaging/pyinstaller/README.md)."""
    assert self_test() == 0
