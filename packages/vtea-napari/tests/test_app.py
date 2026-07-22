from unittest.mock import MagicMock, patch

from vtea_napari.app import main


def test_main_opens_viewer_with_protocol_builder_and_runs_event_loop():
    fake_napari = MagicMock()
    fake_viewer = fake_napari.Viewer.return_value

    with patch.dict("sys.modules", {"napari": fake_napari}):
        main()

    fake_napari.Viewer.assert_called_once_with()
    fake_viewer.window.add_plugin_dock_widget.assert_called_once_with("vtea-napari", "Protocol Builder")
    fake_napari.run.assert_called_once_with()
