"""Verifies the plugin actually loads through napari itself (not just that
the widget class works standalone) - i.e. `napari.yaml` is correctly wired
up and napari's plugin manager can find and instantiate it.
"""

import napari

from vtea_napari.widgets import ProtocolBuilderWidget


def test_protocol_builder_loads_as_a_napari_plugin_dock_widget(qtbot):
    viewer = napari.Viewer(show=False)
    qtbot.addWidget(viewer.window._qt_window)
    try:
        _dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget("vtea-napari", "Protocol Builder")
        assert isinstance(plugin_widget, ProtocolBuilderWidget)
    finally:
        viewer.close()
