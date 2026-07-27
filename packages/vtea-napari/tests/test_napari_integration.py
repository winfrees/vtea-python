"""Verifies the plugin actually loads through napari itself (not just that
the widget class works standalone) - i.e. `napari.yaml` is correctly wired
up and napari's plugin manager can find and instantiate it.
"""

import napari

from vtea_napari.widgets import ExplorerWidget, ProtocolBuilderWidget


def test_protocol_builder_loads_as_a_napari_plugin_dock_widget(qtbot):
    viewer = napari.Viewer(show=False)
    qtbot.addWidget(viewer.window._qt_window)
    try:
        _dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget("vtea-napari", "Protocol Builder")
        assert isinstance(plugin_widget, ProtocolBuilderWidget)
    finally:
        viewer.close()


def test_object_explorer_loads_as_a_napari_plugin_dock_widget_with_viewer_injected(qtbot):
    viewer = napari.Viewer(show=False)
    qtbot.addWidget(viewer.window._qt_window)
    try:
        _dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget("vtea-napari", "Object Explorer")
        assert isinstance(plugin_widget, ExplorerWidget)
        # napari's plugin engine should auto-inject the live viewer via the
        # `napari_viewer` constructor parameter, the same convention
        # magicgui-based widgets use - not a widget we had to pass it to.
        # (napari wraps it in a PublicOnlyProxy, so compare by == not is.)
        assert plugin_widget.viewer == viewer
    finally:
        viewer.close()
