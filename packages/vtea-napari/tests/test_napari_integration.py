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


def test_the_object_explorer_floats_when_opened_as_a_plugin(qtbot):
    """A scatter plot docked into napari's side panel is unusable at the
    width it gets there, and gating means working between the plot and the
    image."""
    viewer = napari.Viewer(show=False)
    qtbot.addWidget(viewer.window._qt_window)
    try:
        dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
            "vtea-napari", "Object Explorer"
        )
        # Floating is deferred to the next event-loop turn, because napari
        # creates the widget before putting it in a dock.
        qtbot.waitUntil(dock_widget.isFloating, timeout=2000)
        assert plugin_widget.float_dock() is True
    finally:
        viewer.close()


def test_both_plugin_widgets_share_one_analysis_session(qtbot):
    """The builder computes; the explorer plots. They are two views of one
    analysis, so results must cross between them without either pane having
    to be open when the other runs."""
    viewer = napari.Viewer(show=False)
    qtbot.addWidget(viewer.window._qt_window)
    try:
        _, builder = viewer.window.add_plugin_dock_widget("vtea-napari", "Protocol Builder")
        _, explorer = viewer.window.add_plugin_dock_widget("vtea-napari", "Object Explorer")
        assert builder.session is explorer.session
    finally:
        viewer.close()


def test_the_builder_can_open_the_explorer(qtbot):
    viewer = napari.Viewer(show=False)
    qtbot.addWidget(viewer.window._qt_window)
    try:
        _, builder = viewer.window.add_plugin_dock_widget("vtea-napari", "Protocol Builder")
        explorer = builder.open_object_explorer()
        assert isinstance(explorer, ExplorerWidget)
        assert explorer.session is builder.session
        # Asking twice raises the existing one rather than stacking a second.
        assert builder.open_object_explorer() is explorer
    finally:
        viewer.close()
