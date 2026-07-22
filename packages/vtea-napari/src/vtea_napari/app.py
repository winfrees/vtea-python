"""Standalone application entry point: launches napari with the VTEA
Protocol Builder already open.

This is the entry script the PyInstaller-built standalone runtime bundles
(see packaging/pyinstaller/), and is also installed as a `vtea-napari`
console script for anyone with a normal pip install who just wants to
launch straight into VTEA rather than opening napari and finding the
plugin manually.
"""

from __future__ import annotations


def main() -> None:
    import napari

    viewer = napari.Viewer()
    viewer.window.add_plugin_dock_widget("vtea-napari", "Protocol Builder")
    napari.run()


if __name__ == "__main__":
    main()
