"""Smoke test: the package installs and imports cleanly (no napari runtime needed)."""

import vtea_napari


def test_version_is_set():
    assert vtea_napari.__version__
