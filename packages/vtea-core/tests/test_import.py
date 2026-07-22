"""Smoke test: the package installs and imports cleanly."""

import vtea_core


def test_version_is_set():
    assert vtea_core.__version__
