"""Standalone application entry point: launches napari with the VTEA
Protocol Builder already open.

This is the entry script the PyInstaller-built standalone runtime bundles
(see packaging/pyinstaller/), and is also installed as a `vtea-napari`
console script for anyone with a normal pip install who just wants to
launch straight into VTEA rather than opening napari and finding the
plugin manually.

`--self-test` runs a headless check of the parts of a packaged build that
PyInstaller can most easily get wrong (dynamically-resolved plugin
commands and the package metadata they need) and exits non-zero if any of
them fail. It exists because those failures are invisible to "does the
app start?" - the app starts fine and then dies the first time a user
opens a file. See packaging/pyinstaller/README.md.
"""

from __future__ import annotations

import sys

# npe2 command ids that a frozen build must be able to resolve. Each one is
# imported lazily at runtime from a string python_name, so PyInstaller's
# static analysis can't see it and a missing module or missing package
# metadata only surfaces when the user triggers that feature.
_REQUIRED_COMMANDS = (
    "napari.get_reader",  # needs napari_builtins.io -> imageio (+ its metadata)
    "vtea-napari.protocol_builder",
    "vtea-napari.object_explorer",
)


def _emit(message: str) -> None:
    """print() that survives a windowed (console=False) PyInstaller build.

    PyInstaller sets sys.stdout/sys.stderr to None when it builds without a
    console - which is how the Windows bundle is built - so a bare print()
    there raises AttributeError on None.write and would crash the very
    check it's reporting on. The exit code is the real verdict; this output
    is diagnostics for platforms that can show it.
    """
    stream = sys.stdout if sys.stdout is not None else sys.stderr
    if stream is None:
        return
    try:
        print(message, file=stream)
    except Exception:  # noqa: BLE001, S110 - see below
        # Deliberately silent: this *is* the output path, so logging a
        # failure to write output would be circular. The self-test's verdict
        # travels by exit code, which is unaffected.
        pass


def self_test() -> int:
    """Headless check that a packaged build can actually resolve its
    plugin commands and read an image. Returns a process exit code."""
    import tempfile
    from pathlib import Path

    import numpy as np
    import tifffile
    from napari.plugins.io import read_data_with_plugins
    from npe2 import PluginManager

    failures: list[str] = []

    plugin_manager = PluginManager.instance()
    plugin_manager.discover()

    for command_id in _REQUIRED_COMMANDS:
        try:
            resolved = plugin_manager.commands.get(command_id)
        except Exception as exc:  # noqa: BLE001 - report every failure, don't stop at the first
            failures.append(f"could not resolve command {command_id!r}: {exc}")
            continue
        # A command that resolves to None is the failure mode that made
        # add_plugin_dock_widget raise a bare "'NoneType' object is not
        # callable" in an earlier build.
        if resolved is None:
            failures.append(f"command {command_id!r} resolved to None")
        else:
            _emit(f"ok: resolved {command_id}")

    # End-to-end image read through napari's reader plugin machinery - the
    # path a user hits by dragging a file onto the viewer.
    try:
        expected = np.arange(16, dtype=np.uint16).reshape(4, 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vtea-self-test.tif"
            tifffile.imwrite(path, expected)
            layer_data, _plugin = read_data_with_plugins([str(path)], plugin="napari", stack=False)
        actual = np.asarray(layer_data[0][0])
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            failures.append(f"read back {actual.shape} data that did not match what was written")
        else:
            _emit("ok: read a TIFF back through napari's reader plugin")
    except Exception as exc:  # noqa: BLE001 - this is the check, report it rather than traceback
        failures.append(f"reading a TIFF failed: {type(exc).__name__}: {exc}")

    if failures:
        _emit("\nself-test FAILED:")
        for failure in failures:
            _emit(f"  - {failure}")
        return 1
    _emit("\nself-test passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()

    import napari

    viewer = napari.Viewer()
    viewer.window.add_plugin_dock_widget("vtea-napari", "Protocol Builder")
    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
