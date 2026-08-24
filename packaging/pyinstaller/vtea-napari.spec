# PyInstaller spec for the standalone VTEA runtime.
#
# Bundles vtea_napari.app:main (launches napari with the Protocol Builder
# already open) into a single-folder distributable per OS. napari's plugin
# discovery (npe2) and Qt/vispy backends are dynamic in ways PyInstaller's
# static import analysis doesn't fully see on its own, hence the explicit
# collect_all()/hiddenimports below - trimmed down to what was actually
# needed by building and running the result, not guessed upfront.
#
# Build with: pyinstaller packaging/pyinstaller/vtea-napari.spec
# (see packaging/pyinstaller/README.md for the full build/verify flow)

from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = []
binaries = []
hiddenimports = []

# napari's own built-in plugins (image reading, layer types, SVG export)
# are separate top-level importable packages within napari's/napari-svg's
# distributions, not submodules of `napari` itself - collect_all("napari")
# doesn't reach them, and without them napari fails at startup with
# "'napari_builtins' declared in entrypoint... could not be imported".
for package in (
    "napari",
    "napari_builtins",
    "napari_svg",
    "napari_console",
    "vispy",
    "magicgui",
    "npe2",
    "vtea_napari",
    "vtea_core",
):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# Entry-point-based plugin discovery (napari's own plugin manager, and
# vtea-napari's own napari.manifest registration) reads package metadata
# via importlib.metadata at runtime - without it explicitly copied into the
# frozen bundle, `add_plugin_dock_widget` fails with a bare
# "TypeError: 'NoneType' object is not callable" (the widget factory
# resolves to None because the plugin's entry point can't be found).
#
# recursive=True is load-bearing, not belt-and-braces: it copies metadata
# for each distribution's dependencies too. Several of them read their own
# metadata at import time - imageio's __init__ does
# `__version__ = importlib.metadata.version("imageio")` - so without the
# recursive walk the app starts fine and then dies with
# "PackageNotFoundError: No package metadata was found for imageio" the
# first time a user opens an image, since that's what finally imports
# napari_builtins.io. Enumerating the affected distributions by hand is
# how that bug shipped in v0.1.0; let the dependency graph decide instead.
for distribution_name in ("napari", "napari-svg", "napari-console", "vtea-napari", "vtea-core"):
    datas += copy_metadata(distribution_name, recursive=True)

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="vtea-napari",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="vtea-napari",
)
