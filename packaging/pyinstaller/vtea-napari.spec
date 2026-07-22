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
for distribution_name in ("napari", "napari-svg", "napari-console", "vtea-napari", "vtea-core"):
    datas += copy_metadata(distribution_name)

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
