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

import sys

from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = []
binaries = []

# Carry the whole standard library, not just the parts this app's own
# imports reach.
#
# vtea_napari.runtime supports resolving torch/cellpose from an install
# outside the bundle (that's what makes GPU acceleration possible - see its
# docstring). Those external packages import stdlib modules that nothing
# inside the bundle does, and a frozen build only ships the trimmed stdlib
# PyInstaller's analysis found. Chasing them one at a time is a losing game:
# a real external cellpose first failed on 'modulefinder', then on
# 'html.parser', with no way to know what came next. Enumerating
# sys.stdlib_module_names covers the category for a few MB.
#
# The exclusions are GUI/dev-only trees that pull weight (or a tk runtime)
# for no benefit here. Missing names are warnings in PyInstaller, not
# errors, so platform-specific entries are harmless.
_STDLIB_EXCLUDES = {
    "antigravity",
    "this",
    "idlelib",
    "tkinter",
    "turtle",
    "turtledemo",
    "lib2to3",
    "test",
    "pydoc_data",
}
hiddenimports = sorted(sys.stdlib_module_names - _STDLIB_EXCLUDES)

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

# Deep-learning variant: build the same spec in an environment where the
# `deeplearning` extra is installed and Cellpose segmentation is bundled
# too. Detected rather than flag-driven, so the workflow only has to decide
# what to pip install.
try:
    import torch  # noqa: F401

    BUNDLE_DEEPLEARNING = True
except ImportError:
    BUNDLE_DEEPLEARNING = False

if BUNDLE_DEEPLEARNING:
    # The CPU-only torch wheel is required here. This refuses to build
    # against a CUDA one rather than producing something unshippable.
    #
    # PyPI's default Linux torch is a CUDA build: it drags in ~2.4 GB of
    # nvidia/* plus 686 MB of triton through PyInstaller's dependency walk,
    # giving a 4.9 GB bundle - past the 2 GiB cap on a GitHub Release asset.
    # Deleting those payloads afterwards does NOT downgrade it to CPU: a
    # CUDA build's torch/__init__.py calls _preload_cuda_deps() on import
    # and dies with "Failed to load dynlib/dll ... libtorch_global_deps.so".
    # That was tried and produced exactly that broken bundle, hence a hard
    # failure here instead.
    if getattr(torch.version, "cuda", None):
        raise SystemExit(
            f"The deep-learning bundle needs the CPU-only torch wheel, but the "
            f"installed torch is a CUDA build ({torch.__version__}). A CUDA build "
            f"cannot be shipped here (~4.9 GB, over the 2 GiB release-asset cap) and "
            f"cannot be stripped down to CPU. Install the CPU wheel instead:\n"
            f"  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            f"For GPU acceleration see packaging/pyinstaller/README.md "
            f"('Using a GPU') - it is deliberately not solved by bundling CUDA."
        )

    for package in ("torch", "cellpose"):
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hiddenimports
    for distribution_name in ("torch", "cellpose"):
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
