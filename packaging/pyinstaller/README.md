# Standalone runtime (PyInstaller)

Bundles `vtea-napari` (napari + the VTEA plugin, opened straight to the
Protocol Builder) into a single-folder distributable per OS, so end users
can run VTEA without installing Python. Built by
`.github/workflows/release.yml` on every push to `main` (and on-demand via
`workflow_dispatch`) and published as GitHub Release assets under an
internally-versioned tag (see that workflow's header comment) - the
binaries themselves are never committed to this repo (they're a few
hundred MB; see the repo root README for why).

Two variants are published per OS:

| Asset | Size (zipped) | Cellpose | GPU |
|---|---|---|---|
| `vtea-napari-<os>.zip` | ~290 MB | via an external torch (below) | **yes**, with a CUDA torch |
| `vtea-napari-<os>-deeplearning.zip` | ~550 MB | built in, no setup | no - CPU-only by construction |

Pick the slim build if you have a GPU or already have a PyTorch
environment; pick the deep-learning build if you want Cellpose to work
offline with nothing to configure.

## Archive formats and file count

Each build is published twice:

| Format | Slim bundle | Notes |
|---|---|---|
| `.7z` | 243 MB | Solid LZMA2. ~31% smaller and ~31% faster to extract. Needs [7-Zip](https://www.7-zip.org/) on Windows. |
| `.zip` | 352 MB | Universal - Windows opens it with no extra tooling. |

A zip stores every file as its own deflate stream, so thousands of small
files cost thousands of independent decompressions; a solid 7z archive
decompresses as one stream. Measured on the slim bundle: 16s → 11s to
extract, on top of the smaller download.

The spec also prunes two trees that are pure build-time weight and dominated
the file count (see `_PRUNE_PREFIXES`): `jedi/third_party` (5,534 typeshed
`.pyi` stubs) and, in the deep-learning build, `torch/include` (9,196 C++
headers). That takes the slim bundle from 9,003 to 3,467 files and the
deep-learning one from 21,009 to 6,277.

**Where that actually helps:** on Linux the two extract in the same time
(15s vs 16s measured - ext4 file creation is cheap, and the time is
dominated by decompressing ~350 MB either way). The win is on Windows,
where each extracted file costs an NTFS create plus a Defender real-time
scan, so file count drives extraction time far more than bytes do. If
extraction is still slow there, adding a Defender exclusion for the folder
you unpack into is the single biggest remaining lever - that is a property
of the machine, not something a build can fix.

## Using a GPU

GPU acceleration is deliberately *not* delivered by bundling CUDA, for two
reasons found by measuring rather than assuming:

- **It doesn't fit.** PyPI's default Linux torch is a CUDA build and pulls
  ~2.4 GB of `nvidia/*` plus 686 MB of `triton` through PyInstaller's
  dependency walk - a 4.9 GB bundle, over the 2 GiB cap on a GitHub Release
  asset. Deleting those payloads afterwards does not yield a CPU build: a
  CUDA torch calls `_preload_cuda_deps()` on import and dies with
  `Failed to load dynlib/dll ... libtorch_global_deps.so`. The spec
  hard-fails on a CUDA torch rather than building something unshippable.
- **A bundled torch can never be replaced.** PyInstaller's frozen importer
  sits ahead of the normal path finder, so `import torch` in a frozen app
  always resolves to the bundled copy - putting another on `PYTHONPATH`
  does nothing (verified against a build that bundled one). Whichever torch
  is baked in is the only one that will ever load, so bundling the CPU
  build would permanently rule out GPU.

So the slim build ships no torch and resolves one at runtime from a
directory you control (`vtea_napari.runtime`). Either:

```bash
# let VTEA install one - pick the CUDA build matching your driver
vtea-napari --install-torch cu121     # or cu124, or cpu
```

```bash
# ...or point it at an environment you already have
export VTEA_TORCH_PATH=/path/to/conda/env/lib/python3.11/site-packages
```

Then confirm what it picked up:

```bash
vtea-napari --self-test
# ok: torch 2.x.y (CUDA 12.1) works - from /path/to/.../torch/__init__.py
#      torch.cuda.is_available() = True
```

`vtea_core.classification`'s CNN follows the same rule as Cellpose: present
in the deep-learning build, and in the slim build once an external torch is
configured.

Note that Cellpose downloads its pretrained weights on first use in either
variant - they are not part of any bundle, so the first segmentation needs
network access.

## Files

- `vtea-napari.spec` — the PyInstaller spec. See its comments for why the
  `collect_all()`/`copy_metadata()` calls are there: napari's own built-in
  plugins (`napari_builtins`, `napari_svg`) live in separate top-level
  packages that PyInstaller's static analysis doesn't reach on its own, and
  npe2's entry-point-based plugin discovery (both napari's and
  `vtea-napari`'s) needs each package's metadata explicitly copied into the
  frozen bundle or it fails at runtime with a bare
  `TypeError: 'NoneType' object is not callable` - the plugin widget
  factory silently resolves to `None`. Both were found and fixed by
  actually building and running the result, not guessed upfront.
- `launcher.py` — the entry script PyInstaller bundles; a thin wrapper
  around `vtea_napari.app:main`. It calls `sys.exit(main())` so a failed
  `--self-test` actually reports a non-zero exit status.

## Missing package metadata: the bug class that has bitten this build three times

Three separate shipped-or-nearly-shipped failures here have all been the
same root cause — a package read its own (or a plugin's) metadata via
`importlib.metadata` at runtime, and that metadata wasn't in the bundle:

1. napari's plugin manager couldn't find `vtea-napari`'s `napari.manifest`
   entry point → `add_plugin_dock_widget` raised a bare
   `TypeError: 'NoneType' object is not callable`.
2. Same for `napari-svg` / `napari-console`.
3. **Shipped in v0.1.0:** `imageio/__init__.py` runs
   `__version__ = importlib.metadata.version("imageio")` at import time.
   `imageio` is only imported when `napari_builtins.io` loads, which only
   happens the first time something reads a file — so the app launched
   perfectly and then died with
   `PackageNotFoundError: No package metadata was found for imageio` the
   moment a user dragged a TIFF in.

The fix for (3) is `copy_metadata(..., recursive=True)`, which walks each
distribution's dependency graph instead of relying on a hand-maintained
list. Enumerating by hand is what let (3) through: the list only contained
the packages whose failures had already been observed.

**If you add a dependency that reads its own version at import time, you
should not need to do anything** — the recursive walk covers it. If you
ever see `PackageNotFoundError` from a frozen build anyway, that package
is reachable at runtime but not a declared dependency of anything in the
`copy_metadata` list, and needs adding there.

## Build locally

```bash
pip install -e "packages/vtea-core" -e "packages/vtea-napari[standalone]"
scripts/build_standalone.sh
# -> dist/vtea-napari/
```

## Verify a build

Start with the self-test — it's the check that catches the failure mode
above, and it needs no display:

```bash
# Linux / macOS
QT_QPA_PLATFORM=offscreen dist/vtea-napari/vtea-napari --self-test
# Windows
dist\vtea-napari\vtea-napari.exe --self-test
```

It resolves every npe2 command the app depends on (napari's file reader
plus both VTEA dock widgets) and reads a real TIFF back through napari's
reader plugin, then exits 0 or 1. Both CI legs run it before the launch
check below.

Note that the Windows bundle is built windowed (`console=False`), so it has
no stdout to print to — there the **exit code is the whole result**. The
self-test is written to survive `sys.stdout` being `None` rather than
crashing on it.

Then confirm it actually launches:

```bash
# Linux, no display available (CI, headless dev box):
QT_QPA_PLATFORM=offscreen dist/vtea-napari/vtea-napari
# Windows:
dist\vtea-napari\vtea-napari.exe
```

A successful run opens a napari window with the Protocol Builder docked.
Launching alone is a weak check — it proves imports at startup work and
nothing more, which is exactly why v0.1.0 passed CI with a broken file
reader. Prefer the self-test.
OpenGL-context warnings in a headless/offscreen environment (no real GPU)
are expected and not a packaging bug - confirmed by comparing against an
unpackaged `pip install`ed run in the same environment, which shows the
identical warnings.

### Known limitation: Windows CI has no headless OpenGL

`release.yml`'s smoke test runs each built binary under
`QT_QPA_PLATFORM=offscreen` and checks its output for a crash. On the
Linux runner this works cleanly - Mesa's llvmpipe backs `offscreen` mode
with real (if slow) software OpenGL. GitHub's `windows-latest` runners
have no such software GL implementation, so a packaged app that reaches
vispy's canvas setup fails there with
`OpenGL.error.GLError(err=1282, description=b'invalid operation', ...)`
from `glGetIntegerv(GL_MAX_TEXTURE_SIZE)` - even though the exact same
`.exe` launches fine on a real desktop with an actual GPU/display. This
was confirmed by pulling the Windows CI job's traceback: the app had
already created the napari main window and vispy canvas (proving the
plugin/DLL bundling itself was fine) before failing on that one GL call.

The smoke-test step recognizes this specific failure signature and treats
it as a passing (but `::warning::`-flagged) result rather than failing the
build - it only hard-fails on tracebacks that don't match this known
signature, so a genuine Windows packaging regression still blocks the
release.
