# Standalone runtime (PyInstaller)

Bundles `vtea-napari` (napari + the VTEA plugin, opened straight to the
Protocol Builder) into a single-folder distributable per OS, so end users
can run VTEA without installing Python. Built by
`.github/workflows/release.yml` on every push to `main` (and on-demand via
`workflow_dispatch`) and published as GitHub Release assets under an
internally-versioned tag (see that workflow's header comment) - the
binaries themselves are never committed to this repo (they're a few
hundred MB; see the repo root README for why).

Deep learning (`cellpose_segmentation`, `vtea_core.classification`'s CNN)
is **not** included - the standalone build deliberately excludes the
`deeplearning` extra to keep the bundle size and build time reasonable.
Those features need a regular `pip install "vtea-core[deeplearning]"`.

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
