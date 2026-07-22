# Standalone runtime (PyInstaller)

Bundles `vtea-napari` (napari + the VTEA plugin, opened straight to the
Protocol Builder) into a single-folder distributable per OS, so end users
can run VTEA without installing Python. Built by
`.github/workflows/release.yml` on tagged releases and published as
GitHub Release assets - the binaries themselves are never committed to
this repo (they're a few hundred MB; see the repo root README for why).

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
  around `vtea_napari.app:main`.

## Build locally

```bash
pip install -e "packages/vtea-core" -e "packages/vtea-napari[standalone]"
scripts/build_standalone.sh
# -> dist/vtea-napari/
```

## Verify a build

```bash
# Linux, no display available (CI, headless dev box):
QT_QPA_PLATFORM=offscreen dist/vtea-napari/vtea-napari
# Windows:
dist\vtea-napari\vtea-napari.exe
```

A successful run opens a napari window with the Protocol Builder docked.
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
