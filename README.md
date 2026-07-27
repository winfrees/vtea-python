# VTEA (Python)

Python port of [VTEA](https://github.com/winfrees/volumetric-tissue-exploration-analysis)
(Volumetric Tissue Exploration and Analysis) — a 3D tissue cytometry tool
originally developed as a Java/ImageJ1 plugin at the Indiana Center for
Biological Microscopy. This port replaces the Java application; see
[`docs/PORT_PLAN.md`](docs/PORT_PLAN.md) for the full rationale, dependency
mapping, and phased roadmap.

## Repository layout

This is a monorepo with two independently installable packages:

- [`packages/vtea-core`](packages/vtea-core) — headless core analysis
  library (data model, I/O, segmentation, measurements, clustering,
  dimensionality reduction, gating, classification). No GUI dependency;
  usable from scripts, Jupyter, a CLI, or HPC batch jobs.
- [`packages/vtea-napari`](packages/vtea-napari) — [napari](https://napari.org)
  plugin providing the interactive GUI, as a thin layer over `vtea-core`.

```
docs/PORT_PLAN.md      Full porting plan and roadmap
tests/golden/          Golden-dataset parity fixtures (Java vs. Python outputs)
packages/vtea-core/    Headless analysis library
packages/vtea-napari/  napari plugin GUI
packaging/pyinstaller/ Standalone runtime build (see "Standalone runtime" below)
```

## Status

**Phases 0-4 done**, plus a standalone runtime (see below). Phases 0-3
(package skeletons/CI; `VolumeDataset`/TIFF/Zarr I/O; the algorithm core -
segmentation, measurements, clustering, reduction, gates, image
preprocessing; and Cellpose segmentation + a `classification` module) - see
`packages/vtea-core/README.md` for the full module-by-module status.

Phase 4 (napari GUI) landed two dock widgets:

- **`ProtocolBuilderWidget`** - the protocol builder. The scope call there
  was **Option A**, a fully functional GUI clone: scoping it against the
  actual Java source corrected an earlier assumption - `vtea.protocol`
  isn't a free-form node-graph editor, it's an ordered stack of step cards
  built from a category menu (no drag-drop/wire code exists in the Java
  source at all), which made Option A smaller than originally estimated.
  Built on `vtea_core.workflow` (`Step`/`Pipeline`, the headless engine,
  shared between the GUI and scripts/notebooks); `run_pipeline()` also
  drives thumbnail previews on each step's card.
- **`ExplorerWidget`** ("Object Explorer") - the `MicroExplorer` equivalent:
  a scatter plot with click-to-draw polygon gating, a gate management
  table, LUT/colormap point-coloring, and a per-object thumbnail gallery.
  Backed by a new `vtea_core.gates.Gate`/`GateSet` model with real gate
  hierarchy, something the Java original never actually had (see
  `docs/PORT_PLAN.md`'s "Object Explorer" section for the full comparison,
  including which Java gate-related classes turned out to be dead code).

Both are registered as real napari plugin dock widgets, verified by
actually loading them through `napari.Viewer` in tests, not just
constructing them standalone. See `docs/PORT_PLAN.md`'s "Protocol builder:
Option A" and "Object Explorer" sections, and `packages/vtea-napari/README.md`,
for the full design writeups.

## Development

Each package uses a standard `src/`-layout with `pyproject.toml` (hatchling).
To install a package in editable mode for development:

```bash
pip install -e "packages/vtea-core[dev]"
pip install -e "packages/vtea-napari[dev]"
```

Run tests with `pytest` from within each package directory, or see
`.github/workflows/ci.yml` for the full matrix.

## Standalone runtime

For users who don't want to install Python: `.github/workflows/release.yml`
builds a standalone `vtea-napari` bundle (PyInstaller, Linux + Windows) and
publishes it as a [GitHub Release](../../releases) - not committed into
this repo. A ~550 MB binary bundle doesn't belong in git history (every
future clone would pay for it permanently); Releases are the standard
place for built artifacts. The deep-learning extra (Cellpose, PyTorch)
isn't included in the standalone build to keep its size/build time
reasonable - install normally via pip for that.

Versioning is internal to the workflow: it reads the highest existing
`vMAJOR.MINOR.PATCH` git tag, bumps the patch number, tags that commit,
and releases it - no one has to pick or push a version number by hand.
A release is cut on every push to `main` (continuous delivery) and
on-demand via the Actions tab ("Run workflow"); pushing a `v*` tag
manually still works too and is used verbatim (skips the auto-bump), for
a deliberate major/minor version bump.

To build one locally: see [`packaging/pyinstaller/README.md`](packaging/pyinstaller/README.md).

## License

GPL-2.0-only, matching the source Java application. See [`LICENSE`](LICENSE).
