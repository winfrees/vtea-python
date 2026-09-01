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
docs/SAVING_AND_ARCHIVING.md
                       Design for reloadable sessions and FAIR publication
                       bundles (planned; gate JSON and the feature catalog
                       are the parts already built)
docs/OBJECT_ASSOCIATION.md
                       Plan for associating segmentations into cells -
                       derived masks, probabilistic parentage, contested
                       voxels (planned, under review)
docs/LARGE_IMAGES.md   Strategy for datasets larger than RAM - memory
                       budgets, tiling, and the rules for objects a tile
                       boundary cuts in half (built; awaiting validation on
                       GPU hardware and real tissue)
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
  shared between the GUI and scripts/notebooks); each step card runs its own
  step and drives its own thumbnail preview.
- **`ExplorerWidget`** ("Object Explorer") - the `MicroExplorer` equivalent,
  floating over the canvas by default: a scatter plot with click-to-draw
  polygon and two-click rectangle gating, a gate manager (list, JSON
  save/open, per-gate count and means), LUT/colormap point-coloring, and a
  per-object thumbnail gallery. Backed by a new
  `vtea_core.gates.Gate`/`GateSet` model with real gate hierarchy, something
  the Java original never actually had (see `docs/PORT_PLAN.md`'s "Object
  Explorer" section for the full comparison, including which Java
  gate-related classes turned out to be dead code).

Since then the builder has grown four things the Java original has no
answer to, all described in `docs/PORT_PLAN.md`'s "Beyond the port"
section: every segmentation in a protocol is measured under its own step's
name (so a derived cytosol ring is no longer silently unmeasured, and
renaming or deleting a segmentation carries its measurement with it);
editing a step's settings re-runs it and what depends on it; every step
card carries a small progress bar and every step runs off the GUI thread,
with a real time estimate where the work's size implies one and a
continuous bar where it does not; and reductions and clusterings - t-SNE,
UMAP - are added to the data as features rather than to the viewer as
layers. The analysis menus also gained UMAP, Louvain and Leiden.

The Object Explorer has since grown the other half of that work: gate
highlights that occupy the same volume and place as the data rather than one
flat section of it, a plot fixed at 4:3 with the gate list beneath it, a LUT
that colours categories (clusters, classes, ROIs) as categories and
measurements as gradients, and **image gates** - a region painted on a napari
Labels layer, whose objects are ringed on the plot in that region's own
colour. The protocol's `gates` category became **`classes`**: a class is a
rule (a range of a feature, a gate, an ROI, a cluster id, or any boolean
combination of them) rather than a shape somebody drew, objects carry as
many labels as apply, and two label sets cross into the hierarchy that makes
"immune > CD3+" a population you can count and map. See
`docs/PORT_PLAN.md`'s "Classes, label sets and image gates" section.

The two panes are views of one analysis, sharing a
`vtea_napari.session.AnalysisSession` keyed by the napari viewer: the
builder publishes each run into it, the explorer plots and gates it, and
hiding or closing either pane loses nothing.

**Data larger than memory** runs through the same protocol, a tile at a
time: `vtea_core.blocked` carries the memory budget, the tile plan, the
reconciliation of objects a tile boundary cuts in half (four selectable
strategies, defaulting to overlap matching), streaming measurements,
scaled estimators, blocked association and cell composition through
DuckDB. The builder decides from the plan - data that fits runs in memory
as it always has - shows the budget and what it divided the data into,
runs off the GUI thread with a Cancel button, and previews the protocol
over the region on screen. What the numbers are worth is pinned by
invariance tests: one tile is bit-identical to the whole image, and the
same objects and measurements come out at 1, 8 and 27 tiles. See
[`docs/LARGE_IMAGES.md`](docs/LARGE_IMAGES.md), including what still needs
real GPU hardware and real tissue to validate.

Both panes are registered as real napari plugin dock widgets, verified by
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
builds standalone `vtea-napari` bundles (PyInstaller, Linux + Windows) and
publishes them as a [GitHub Release](../../releases) - not committed into
this repo. A few-hundred-MB binary bundle doesn't belong in git history
(every future clone would pay for it permanently); Releases are the
standard place for built artifacts.

Two variants are published per OS:

| Asset | Cellpose | GPU |
|---|---|---|
| `vtea-napari-<os>.zip` | via `--install-torch` (one command) | **yes** |
| `vtea-napari-<os>-deeplearning.zip` | built in, works offline | no - CPU-only |

**Using a GPU: see [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md).** Short
version - use the slim download and run `vtea-napari --install-torch`,
which detects your NVIDIA driver and installs the matching PyTorch build;
or point `VTEA_TORCH_PATH` at a conda/venv environment you already have.
`vtea-napari --gpu-status` reports what's actually in use. GPU is
deliberately not delivered by bundling CUDA - that build is 4.9 GB (over
the 2 GiB release-asset cap), and a bundled PyTorch can never be replaced
by one on `sys.path`, so baking one in would rule out GPU permanently.

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
