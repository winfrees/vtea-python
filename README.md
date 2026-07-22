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
```

## Status

**Phase 4 — napari GUI.** Phases 0-3 (package skeletons/CI; `VolumeDataset`/
TIFF/Zarr I/O; the algorithm core - segmentation, measurements, clustering,
reduction, gates, image preprocessing; and Cellpose segmentation + a
`classification` module) are done - see `packages/vtea-core/README.md` for
the full module-by-module status.

The protocol-builder scope call is decided: **Option A**, a fully
functional GUI. Scoping it against the actual Java source corrected an
earlier assumption - `vtea.protocol` isn't a free-form node-graph editor,
it's an ordered stack of step cards built from a category menu (no
drag-drop/wire code exists in the Java source at all), which made Option A
smaller than originally estimated. Landed so far: `vtea_core.workflow`
(`Step`/`Pipeline`, the headless engine, shared between the GUI and
scripts/notebooks) and `vtea-napari`'s `ProtocolBuilderWidget` - registered
as a real napari plugin dock widget, verified by actually loading it
through `napari.Viewer` in tests, not just constructing it standalone. See
`docs/PORT_PLAN.md`'s "Protocol builder: Option A" section and
`packages/vtea-napari/README.md` for what's built and what's still open
(a `MicroExplorer`-equivalent plot view, gate manager, LUT controls).

## Development

Each package uses a standard `src/`-layout with `pyproject.toml` (hatchling).
To install a package in editable mode for development:

```bash
pip install -e "packages/vtea-core[dev]"
pip install -e "packages/vtea-napari[dev]"
```

Run tests with `pytest` from within each package directory, or see
`.github/workflows/ci.yml` for the full matrix.

## License

GPL-2.0-only, matching the source Java application. See [`LICENSE`](LICENSE).
