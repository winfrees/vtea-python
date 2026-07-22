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

**Phase 3 — deep learning consolidation.** Phases 0-2 (package skeletons/CI,
`VolumeDataset`/TIFF/Zarr I/O, and the algorithm core - segmentation,
measurements, clustering, reduction, gates, image preprocessing) are done.
Phase 3 does *not* introduce a separate `deeplearning` module - see
`docs/PORT_PLAN.md`'s "Why deep learning isn't a separate module". Instead:
`cellpose_segmentation` landed in `vtea_core.segmentation`, and a new
`vtea_core.classification` module (parallel to `clustering`/`reduction`)
holds `class_map` plus a small torch 3D CNN (`train_classifier`/`predict`)
for supervised object classification. Both `torch` and `cellpose` stay
behind the `deeplearning` extra, and the rest of `vtea-core` works without
it. Still open in Phase 3: `bioimageio.core`-based generic model inference
(the DeepImageJ replacement). See `packages/vtea-core/README.md` for the
full module-by-module status and `docs/PORT_PLAN.md` for the phase
breakdown and current open decisions (notably, the protocol-builder UI
scope call ahead of Phase 4).

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
