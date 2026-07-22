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
  dimensionality reduction, gating, deep learning). No GUI dependency;
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

**Phase 1 — core data model & I/O.** Phase 0 (package skeletons, CI, parity
harness scaffolding) is done. `vtea_core.data` now has a working
`VolumeDataset` abstraction (`InMemoryVolumeDataset` / `ChunkedVolumeDataset`,
canonical (C, Z, Y, X) axis order) and label-mask object helpers;
`vtea_core.io` reads/writes TIFF (via tifffile, including ImageJ hyperstacks)
and Zarr (via Dask); `vtea_core.measurements.MeasurementStore` is a DuckDB-
backed table store replacing the Java H2 database. Segmentation, clustering,
and reduction algorithms are still unimplemented (Phase 2). See
`docs/PORT_PLAN.md` for the phase breakdown and current open decisions
(notably, the protocol-builder UI scope call ahead of Phase 4).

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
