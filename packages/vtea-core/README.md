# vtea-core

Headless core analysis library for VTEA (Volumetric Tissue Exploration and
Analysis) — 3D tissue cytometry. Ported from the Java/ImageJ1 application at
[winfrees/volumetric-tissue-exploration-analysis](https://github.com/winfrees/volumetric-tissue-exploration-analysis).

This package has no GUI dependency and is usable from scripts, Jupyter, a
CLI, or HPC batch jobs. The GUI lives in the separate `vtea-napari` package
in this repo, which is a thin napari plugin layer over this library.

See [`/docs/PORT_PLAN.md`](../../docs/PORT_PLAN.md) in the repo root for the
full porting plan, architecture mapping, and phased roadmap.

## Status

Phase 1 (core data model & I/O) in progress. Implemented and tested:
`data.VolumeDataset`/`InMemoryVolumeDataset`/`ChunkedVolumeDataset`,
`data.object_ids`/`object_pixel_indices`/`object_intensity_values`,
`io.read_tiff`/`write_tiff`/`read_zarr`/`write_zarr`/`open_volume`,
`measurements.MeasurementStore`. Segmentation, clustering, reduction, gates,
and deep learning are still unimplemented (Phase 2+).

## Layout

```
src/vtea_core/
  data/           VolumeDataset abstraction (in-memory + Dask/Zarr-chunked),
                  label-mask object helpers
  io/             Readers/writers (TIFF/ImageJ hyperstack, Zarr; proprietary
                  formats via bioio are not implemented yet)
  segmentation/   Segmentation algorithms (not implemented yet)
  measurements/   Per-object feature/measurement extraction; MeasurementStore
                  (DuckDB-backed table storage) is implemented, the
                  regionprops-based extraction algorithms are not yet
  clustering/     KMeans, GMM, hierarchical, X-Means/G-Means, ... (not implemented yet)
  reduction/      PCA, t-SNE, Isomap, Laplacian Eigenmap (not implemented yet)
  gates/          Boolean gate math over measurement tables (not implemented yet)
  deeplearning/   Cellpose, native PyTorch models, bioimageio.core inference (not implemented yet)
```

Each subpackage's built-in implementations register into an
`vtea_core.<group>` entry-point group (see `pyproject.toml`), mirroring the
Java `vtea.services` plugin registry so the same algorithm-discovery pattern
carries over.
