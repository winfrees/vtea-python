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

Phase 0 (foundations + parity harness) — package skeleton only. No
functionality has been ported yet.

## Layout

```
src/vtea_core/
  data/           VolumeDataset abstraction (in-memory + Dask/Zarr-chunked)
  io/             Readers/writers (TIFF, OME-TIFF/Zarr, proprietary formats)
  segmentation/   Segmentation algorithms
  measurements/   Per-object feature/measurement extraction
  clustering/     KMeans, GMM, hierarchical, X-Means/G-Means, ...
  reduction/      PCA, t-SNE, Isomap, Laplacian Eigenmap
  gates/          Boolean gate math over measurement tables
  deeplearning/   Cellpose, native PyTorch models, bioimageio.core inference
```

Each subpackage's built-in implementations register into an
`vtea_core.<group>` entry-point group (see `pyproject.toml`), mirroring the
Java `vtea.services` plugin registry so the same algorithm-discovery pattern
carries over.
