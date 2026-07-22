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

Phase 4 (napari GUI) in progress. Phases 0-3 are done. Implemented and tested:

- **data**: `VolumeDataset`/`InMemoryVolumeDataset`/`ChunkedVolumeDataset`,
  `object_ids`/`object_pixel_indices`/`object_intensity_values`
- **io**: `read_tiff`/`write_tiff`/`read_zarr`/`write_zarr`/`open_volume`
- **segmentation**: `threshold_mask`, `label_components`, `watershed_split`,
  `filter_by_size`, `labels_from_points`, `import_labels`,
  `cellpose_segmentation`
- **measurements**: `MeasurementStore` (DuckDB-backed), `extract_measurements`
  (regionprops-based), `threshold_mean`
- **clustering**: `kmeans`, `gaussian_mixture`, `hierarchical`, `auto_k_kmeans`
- **reduction**: `pca`, `pca_explained_variance`, `isomap`,
  `laplacian_eigenmap`, `tsne`
- **gates**: `polygon_gate`, `rectangle_gate`
- **imageprocessing**: `gaussian_blur`, `median_filter`, `enhance_contrast`,
  `subtract_background`
- **classification**: `class_map` (no extra dependencies);
  `Cell3DClassifier`/`train_classifier`/`predict` (require the
  `deeplearning` extra - torch)
- **workflow** (new): `Step`/`Pipeline` - the headless engine behind
  `vtea-napari`'s protocol builder widget, and `STEP_REGISTRY`/
  `available_steps`/`get_step_function`, the category -> function registry
  both the GUI and scripts draw on

There is no separate `deeplearning` module - see PORT_PLAN.md's "Why deep
learning isn't a separate module". `cellpose_segmentation` lives in
`segmentation` next to the other volume→label-mask functions; the
supervised classification work lives in the new `classification` module,
parallel to `clustering`/`reduction`.

Not yet ported: `bioimageio.core`-based generic model inference (the
DeepImageJ replacement - deferred, more involved than Cellpose), spatial
statistics (`vtea.spatial` - unregistered/non-plugin utility classes in the
Java source, lower priority), linear unmixing and ImageJ macro execution
(`vtea.imageprocessing.builtin.LinearUnmixing`/`IJMacro` - deferred per
PORT_PLAN.md's open question on macro compatibility), and ImageJ ROI-file
import (`vtea.objects.Segmentation.ImageJROIBased` - an I/O format concern,
not an algorithm).

## Layout

```
src/vtea_core/
  data/             VolumeDataset abstraction (in-memory + Dask/Zarr-chunked),
                    label-mask object helpers
  io/               Readers/writers (TIFF/ImageJ hyperstack, Zarr; proprietary
                    formats via bioio are not implemented yet)
  segmentation/     threshold -> label -> (optional watershed split) -> size
                    filter, plus cellpose_segmentation (deep-learning-based)
  measurements/     MeasurementStore (DuckDB) + regionprops-based extraction
  clustering/       KMeans, GMM, hierarchical, BIC-based automatic-k selection
  reduction/        PCA, Isomap, Laplacian Eigenmap, t-SNE
  gates/            Boolean gate math (polygon/rectangle point-membership tests)
  imageprocessing/  Gaussian blur, median filter, contrast, background subtraction
  classification/   class_map (label-remap) + a small torch 3D CNN
                    (train_classifier/predict) for supervised object classification
  workflow/         Step/Pipeline engine + the category -> function step registry
                    driving vtea-napari's protocol builder widget
```

Each subpackage's built-in implementations register into an
`vtea_core.<group>` entry-point group (see `pyproject.toml`), mirroring the
Java `vtea.services` plugin registry so the same algorithm-discovery pattern
carries over.

## The `deeplearning` extra

`torch` and `cellpose` are optional (`pip install "vtea-core[deeplearning]"`)
so a plain `pip install vtea-core` doesn't force a multi-GB PyTorch install.
`vtea_core.segmentation.cellpose_segmentation` only imports `cellpose`
inside the function body (only reached if you don't pass your own `model`),
so importing `vtea_core.segmentation` itself never requires the extra.
`vtea_core.classification`'s CNN pieces do require it at import time (an
`nn.Module` subclass needs `torch` importable to be defined at all) - the
module degrades gracefully: `class_map` is always available, and
`Cell3DClassifier`/`train_classifier`/`predict` are only exposed if `torch`
is installed.
