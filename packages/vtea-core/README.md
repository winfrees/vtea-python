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

Phases 0-4 are done. Implemented and tested:

- **data**: `VolumeDataset`/`InMemoryVolumeDataset`/`ChunkedVolumeDataset`,
  `object_ids`/`object_pixel_indices`/`object_intensity_values`; `Spacing`
  (new) - the physical size of a voxel, its unit, and whether it came from
  the image, from the user, or from nowhere. `spacing_from_scale` reads a
  napari layer's scale and treats all-ones as unknown, since that is what
  napari fills in when a file records no scale
- **io**: `read_tiff`/`write_tiff`/`read_zarr`/`write_zarr`/`open_volume`
- **segmentation**: `threshold_mask`, `label_components`, `watershed_split`,
  `filter_by_size`, `labels_from_points`, `import_labels`,
  `cellpose_segmentation`
- **measurements**: `MeasurementStore` (DuckDB-backed), `extract_measurements`
  (regionprops-based - object_id, centroid-*, count, mean, sum, stddev, min,
  max, threshold_mean), `extract_measurements_by_channel` (one segmentation
  against every channel as one flat table, intensity columns suffixed with
  the channel they were measured on: `mean_ch0`, `mean_ch2`, ...),
  `feature_matrix` (that table as the float array clustering and reduction
  take as `data`), a physical `volume` column when the voxel size is known,
  `threshold_mean`; `FeatureCatalog`/`FeatureDescriptor`
  (new) - what each column of the table is and how it was produced (what was
  measured, on which channel and segmentation, by which step with what
  parameters, and for a derived feature which features were fed to it),
  saved as JSON and rendered as the publication data dictionary
- **clustering**: `kmeans`, `gaussian_mixture`, `hierarchical`, `auto_k_kmeans`
- **reduction**: `pca`, `pca_explained_variance`, `isomap`,
  `laplacian_eigenmap`, `tsne`
- **gates**: `polygon_gate`, `rectangle_gate`, `rectangle_vertices`
  (boolean-array primitives);
  `Gate`/`GateSet` (new) - named, stateful gates with real hierarchy
  (a subgate's membership is intersected with its parent's) and per-gate
  statistics (`GateSet.statistics` - the gated cell count plus the mean of
  each plotted feature over those cells); `save_gates`/`load_gates`, plain
  versioned JSON so the polygons behind a figure can be reopened and
  archived with it
- **imageprocessing**: `gaussian_blur`, `median_filter`, `enhance_contrast`,
  `subtract_background`
- **classification**: `class_map` (no extra dependencies);
  `Cell3DClassifier`/`train_classifier`/`predict` (require the
  `deeplearning` extra - torch)
- **workflow** (new): `Step`/`Pipeline` - the headless engine behind
  `vtea-napari`'s protocol builder widget, and `STEP_REGISTRY`/
  `available_steps`/`get_step_function`, the category -> function registry
  both the GUI and scripts draw on. `wiring.STEP_IO` declares each step's
  data inputs, its output key, and how it relates to the channel axis
  (`CHANNEL_SLICE` for image steps, `CHANNEL_ARGUMENT` for one that handles
  channels itself, `CHANNEL_NONE` for the clustering/reduction/gating steps
  that consume the per-object feature table, which has no channel axis).
  A step declaring a `feature_input` is handed the measurement *table* and
  narrows it to its own `Step.features` selection, so "this clustering used
  these six of the forty measured features" is part of the protocol rather
  than something the caller did on the way in

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
                    plus Gate/GateSet (named, hierarchical gates over a
                    measurement DataFrame)
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
