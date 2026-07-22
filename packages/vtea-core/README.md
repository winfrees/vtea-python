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

Phase 2 (algorithm core) in progress. Implemented and tested:

- **data**: `VolumeDataset`/`InMemoryVolumeDataset`/`ChunkedVolumeDataset`,
  `object_ids`/`object_pixel_indices`/`object_intensity_values`
- **io**: `read_tiff`/`write_tiff`/`read_zarr`/`write_zarr`/`open_volume`
- **segmentation**: `threshold_mask`, `label_components`, `watershed_split`,
  `filter_by_size`, `labels_from_points`, `import_labels`
- **measurements**: `MeasurementStore` (DuckDB-backed), `extract_measurements`
  (regionprops-based), `threshold_mean`
- **clustering**: `kmeans`, `gaussian_mixture`, `hierarchical`, `auto_k_kmeans`
- **reduction**: `pca`, `pca_explained_variance`, `isomap`,
  `laplacian_eigenmap`, `tsne`
- **gates**: `polygon_gate`, `rectangle_gate`
- **imageprocessing**: `gaussian_blur`, `median_filter`, `enhance_contrast`,
  `subtract_background`

Most of these consolidate several near-duplicate Java classes into one
function - see each module's docstring for the specific mapping. Not yet
ported: deep learning (Cellpose/PyTorch/bioimageio, Phase 3), spatial
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
  segmentation/     threshold -> label -> (optional watershed split) -> size filter
  measurements/     MeasurementStore (DuckDB) + regionprops-based extraction
  clustering/       KMeans, GMM, hierarchical, BIC-based automatic-k selection
  reduction/        PCA, Isomap, Laplacian Eigenmap, t-SNE
  gates/            Boolean gate math (polygon/rectangle point-membership tests)
  imageprocessing/  Gaussian blur, median filter, contrast, background subtraction
  deeplearning/     Cellpose, native PyTorch models, bioimageio.core inference (not implemented yet)
```

Each subpackage's built-in implementations register into an
`vtea_core.<group>` entry-point group (see `pyproject.toml`), mirroring the
Java `vtea.services` plugin registry so the same algorithm-discovery pattern
carries over.
