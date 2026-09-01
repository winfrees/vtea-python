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
  `cellpose_segmentation`; plus the derived segmentations (new), which build
  one label image out of another by morphology rather than by intensity -
  `expand_labels`, `label_ring` (a cytosol band around a nucleus),
  `label_shell` (a nuclear envelope straddling the boundary),
  `subtract_labels`, `restrict_labels_to`, and `watershed_ownership` (divide
  one region among the objects inside it, splitting at its own narrow waist).
  Each preserves label identity, and each takes a `Spacing`, so a 2 µm band is
  2 µm in the specimen rather than 2 voxels in an anisotropic stack
- **objects** (new): `ObjectRef`/`Association`/`AssociationSet` - which object
  of one segmentation belongs to which object of another, keeping the whole
  posterior (`alternatives`) rather than only the winner, along with how the
  link was made (`relationship`, `method`, `params`);
  `save_associations`/`load_associations` as versioned JSON, and `unassigned`
  for the children deliberately left without a parent - "17 of 400 cytoplasms
  had no nucleus" is a finding, and invisible without it.
  `associate_by_identity` covers the exact case: a derived segmentation keeps
  its parent's ids, so every link is certain by construction.
  `associate_objects` covers the inferred one - two channels segmented
  independently - through `scoring` (`containment`, `centroid_distance`,
  `boundary_distance`, each a sparse child x parent affinity) and
  `assignment` (a posterior with an explicit orphan option, then either
  `many_to_one` per child or a global `one_to_one` solve that is the only way
  to honour "one and only one"; the sparse candidate graph is split into
  independent blocks first, so the O(n^3) solve stays affordable).
  `Cell`/`CellSet`/`build_cells` then compose those links into cells - a
  chosen root segmentation identifies them, ids come from the root object so
  they survive a re-run, cycles are refused and objects that reach no root are
  kept as `unclaimed`; `merge_associations` joins two association steps into
  the one hierarchy a cell spans, and `cell_features` turns one measurement
  table per segmentation into one row per cell, columns namespaced by role
  (`nuclei_1.mean_ch0`) and one-to-many roles aggregated (`lysosome_1.n`,
  `lysosome_1.mean_count`). `Ownership`/`distance_ownership` take the same
  question down to the voxel: rather than one answer per voxel, the best *k*
  owners and their probabilities, so a boundary the stain never resolved
  reads as a coin toss instead of a confident split. `owners[0]` is the hard
  label image and `probabilities[0]` the confidence map; `margin()` separates
  "two cells claim this equally" from "one cell weakly claims it and nobody
  contests"; `override()` gives a region to one owner and records that a
  person said so. `AssociationSet.set_parent`/`unassign` are the same idea
  one level up - a hand-made link keeps the answer it replaced in `params`,
  is marked `manual` forever after, and drops out of `uncertain()` because a
  person's decision is not a posterior
- **measurements**: `MeasurementStore` (DuckDB-backed), `extract_measurements`
  (regionprops-based - object_id, centroid-*, count, mean, sum, stddev, min,
  max, threshold_mean), `extract_measurements_by_channel` (one segmentation
  against every channel as one flat table, intensity columns suffixed with
  the channel they were measured on: `mean_ch0`, `mean_ch2`, ...),
  `feature_matrix` (that table as the float array clustering and reduction
  take as `data`), `weighted_measurements`/`weighted_measurements_by_channel`
  (the same measurements over a probabilistic `Ownership` - a count becomes
  the expected voxel count and a mean a probability-weighted mean, under the
  same column names, so a protocol can swap a hard measurement for a weighted
  one without every plot axis changing underneath), a physical `volume`
  column when the voxel size is known,
  `threshold_mean`; `FeatureCatalog`/`FeatureDescriptor`
  (new) - what each column of the table is and how it was produced (what was
  measured, on which channel and segmentation, by which step with what
  parameters, and for a derived feature which features were fed to it),
  saved as JSON and rendered as the publication data dictionary
- **clustering**: `kmeans`, `gaussian_mixture`, `hierarchical`,
  `auto_k_kmeans`; plus `louvain` and `leiden` (new) - community detection
  over a shared-nearest-neighbour graph (`shared_neighbor_graph`), which
  decide how many populations there are rather than being told. Every other
  method here has to be given `k` before it has looked at the tissue, and
  the number of cell types in a biopsy is the thing being measured.
  `resolution` moves the granularity instead. Needs the `graph` extra
  (python-igraph, leidenalg); the steps stay in the menu without it and say
  what to install when run
- **reduction**: `pca`, `pca_explained_variance`, `isomap`,
  `laplacian_eigenmap`, `tsne`, and `umap` (new) - the projection the Java
  original predates: it keeps global structure t-SNE discards (the distance
  between two t-SNE islands means nothing), runs an order of magnitude
  faster at tissue-scale object counts, and can project new objects into an
  existing embedding, which is what makes a second acquisition comparable
  to the first. Needs the `umap` extra (umap-learn)
- **classes** (new): what an object *is*, as a rule that re-runs rather than
  a shape somebody drew. `class_from_range` ("mean_ch2 from 50 to 150"),
  `class_from_values` (cluster 3 and 7; ROI 2; a gate's members) and
  `class_from_expression`, which reads a small boolean language
  (`expression.py`: AND / OR / NOT / XOR / XNOR / NAND / NOR, comparisons,
  `in [3, 7]`, chained ranges) over the table's columns - parsed, never
  `eval`'d, because these definitions are saved in protocol files and mailed
  around. `LabelSet`/`ObjectLabel` then hold the *n* labels an object
  carries (an object is a cell, an immune cell, a CD3+ T cell and inside the
  tubule ROI, all at once), `combine_label_sets` builds the hierarchy out of
  two sets - `cross` refines "immune" by "CD3+" into "immune > CD3+",
  keeping the objects the finer set says nothing about rather than dropping
  them - and `label_image` paints a set back onto the segmentation as a
  lookup-table remap. This replaces the `gates` protocol category (see the
  napari README); the gate *primitives* below stay where they are, for the
  Object Explorer
- **gates**: `polygon_gate`, `rectangle_gate`, `rectangle_vertices`
  (boolean-array primitives);
  `objects_in_rois`/`image_gate` (new) - the gate drawn on the *image*
  rather than the plot: which region of a napari Labels layer each object
  lies in, by centroid (one voxel read per object, so it costs nothing at
  any volume) or by majority overlap, answered with the region's own id so
  the objects and the region can be drawn in the same colour;
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
  than something the caller did on the way in.
  `wiring.produces_image`/`Step.produces_image` separate the steps whose
  result is a picture from the ones whose result is per-object numbers - a
  t-SNE embedding of nine thousand nuclei is a (9000, 2) array, and without
  that distinction it lands in the viewer as a two-pixel-wide stripe rather
  than as the two features it is.
  `measure.sync_measurement_steps` (new) keeps one measurement step per
  segmentation, named after it (`measure_<segmentation>`) and recorded on
  the step (`Step.auto_for`), so a protocol that derives a cytosol ring from
  a nucleus measures both without anyone having to remember to;
  `rename_segmentation` follows a renamed segmentation into the step raised
  for it. `cost.estimate_seconds`/`StepCost`/`Calibration` (new) say how
  long a step should take before it is run - from the size of the tile and
  a per-step cost, calibrated against what the steps actually took on this
  machine - and return None for the steps whose runtime genuinely cannot be
  predicted (t-SNE, UMAP, Leiden, agglomerative clustering), which is what
  a GUI needs in order to show a continuous progress bar instead of an
  invented fraction. `Step.settings_signature` is what "a setting changed"
  means: everything about a step that decides what it computes, and
  deliberately not its name

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
                    and the identity-preserving derived segmentations
                    (expand/ring/shell/subtract/restrict/ownership)
  objects/          Association model: which object belongs to which, with the
                    posterior behind the link and how it was made, the scoring
                    and assignment that infer it between two independently
                    segmented channels, the cells those links compose into,
                    and the per-voxel ownership underneath them
  measurements/     MeasurementStore (DuckDB) + regionprops-based extraction
  clustering/       KMeans, GMM, hierarchical, BIC-based automatic-k selection,
                    and graph-based community detection (Louvain, Leiden)
  reduction/        PCA, Isomap, Laplacian Eigenmap, t-SNE, UMAP
  gates/            Boolean gate math (polygon/rectangle point-membership tests)
                    plus Gate/GateSet (named, hierarchical gates over a
                    measurement DataFrame) and image gates (which napari ROI
                    each object is in)
  classes/          Class rules (range / values / boolean expression), the
                    label sets an object's n labels live in, and the
                    hierarchies two sets combine into
  imageprocessing/  Gaussian blur, median filter, contrast, background subtraction
  classification/   class_map (label-remap) + a small torch 3D CNN
                    (train_classifier/predict) for supervised object classification
  workflow/         Step/Pipeline engine + the category -> function step registry
                    driving vtea-napari's protocol builder widget, the
                    measurement-per-segmentation rules (measure.py) and the
                    a priori step-duration estimates behind its progress
                    bars (cost.py)
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

## The `umap` and `graph` extras

`umap-learn` (for `reduction.umap`) and `python-igraph`/`leidenalg` (for
`clustering.louvain`/`leiden`) are optional the same way:
`pip install "vtea-core[umap]"` and `pip install "vtea-core[graph]"`.

They differ from `deeplearning` in one deliberate way: those steps stay in
`STEP_REGISTRY` - and so in the protocol builder's menu - whether or not the
backend is installed, because the import that would fail is inside the
function. Picking UMAP without umap-learn therefore fails with the command
to install it, rather than the step quietly not being there and the user
concluding VTEA has no UMAP. Louvain falls back to `networkx` where igraph
is missing, and Leiden to python-igraph's own implementation where
`leidenalg` is.
