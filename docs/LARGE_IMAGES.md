# Processing images that do not fit in RAM

Plan for running every protocol step over a dataset larger than the memory
available to it — real, cgroup-imposed, or chosen by the user — while
keeping the answers the same ones a whole-image run would have given, and
being explicit about the places where they cannot be.

Nothing here is built yet. This document is the design under review.

## What "large" actually means

Not a file size. A dataset is large when some step's **working set** —
its inputs, its outputs, and every intermediate the algorithm allocates
while running — exceeds the budget. That is a per-step property, not a
per-dataset one, which is the first thing the design has to take seriously:
a volume that thresholds comfortably in memory will not survive
`watershed_split`, because a `float64` distance transform is four times the
size of the `uint16` image it came from and scipy allocates more than one of
them.

A worked example, to keep the numbers concrete. A cleared-tissue volume of
2048 × 2048 × 2000 voxels, two channels, `uint16`:

| Thing | Size |
| --- | --- |
| The image | 33.5 GB |
| One channel of it | 16.8 GB |
| The `int32` label image | 33.5 GB |
| A dense top-3 `Ownership` over it | 201 GB |
| `watershed_split`'s live working set (~35 B/voxel) | 294 GB |

The label image being as large as the image is the point most easily
missed: **outputs go out of core too.** A design that streams the input and
returns a NumPy label array has not solved anything.

## What breaks today

The audit, so the plan is scoped against the code rather than against an
impression of it:

- `ChunkedVolumeDataset` (`data/volume.py`) exists and wraps a Dask array,
  but nothing consumes it lazily. `VolumeDataset.to_numpy()` and
  `.subvolume()` are the only ways data leaves it, and `fits_in_memory()`
  compares against a hardcoded `DEFAULT_MEMORY_BUDGET_BYTES = 4 GiB`
  described in its own comment as a placeholder.
- `read_tiff` (`io/tiff.py`) calls `series.asarray()` — a full read, always.
  `read_zarr` is lazy, and `write_zarr` accepts either, so Zarr is already
  the half-open door.
- `Pipeline.run` (`workflow/pipeline.py`) threads NumPy arrays through a
  `dict` context, holds every intermediate for the life of the run, and
  `Step._select_channel` calls `np.take` — a copy. A four-step protocol
  keeps four full-size arrays alive.
- The napari builder's `active_image()`
  (`widgets/protocol_builder.py:459`) is `np.asarray(layer.data)`, which
  materializes a lazily-loaded layer the moment a protocol runs, undoing
  whatever the reader was careful about.
- Every algorithm function takes and returns `np.ndarray`. That is the
  right signature and should not change; what changes is who calls them and
  with what.
- `extract_measurements` calls `regionprops_table` over the whole label
  array at once.
- `Ownership` (`objects/ownership.py`) is dense top-k over the full volume —
  see the 201 GB row above.
- `MeasurementStore` is DuckDB, but registers an in-memory DataFrame.
  DuckDB is the piece that already scales; the DataFrame is not.

So: the data model anticipated this, and no execution path uses it.

## Principles

**Do not write what someone else maintains.** The genuinely novel work here
is reconciling objects across tile seams and being honest about it. Almost
everything else exists:

| Need | Use | Not |
| --- | --- | --- |
| Chunked array, lazy graph, overlap | `dask.array`, `map_overlap` | a `Chunk`/`ChunkIterator` port |
| Connected components across chunks | `dask_image.ndmeasure.label` | our own union-find for this one case |
| Chunked storage, compression, pyramids | `zarr`, `ome-zarr-py` | anything |
| Changing chunking without an OOM | `rechunker` | a hand-rolled shuffle |
| Lazy display, pyramid rendering, ROI | napari `multiscale`, `corner_pixels` | a custom viewer path |
| Reading vendor formats lazily | `bioio` (already an extra) | — |
| Tables larger than RAM | DuckDB over Parquet (already a dep) | a custom spill-to-disk table |
| Streaming estimators | `MiniBatchKMeans`, `IncrementalPCA` | reimplementation |
| 2D-to-3D mask linking in Cellpose | cellpose's own `stitch_threshold` | our own for the z axis |

**The algorithm functions stay pure.** `label_components(mask)` keeps taking
a NumPy array. Blocking is a property of the *executor*, declared by the
step, not baked into each function. This is the same split the codebase
already makes between `Step`/`Pipeline` and the functions they call, and it
is what keeps a headless script, a notebook and the GUI on one code path.

**Exactness is a stated property, not a hope.** Each step declares whether
its blocked form is exact, exact-given-a-sufficient-halo, or approximate.
Where it is the middle one, the executor *verifies* the halo was sufficient
rather than trusting the parameter, and says so when it wasn't.

**Ambiguity is recorded, not resolved silently.** The codebase already does
this twice — `Association.alternatives` keeps the posterior, and
`Ownership.confidence()` keeps the doubt. Objects cut by a tile boundary get
the same treatment: a ledger of how each one was put back together, and a
per-object confidence that a gate can select on for review.

## The memory budget

One object, resolved once per run, from the first of these that answers:

1. An explicit setting — the GUI's budget field, `VTEA_MEMORY_BUDGET`, or
   the `budget=` argument to a headless call.
2. The cgroup limit (`/sys/fs/cgroup/memory.max`, and the v1 path), which is
   the real ceiling inside a container and is *lower* than what `psutil`
   reports. Getting this wrong is how a job gets OOM-killed on a cluster
   while claiming 64 GB is free.
3. `psutil.virtual_memory().available`, times a fraction.

```python
@dataclass(frozen=True)
class MemoryBudget:
    total_bytes: int
    fraction: float = 0.6      # leave room for the interpreter, Qt, and the OS
    workers: int = 1           # the per-tile budget is divided by this
    gpu_bytes: int | None = None   # torch.cuda.mem_get_info()[0], when relevant
    source: str = DETECTED     # DETECTED | CGROUP | USER | ENV
```

`source` is there for the same reason `Spacing.source` is: "8 GB because you
said so" and "8 GB because that is all we could find" are different facts,
and only one of them should be silently believed.

The GPU budget is separate and smaller, and it is the binding constraint for
Cellpose. It is also the one that cannot be computed from first principles —
what fits depends on the model, the version, and the driver. So: a
**calibration probe**, run once per (GPU, model) pair and cached in the user
config, that grows a synthetic tile until it OOMs and records the largest
that worked. The Java codebase had bespoke GPU-OOM detection and restart
logic for exactly this; measuring once beats catching the failure forever.

## The tile plan

A step's blocked form needs three numbers, and none of them can be guessed
from its signature. They go where the rest of a step's non-obvious I/O
already lives — `wiring.StepIO`, which already carries `inputs`, `output`,
`channel_mode` and `feature_input`:

```python
@dataclass(frozen=True)
class StepIO:
    ...                        # everything it has today
    block_mode: str = ELEMENTWISE
    halo: HaloSpec | None = None      # fixed voxels, or derived from a param
    bytes_per_voxel: int = 4          # live intermediates, per input voxel
    exactness: str = EXACT            # EXACT | EXACT_WITH_HALO | APPROXIMATE
```

`bytes_per_voxel` is what turns a budget into a tile shape:

```
usable = budget.total_bytes * budget.fraction / budget.workers
voxels_per_tile = usable / step.bytes_per_voxel
```

For `watershed_split` at ~35 B/voxel (uint16 input, bool mask, float64 EDT,
int32 markers, int32 labels, plus scipy's internal copies):

| Machine | Usable | Voxels | Tile (rounded to chunks) |
| --- | --- | --- | --- |
| 8 GB laptop | 4.8 GB | 137 M | 512 × 512 × 512 |
| 32 GB workstation | 19.2 GB | 549 M | 768 × 768 × 768 |
| 128 GB server | 76.8 GB | 2.2 G | 1024 × 1024 × 1024 (or 2048² × 512) |

The plan for a *pipeline* is the tightest tile any of its steps needs,
snapped to a multiple of the store's chunk shape, and enlarged in the axes
where the data is thin (a 2048 × 2048 × 24 slab tiles in XY only). It is
computed once, recorded in the protocol, and reported — "512³ tiles, 64-voxel
halo, 4,096 tiles, bounded by watershed_split_1" — because a user who cannot
see the plan cannot tell a slow run from a stuck one.

**The plan is part of the result's provenance.** Change the tile size and
seam-crossing objects can change; a protocol that does not record its plan
cannot be reproduced. See "Invariance" below for the tests that keep that
honest.

## Block modes, and every step

Five modes. The whole protocol classifies into them.

- **ELEMENTWISE** — output voxel depends on its input voxel. No halo, exact,
  trivially blocked.
- **NEIGHBORHOOD(d)** — output voxel depends on inputs within `d`. Blocked
  with a halo of `d`, exact given a sufficient halo; at the dataset border
  the halo is synthesized by padding (see "Reflection" below).
- **GLOBAL_STAT** — needs a statistic over the whole image (Otsu's
  threshold, a percentile, min/max). Split into a streaming statistics pass
  followed by an ELEMENTWISE apply.
- **OBJECT_LOCAL** — works per object, within its bounding box. Scheduled by
  object, not by grid; a bbox that fits one object fits in RAM by
  definition.
- **TABLE** — consumes the feature table, not voxels. Scales with object
  count, and is a different problem (see "Analysis at scale").

### Image processing

| Step | Mode | Halo | Exactness | Notes |
| --- | --- | --- | --- | --- |
| `gaussian_blur` | NEIGHBORHOOD | `ceil(4σ)` | exact w/ halo | scipy truncates at 4σ; halo follows `sigma` |
| `median_filter` | NEIGHBORHOOD | `radius` | exact w/ halo | |
| `enhance_contrast` (`normalize`) | GLOBAL_STAT | — | exact | streaming min/max, then rescale per block |
| `enhance_contrast` (`equalize`) | NEIGHBORHOOD | kernel | **approximate** | CLAHE is tile-local by construction; tiles must align to the CLAHE kernel grid or seams show. Recommend: require `kernel_size` to divide the tile shape, and warn otherwise |
| `subtract_background` | NEIGHBORHOOD | `radius` | **approximate** | A rolling ball is not strictly local. The exact-enough alternative, and the faster one: estimate the background on a coarse pyramid level, upsample, subtract. Offer both, default to the pyramid estimate above a size threshold |

### Segmentation

| Step | Mode | Halo | Exactness | Notes |
| --- | --- | --- | --- | --- |
| `threshold_mask` (`fixed`) | ELEMENTWISE | — | exact | |
| `threshold_mask` (`otsu`) | GLOBAL_STAT | — | exact | Exact for integer dtypes: accumulate a full histogram over blocks, run Otsu on it. Not a sample |
| `threshold_mask` (`percentile`) | GLOBAL_STAT | — | exact (int) | Same histogram; a t-digest for float data, which is approximate and should say so |
| `label_components` | NEIGHBORHOOD | 1 | exact | `dask_image.ndmeasure.label` already merges across chunks. Use it; do not port `ObjectStitcher` for this case |
| `watershed_split` | NEIGHBORHOOD | max object radius | exact w/ **verified** halo | The distance transform is global in principle. See "Verifying the halo" |
| `filter_by_size` | TABLE + ELEMENTWISE | — | exact | Two phases: sizes come from the reconciled ledger (already counted), then a block-wise remap |
| `labels_from_points` | OBJECT_LOCAL | `radius` | exact | Points are sparse; scatter each into its own block |
| `cellpose_segmentation` | NEIGHBORHOOD | `1.5 × diameter` | **approximate** | See "Deep learning" |
| `expand_labels` | NEIGHBORHOOD | `ceil(distance/min voxel)` | exact w/ halo | Growth is capped by the parameter, so a halo above it is provably sufficient |
| `label_ring` | NEIGHBORHOOD | as `expand_labels` | exact w/ halo | |
| `label_shell` | NEIGHBORHOOD | max object radius | exact w/ verified halo | Inward distance is bounded by object size, not by a parameter |
| `subtract_labels` | ELEMENTWISE | — | exact | |
| `restrict_labels_to` | ELEMENTWISE | — | exact | |
| `watershed_ownership` | OBJECT_LOCAL | region extent | exact | Scheduled per connected region of the mask — a cell-sized window, not a grid tile |

### Objects, association, cells

| Step | Mode | Notes |
| --- | --- | --- |
| `associate_by_identity` | TABLE | Needs the set of ids per segmentation, which the ledger already holds. Cheap |
| `associate_objects` | OBJECT_LOCAL + TABLE | Centroid scoring is a table op over a kd-tree. Boundary-distance scoring reads each candidate pair's bboxes from the store. Spatially partitioned with a halo of `max_distance`; the sparsity that `scoring.py` already relies on is what makes this work at 10⁷ objects |
| `merge_associations` | TABLE | |
| `distance_ownership` | OBJECT_LOCAL | Already windows per marker (`_insert_claim`). What changes is the output: a dense top-k array becomes **mask-restricted sparse** — coordinates and claims only where the mask is true. For a 5%-foreground volume that is 201 GB → 10 GB, and it is the only version that is usable at all |
| `build_cells` | TABLE | A graph over associations. Memory scales with object count; fine to ~10⁷, worth measuring beyond |
| `cell_features` | TABLE | Grouped aggregation — push into DuckDB rather than pandas |

### Measurements

| Step | Mode | Notes |
| --- | --- | --- |
| `extract_measurements` | accumulate + exact second pass | See "Measuring an object that spans tiles" |
| `extract_measurements_by_channel` | as above, per channel | |
| `weighted_measurements_by_channel` | accumulate | Weighted sums are additive by construction, so this one streams exactly with no second pass |

### Analysis

| Step | Strategy at scale |
| --- | --- |
| `kmeans` | `MiniBatchKMeans` above a row threshold; identical interface, near-identical result |
| `gaussian_mixture` | Fit on a stratified subsample, `predict` over all rows in batches. Exact for the assignment given the fitted model |
| `hierarchical` | O(n²) — **not feasible** above ~10⁵ objects and should say so rather than hang. Offer: fit on a subsample, assign the rest to the nearest cluster centroid, and label the column as what it is |
| `auto_k_kmeans` | BIC over minibatch fits; the existing `_kmeans_bic` works on a subsample |
| `pca` | `IncrementalPCA` over row batches — exact to floating point |
| `isomap` | Fit on a subsample; `Isomap.transform` extends to the rest |
| `laplacian_eigenmap` | sklearn's `SpectralEmbedding` has no `transform`. Nyström extension, or plot the subsample and say so |
| `tsne` | sklearn's `TSNE` has no `transform` either. Either add `openTSNE` as an optional dep (it does) or subsample honestly |
| `polygon_gate` / `rectangle_gate` | Unchanged — a boolean over 10⁷ rows is 10 MB |
| `class_map` | ELEMENTWISE block-wise remap through a lookup table |
| `train_classifier` / `predict` | Crops are read from the store by bbox, batched. A `CropDataset` over Zarr replaces the `crops` array argument; the model never sees the volume |

### I/O and display

| Step | Notes |
| --- | --- |
| `read_tiff` | Add a lazy path: `TiffFile.aszarr()` → `da.from_zarr`. Keep the eager one for small files |
| `read_zarr` | Already lazy. Gains multiscale awareness |
| OME-Zarr read/write | New, via `ome-zarr-py`: the working format, with a pyramid |
| Ingest | New: convert anything readable into a chunked OME-Zarr with a pyramid, once, before analysis. This is the honest answer to "my data is a 40 GB TIFF" |
| Thumbnails, gallery | Read from a coarse pyramid level and by bbox — already the cheapest possible access pattern |
| Session save | Table to Parquet, labels to Zarr, ledger to JSON. Slots into `SAVING_AND_ARCHIVING.md` |

## Reflection rules: the object the tiling cut in half

This is the part that matters, and the part with no library to defer to.

Two unrelated problems share the word:

**At the dataset border**, a neighborhood step's halo does not exist. A
blur at the edge of a 512³ tile in the middle of the volume reads real
neighbors; at the volume's own edge there are none. Padding mode has to be
declared and has to match what a whole-image run does — scipy's filters
default to `reflect`, so the executor pads dataset-border halos by
reflection and passes `mode="constant"` to the inner call, which makes the
tiled result *identical* to the whole-image result rather than merely
similar. One line, easy to get wrong, worth stating.

**At a tile seam**, an object is in two places at once. This is the real
problem.

```
   tile A                          tile B
 ┌─────────────────┬──────┐┌──────┬─────────────────┐
 │      core A     │ halo ││ halo │      core B     │
 │                 │  ▓▓  ││  ▓▓  │                 │
 │            ○────┼──▓▓──┼┼──▓▓──┼────○            │
 │          nucleus│  ▓▓  ││  ▓▓  │  the same       │
 │            (7)  │      ││      │  nucleus (3)    │
 └─────────────────┴──────┘└──────┴─────────────────┘
```

Both tiles segment it. A gets object 7, B gets object 3. They are one
nucleus, they have different local ids, each has a truncated shape, and
neither tile knows the other exists.

### The rules

A `SeamPolicy`, recorded on the run like a `Spacing` is, with a default that
is right for most data and named alternatives that are right for the rest:

```python
@dataclass(frozen=True)
class SeamPolicy:
    rule: str = OWN_BY_CENTROID
    halo: int | None = None          # None → from the step's contract
    max_object_extent: float | None = None   # physical; overrides the halo
    min_overlap: float = 0.5         # IoU in the shared halo, for MERGE_BY_OVERLAP
    border_objects: str = KEEP       # KEEP | DROP  — the *dataset* border
    pad_mode: str = "reflect"
    on_halo_exceeded: str = FLAG     # FLAG | RAISE | RETILE
```

**`OWN_BY_CENTROID`** (default). Every tile segments its core plus its halo.
An object is *kept* by exactly one tile: the one whose core contains its
centroid. Every other tile drops it. No merging, no matching, no
double-counting, and the kept copy is the complete object because it lay
entirely within core-plus-halo. Exact whenever the halo exceeds the object's
extent, which is the same condition the neighborhood steps already need.
Cheap: a centroid comparison, no cross-tile voxel reads.

**`MERGE_BY_OVERLAP`**. For objects genuinely larger than any affordable
halo — a vessel, a tubule, a whole glomerulus. Fragments are matched across
the seam by IoU within the shared halo region; a match above `min_overlap`
unions their ids. Union-find over the whole grid gives global ids in one
pass. This is the case the Java `ObjectStitcher` (a Smile kD-tree over
boundary objects) was written for, and the only one where its logic needs a
real port rather than a library.

**`RESEGMENT_SEAM`**. Neither of the above works when the segmenter itself
is not translation-invariant — which is exactly the deep-learning case, and
why it gets its own rule. Fragments touching a seam are re-segmented in a
window *centred on the seam*, so the object that was cut is now interior,
and the new answer replaces both fragments. Costs one extra inference per
seam-crossing object; buys a mask that no boundary ever ran through.

**`FLAG_ONLY`**. Keep both fragments, link them, mark them contested,
resolve nothing. For QC, for a first look at unfamiliar data, and as the
fallback when a rule's own precondition fails.

**`border_objects`** is a separate axis and must not be confused with the
above: dropping objects that touch the *dataset* boundary is a standard
cytometry choice (they are genuinely truncated specimens). Dropping objects
that touch a *tile* boundary would be a bug. The ledger distinguishes them
because the executor knows which faces are which.

### Verifying the halo

"Exact given a sufficient halo" is worth nothing if nobody checks the halo
was sufficient. The check is cheap and should always run:

- After segmenting core-plus-halo, any object whose voxels reach the outer
  edge of the halo was *not* fully contained. That is a definitive
  measurement, not an estimate.
- For distance-transform steps, an EDT value at the core boundary that has
  not saturated means the true nearest background lay outside the tile.

On a failure, `on_halo_exceeded` decides: `FLAG` (default — record it,
continue, and report the count), `RAISE`, or `RETILE` (re-run those tiles
with a doubled halo, which is bounded because the objects that need it are
few). A run that reports "3 objects of 412,000 exceeded the halo, flagged"
is a usable result. One that silently truncated three vessels is not.

### The ledger

The record of what was done, kept for the same reason
`Association.alternatives` is kept:

```python
@dataclass(frozen=True)
class Fragment:
    tile: tuple[int, ...]
    local_id: int
    voxels: int
    faces: frozenset[str]        # which tile faces it reaches: "-z", "+x", ...
    at_dataset_border: bool

@dataclass
class LabelLedger:
    """global id -> the fragments it was assembled from, and how."""
    fragments: dict[int, list[Fragment]]
    rule: dict[int, str]         # which rule decided this object
    evidence: dict[int, float]   # the IoU that merged it, 1.0 when uncut
    exceeded_halo: set[int]
```

Three columns join the measurement table from it — `n_fragments`,
`seam_rule`, `seam_confidence` — so a seam-crossing object is *gateable*.
Drawing a gate on `seam_confidence < 0.8` and opening the gallery is then
the review workflow, with no new UI at all; the existing
`AssociationReviewWidget` is the natural home for a richer one.

### Measuring an object that spans tiles

`regionprops_table` over the whole label array is not available. Two passes:

**Pass 1, streaming, over every tile.** Accumulate per global id:
`count`, `sum`, `sum of squares`, `min`, `max`, `sum of coordinates`, and
the union of bounding boxes. These compose across fragments exactly —
`count`, `mean`, `sum`, `stddev`, `min`, `max` and `centroid` all fall out
of them with no error term. Roughly 100 bytes per object; ten million
objects is a gigabyte, and it goes to Parquet if it has to.

**Pass 2, only for objects the ledger says were cut.** Everything else is
already exact. `threshold_mean` (its cutoff depends on the object's global
intensity range), any percentile, and every shape feature —
surface area, sphericity, skeleton length — are not additive and cannot be
recovered from fragment values. So for those objects, read the union bbox
back from the store and measure it directly. A single object's bbox fits in
RAM by definition; there are typically a few thousand of them; the cost is
minutes, and the result is exact rather than approximately-merged.

This two-pass split is the whole trick: **stream what composes, re-read what
doesn't.** It is why the answer can be exact rather than nearly right.

### Deep learning, specifically

Cellpose is the case the user's question is really about, and it is the
hardest one, because it breaks the assumption the other rules rest on:
segmenting core-plus-halo and segmenting the whole image do not give the
same masks. The flow field near a tile edge is computed from truncated
context, so a cell at the edge can be shaped differently, split, or missed —
and `OWN_BY_CENTROID` faithfully keeps a wrong mask.

What the plan does about it:

- **Halo from `diameter`, not from the budget.** The user already sets the
  expected object diameter; `1.5 × diameter` is a defensible default halo
  and can be derived automatically. If the GPU budget cannot fit
  core-plus-halo, the *core* shrinks — the halo does not.
- **`RESEGMENT_SEAM` is the default rule for this step**, not
  `OWN_BY_CENTROID`. Any object touching the seam is re-run in a
  seam-centred window where it is interior. This is the reflection rule the
  question asks for, and it is the only one that produces a mask no tile
  boundary influenced.
- **Reuse cellpose's own z-stitching.** For a 2D-plane-wise run,
  `stitch_threshold` already links masks across z by IoU. Do not duplicate
  it for the z axis; apply our rules to XY only.
- **GPU budget is its own budget**, from `torch.cuda.mem_get_info` plus the
  cached calibration probe, and it is usually far smaller than the CPU one.
  The tile plan carries both.
- **Batch, and resume.** Inference over 4,096 tiles takes hours. Each tile's
  result is written to the label store as it completes and recorded in a
  manifest, so a crashed or cancelled run resumes rather than restarts.
  Non-negotiable for a step measured in hours.
- **`do_3D` is memory-hostile** and scales worse than the tile volume.
  Default to plane-wise plus `stitch_threshold` above a size threshold, and
  say which mode ran in the provenance, because they do not give the same
  answer.

### Invariance: the test that makes any of this trustworthy

Two properties, as golden tests against the existing fixtures:

1. **Single tile equals whole image.** With a tile large enough to hold the
   dataset, the blocked executor's output is bit-identical to the direct
   call. This catches padding, halo-trimming and off-by-one errors, and it
   costs nothing to run in CI on a small volume.
2. **Tiling is invariant.** The same volume processed at three different
   tile sizes (and at a deliberately adversarial offset, so seams fall
   through objects) gives the same object count, and identical measurements
   for every object the ledger says was uncut. Seam-crossing objects are
   compared against the whole-image run within a stated tolerance — and
   with the exact second pass, that tolerance should be zero for everything
   except the approximate steps, which are the ones already labelled
   approximate.

If (2) fails, the reconciliation is wrong. It is the acceptance criterion
for the whole of Phase L3.

## Analysis at scale

Object count, not voxel count, is the constraint here, and the numbers are
smaller than they look: 10⁷ objects × 40 `float32` features is 1.6 GB, which
mostly fits. What does not fit is 10⁸, or the O(n²) methods at any size.

- **The table goes to Parquet, queried through DuckDB** — which is already a
  dependency and already the store. `MeasurementStore.register` gains a
  path-backed sibling; the SQL interface does not change. DuckDB handles
  aggregations and joins larger than memory itself, so `cell_features` and
  every gate count come along for free.
- **Estimators**: as the table above.
- **The scatter plot** cannot draw 10⁷ points, and should not try. Above a
  threshold, render a 2D histogram (`histogram2d` + `imshow`, no new
  dependency) with the same axes and colormap; gate *membership* is still
  evaluated exactly on every row, so the gate is not an approximation even
  though its backdrop is.
- **The gallery** reads crops by bbox from the store, which is the access
  pattern Zarr is best at. It gets faster on large data, not slower.

## napari integration

What to lean on rather than build:

- **Lazy layers.** napari renders a Dask- or Zarr-backed layer by pulling
  the chunks it needs. Adding an OME-Zarr as `multiscale=True` gives
  pyramid rendering for free; `napari-ome-zarr` is an existing reader.
  The builder's `active_image()` must stop calling `np.asarray` and hand
  the executor the lazy array instead — that one line is most of the GUI
  change.
- **`layer.corner_pixels`** gives the currently displayed extent. That is
  the hook for the feature this design most wants: **preview on what you
  are looking at.** Step cards run their preview over the visible ROI at
  the current zoom level, interactively, and the full run is a separate,
  explicit, long-running commit. Tuning a threshold against a 40 GB volume
  is otherwise unusable.
- **Progress and cancellation** via `napari.utils.progress` and a worker
  thread (`superqt`/`thread_worker`), reporting tiles completed, elapsed,
  and estimated remaining — driven by the tile plan, which knows the
  denominator.
- **Labels layers** display from Zarr the same way images do, so results
  come back into the viewer without materializing.

## Module layout

One new package in `vtea-core`, plus small changes at the seams of existing
ones:

```
vtea_core/blocked/
    budget.py      MemoryBudget: detection (cgroup, psutil, torch), overrides
    plan.py        TilePlan: tiles, halos, chunk snapping, the human summary
    store.py       Zarr-backed scratch for out-of-core intermediates; lifecycle
    executor.py    BlockedPipeline: runs a Pipeline over a TilePlan; resume
    stats.py       The streaming statistics pass (histograms, min/max)
    reconcile.py   SeamPolicy, Fragment, LabelLedger, the rules, halo checks
    measure.py     Accumulators, the exact second pass, table assembly
    table.py       Parquet/DuckDB-backed feature table
```

Changed elsewhere, and deliberately little:

- `workflow/wiring.py` — the four new `StepIO` fields. Declarative, no
  behavior.
- `workflow/pipeline.py` — `Pipeline.run` gains an optional executor; the
  in-memory path is unchanged and stays the default.
- `io/` — lazy TIFF, OME-Zarr read/write, the ingest entry point.
- `objects/ownership.py` — the sparse, mask-restricted representation.
- `measurements/store.py` — the Parquet-backed registration.
- `vtea_napari` — lazy `active_image()`, the budget control, ROI preview,
  the progress/cancel worker, seam review in the association review widget.

## Phases

| Phase | Content | Est. |
| --- | --- | --- |
| **L0 — Budget and plan** | `MemoryBudget`, `TilePlan`, the `StepIO` scaling fields, the plan summary in the GUI. No execution change; everything downstream depends on it | 1–2 wk |
| **L1 — Lazy I/O and the store** | OME-Zarr read/write with pyramids, lazy TIFF, ingest, `ZarrScratch`, lazy napari layers | 2–3 wk |
| **L2 — The executor: elementwise, neighborhood, global stats** | Covers all of `imageprocessing` and `threshold_mask`. Padding, halo trimming, and the "single tile equals whole image" test | 2–3 wk |
| **L3 — Labels across tiles** | The ledger, the four rules, halo verification, blocked connected components / watershed / derived segmentations. **The crux**; gated on the invariance test | 4–6 wk |
| **L4 — Measurements at scale** | Accumulators, the exact second pass, the Parquet/DuckDB table, the seam columns | 2–3 wk |
| **L5 — Deep learning** | Blocked Cellpose, GPU budget and calibration, `RESEGMENT_SEAM`, resumable runs | 2–3 wk |
| **L6 — Objects at scale** | Sparse ownership, blocked association scoring, `build_cells`/`cell_features` through DuckDB | 2–3 wk |
| **L7 — Analysis and explorer** | Streaming estimators, binned scatter, gallery from the pyramid | 2–3 wk |
| **L8 — GUI** | ROI preview, background runs with progress and cancellation, resume, seam review | 2–3 wk |

**Total: roughly 18–28 engineer-weeks.** L0–L2 are worth landing on their
own — they make a large dataset *openable and preprocessable*, which is
most of the day-to-day pain — and L3 is where the intellectual risk is
concentrated.

## Open questions

- **Zarr 2 or 3.** `vtea-core` pins `zarr>=2.16,<3`. Zarr 3 brings sharding,
  which matters for a pyramid with millions of chunks on a network
  filesystem, but `ome-zarr-py` and the wider ecosystem moved at their own
  pace. Decide before L1 rather than during it.
- **Dask scheduler.** Threads are enough for a workstation; `distributed`
  buys a cluster and a dashboard, and costs a dependency and a failure mode.
  Recommend: threaded by default, `distributed` opt-in, and keep the
  executor agnostic.
- **Which reconciliation rule is the default per step.** Proposed above
  (`OWN_BY_CENTROID` generally, `RESEGMENT_SEAM` for Cellpose), but this is
  a scientific judgement as much as an engineering one and should be
  confirmed against real tissue with vessels and tubules in it, not only
  against nuclei.
- **Does anything need true random-access editing of a large label image?**
  Manual correction of a 33 GB label store is a different problem (chunk
  write amplification, undo). Out of scope here; flagged so it is not
  assumed to be in.
- **Time series.** `_to_czyx` currently rejects `T` outright. Large data and
  time are likely to arrive together; the axis model should be decided
  before L1 fixes a store layout, even if the implementation waits.
