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
| Reading back what we wrote, in a test | `ome-zarr-py` | trusting our own writer |
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
    scaling: Scaling = DEFAULT_SCALING

@dataclass(frozen=True)
class Scaling:
    mode: str = ELEMENTWISE           # see "Block modes" below
    halo: HaloSpec = HaloSpec()       # fixed voxels, or derived from a param
    bytes_per_voxel: int = 8          # live intermediates, per input voxel
    exactness: str = EXACT            # EXACT | EXACT_WITH_HALO | APPROXIMATE
    needs_reconciliation: bool = False     # objects have to be joined across seams
    variants: Mapping[str, Scaling] = {}   # keyed on a parameter's value
```

`needs_reconciliation` turned out to be the field that decides whether a
step can be tiled at all, and it is *not* implied by the block mode — a
distinction L2 found the hard way. `label_components` is a one-voxel
neighbourhood step by every mechanical measure, shape-preserving and
cheap, and tiling it would still be wrong: every tile numbers its objects
from 1, so object 7 means something different in each. `filter_by_size` is
elementwise and equally unsafe, because the size it filters on belongs to a
whole object. Both wait on the ledger. Steps that only *carry* ids forward —
`expand_labels`, `subtract_labels`, `class_map` — assign nothing and need
nothing global, so they tile today. The executor refuses on this flag by
name, because a plausible-looking wrong label image is the worst outcome
available.

`variants` earns its place on the two steps whose scaling depends on a
parameter rather than on the function: `threshold_mask` is elementwise with
`method="fixed"` and needs a whole-image histogram with `method="otsu"`, and
those are not the same step to a planner.

`bytes_per_voxel` is what turns a budget into a tile shape:

```
usable = budget.total_bytes * budget.fraction / budget.workers
voxels_per_tile = usable / step.bytes_per_voxel
```

For `watershed_split` at ~35 B/voxel (uint16 input, bool mask, float64 EDT,
int32 markers, int32 labels, plus scipy's internal copies) over the 2048 ×
2048 × 2000 volume above, with a 64-voxel halo and 128³ storage chunks -
these are the planner's actual output, not an illustration:

| Machine | Usable | Tile | Tiles | Read amplification |
| --- | --- | --- | --- | --- |
| 8 GB laptop | 4.8 GB | 384³ | 216 | 3.5× |
| 32 GB workstation | 19.2 GB | 512 × 768 × 768 | 36 | 2.2× |
| 128 GB server | 76.8 GB | 1024³ | 8 | 1.5× |

The last column is the halo's real cost and the argument for more memory:
the laptop reads the dataset three and a half times over, the server one and
a half.

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

Six modes. The whole protocol classifies into them.

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
- **ACCUMULATE** — reduces voxels to per-object rows. Blocked by
  accumulating partial sums per object and merging them, with a second pass
  over the objects a seam cut (see "Measuring an object that spans tiles").
- **TABLE** — consumes the feature table, not voxels. Scales with object
  count, and is a different problem (see "Analysis at scale").

### Image processing

| Step | Mode | Halo | Exactness | Notes |
| --- | --- | --- | --- | --- |
| `gaussian_blur` | NEIGHBORHOOD | `ceil(4σ)` | exact w/ halo | scipy truncates at 4σ; halo follows `sigma` |
| `median_filter` | NEIGHBORHOOD | `radius` | exact w/ halo | |
| `enhance_contrast` (`normalize`) | GLOBAL_STAT | — | exact | streaming min/max, then rescale per block |
| `enhance_contrast` (`equalize`) | NEIGHBORHOOD | kernel | **approximate** | CLAHE adapts per kernel window, and skimage derives that window from the image it is handed - so the same data equalizes differently depending on how it was divided. `enhance_contrast` therefore takes an explicit `kernel_size` (added in L0), which should divide the tile shape |
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
| `distance_ownership` | OBJECT_LOCAL | Already windows per marker (`_insert_claim`). What changes is the output: a dense top-k array becomes **mask-restricted sparse** — a sorted array of flat indices with owners and claims beside them, grouped by tile so a later pass finds one tile's voxels without searching. For a 5%-foreground volume that is 201 GB → about 13 GB, the same order as the image rather than six times it. Probabilities go to float32: a posterior meaningful to seven decimal places is not a posterior anybody has |
| `build_cells` | TABLE | A graph over associations. Memory scales with object count; fine to ~10⁷, worth measuring beyond |
| `cell_features` | TABLE | Grouped aggregation — push into DuckDB rather than pandas |

### Measurements

| Step | Mode | Notes |
| --- | --- | --- |
| `extract_measurements` | accumulate + a second streaming pass | See "Measuring an object that spans tiles" |
| `extract_measurements_by_channel` | as above, per channel | Geometry appears once; every intensity column carries the channel it was measured on. One channel is read at a time, so a four-channel volume costs the same per tile as a one-channel one |
| `weighted_measurements_by_channel` | accumulate | Weighted sums are additive, and `min`/`max` are over every voxel an owner has a claim on, so unlike `threshold_mean` nothing depends on knowing the whole object first — one pass is the entire calculation |

### Analysis

| Step | Strategy at scale |
| --- | --- |
| `kmeans` | `MiniBatchKMeans` above a row threshold; identical interface, near-identical result |
| `gaussian_mixture` | Fit on a stratified subsample, `predict` over all rows in batches. Exact for the assignment given the fitted model |
| `hierarchical` | O(n²) — **not feasible** above ~10⁵ objects and should say so rather than hang. Offer: fit on a subsample, assign the rest to the nearest cluster centroid, and label the column as what it is |
| `auto_k_kmeans` | BIC over minibatch fits; the existing `_kmeans_bic` works on a subsample |
| `pca` | `IncrementalPCA` over row batches. Close but *not* the same arithmetic: it merges a partial SVD per batch, so where two eigenvalues are nearly equal the subspace is right and the axes inside it can rotate. Measured on a case with one dominant direction, the leading component agrees to 1 − 2e−9 and the second to r = 0.94 |
| `isomap` | Fit on a subsample; `Isomap.transform` extends to the rest |
| `laplacian_eigenmap` | sklearn's `SpectralEmbedding` has no `transform`, and inventing one is a research decision wearing an implementation's clothes. Rows outside the sample come back **NaN** — what "not embedded" honestly looks like, rather than a fabricated position a plot would treat as a measurement |
| `tsne` | sklearn's `TSNE` has no `transform` either. Either add `openTSNE` as an optional dep (it does) or subsample honestly |
| `polygon_gate` / `rectangle_gate` | Unchanged — a boolean over 10⁷ rows is 10 MB |
| `class_map` | ELEMENTWISE block-wise remap through a lookup table |
| `train_classifier` / `predict` | Crops are read from the store by bbox, batched. A `CropDataset` over Zarr replaces the `crops` array argument; the model never sees the volume |

### I/O and display

| Step | Notes |
| --- | --- |
| `read_tiff` | Keep the eager path for files that fit. `open_tiff` (new) maps the file a plane at a time, via Dask over `tifffile`'s pages |
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

### Two decisions, not one

How a protocol puts a cut object back together is really two settings, and
collapsing them into one list of rules (as an earlier draft of this document
did) hides the more consequential of the two.

**Does the tiling overlap at all?** An overlapping tiling gives every tile a
halo, so a tile can hold a *complete* copy of an object that straddles its
core boundary. It is paid for in redundant computation — the planner reports
that as a read-amplification factor, and on the worked example it runs from
1.5× on a large budget to 3.5× on a laptop. An abutting tiling computes every
voxel exactly once, which for an expensive segmenter is the difference
between a nine-hour run and a three-hour one; the price is that no tile has
a complete copy of anything a seam crosses, so a cut object can only be
reassembled from its pieces, never chosen.

**How is a fragment in one tile recognised as the same object as a fragment
in another?** Four answers, of which the last is "it isn't".

Not every pairing of the two is valid — overlap matching needs an overlap,
face-adjacency matching only means anything without one — and the presets
below are the combinations that are.

### The strategies

```python
@dataclass(frozen=True)
class SeamPolicy:
    tiles: str = OVERLAPPING      # OVERLAPPING | ABUTTING
    matching: str = OVERLAP       # OVERLAP | CENTROID | TOUCHING | NONE
    resolution: str = OWN         # OWN | MERGE | RESEGMENT | FLAG

    halo: int | None = None            # None → from the step's contract
    max_object_extent: float | None = None  # physical; sets the halo directly
    min_overlap: float = 0.5           # IoU, for matching = OVERLAP
    max_centroid_distance: float | None = None  # physical, for matching = CENTROID
    border_objects: str = KEEP         # KEEP | DROP — the *dataset* border
    pad_mode: str = "reflect"
    on_halo_exceeded: str = FLAG       # FLAG | RAISE | RETILE

    # The four a user actually picks from, plus the two specialisations.
    @classmethod
    def overlap_match(cls, *, merge: bool = False) -> SeamPolicy: ...
    @classmethod
    def centroid_match(cls) -> SeamPolicy: ...
    @classmethod
    def touching_merge(cls) -> SeamPolicy: ...
    @classmethod
    def no_merge(cls) -> SeamPolicy: ...
```

| Preset | Tiles | Matching | Result | Cost | Fails by |
| --- | --- | --- | --- | --- | --- |
| **1. Overlap matching** *(the default)* | overlapping | IoU in the shared region | one complete copy kept, or fragments unioned | halo, plus one comparison pass over the halo regions | needing the halo to exceed the object |
| **2. Centroid matching** | overlapping | nearest centroid within a distance | fragments unioned | halo, plus a kd-tree over a table — no voxel reads | over-merging, and under-merging elongated objects |
| **3. Edge-touching merge** | abutting | voxels face-adjacent across the seam plane | fragments unioned | none beyond the seam planes | merging cells that merely abut at the seam |
| **4. No merge** | abutting | none | every fragment is its own object | nothing | inflating the object count by the seam-exposed fraction |

**1 — Overlap matching. This is the default**, and a user who never opens
the setting gets it. It is the most accurate of the four and the only one
whose correctness does not depend on a property of the specimen — the other
three each trade accuracy for time in a way that is right for some tissue
and wrong for other tissue, which is a judgement a default should not make
on someone's behalf. It costs a halo, and the plan's position is that
paying it is the right thing to do until a user says otherwise.

Compare the two tiles' labellings of the region they both segmented; IoU
above `min_overlap` says two fragments are one object. Unambiguous, because it is the same voxels being compared, and it is
the only matching that stays correct when objects are packed tightly.
Two things can then be done with the correspondence, and the choice is the
`resolution` setting rather than a separate rule:

- `OWN` — exactly one tile keeps the object and the rest drop it, the tile
  holding the most of it in its core winning. Right when the halo exceeds
  the object, so at least one copy is complete.
- `MERGE` — union the fragments. Right when nothing is complete anywhere: a
  vessel, a tubule, a whole glomerulus, larger than any affordable halo.
  Union-find over the grid gives global ids in one pass. This is the case the
  Java `ObjectStitcher` was written for.

With a sufficient halo and a translation-invariant segmenter, every tile's
copy is the *same* copy, so `OWN` can skip the comparison entirely and pick
by centroid position — the cores partition the volume exactly, so exactly one
core contains any given centroid. That fast path is worth having for nuclei
and worth distrusting elsewhere: see "Verifying the halo" for the silent
double-count it produces when the halo turns out to be too small.

One cost of overlap matching that the read-amplification figure does not
show: each tile's own labelling of its whole block has to be *kept* until
its neighbours have been compared with it, because that is what the IoU
compares. That is the same factor again in scratch - 1.5x to 3.5x the label
volume, on disk, temporarily. It is the price of the only unambiguous
matching, and it is the reason the other three exist.

**2 — Centroid matching.** Pair fragments across a seam by centroid
proximity instead of by voxels. It reads a table rather than an image, so it
costs a kd-tree query per boundary object and no I/O — `scipy.spatial.cKDTree`,
which `objects/scoring.py` already uses for candidate association, and the
same structure the Java `ObjectStitcher` used for exactly this job.

It **may over-merge**, and the plan should say why rather than only that:
two genuinely distinct cells lying either side of a seam can have centroids
closer together than one cut cell's two halves. Densely packed tissue is
where that happens, which is most tissue. It also *under*-merges the
opposite case — an elongated object cut across its long axis has two
fragment centroids far apart. So it is the right choice when objects are
round, well separated, and numerous enough that the voxel comparison is the
bottleneck, and the wrong one for packed epithelium. `max_centroid_distance`
is the knob, and it is physical, so it means the same thing in an
anisotropic stack.

**3 — Edge-touching merge.** No halo: tiles abut, every voxel is segmented
once. Fragments whose voxels are face-adjacent across the shared plane are
unioned. This is the cheapest way to get whole objects out of a tiled
segmentation, and for an expensive segmenter the saving is the whole point —
no redundant inference at all.

What it gets wrong is specific and unavoidable: **anything touching across
the seam is merged, including two cells that were merely adjacent there.**
The seam plane is a line of systematic under-segmentation, and it cannot be
distinguished from correct merging by any amount of care, because the
information that would settle it — what the region looks like with context
on both sides — was never computed. The second-order problem is that each
half was segmented from truncated context, so the two halves may not even
line up; the merged object can have a visible step at the seam. Best suited
to sparse objects and a segmenter that does not need much context.

**4 — No merge.** Abutting tiles, no correspondence, every fragment its own
object. Nothing is reconciled and nothing pretends to be.

This is a real choice, not a null option, and the arithmetic says when. The
fraction of objects a seam touches is about `1 - ((L - d) / L)³` for tiles of
edge *L* and objects of diameter *d*:

| | 10-voxel objects | 20-voxel | 40-voxel |
| --- | --- | --- | --- |
| 384³ tiles | 7.6% | 14.8% | 28.1% |
| 512³ tiles | 5.7% | 11.3% | 21.7% |
| 1024³ tiles | 2.9% | 5.7% | 11.3% |

So at a laptop's tile size, one nucleus in seven is cut. That is not a
rounding error, and "no merge" is only defensible when it is paired with
`border_objects = DROP` extended to tile faces — the standard cytometry
exclusion, applied to seams as well as to the specimen edge. Then the count
is honest and the sample is smaller, which is a trade a cytometrist already
understands. Used *without* that exclusion it inflates the object count and
skews the size distribution low, so the ledger reports the fragmented
fraction whether or not anyone asked. It is also the right setting for a
fast preview, where a seam artefact costs nothing and an hour does.

One interaction to know about, because it is silent and it compounds: **a
size filter after a no-merge segmentation deletes pieces of real objects.**
A minimum size is chosen for whole objects; the things being filtered are
fragments, each smaller than the object it came from. So a threshold that
would have kept everything removes parts of several, and the resulting
count can come out *lower* than the correct answer rather than higher - the
inflation and the filtering pull in opposite directions and neither is
visible in the output. Pair no-merge with dropping seam-touching objects,
which removes whole fragments rather than trimming them, or filter by size
under a merging strategy.

Note what the table also says: **more memory is a merge strategy.** Doubling
the tile edge roughly halves the number of objects any of this applies to.

**`RESEGMENT`** is a `resolution`, not a matching mode, and it is not
based on overlap. The strategies above all assume at least one tile's answer
is worth keeping or joining. When the segmenter is not translation-invariant
— the deep-learning case — that is what fails: the flow field near a tile
edge is computed from truncated context, so every copy is shaped by a
boundary that has nothing to do with the specimen, and the better of two
wrong masks is still wrong. The fix is to remove the thing being chosen
between: re-run on a window *centred on the seam*, where the object is
interior, and let that replace the fragments. Matching is used only to find
which objects need it. Note the corollary for strategies 3 and 4 — abutting
tiles give a learned segmenter *no* context at a seam, so that pairing is
the least accurate available, and that is the price of the inference it
saves.

Two things building it added. A window larger than a tile is **declined
rather than attempted**: it would need more memory than the plan was built
for, which is the one thing the plan exists to prevent, and the stitched
answer stands with the reason recorded. And a re-segmented object
**absorbs any object lying wholly inside it**, where the window contains
all of that object — because the re-segmentation is the better evidence,
and a sliver a tile boundary left behind is a piece of the object rather
than an object. Without that rule the count is inflated by exactly the
fragments the strategy was supposed to fix, and they hold voxels the real
object should own.

**`FLAG_ONLY`** keeps everything, marks it contested, and resolves nothing.
For QC, for a first look at unfamiliar data, and as the fallback when a
strategy's own precondition fails.

**A fifth, named but not proposed.** Several tiles each segmenting the same
seam region is a multi-rater consensus problem rather than a choice, and
`vtea.objects` already ships STAPLE (see PORT_PLAN.md's inventory). A
`CONSENSUS` resolution combining the overlapping copies rather than picking
one would be a port rather than research, if seam review ever shows that
picking is what costs accuracy. Out of scope for L3; recorded so it is not
reinvented.

### What the strategy does to the tile plan

Choosing `tiles = ABUTTING` is not "set the halo to zero", and getting that
wrong would break every filter in the protocol. A halo exists for two
unrelated reasons, and `HaloSpec` already distinguishes them:

- **Because a kernel reaches** — 4σ for a Gaussian, the radius for a median
  filter, the distance for `expand_labels`. These are correctness
  requirements of the function and are not the seam policy's to negotiate.
  They stay.
- **Because objects are big** — the `object_extent` term, which exists only
  so that a tile can hold a whole object. That is the one an abutting
  strategy drops.

So `ABUTTING` means the seam policy contributes no object-scale halo; a
protocol that blurs before it segments still gets its 4σ. The plan records
which strategy produced it, because a result computed under one and compared
against a result computed under another is not a comparison.

**`border_objects`** is a separate axis from all of the above and must not be
confused with them: dropping objects that touch the *dataset* boundary is a
standard cytometry choice, since they are genuinely truncated specimens.
Dropping objects that touch a *tile* boundary is a bug under strategies 1–3
and a requirement under strategy 4. The ledger distinguishes the two kinds of
face because the executor knows which is which.

### Verifying the halo

"Exact given a sufficient halo" is worth nothing if nobody checks the halo
was sufficient. The check is cheap and should always run:

- After segmenting core-plus-halo, any object whose voxels reach the outer
  edge of the halo was *not* fully contained by that tile. That is a
  definitive measurement, not an estimate.
- **An object is only in trouble when *every* tile that saw it saw it cut
  off.** One tile's truncated view is normal - it is what a halo is for, and
  the neighbour holding the rest may well have the whole thing. Reporting
  per fragment rather than per object turns an ordinary reconciliation into
  a page of warnings: on a 27-tile run of well-behaved nuclei it flagged a
  third of them, all of which were fine.
- For distance-transform steps, an EDT value at the core boundary that has
  not saturated means the true nearest background lay outside the tile.

On a failure, `on_halo_exceeded` decides: `FLAG` (default — record it,
continue, and report the count), `RAISE`, or `RETILE` (re-run those tiles
with a doubled halo, which is bounded because the objects that need it are
few). A run that reports "3 objects of 412,000 exceeded the halo, flagged"
is a usable result. One that silently truncated three vessels is not.

Under an abutting strategy (3 or 4) there is no halo to exceed, so the check
has nothing to measure and the equivalent fact is simply *which objects
touch a tile face* - which the ledger records for every object regardless.
That is the number strategy 4 has to report, and the one strategy 3 uses to
decide what to try to merge.

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
    policy: SeamPolicy           # the strategy the whole run used
    fragments: dict[int, list[Fragment]]
    decided_by: dict[int, str]   # matching + resolution, per object
    evidence: dict[int, float]   # the IoU or distance that joined it, 1.0 when uncut
    exceeded_halo: set[int]
    touches_seam: set[int]       # every object on a tile face, merged or not

    @property
    def seam_exposed_fraction(self) -> float: ...
```

`policy` is on the ledger and not only in the run's configuration because a
result computed under strategy 1 and a result computed under strategy 4 are
different measurements of the same specimen, and a table that cannot say
which it is cannot be compared with anything.
`seam_exposed_fraction` is what strategy 4 has to report whether or not
anyone asked: at a laptop's tile size it is one object in seven.

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

**Pass 2, for the features that do not compose.** `threshold_mean` (its
cutoff depends on the object's global intensity range), any percentile, and
every shape feature — surface area, sphericity, skeleton length — cannot be
recovered from fragment values.

The plan originally proposed reading each cut object's bounding box back
from the store for these. Building it corrected that for the case that
actually exists today: **a second streaming pass is simpler and cheaper than
random access, wherever the missing quantity is derivable from the
accumulators.** `threshold_mean` is exactly that case — once every object's
minimum and maximum are known, its cutoff is known, so a voxel counts if it
clears the cutoff of the object it belongs to. One lookup per voxel, one
masked `bincount` per tile, fully vectorized, no sorting and no per-object
loop. Two sequential passes over the data, and a result identical to the
whole-image call.

Random access earns its place when shape features arrive, since no
accumulator can hold a surface area — and the ledger already knows which
objects would need it.

So the trick is still **stream what composes**; what changed is that the
second half is usually "stream it again" rather than "re-read what
doesn't". It is why the answer is exact rather than nearly right.

### Deep learning, specifically

Cellpose is the case the user's question is really about, and it is the
hardest one, because it breaks the assumption the other rules rest on:
segmenting core-plus-halo and segmenting the whole image do not give the
same masks. The flow field near a tile edge is computed from truncated
context, so a cell at the edge can be shaped differently, split, or missed —
and any strategy that picks between the copies faithfully keeps a wrong
mask.

What the plan does about it:

- **Halo from `diameter`, not from the budget.** The user already sets the
  expected object diameter; `1.5 × diameter` is a defensible default halo
  and can be derived automatically. If the GPU budget cannot fit
  core-plus-halo, the *core* shrinks — the halo does not.
- **`RESEGMENT` is the default resolution for this step**, over overlap
  matching. Any object touching the seam is re-run in a
  seam-centred window where it is interior. This is the reflection rule the
  question asks for, and it is the only one that produces a mask no tile
  boundary influenced.
- **Reuse cellpose's own z-stitching.** For a 2D-plane-wise run,
  `stitch_threshold` already links masks across z by IoU. Do not duplicate
  it for the z axis; apply our rules to XY only. Linking objects between
  adjacent planes is the same problem VTEA solves across tile boundaries,
  and where the library already has an answer for one axis, using it beats
  having two implementations that can disagree.
- **GPU budget is its own budget**, from `torch.cuda.mem_get_info` plus the
  cached calibration probe, and it is usually far smaller than the CPU one.
  The tile plan carries both. The probe doubles a synthetic tile until the
  device refuses, then bisects once - which turns a factor-of-two answer
  into a few-percent one for one extra attempt, and a factor of two in tile
  volume is a factor of two in the number of inferences. It distinguishes
  an out-of-memory refusal from every other failure: a broken driver
  treated as "too big" would shrink the tile to nothing and blame the data.
  A calibration records *how much of the card was free when it was taken*,
  so that a measurement from an empty card is scaled down on a busy one
  rather than believed - and never scaled up, since free memory is not
  evidence the model would use it.
- **Batch, and resume.** Inference over 4,096 tiles takes hours. Each tile's
  result is written to the label store as it completes and recorded in a
  manifest, so a crashed or cancelled run resumes rather than restarts.
  Non-negotiable for a step measured in hours. The manifest is append-only
  JSON Lines rather than a file rewritten per tile: rewriting is quadratic
  in the number of tiles, and a rewrite interrupted halfway is exactly the
  failure it exists to survive. A torn final line costs one tile and keeps
  the rest. It refuses to resume a run whose tile size, halo, policy or
  segmenter differs, because those change which objects a seam cuts and a
  manifest from a different plan describes different objects wearing the
  same tile indices.
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

**Property (2) applies to the strategies that claim it, and only those.**
Writing it as a blanket rule would make two legitimate settings look like
bugs:

| Strategy | Invariant under retiling? |
| --- | --- |
| 1. Overlap matching | **yes**, exactly — the acceptance criterion for L3 |
| 2. Centroid matching | yes for well-separated objects; a disagreement in packed tissue is the over-merge, and the test measures its rate rather than forbidding it |
| 3. Edge-touching merge | **no**, by construction — a seam merges what it runs through, so the answer depends on where it runs. The test pins the *direction*: object count never rises when tiles get larger |
| 4. No merge | **no**, by construction — the test asserts the count matches the seam-exposed fraction predicted from the tile size, which is what makes the inflation a known quantity rather than a surprise |

So: if (2) fails under strategy 1, the reconciliation is wrong, and that is
the acceptance criterion for the whole of Phase L3. Under 3 and 4 the same
comparison is not a pass/fail but the measurement that tells a user what
they gave up for the speed — which is worth running on their own data, not
only in CI.

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

## The store: Zarr 2 now, Zarr 3 ready

**Decided: Zarr 2 (`zarr>=2.16,<3`), writing OME-NGFF 0.4.** The pin stays
where it is.

The reason is not inertia. NGFF 0.4 *is* a Zarr-2 spec, and the tools a
collaborator will open the store with — `ome-zarr-py`, `napari-ome-zarr`,
Fiji's N5/Zarr reader, `bioio` — read v2 first and v3 with varying
enthusiasm. A v3 store written today is one that the people the data is
shared with cannot open, which is the opposite of the point of adopting a
community format at all.

What Zarr 3 buys, so the decision is revisitable rather than forgotten:
**sharding**. A 33 GB volume at 256³ chunks is ~500,000 chunk files on the
finest pyramid level alone. On a laptop's SSD that is untidy; on a network
filesystem or object store it is the actual bottleneck, and sharding is the
fix. NGFF 0.5 also lands there.

Being "ready" has to mean concrete rules, not an intention:

1. **One module touches `zarr`.** All of it — including `da.from_zarr` /
   `da.to_zarr`, which are zarr use by another name — lives in
   `io/store.py`. Nothing else in either package imports `zarr`. The
   migration is then a module rewrite with a test suite already pointed at
   it, not an archaeology exercise.
2. **Only API that exists in both versions.** `zarr.open_group`, array and
   group `attrs`, array creation through our own wrapper. No `.store`
   internals, no reading `.zarray` by hand, no `numcodecs` objects crossing
   the module boundary — a compressor is named by string in our config and
   translated inside.
3. **The NGFF version written is a constant** (`NGFF_VERSION = "0.4"`), and
   **the reader accepts 0.4 and 0.5 from the start.** Reading is where a
   version mismatch actually costs a user something; writing can lag safely.
4. **Every store carries `vtea_format_version` in its root attrs**, so a
   future reader knows what it is looking at instead of inferring.
5. **Chunk shapes are chosen to survive becoming shard-inner chunks** —
   128–256 voxels per axis, not 1024³ blocks that a shard could never
   usefully subdivide. This costs nothing now and is the difference between
   "enable sharding" and "rewrite every store" later.
6. **Chunk keys are written nested** (`0/1/2`, not zarr 2's default
   `0.1.2`). NGFF asks for it, Zarr 3 does it as standard, and a store
   written the other way is worse than broken: it opens, reports the right
   shape, and hands back fill values, because the chunk files are not where
   the reader looks. This was found by the interoperability test below and
   not by reading the spec, which is the argument for having the test.

Revisit when `ome-zarr-py` and `napari-ome-zarr` both read v3 by default, or
when chunk count becomes a measured bottleneck on real data — whichever
comes first.

Two things L1 turned up that are worth recording against that decision,
because both are the ecosystem moving:

- **`ome-zarr-py` 0.11 and later require Zarr 3.** The interoperability
  test therefore pins `ome-zarr<0.11`, test-only. This is a real cost of
  the Zarr 2 choice and the clearest signal so far of when to revisit it.
- **`tifffile`'s `aszarr` store now requires Zarr 3 too**, which took the
  originally planned lazy-TIFF path off the table. What replaced it is
  better: `open_tiff` builds a Dask array from the file's own pages, one
  plane per chunk, which keeps zarr out of the TIFF path entirely. It also
  makes the case for `ingest` sharper — a TIFF's finest unit is a plane, so
  pulling a small cube out of a large stack still costs a full plane per
  slice until the data is converted.

## Axes: five in the store, four in memory

**Decided: the store is 5D `TCZYX` from day one; the in-memory model stays
4D `CZYX`; `T` is squeezed on read; `T > 1` raises a specific, named
error.** Time-series *compatibility* is built now. Time-series *analysis* is
not, and the plan says so rather than leaving a half-built axis around.

Why build the compatibility now, when nothing uses it:

- OME-NGFF's canonical order is `TCZYX` and readers expect it. Writing 4D
  today and 5D later means converting every store written in between.
  Writing 5D with `T=1` costs one axis of length one and nothing else — it
  is already a valid time-series store that happens to have one timepoint.
- The changes real time support needs are individually small and spread
  wide: `_to_czyx`, `VolumeDataset`, `Spacing.for_ndim`, the measurement
  table's id space (an object becomes `(t, id)`), the tile plan's axis
  handling, and the GUI's axis pickers. Fixing the axis model once, now,
  while there is only one caller of each, is far cheaper than after the
  blocked executor has hardcoded four dimensions in nine places.
- And one honest caveat: **linking objects across timepoints is tracking,
  not association.** It is a different problem with different algorithms,
  and nothing in this plan should imply that a `T` axis brings it along.

| Built now | Deliberately deferred |
| --- | --- |
| `vtea_core.data.axes.Axes` — a validated axis-order string, canonical `TCZYX`, generalising the reordering `_to_czyx` does by hand | Any per-timepoint execution loop |
| Readers parse and record the timepoint count even when it is 1 | Per-timepoint measurement tables and their id space |
| `TimeSeriesNotSupported`, naming the axis and its length, raised at read | Tracking: linking an object at *t* to the same object at *t+1* |
| The OME-Zarr writer emits five axes with correct types, units and a `coordinateTransformations` scale from `Spacing` | Time-aware gating, plots and galleries |
| The tile plan treats `T` as a non-tiled axis of length 1 | Making `T` the outermost tiling loop (the cheap part, once the rest exists) |

The test that keeps this honest is small and worth writing in L1: a
round-trip of a `T=1` store through `ome-zarr-py`'s own reader, asserting
the axis metadata is what the spec says it should be. Compatibility claimed
and never checked against another implementation is not compatibility.

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
    measure.py     Accumulators, the second streaming pass, table assembly
    gpu.py         The calibration probe, its cache, and GPU-sized plans
    resume.py      The append-only manifest that makes a long run restartable
    ownership.py   SparseOwnership, and building one a tile at a time
    analysis.py    Scaled estimators, and the record of which one ran
```

In `vtea-napari`, `widgets/memory_control.py`: the budget and the tile plan
it implies, as a button beside the voxel size.

The Parquet/DuckDB-backed table originally listed here as `blocked/table.py`
went into `measurements/store.py` instead. `MeasurementStore` is already the
store, already DuckDB, and already the thing every consumer talks to; a
second parallel store would have been two of everything for one new method.
`register_parquet` points a table at a file rather than holding a DataFrame,
and the SQL is identical either way — which is the point, since nothing
downstream should need to know which kind of table it is looking at.

Plus two new modules that are not about blocking as such, but that the
decisions above require:

```
vtea_core/data/axes.py     Axes: canonical TCZYX, validation, reordering
vtea_core/io/store.py      The only module that imports zarr (see above)
vtea_core/io/ome_zarr.py   OME-NGFF 0.4 read/write, pyramids, ingest
```

Changed elsewhere, and deliberately little:

- `workflow/wiring.py` — the four new `StepIO` fields. Declarative, no
  behavior.
- `workflow/pipeline.py` — `Pipeline.run` gains an optional executor; the
  in-memory path is unchanged and stays the default.
- `io/tiff.py`, `io/zarr_io.py` — a lazy TIFF path; the direct `zarr` and
  `da.from_zarr` calls move behind `io/store.py`.
- `objects/ownership.py` — the sparse, mask-restricted representation.
- `measurements/store.py` — `write_measurements`/`read_measurements` and the
  Parquet-backed registration. Written through DuckDB rather than pandas,
  which would pull in `pyarrow` — a hundred megabytes of dependency for a
  format the database already in this package reads and writes natively.
- `vtea_napari` — lazy `active_image()`, the budget control, ROI preview,
  the progress/cancel worker, seam review in the association review widget.

## Phases

| Phase | Content | Est. |
| --- | --- | --- |
| **L0 — Budget and plan** **done** | `MemoryBudget`, `TilePlan`, the `StepIO` scaling fields, the plan summary in the GUI. No execution change; everything downstream depends on it | 1–2 wk |
| **L1 — Lazy I/O and the store** **done** | `io/store.py` as the single zarr seam, `Axes`, OME-NGFF 0.4 read/write with pyramids and a 5D `TCZYX` layout, lazy TIFF, ingest, `ZarrScratch`, lazy napari layers | 2–3 wk |
| **L2 — The executor: elementwise, neighborhood, global stats** **done** | Covers all of `imageprocessing` and `threshold_mask`. Padding, halo trimming, and the "single tile equals whole image" test | 2–3 wk |
| **L3 — Labels across tiles** **done** | The ledger, the four selectable strategies (overlap, centroid, edge-touching, none) and the resolutions over them, halo verification, blocked connected components / watershed / derived segmentations. **The crux**; gated on the invariance test | 4–6 wk |
| **L4 — Measurements at scale** **done** | Accumulators, the exact second pass, the Parquet/DuckDB table, the seam columns | 2–3 wk |
| **L5 — Deep learning** **done** | Blocked Cellpose, GPU budget and calibration, `RESEGMENT_SEAM`, resumable runs | 2–3 wk |
| **L6 — Objects at scale** **done** | Sparse ownership, blocked association scoring, `build_cells`/`cell_features` through DuckDB | 2–3 wk |
| **L7 — Analysis and explorer** **done** | Streaming estimators, binned scatter, gallery from the pyramid | 2–3 wk |
| **L8 — GUI** **done** | ROI preview, background runs with progress and cancellation, resume, seam review | 2–3 wk |

L0-L4 are built. A protocol of blur, Otsu threshold, connected components
and per-object measurement now runs entirely out of core, returning the same
objects voxel for voxel and the same measurement table column for column as
the in-memory run - at one tile and at sixteen. Only `stddev` differs, by
about one part in 10^13, because it comes from a sum of squares rather than
a second pass over the values.

L8's first half is built. `active_image()` no longer materializes a lazily
loaded layer — `source_data()` hands the protocol the array as the layer
holds it, and `run_processing` decides from the plan: data that fits runs
in memory as it always has, and data that does not runs a tile at a time
through `vtea_core.blocked`, with the results staying in the scratch store.
A `MemoryControl` beside the voxel-size button shows the budget and what it
divided the data into ("216 tiles of 384x384x384", with what bounded it in
the tooltip), and opens a dialog to change it — because a user who cannot
see that number cannot tell a slow run from a stuck one, and one who cannot
change it cannot trade time for memory on the machine they actually have.
L8 is complete: cancellation off the GUI thread, the seam-review view over
`seam_confidence`, and the ROI preview driven by `corner_pixels` are all
built (items 2, 3 and 6 below).

One environment note for whoever runs these tests: napari's `add_image`
tears down through a vispy path that needs a GL context, so the viewer
tests here use `add_labels`, as the rest of this package's viewer tests
already do. The builder reads `layer.data` and does not care which kind of
layer produced it.

L7's estimator half is built. The pattern there is `EstimatorChoice`: every
scaled fit returns which method ran, how many rows it was fitted on, and
whether the result is the whole-table answer — because "k-means on ten
million objects" and "k-means fitted on fifty thousand of them" are
different claims, and a table that cannot tell them apart cannot be
compared with one computed the other way. `exact` is deliberately strict:
seeing every row is not the same as running the same algorithm, so
`IncrementalPCA` reports as streamed however close it lands. The gallery
reading crops from the pyramid completes it (item 1 below).

L6's ownership half is built, which is the phase's headline: the largest
thing a protocol produces now costs the same order as the image rather than
six times it. The check that keeps it exact is a run-time one, because a
static contract cannot express the rule — a marker's claim carries four
falloffs by default, so `ownership_blocked` computes the reach from the
parameters actually passed and refuses to run when the tiles do not overlap
by at least that much. Anisotropically: eight microns of reach is four
voxels along a 2 µm z-step and sixteen in x at 0.5, so a scalar check would
pass on z and be wrong in x.

L6's other two halves are built too (items 4 and 5 below). The table side —
association scoring and cell composition — scales with object count rather
than with voxels, which is a different problem with a different ceiling,
and it is answered with the two shapes a database is for: additive counting
and a recursive join.

L5 completes the reconciliation: `RESEGMENT` is built, so a learned
segmenter can be tiled rather than refused, and the tests show the
difference it makes — against a stand-in segmenter that needs context, a
strategy that picks between tile copies returns truncated objects where
re-segmenting returns the whole-image answer exactly. The GPU calibration
probe and the resume manifest are built and tested against injected
devices and induced crashes; what cannot be exercised here is a real CUDA
device, so the probe's `torch` path awaits hardware.

**Total: roughly 18–28 engineer-weeks**, and all of it is now built. L0–L2
were worth landing on their own — they make a large dataset *openable and
preprocessable*, which is most of the day-to-day pain — and L3 was where
the intellectual risk was concentrated. What is left is not code: see
"Validation this environment cannot do" below.

## Finishing L6–L8

Six items were outstanding when this section was written, plus two things
that can only be checked on hardware this was not built on. All six are
built; the hardware validation is not, and should not be assumed.

Scoped against the code rather than against the phase headings, they were
smaller and more separable than "three unfinished phases" suggested — and
two were much smaller than the plan implied, because the existing code
already did most of the work.

Ordered by what a user gets soonest, not by phase number.

### 1. The gallery reads crops rather than volumes — **done**

`GalleryWidget._crop_2d` already slices `volume[..., r0:r1, c0:c1]` — a
bounding-box read, which is what a chunked store is best at. Almost nothing
has to change in it; what has to change is around it.

- `show_objects` is typed `volume: np.ndarray` and its callers hand it a
  materialized array. Take whatever the layer holds, and `np.asarray` the
  *crop* rather than the volume.
- **Pick a pyramid level.** A 64-pixel thumbnail from a 40-pixel crop wants
  level 0; a thumbnail of a 400-pixel region wants level 2, and reading
  level 0 for it is sixteen times the I/O for the same picture. Read
  `MultiscaleInfo` and choose the level where the crop is about the
  thumbnail's size.
- **Crop z as well.** `max_projection` over the full depth of a 2,000-slice
  stack is not a thumbnail, it is a reduction over the whole volume. Add a
  `z_radius` around the object's own centroid.

Small, self-contained, and the thing that makes a large result *lookable
at*. Do it first.

### 2. A run you can cancel, off the GUI thread — **done**

`run_processing` blocks Qt for the length of the run. That was tolerable
when a run was seconds; it is not when the status line says "tile 340 of
4,096".

- `BlockedPipeline.run` gains a `should_stop` callable checked between
  tiles. Small, and in the core rather than the GUI so a script can use it.
- The builder runs it on a background thread and pumps the Qt event loop
  while it waits, rather than on a `thread_worker`. A worker would make
  `run_processing` asynchronous and every caller that treats it as a
  function returning a result would have to change; this keeps the return
  value, keeps napari repainting, and makes Cancel a button that can
  actually be clicked. NumPy and scipy release the GIL for the heavy
  operations, so the two threads genuinely overlap.
- **Cancellation composes with L5's manifest**: a cancelled run has written
  every tile it finished, so pointing the same manifest at it resumes rather
  than restarts. That is worth wiring deliberately rather than discovering —
  it turns Cancel from "throw away an hour" into "stop for now".

The one design point: a cancelled run leaves a partial label array in
scratch. It must not be published as a result. Publish on completion only,
and say what was cancelled — `Cancelled` is its own exception type for
exactly that, so nothing downstream can mistake a partial result for a
finished one. The scratch store is kept rather than deleted: it is the
user's to discard, and with a manifest it is the start of a resume.

Two things pumping the event loop makes possible that a worker would not,
and both are guarded. A user can click Run again mid-run, so a run in
progress refuses to start another. And `cancel()` can arrive from any
thread — a timer, a worker — so it sets the flag and touches no widget;
the button's feedback is applied by the pump loop, which is on the GUI
thread by construction. Qt does not merely misbehave when a widget is
touched from the wrong thread, it segfaults, which is how that was found.

### 3. Seam review — **done**

The plan claimed this would need "no new UI at all", and checking that
against the code, it is nearly true: `LabelLedger.to_frame` already joins
`seam_confidence` onto the measurement table, and `ExplorerWidget` already
gates on any column and shows a gallery of what a gate selects. So drawing
a gate on low confidence and looking at the crops works today.

What is missing is the shortcut and the record:

- A one-click "show seam objects" preset — a gate on
  `seam_confidence < threshold`, so a reviewer does not have to know the
  column exists.
- A seam table beside the association review: object, `n_fragments`,
  `seam_rule`, `seam_confidence`, sorted by confidence.
- **Reject, do not edit.** Correcting a seam decision means rewriting voxels
  in a label array that may be 33 GB, which is the random-access editing
  problem this document explicitly puts out of scope (chunk write
  amplification, undo). What a reviewer can do is *exclude* an object —
  a table operation, recorded in `LabelLedger.dropped` beside the objects
  the policy dropped. Say so in the interface, so nobody looks for an edit
  tool that is deliberately absent.

Built as `vtea_core.gates.seam` and `SeamReviewWidget`. `seam_gate(frame)`
returns an ordinary `Gate` over `seam_confidence` against `n_fragments` —
deliberately a gate rather than a selection mode, so a reviewer can
intersect it with a size or brightness gate and open the gallery on what
survives. Its fragment ceiling is read from the data rather than fixed,
because the vessel that ended up in nine tiles is exactly the object worth
looking at and a constant would silently miss it; and its lower edge starts
below zero, so an object no tile contained (confidence 0.0) falls inside
the gate rather than on its boundary.

The pane itself lists the objects worst-first with the rule that decided
each, and offers one action: reject. A rejection is written to
`LabelLedger.dropped` with a reason naming a person, the same distinction
`Association.MANUAL` draws and for the same reason — a correction
indistinguishable from an inference is worse than no correction. The tab
appears only when the table has seam columns, since an in-memory run has no
tile boundaries and a permanently empty tab reads as a broken feature
rather than an absent condition. The ledger reaches it through the session
(`set_ledger`), so a seam can be reviewed with the builder closed, and an
in-memory run clears it rather than leaving the previous run's ledger
describing objects this table has not got.

### 4. Association, spatially partitioned — **done**

Three scoring methods with three different answers, and one of them needs
no image at all:

- **`containment` is already additive.** It is a `bincount` over paired
  label arrays: per-(child, parent) overlap counts sum across tiles, and so
  do the per-child totals. It is the `ACCUMULATE` pattern exactly — one
  streaming pass, exact, no halo.
- **`centroid_distance` needs no voxels.** The centroids are already in the
  measurement table. Read them, build one kd-tree, done — this is a table
  operation that has been reading images out of habit.
- **`boundary_distance` is already object-local**: `find_objects` then an
  EDT per parent within its window. Blocked, it reads each parent's bounding
  box grown by `max_distance` from the store. Exact given that window.

The real scaling work is neither of those. `CandidateScores` holds a
`dict[int, dict[int, float]]`, which is roughly 200 bytes per candidate
against 16 for the same thing in three arrays; at 10⁷ children with a few
candidates each, the dictionary *is* the ceiling. So:

- A COO-backed `CandidateScores` — `child_index`, `parent_index`, `score`
  arrays — keeping the existing accessors so `posterior` and `assign` do not
  change.
- `assignment._blocks` walks that with union-find instead of traversing
  dictionaries. The block decomposition itself is already right, and it is
  what keeps the O(n³) Hungarian solve tractable; only its input
  representation changes.

Built as `vtea_core.blocked.associate`, with the representation change in
`objects/scoring.py` and `objects/assignment.py`. `Posterior` moved to the
same three arrays for the same reason — it is one number per candidate pair
too, so leaving it as a dict of dicts would have moved the ceiling by one
step and no further. `posterior()` is now one `bincount` and one
elementwise division, with no per-child Python loop at all.

The trap, found by a test and worth stating because any COO structure with
a sorted id list has it: sorting the ids while accepting index arrays the
caller built against *their* order silently points every pair at a
different object. Indices are remapped through the sort, and the common
path — ids from `np.unique`, already sorted — costs nothing. A second one
in the same family: `searchsorted` on a label that is not in the id list
returns the slot it would occupy, which belongs to some other object, so a
mismatched id list produced a plausible wrong answer rather than an error.
The blocked scorers check rather than trust, because the ids and the labels
genuinely come from different places — a ledger, a table, a scan — and only
have to agree.

Three invariance claims, each pinned at one tile and at many: containment
is bit-identical because overlap counts and child totals are integer sums;
centroid distance is identical because it is the same centroids and the
same arithmetic, read from the measurement table the run already produced;
boundary distance is identical because a parent's window already holds
everything within reach of it. Where a blocked segmentation is in the same
run, the object ids and the bounding boxes come from its `LabelLedger`
rather than from a scan — the ledger already knows both.

`associate_by_identity` came along for a small price: it only ever compared
two id sets, so `associate_ids` takes the sets and the blocked form hands
it ids it got without reading a voxel. The executor now runs the whole
`association` category rather than refusing it as OBJECT_LOCAL.

### 5. Cells through DuckDB — **done**

`build_cells` walks the association graph and builds a `Cell` object per
cell with an `ObjectRef` per part. At ten million cells that graph is
several gigabytes of Python objects before any measurement is joined to it.

- **Associations become a table** (`child_segmentation`, `child_id`,
  `parent_segmentation`, `parent_id`, `probability`, `relationship`) —
  `AssociationSet.to_frame`, which is useful on its own for saving and
  diffing.
- **`build_cells` becomes a recursive CTE.** Following links out from the
  roots until nothing new joins is what `WITH RECURSIVE` is for, and DuckDB
  spills it. The output is a `(cell_id, role, object_id)` mapping table
  rather than an object graph.
- **`cell_features` becomes a join and a group-by**, which is the operation
  DuckDB exists to do and pandas does in memory.

Two things must survive the port exactly, and they are the parts worth
testing hardest. `single_roles` decides whether a role's columns are
`nuclei.mean` or `lysosomes.n` + `lysosomes.mean_mean`, and it comes from
how the association was made rather than from what this field happens to
contain — a shape that varied with the data could not be pooled across
fields. And a cell missing a part gets NaN and a count of 0 rather than
being dropped, which is a `LEFT JOIN` and an explicit role list rather than
whatever the data brings.

Built as `vtea_core.blocked.cells`. `CellMembership` holds the membership
table and answers everything a cell result is asked to report; `CellSet`
holds the object graph. Both are `CellCollection`, which is what the GUI
now checks for, so a blocked run's cells behave like any other in the panes
that only want to say how many cells there are and how many are missing a
part. `CellMembership.to_cell_set()` materializes the graph for a result
small enough to want one, and is deliberately explicit — it is exactly the
materialization the table form exists to avoid.

The per-cell table comes back *identical* to the in-memory one: same
values, same column names, same column order, same dtypes, pinned with
`assert_frame_equal`. Two things had to be made to agree for that to be
true, and both are worth naming. DuckDB returns a missing integer
measurement as `<NA>` in a nullable column where pandas gives `NaN` in a
float one, so the blocked path converts — a column whose dtype depends on
which code path produced it cannot be pooled with one that did not. And the
in-memory path had a latent wart in the same place: mapping objects to
cells produces NaN for the ones in no cell, which made the join key a float
and quietly turned `cell_id` into `1.0` whenever any object was unclaimed.
Fixed there rather than reproduced here.

Cycles are refused as they were, but found differently. Following links out
from the roots cannot see a loop that goes back through the root
segmentation, which is the very case the in-memory form catches, so the
check is made up front and on the whole link table: a child has at most one
parent, so the links are a functional graph, and following every child's
parent pointer a bounded number of times at once settles whether any chain
is still climbing. Vectorized and keyed on integers rather than on tuples,
because it runs on the same ten million links everything else here is
shaped around.

A measurement table may be a Parquet file rather than a DataFrame, which is
what makes this worth doing in a database at all: the join reads the table
as the query needs it instead of loading ten million rows to group them.

### 6. ROI preview — **done**

`layer.corner_pixels` gives the extent currently on screen, which is the
hook for running a step over what the user is looking at instead of over
forty gigabytes. Tuning a threshold any other way on data this size is not
usable.

Left until last deliberately: it is a convenience, and every item above it
is either a correctness gap or the difference between a feature being
reachable and not. Three things it needs — the displayed pyramid level (not
level 0), a preview layer named so it cannot be mistaken for a committed
result, and debouncing so panning does not queue a hundred runs.

Built as `vtea_napari.widgets.roi_preview` plus `run_preview` on the
builder, with all three, and a fourth that turned out to matter more than
any of them: **the preview is a tile of the protocol's own tiling.** The
visible box is grown by the same halo every other tile gets, run, and
trimmed back — so the preview *is* what a full run would write there,
rather than what a filter computes when it can see nothing past the edge of
the view. `tile_for_region` in `blocked/plan.py` is the piece that makes
that a two-line change rather than a second implementation, and the test
that pins it has the negative control beside it: the same region run in
isolation genuinely disagrees with the run at its edges, which is exactly
where a person looks.

The one thing a preview cannot promise is stated rather than hidden. A step
whose parameter comes from a statistic over the whole image — an Otsu
threshold, a percentile — computes it over the region on screen instead,
because that is all it was given. Often that is what a user tuning it wants
to see; it is never what the full run will do, so the status line says
which steps it applies to. The scaling contract already knew: `Scaling.
resolve` reports `threshold_mask` as elementwise at a fixed value and a
global statistic at otsu, which is the whole question.

A view too large for the budget is refused with the reason and the remedy
("zoom in to preview it") rather than freezing the window, and the preview
never touches the run context — the Show buttons, the plot and the tables
go on reading whatever the last real run produced.

### 7. Validation this environment cannot do

Not code, and not optional before anyone trusts this with real data.

- **The GPU probe on real hardware.** `device_name()` and
  `torch.cuda.mem_get_info` have never run against a device here; everything
  around them is tested against injected ones. One run on a CUDA box, and a
  cached calibration checked against what the card actually manages.
- **The strategy comparison on real tissue.** The open question below asks
  how often centroid and overlap matching disagree, and on what. The ledger
  was built to make that a query rather than a study; it needs tissue with
  vessels and tubules in it, not only nuclei.
- **`napari.add_image` under a headless runner.** It tears down through a
  vispy path needing a GL context, which is why the viewer tests here use
  `add_labels`. Worth checking against a napari release with a display
  attached before assuming it is only a test-environment quirk.

### Order and total

| # | Item | Est. | Why here |
| --- | --- | --- | --- |
| 1 | Gallery from the pyramid | **done** | Smallest; makes a large result lookable at |
| 2 | Worker thread and cancel | **done** | A run nobody can cancel is a run nobody starts |
| 3 | Seam review | **done** | Makes L3's ledger reachable |
| 4 | Association partitioned | **done** | Unblocks the table forms item 5 needs |
| 5 | Cells through DuckDB | **done** | Depended on 4's association table |
| 6 | ROI preview | **done** | A convenience, and the only one |
| 7 | Hardware validation | — | Gating for production use, not for merging |

**Items 1–6 are built.** What remains is item 7, which is not code: a run
on a CUDA device, a strategy comparison on real tissue, and a check of
`napari.add_image` under a display. None of it can be done here, and none
of it should be assumed.

## Open questions

- **Dask scheduler.** Threads are enough for a workstation; `distributed`
  buys a cluster and a dashboard, and costs a dependency and a failure mode.
  Recommend: threaded by default, `distributed` opt-in, and keep the
  executor agnostic.
- **When to move off the default.** Overlap matching is the default and
  the other three are selectable (decided; see "The strategies"). What is
  still open is guidance on when a user should switch, which is a question
  about tissue rather than about code. The ledger is built to make that
  comparison a query rather than a study, and three things are worth
  measuring on real data: how often centroid matching over-merges relative
  to overlap matching in packed epithelium; whether edge-touching merge is
  within tolerance for sparse puncta, where it saves the whole halo; and
  whether "no merge plus drop seam-touching objects" costs less accuracy
  than it saves time on a first pass.
- **Does anything need true random-access editing of a large label image?**
  Manual correction of a 33 GB label store is a different problem (chunk
  write amplification, undo). Out of scope here; flagged so it is not
  assumed to be in.
- **When time actually lands.** The axis model and the store layout are
  settled above; what is not settled is whether the first real time support
  is "run the whole protocol per timepoint independently" (cheap, useful,
  no tracking) or waits for tracking to exist. The former is a few days'
  work on top of L0-L4 and is probably what most users mean; confirm that
  before building either.
