# Associating segmentations into cells

Plan for turning independent segmentations into a hierarchy that describes a
cell: a nucleus with its envelope and cytosol, a cytoplasm assigned to one
nucleus, organelles belonging to that cytoplasm — and an honest account of
what happens where the evidence runs out.

Phases 0 to 3 are built; each phase below records what it actually turned
out to be, including where the plan was wrong. Phases 4 and 5 are still
a plan.

## What is actually being asked

Three problems that look like one and are not:

| | Problem | What association means | Where uncertainty lives |
| --- | --- | --- | --- |
| **1** | A segmentation **derived** from another by morphology — an annulus for nuclear envelope, a dilation-minus-nucleus for cytosol | Exact, by construction: derived object *k* came from parent *k* | None. The label identity carries it |
| **2** | Segmentations made **independently** in different channels — DAPI nuclei, a cytoskeletal cytoplasm, organelle puncta | Inferred: which nucleus does this cytoplasm belong to? | Which parent, per object |
| **3** | **Contested voxels** — a fuzzy plasma membrane, two cells touching, overlapping masks | Inferred per voxel: which cell owns this bit of area? | Which cell, per voxel |

They need different machinery and they should be built in that order,
because 1 is nearly free, 2 is the one people will use every day, and 3 is
the one that can quietly consume a month.

The Java VTEA has nothing to port here. `MicroNeighborhoodObject` and
`vtea.objects.neighborhoodmeasurements` group objects that are *near* each
other and report distance statistics; there is no parent/child relationship
between segmentations anywhere in that codebase. This is a design from
scratch, which is a reason to keep the model small and the algorithms
replaceable.

## Prerequisite: the pixel size is missing

`grep -rn "spacing\|voxel_size\|pixel_size"` across both packages returns
nothing. Every measurement today is in voxels, and that has been fine
because nothing compared distances.

It stops being fine here. Confocal z-steps are routinely 3–10× the lateral
pixel size, so "dilate the nucleus by 5" is a sphere in index space and a
flattened disc in the specimen. Every part of this plan — annulus
thickness, centroid distance, the falloff in a soft assignment — is wrong in
anisotropic 3D without it, and wrong in a way that looks plausible.

So **Phase 0 is voxel spacing**, and it is small:

- `AnalysisSession.spacing` (z, y, x), seeded from the napari layer's
  `.scale` (napari already reads it from OME-TIFF where present) and
  overridable in the builder's top row next to the axis pickers.
- Threaded into the run context as `spacing`, the way `channel_axis` is.
- `extract_measurements*` gains physical-unit columns where spacing is
  known (`volume_um3` alongside `count`), recorded in the FeatureCatalog
  with real units instead of "voxels".

This is worth doing even if the rest of the plan is deferred.

## The model

One idea, deliberately small:

```python
@dataclass(frozen=True)
class ObjectRef:
    segmentation: str      # a step name: "watershed_split_1"
    object_id: int

@dataclass
class Association:
    child: ObjectRef
    parent: ObjectRef
    relationship: str      # "derived" | "contained" | "assigned"
    probability: float     # 1.0 for derived
    method: str            # "identity" | "containment" | "hungarian" | ...
    params: dict
    alternatives: list[tuple[ObjectRef, float]]  # the rest of the posterior
```

`AssociationSet` holds these, indexes them both ways, and serialises to
versioned JSON exactly as `GateSet` does.

Four things this buys, each of which is why it is a first-class object
rather than an extra column on the measurement table:

- **`alternatives` keeps the posterior, not just the winner.** "Cytoplasm 12
  belongs to nucleus 7 (p=0.55), or nucleus 9 (p=0.44)" is a different
  statement from "cytoplasm 12 belongs to nucleus 7", and it is the one that
  lets a QC view surface the 3% of cells worth looking at by eye.
- **`method` and `params` make it reproducible** on the same terms as
  everything else in the feature catalog.
- **It is separable from the images**, so it saves, reloads and diffs.
- **It is agnostic about direction.** Nothing in it says nuclei are the
  root. A whole-cell segmentation can be the parent of everything, which is
  the workflow you asked about at the end.

### One decision worth flagging now

A `Step` produces exactly one output. Rather than change that, association
is **its own step category** consuming two named segmentations and producing
an `AssociationSet`. So a derived segmentation is two steps — make the
annulus, then associate it — even though the second is trivial for the
derived case. That is one extra card on screen in exchange for not
complicating the workflow engine, and it keeps association reviewable and
re-runnable on its own.

## Phases

### Phase 0 — Voxel spacing *(small)* — **done**

`vtea_core.data.Spacing` carries the voxel size, its unit, and where it came
from; `spacing_from_scale` reads a napari layer's `.scale` and reports an
all-ones scale as *unknown* rather than as one micron isotropic. The
builder's top row gains a voxel-size button that reads the image, says so
when nothing was recorded, and opens a dialog to set it by hand — a value
typed there outlives switching between layers. It threads into the run
context beside `channel_axis`, and `extract_measurements*` add a physical
`volume` column when — and only when — the spacing is actually known.

### Phase 1 — The model, and derived segmentation *(small–medium)* — **done**

The cheap, high-value half of the problem.

`vtea_core.objects` holds `ObjectRef`, `Association` and `AssociationSet`
as described above, with versioned JSON I/O and the same
newer-format-refused check the gates use. `vtea_core.segmentation.derived`
adds `expand_labels`, `label_ring` (cytosol), `label_shell` (nuclear
envelope, with independent inward and outward thicknesses),
`subtract_labels` and `restrict_labels_to` — every one of them preserving
label identity, which is what makes `associate_by_identity` exact rather
than inferred. Association is its own step category, and the derived steps
sit in the segmentation menu; the Edit dialog resolves `child_labels` and
`parent_labels` to named segmentations the way `labels` already was.

The anticipated risk was real: skimage's `expand_labels` takes no
`sampling`, so this is our own distance transform
(`distance_transform_edt(..., sampling=spacing)`) and every thickness is
physical when the spacing is known and in voxels when it is not. The
inward half of `label_shell` measures from `find_boundaries(mode="inner")`
rather than from background, so two touching nuclei each keep an envelope
on the face they share — a distance-to-background would have dropped it.
The processing pane's seeded context gained `spacing` alongside it, since
a `label_ring` added there would otherwise have silently run in voxels.

**Done:** a protocol produces nucleus → envelope → cytosol from one DAPI
channel, measures each, and says which envelope belongs to which nucleus.

### Phase 2 — Probabilistic association between independent segmentations *(medium)* — **done**

The heart of it.

`vtea_core.objects.scoring` measures the evidence, each function returning a
sparse child × parent affinity in [0, 1]: `containment` (the fraction of the
child's voxels inside each parent, needing no distance parameter at all),
`centroid_distance`, and `boundary_distance` — the gap between the two
objects' nearest voxels, which is the one that matches how a person decides
these by eye and the one that is not fooled by a parent much larger than its
children. The `size_ratio` prior was not built; nothing needed it yet.

`vtea_core.objects.assignment` turns those into an answer.
`posterior()` normalises a child's affinities against an explicit
`orphan_score`, so "this cytoplasm has no nucleus" is a probability rather
than an absence — without it a cytoplasm with one distant nucleus comes out
certain of it. `assign()` then resolves the competition: `many_to_one` per
child for organelles, `one_to_one` globally through
`scipy.optimize.linear_sum_assignment` for the nucleus/cytoplasm case, and a
`min_probability` below which a child is left unassigned rather than forced.

The anticipated O(n³) risk is handled where it arises rather than
retrofitted: because scoring only proposes nearby parents, the candidate
graph splits into connected blocks, and the global solve runs per block. A
field of 50,000 objects is thousands of tiny problems instead of one
impossible matrix, and a test pins that blocking does not change the answer.

`AssociationSet` gained `unassigned` (format version 2) so a run that linked
383 of 400 children says so; `summary()` is what the builder's log prints,
because an association draws nothing and "it ran" would hide the one number
that says whether the parameters were right.

`segmentation.watershed_ownership` is the deterministic ownership baseline,
brought forward as planned: it divides a region among the objects inside it,
splitting at the region's own narrow waist rather than midway between
markers, and keeps each owner's id so the territories associate back by
identity.

In the GUI, `child_name`/`parent_name` are no longer form fields — they are
filled from the wiring (`StepIO.names_from`), so an `ObjectRef` names the
step the input is actually pointed at instead of the function's `"child"`
default, and re-pointing a step moves the names with it. `method` and `mode`
render as dropdowns, from `Literal` annotations rather than a hard-coded
special case.

**Done:** DAPI nuclei and a cytoplasm segmentation produce a 1:1 assignment
with a recorded posterior, and the unassigned ones are visible as such.

### Phase 3 — Cells: hierarchy and per-cell features *(medium)* — **done**

Where it becomes useful rather than merely correct.

`vtea_core.objects.cells` composes associations into cells. `build_cells`
follows the links out from a chosen root segmentation, so
nucleus → cytoplasm → lysosome works and so does whole-cell → nucleus —
the same model read the other way, which is also how a multinucleate cell is
expressed. Cell ids are the root object's id, so a gate drawn on cell 412
still means that cell after a re-run. The root's *label image* is what names
its objects, because a nucleus nothing was assigned to is still a cell, and
dropping exactly the cells that lost a part biases every statistic after it.
Cycles are refused, a root that is only ever a child is refused with an
explanation, and objects whose chain never reaches a root are kept as
`unclaimed` rather than silently dropped. `merge_associations` folds two
association steps into the one hierarchy a cell spans.

`cell_features` builds the per-cell table from one measurement table per
segmentation: columns namespaced by role (`nuclei_1.mean_ch0`,
`cytoplasm_1.count`), one-to-many roles aggregated (`lysosome_1.n`,
`lysosome_1.mean_count`, `lysosome_1.sum_mean_ch2`), and a cell missing a
part given NaN and a count of 0 rather than being dropped.

One decision worth recording: **which roles are aggregated comes from how
the association was made, not from the data.** A `one_to_one` assignment or
a derived part is single; anything else is aggregated. Deciding it by
looking at whether this particular field happens to contain a cell with two
of something would make the same protocol produce differently shaped tables
on different fields, and they could not be pooled.

The Object Explorer plots and gates that table. It is a second table rather
than more columns on the first — its rows are cells, not objects — so the
session carries both, each with what its rows are: the id column that names
a row, the segmentation a gate on it should light up, and its own gates,
since a polygon drawn over cell features selects nothing on a per-object
table. The pane gains a table picker once there is more than one to choose
between, and the gallery crops around the root segmentation's centroids,
which is where a cell's id points anyway.

**Still to do:** recording the role on each column in the `FeatureCatalog`,
so the data dictionary says which association a per-cell column came from.

### Phase 4 — Pixel-level ownership *(large)*

The genuinely hard one, and the one I would most like to keep small.

- The deterministic baseline, `watershed_ownership`, shipped in **Phase 2**:
  watershed the candidate region from the parent objects as markers. It is
  the standard answer for "split the cytoplasm between these nuclei", and for
  many datasets it is enough. Everything below builds on having it already.
  What it does not do is say a division was a close call — it answers every
  voxel with the same confidence, which is exactly the gap this phase fills.
- `distance_ownership`: per contested voxel, a posterior over nearby
  cells from spacing-aware distance with a tunable falloff.
- Optionally `membrane_ownership`, where a membrane channel raises the cost
  of crossing a boundary.
- **Representation is the design decision.** A dense cell × voxel posterior
  is out of the question: 2,000 cells over a 2048×2048×24 volume is ~10¹¹
  floats. Store instead the top-*k* (k=2 or 3) owner ids and probabilities
  per voxel — about four times the size of the label image, and it captures
  essentially all real ambiguity, since a voxel contested by four cells is
  rare and not usefully resolved anyway. Plus a scalar confidence map
  (max posterior) for display.
- **Weighted measurements**: with soft ownership, `mean` becomes a
  probability-weighted mean and `count` becomes an expected volume
  (Σp). `regionprops_table` cannot do this, so it needs our own reducer
  alongside the existing one — a real piece of work, not a parameter.
- napari: the hard argmax as a Labels layer, the confidence map as an Image
  layer, so contested regions are visible.

**Risk:** this is where scope grows without bound. The deterministic
baseline is the hedge — if the soft version proves not worth its cost on
real data, Phase 4 still delivered something usable.

### Phase 5 — QC, editing and persistence *(medium)*

- A **Cells pane**: the association graph, per-link confidence, counts of
  assigned/orphaned/contested, and the ambiguous cells sorted worst-first.
- **Manual override**: reassign or break a link by hand, recorded as
  `method="manual"` so it is distinguishable from an inferred one forever
  after. Non-negotiable for real use — every automated assignment is wrong
  somewhere, and an analysis you cannot correct is one you cannot publish.
- Associations and cells fold into the saved protocol and the publication
  bundle, extending `docs/SAVING_AND_ARCHIVING.md`.

## Validation

Assignment quality is not self-evident from looking at it, so:

- **Synthetic fixtures with known truth** — generated nuclei with generated
  cytoplasms at controlled separations and overlaps — as the primary tests.
  Accuracy against ground truth, not "it ran".
- **Degenerate cases as first-class tests**: two nuclei in one cytoplasm, a
  cytoplasm with no nucleus, a nucleus with no cytoplasm, touching cells,
  and a child overlapping two parents equally.
- **Reported metrics** on real data, where truth is unavailable: fraction
  assigned, mean posterior, count of contested voxels, count of cells
  missing a part. These are what tell you the parameters are wrong.

## Decisions

Answered, and folded into the phases above.

**Multinucleate cells are allowed, behind a flag — and the flag moves the
root.** This is more than relaxing a constraint, so it is worth being
explicit. With the flag off, a nucleus defines a cell and a cytoplasm is
assigned to exactly one of them (the Hungarian case). With it on, one
cytoplasm may hold several nuclei — which means the nucleus can no longer
be what identifies the cell, because there are several. The cell root moves
to the cytoplasm, nuclei become its children (an ordinary many-to-one
assignment, no special algorithm), and a cell gains an `n_nuclei` feature.
So the flag reads as "multinucleate" to the user and as "which segmentation
identifies a cell" to the model.

**Pixel size comes from the image metadata, and is asked for when it is
not there.** With a wrinkle: napari sets `layer.scale` to `(1, 1, 1)` when a
file carries no scale, so "one micron isotropic" and "nobody said" are the
same value. The session therefore tracks *where* the spacing came from -
metadata, the user, or unknown - and treats an all-ones scale as unknown.
Steps whose result depends on it (any dilation thickness, any distance)
prompt for it rather than silently running in voxels.

**One object, one cell.** The alternatives stay on the association for
inspection, but a child has one parent.

**Cell ids are stable, derived from the root object's id**, so gates and
annotations survive a re-run.

**Contested voxels are in scope, user-selected, with manual override.**
Phase 2 brings the deterministic watershed baseline forward so there is a
usable answer early; Phase 4 adds the probabilistic framework and the
override, which is where "user selected" is honoured - the method is a
choice per protocol, not a fixed behaviour.

## Build order

Phase 0 and Phase 1 first: small, prerequisites for everything else, and
they deliver the nuclear-envelope/cytosol workflow — the half of the
request that needs no inference at all and should not wait behind the half
that does. Then Phase 2 (association plus the deterministic ownership
baseline), Phase 3 (cells), Phase 4 (probabilistic ownership and override),
Phase 5 (QC and persistence).
