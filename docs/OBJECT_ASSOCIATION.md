# Associating segmentations into cells

Plan for turning independent segmentations into a hierarchy that describes a
cell: a nucleus with its envelope and cytosol, a cytoplasm assigned to one
nucleus, organelles belonging to that cytoplasm — and an honest account of
what happens where the evidence runs out.

Nothing here is built. This document is for review before any of it is.

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

### Phase 0 — Voxel spacing *(small)*

As above. Prerequisite for 1, 2 and 4; independently useful today.

### Phase 1 — The model, and derived segmentation *(small–medium)*

The cheap, high-value half of the problem.

- `vtea_core.objects.association`: `ObjectRef`, `Association`,
  `AssociationSet`, JSON I/O, provenance.
- `vtea_core.segmentation.derived`, all identity-preserving so association
  is exact: `expand_labels` (skimage's, which grows each label into
  background and stops where two meet), `label_shell` (an annulus of given
  inner/outer thickness — nuclear envelope), `label_ring` (expand minus
  original — cytosol), `subtract_labels`, `restrict_labels_to` (mask one
  segmentation by another).
- An `association` step category with `associate_by_identity` for this case.
- GUI: the new steps appear in the segmentation menu; nothing else changes.

**Done when** a protocol can produce nucleus → envelope → cytosol from one
DAPI channel, measure each, and say which envelope belongs to which nucleus.

**Risk:** thickness is in physical units via Phase 0, but `expand_labels`
works in voxels; it needs an anisotropy-aware wrapper, and skimage's
implementation doesn't take spacing. Likely a small distance-transform of
our own rather than a call-through.

### Phase 2 — Probabilistic association between independent segmentations *(medium)*

The heart of it.

- **Candidate scoring** (`vtea_core.objects.scoring`), each returning a
  sparse child × parent score matrix: `containment` (fraction of the child's
  voxels inside the parent), `centroid_distance`, `boundary_distance`,
  optionally a `size_ratio` prior.
- **Posterior**: scores → a normalised distribution over candidate parents
  per child, plus an explicit *orphan* probability so "this cytoplasm has no
  nucleus" is representable rather than forced onto the nearest one.
- **Assignment modes**, and the distinction matters:
  - `many_to_one` — argmax per child. Right for organelles: one cytoplasm
    holds many lysosomes.
  - `one_to_one` — a global optimum over the whole cost matrix
    (`scipy.optimize.linear_sum_assignment`). Right for your nucleus ↔
    cytoplasm case, and **not** the same as argmax: per-child argmax will
    happily hand the same nucleus to two cytoplasms, which is exactly the
    constraint you said should hold. A global assignment is the only way to
    honour "one and only one".
  - A threshold below which a child is left unassigned rather than forced.
- GUI: an "Association" category in the analysis pane, with the two
  segmentations picked by name from the same dropdown that measurement steps
  use.

**Done when** DAPI nuclei and a cytoskeletal cytoplasm segmentation produce
a 1:1 assignment with a recorded posterior, and the unassigned ones are
visible as such.

**Risk:** the Hungarian algorithm is O(n³). 5,000 objects is seconds;
50,000 is not. Mitigate by restricting candidates to a spatial neighbourhood
first, which makes the matrix sparse and blocks it into independent
components — worth building in from the start rather than retrofitting.

### Phase 3 — Cells: hierarchy and per-cell features *(medium)*

Where it becomes useful rather than merely correct.

- `CellSet`: compose associations into cells. A chosen root segmentation
  defines cell identity; children inherit through the chain, so
  nucleus → cytoplasm → lysosome works, and so does whole-cell → nucleus.
- Handle honestly: orphans, cells missing a part, contradictory links,
  and cycles (refuse them).
- **The cell feature table** — one row per cell, with columns namespaced by
  role: `nucleus.mean_ch0`, `cytoplasm.count`, `lysosome.n`,
  `lysosome.total_count`, `lysosome.mean_of_mean_ch2`. One-to-many children
  are aggregated (count / sum / mean / median), which is where "how many
  endolysosomes per cell, and how bright" finally becomes a number.
- `FeatureCatalog` gains the role and the relationship, so the data
  dictionary says which segmentation *and* which association a column came
  from.
- The Object Explorer plots and gates this table, so gating on cells rather
  than on objects is free.

**Done when** the scatter plot's X/Y menus offer `nucleus.mean_ch0` against
`lysosome.n`.

### Phase 4 — Pixel-level ownership *(large)*

The genuinely hard one, and the one I would most like to keep small.

- Deterministic baseline first: `watershed_ownership` — watershed the
  candidate region from the parent objects as markers. This is the standard
  answer for "split the cytoplasm between these nuclei", it is one skimage
  call, and for many datasets it is enough. Ship it before anything
  probabilistic.
- Then `distance_ownership`: per contested voxel, a posterior over nearby
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

## Questions I need answered before starting

1. **Is nucleus ↔ cytoplasm strictly 1:1?** Multinucleate cells —
   hepatocytes, syncytia, osteoclasts — break it, and the constraint is
   baked into the algorithm choice, not a parameter. If they matter, the
   model needs one-to-many in that direction too.
2. **Where does pixel size come from?** napari's `layer.scale` where the
   file provides it, with manual override, is my proposal — but if your
   TIFFs generally lack it, the override becomes the main path and deserves
   to be more prominent.
3. **Can one object belong to two cells?** My default is no — one parent,
   with the alternatives retained for inspection. Shared or ambiguous
   structures would need a different answer.
4. **Should cell ids be stable across re-runs?** Stable ids mean gates and
   annotations survive a re-run; unstable ones are simpler. I lean stable,
   derived from the root object's id.
5. **How much does Phase 4 matter to you?** If contested membranes are
   central to the science, it moves earlier and the deterministic watershed
   baseline should land in Phase 2. If it is a refinement, the order above
   is right.

## What I would build first

Phase 0 and Phase 1 together: they are small, they are prerequisites for
everything else, and they deliver the nuclear-envelope/cytosol workflow —
which is the half of your request that needs no inference at all and should
not wait behind the half that does.
