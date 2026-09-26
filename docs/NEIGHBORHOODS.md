# Neighbourhoods: a second level of objects

A segmented nucleus is an object. The dozen cells around it are a
neighbourhood, and a neighbourhood is an object too: it has members, a
place and measurements of its own (how much of it is each cell type), and it
can be plotted, gated and clustered exactly as a cell can. That is the first
half of the model.

The second half is that the relationship is **reflective**. A cell belongs
to neighbourhoods, so it carries what they are: the composition around it,
and once neighbourhoods are clustered into types, which kind of
neighbourhood it lives in. That is what lets a gate say "CD8 T cells in
tumour-rich neighbourhoods", a statement about a cell that no measurement of
the cell itself can make.

```
 objects (level 1)              neighbourhoods (level 2)
 ┌──────────────┐  build        ┌─────────────────────────┐
 │ object table │ ────────────▶ │ NeighborhoodSet          │
 │  centroid-*  │  (membership, │  members, centre, seed   │
 │  class col.  │   many-to-    ├─────────────────────────┤
 │              │   many)       │ neighbourhood table      │
 │              │  measure ───▶ │  n_objects, Class_c_*,   │
 │              │               │  mean_distance, <agg>_f  │
 │              │               │  neighborhood_type ◀─ classify
 │  nbhd.*  ◀───┼── reflect ────┤                          │
 └──────────────┘               └─────────────────────────┘
```

## The Java VTEA had half of this

The Java Object Explorer's "Neighborhood analysis"
(`XYExplorationPanel.addNeighborhoodFromGate`) built neighbourhoods
(`MicroNeighborhoodObject`), measured them (`NeighborhoodMeasurements`:
`ClassFraction`, `ClassSums`, the disabled `TotalObjects`) and opened them
as a new explorer window. Nothing flowed back: a cell never knew which
neighbourhoods it was in, so the neighbourhood table was a dead end for
anyone asking a question about cells.

The port keeps the Java construction methods and column names, so a table
exported here reads like one exported there, and adds the reflection.

| Java | here |
| --- | --- |
| Nearest-k | `build_neighborhoods(method="nearest", k=...)` |
| Spatial by cell | `build_neighborhoods(method="radius", radius=...)` |
| Spatial by point | `build_neighborhoods(method="grid", radius=..., interval=...)` |
| Randomize | `randomize=True` (a permutation of positions, see below) |
| Voxel Z scale | the protocol's voxel size (`spacing`) |
| `ClassFraction` | `Class_<c>_ClassFraction`, percent (0-100) |
| `ClassSums` | `Class_<c>_ClassSums`, count |
| `TotalObjects` | `n_objects` |
| (none) | `classify_neighborhoods` → `neighborhood_type` |
| (none) | `reflect_neighborhoods` → per-object columns |

Classes run from 0 (or the smallest class, if it is negative) to the
largest, whether or not a class in between is seen, as the Java enumerated
them. A class column that is not integer-valued is enumerated as its sorted
distinct values.

## Deliberate differences

- **Distances are physical** when the voxel size is known (a 5 µm z-step
  is five times a 1 µm x-step), and in voxels otherwise. The Java had a
  single "Voxel Z scale" factor, which is the special case where x and y
  are 1.
- **Randomise permutes** which object sits at which position rather than
  drawing new positions. A permutation keeps the tissue's density (crowded
  where the tissue is crowded), so what the null model removes is who is
  next to whom and nothing else.
- **The grid** spans the objects' own extent, starting half an interval in,
  rather than the image width from pixel `interval`: the table does not
  carry the image size, and a grid that starts in an empty margin samples
  the margin. `extent=` bounds it explicitly.
- **`mean_distance`** (members' mean distance from the centre) is filled
  in. The Java declared `meanDistances` and never computed it.

## Why membership is not an `AssociationSet`

An association gives a child one parent (see `OBJECT_ASSOCIATION.md`).
Neighbourhoods overlap by design: a cell is a member of its own
neighbourhood and of the neighbourhood of every cell near it. So membership
is a many-to-many table (`NeighborhoodSet.membership()`: one row per
neighbourhood-member pair, `is_seed` marking the object a neighbourhood is
centred on) rather than a shape that would have to pick one parent.

For a centred method (`nearest`, `radius`) a neighbourhood's id **is** its
seed object's id. The neighbourhood table and the object table join on id,
and a gate drawn on neighbourhoods in the Object Explorer lights up the
cells at their centres.

## Reflection: two relations

An object relates to neighbourhoods in two ways, and they answer different
questions:

- **its own** neighbourhood, the one centred on it. "What surrounds this
  cell?" Reflected one to one.
- **the neighbourhoods it is a member of**, its own included. Grid
  neighbourhoods only have this kind. "What is this cell part of?"
  Quantities are averaged over them, categories (a neighbourhood type) take
  the most frequent value (ties go to the smallest), and `n_neighborhoods`
  says how many there were.

`relation="auto"` takes the own neighbourhood where there is one and the
membership otherwise. An object in no neighbourhood gets NaN for a quantity
and -1 for a category; it is never dropped.

## Where it runs

**As steps**: `build_neighborhoods`, `neighborhood_features`,
`classify_neighborhoods` and `reflect_neighborhoods` are registered
pipeline steps (the `neighborhoods` category), usable from a script or a
saved protocol. They are deliberately *not* in the protocol builder's
analysis menu: offering the same analysis in two places was confusing, and
the pane below is where it is built. A protocol that carries the steps
still opens and runs in the builder: the neighbourhood table is published
to the Object Explorer as its own table (rows are neighbourhoods), and the
reflected columns join the object table as
`reflect_neighborhoods_1.neighborhood_type` and so on.

**Interactively**: the **Neighborhoods** pane (Plugins → VTEA →
Neighborhoods, or the button beside "Object Explorer" in the builder) does
the four steps in one go from any table the protocol produced, objects or
cells, and draws both levels on the same napari viewer:

- a Points layer, one point per neighbourhood at its centre, sized to its
  reach and coloured by any neighbourhood feature. Clicking a point selects
  that neighbourhood;
- a Labels layer painting every object by what it took on (its
  neighbourhood type, by default), so the tissue reads as regions of one
  kind of neighbourhood or another;
- a members layer showing only the selected neighbourhood's objects.

What the pane builds is held in the shared session as a
`NeighborhoodResult`, as image gates are. It is published beside the
builder's tables and its reflected columns are merged back onto their table
**by id** on every builder run, so a re-run that adds or drops objects
leaves the rest with their values. Each analysis is built from the table
as the builder published it, never from another analysis's reflected
columns. Opening a protocol clears them: they describe results that are
gone.

The pane's reflected columns are for the Object Explorer: they are
deliberately *not* handed to the builder's steps, because a clustering with
no feature selection uses every numeric column and would change silently
the moment a pane analysis existed. A protocol that wants to classify cells
by their neighbourhood (`cd8 AND reflect_neighborhoods_1.neighborhood_type == 2`)
has to carry the four neighbourhood steps itself, which today means adding
them from a script (see "As steps" above).

## Not done

- **Neighbourhoods of neighbourhoods.** The pane offers only object and
  cell tables as members. The core functions take any table with
  `centroid-*` columns and an id column, so the pane change is small, but
  nobody has asked for the analysis yet.
- **Saving a pane-built analysis.** It is not in the protocol file: a
  protocol carries steps, and the pane's analysis is a result. Add the four
  steps to the protocol to carry it. `save_neighborhoods`/
  `load_neighborhoods` persist a `NeighborhoodSet` from a script.
- **Parity.** No Java neighbourhood fixture exists (see `PORT_PLAN.md`
  M1/M2). The composition columns are simple enough to check by hand, and
  the tests do.
