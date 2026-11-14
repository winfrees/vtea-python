# The spatial hierarchy

Neighbourhoods are objects (see [NEIGHBORHOODS.md](NEIGHBORHOODS.md)). This
module takes the next step: since a level of neighbourhoods is a table of
objects like any other, it can be built on again, to any depth.
`vtea_core.hierarchy` holds the levels, builds them, and moves information
between them.

```
Objects (cells)                 level 0: the segmentation (masks)
├── Neighbourhoods 1            windows on cells
│   ├── Neighbourhoods 2        windows on those windows
│   └── Regions 2               regions of contiguous same-type windows
└── Regions 1                   regions of touching cells of one type
```

Levels branch: one level can be the parent of several.

## A level

`ObjectLevel` is a table (one row per object: an id, `centroid-*` in voxels so
it can be drawn on the image, and features), a **membership** to its parent and
a **provenance**. The membership is a sparse `(objects x parent objects)` 0/1
matrix in row order, many-to-many, so walking down several levels is a product
of matrices. Three kinds:

| kind | made of | overlap | answers |
| --- | --- | --- | --- |
| `objects` | the segmentation | - | - |
| `windows` | a window around each seed object of the parent (radius or k nearest) | yes: an object is in many | what surrounds each object, and how that changes (an infiltrate's front and its hinterland) |
| `regions` | contiguous parent objects of one type, merged | no | which functional units exist (a tubule, a glomerulus), their size, shape and makeup |

Windows on windows stay windows: a level-2 window describes the surroundings of
the surroundings. Regions are a separate builder, not what deeper levels turn
into.

## Building

```python
from vtea_core.hierarchy import Hierarchy

h = Hierarchy(cells, spacing=spacing, labels=label_image)      # level 0
w1 = h.add_windows("Objects", "NH1", radius=50, classes="phenotype")
w1.table["type"] = ...                                        # e.g. classify_neighborhoods
w2 = h.add_windows("NH1", "NH2", radius=50, classes="type")    # windows on windows
r = h.add_regions("NH1", "Units", types="type")               # functional units
```

**Windows** take optional `seeds` and `members` populations of the parent: which
objects get a window, and which can be counted in one. Seeding on a subset while
counting everything is usually what is wanted (windows around tubular
neighbourhoods, with whatever infiltrates them). A seed is always a member of its
own window; k nearest gives exactly k others, ties included.

**Regions** join parent objects of the same type in two stages:

1. **Candidates.** The two *touch* - masks touching at level 0 (from the label
   image, 3D included); above it, windows whose seeds touch and regions whose
   members touch, so touching always comes down to masks - or their centres are
   within a distance `d` (k-d tree pairs). `d = 0` is touching only; a larger `d`
   bridges small gaps. Sharing members never makes a candidate: overlapping
   windows share members almost everywhere.
2. **Weight** (when the parent objects have members of their own):

   ```
   w(A, B) = mean over m in A ∪ B of exp(−δ(m)² / 2σ²)
   δ(m)    = distance from m to the nearest member of the other (0 if shared)
   ```

   computed with k-d tree queries on the members' positions. Joined when
   `w >= tau` (default 0.5). `sigma` defaults to the median nearest-neighbour
   distance among the members (one cell spacing). Two windows whose seeds touch
   but whose members extend in different directions - two tubules side by side -
   score low and stay apart: on two rows of cells 10 apart with the rows 16
   apart, windows along a row score 0.87 and windows across the rows 0.28.

Connected components of the joined pairs, at least `min_size` objects, are the
regions, numbered from the largest. Each gets `n_objects`, its centre, and its
shape from its members' positions: `radius_of_gyration` and `elongation` (square
root of the ratio of the two largest gyration eigenvalues; a strand is large).

Both builders measure composition over a parent `classes` column (the
`Class_<c>_ClassFraction` / `ClassSums` columns of `neighborhoods`) and can
aggregate parent `features`.

## Moving between levels

| method | direction | gives |
| --- | --- | --- |
| `expand(level, to)` | down | which objects of the ancestor `to` each object contains, through every level between |
| `reflect(level, onto, columns, how="member")` | down | per object of `onto`: mean (or most common category) over the objects containing it, and `n_containers` |
| `reflect(..., how="own")` | down | its own window's value, following seeds through every windows level |
| `aggregate(level, of, columns)` | up | each object's summary of the `of` objects it contains |
| `members_of` / `containers_of` | down / up | a selection followed between levels |

## Bookkeeping

- `is_stale(name)`: whether the parent's objects, members or the columns a level
  was built from changed since it was built (or an ancestor's did). Adding an
  unrelated column does not make a child stale.
- `remove(name)` removes a level and everything built on it.
- `tree()` lists `(depth, name)` depth-first: what a hierarchy map draws.
- `save(folder)` / `Hierarchy.load(folder)`: one sub-folder per level
  (`table.csv`, `membership.npz`, `seeds.npy`), `contacts.npy` and
  `hierarchy.json`.

## Protocol steps

Category `hierarchy`, one `Hierarchy` threaded through under the context key
`hierarchy`: `hierarchy_from_objects` (from `data`, optional `spacing` and
`labels`), `add_windows_level`, `add_regions_level`, `classify_level` (types a
level, by default on its composition) and `reflect_level` (to
`level_reflection`, one row per object). Levels are named; an empty `parent` or
`level` means the newest. Re-running a protocol replaces levels of the same name.

## Scale

On 100,000 cells in 3D (one core): windows 1.8 s, windows on windows 1.6 s,
regions of cells 0.3 s, regions of windows 3.4 s (35,000 candidate pairs; the
member weighting is per pair, about 0.1 ms each), expand through two levels
0.6 s.

## References

The neighbourhood analyses this builds on (see also [NEIGHBORHOODS.md](NEIGHBORHOODS.md)):

- Winfree S, McNutt AT, Khochare S, et al. Integrated cytometry with machine learning
  applied to high-content imaging of human kidney tissue for in situ cell
  classification and neighborhood analysis. *Laboratory Investigation* 103(6), 100104
  (2023). doi:10.1016/j.labinv.2023.100104. VTEA tissue cytometry with NephNet3D
  classification of kidney cell types from a nuclear stain, and neighbourhood analysis
  of the classified cells in situ.
- Lake BB, Menon R, Winfree S, et al. An atlas of healthy and injured cell states and
  niches in the human kidney. *Nature* 619, 585–594 (2023).
  doi:10.1038/s41586-023-05769-3. Injury neighbourhoods located by spatial
  transcriptomics, and large-scale 3D imaging analysis of about 1.2 million
  neighbourhoods linking them to active immune responses.
- Asghari M, Sabo AR, Barwinska D, et al. Integration of spatial protein imaging and
  transcriptomics in the human kidney tracks the regenerative potential of proximal
  tubules. *Science Advances* 11(33), eadv8918 (2025). doi:10.1126/sciadv.adv8918.
  Cell-centric neighbourhoods built with VTEA's "spatial by cell" method (55 µm
  diameter, matched to the Visium spot), described by class sums and fractions,
  clustered with Louvain into 19 niches, compared between reference and disease
  tissue by odds ratio.
