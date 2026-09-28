# Contexts: moving between length scales

A tissue analysis is not about one kind of object. A lysosome is part of a
cell; the cell is a member of a neighbourhood; neighbourhoods group into
larger structures. Each is worth defining, typing and gating in its own
right, and each is described by the others: a cell by the neighbourhood it
lives in, a neighbourhood by the cells it holds. A **context** is one level
of that hierarchy, and the analysis should be able to switch between them
the way a microscope switches objectives.

## Decisions

- **Three tiers, fixed:** *subcellular* (the objects of one segmentation
  that are pieces of a cell), *cellular* (cells, as `build_cells` composes
  them), *neighborhood* (built from a lower level: depth 1 from cells or
  pieces, depth 2 from neighbourhoods, and so on to any depth).
- **Level definitions are saved in the protocol**, in a `context` section,
  so the same hierarchy is rebuilt from the next acquisition. The levels
  themselves are results and are not saved.
- **Higher levels are drawn as outlines**: a hull per neighbourhood, with a
  user-chosen fill pattern (none, solid, hatch, cross-hatch, dots) and colour.
- **Neighbourhood overlays come from gating.** By default only the
  neighbourhoods a gate selects in the Object Explorer are drawn; drawing
  all of them (`draw="all"`) has to be asked for, because neighbourhoods
  overlap by design and a screen of overlapping hulls shows nothing.
- **Levels need not nest.** Membership is many-to-many where neighbourhoods
  overlap, and the levels are a stack rather than a strict tree.

## The model (`vtea_core.context`) - built

- `ContextSpec` / `LevelSpec` / `LevelDisplay`: the definitions a protocol
  saves. A neighbourhood level carries its build settings (method, radius,
  k), what it is measured on (class column, member features), optional
  typing (clustering method, number of types), which of its columns are
  handed down (default: its type), and how it is drawn.
- `ContextGraph`: the built levels (`Level`: a table, its id column, its
  rank, its label image) and the links between them (`Link`: a membership
  table, `part_of` or `member_of`).
- Two operators, defined once for every link and composed along chains:
  - **aggregate up**: a parent's features from its children (count, mean,
    sum, min, max);
  - **reflect down**: a child takes on its parents' features. It uses the
    value of its own neighbourhood where there is one, and otherwise the
    mean of a quantity or the most common category.

  Each link is a sparse matrix, and a chain is their product. A lysosome
  therefore takes on the type of its cell's neighbourhood of neighbourhoods
  without any code that knows about lysosomes or depth.
- **The circularity rule:** a level may not be defined or typed on a column
  handed down from its own rank or above, since the result would be its own
  type back again. `typing_features` leaves such columns out, and
  `build_context` refuses a definition that names one, by name.
- `build_context(spec, measurement_tables=..., cells=..., cell_tables=...)`
  builds the stack from a run's results, in order, and hands each
  neighbourhood level's type down through every level beneath it.

Protocols with a context section are written as version 2. Those without
one are still written as version 1, so earlier VTEA builds keep opening them.

## Phases

| Phase | Scope | Status |
|---|---|---|
| C1 | `ContextGraph`, sparse aggregate/reflect operators, circularity rule | **done** |
| C2 | Subcellular levels linked to cells; reflection cell → piece, and chained | **done** (in `build_context`) |
| C3 | Neighbourhoods of neighbourhoods to any depth | **done**, with a fix to `NeighborhoodSet.membership()` (members that are themselves neighbourhoods now use `member_id` instead of colliding with `neighborhood_id`) |
| C6 | Level definitions in the protocol's `context` section | **done**: the builder's Save… writes it and Open… restores it |
| C4 | The session holds the graph, rebuilt from every builder run; each level is a table in the Object Explorer; the Neighborhoods pane and `NeighborhoodResult` are retired | **done** |
| C5 | The Context pane: the level slider; defining the next level; hull outlines with pattern and colour; gated-only drawing; carrying a selection across levels; a radius preview | **done**; see "What was built" below |

### What was built (C4, C5)

- **Levels come from the run.** Every measured segmentation is a
  subcellular level and every `build_cells` result a cellular one. With no
  cells, each segmentation is the cellular level. Levels a person defines
  are built on these and saved. A level whose source is missing after a
  re-run (a renamed segmentation) is left out with its reason shown, and
  the other levels are still built.
- **Level tables in the explorer** appear once a level above the cells
  exists. Before that, each would repeat a table the builder already
  published.
- **The Context pane** (Plugins → VTEA → Context, or the builder's Context
  button):
  - the slider runs through the levels, and the explorer's table follows
    it (changing the table in the explorer moves the slider too);
  - **Define neighbourhoods of this level** builds the next level from the
    one showing. Class columns handed down from that rank or above are not
    offered, since a level cannot be defined on its own type;
  - the radius slider previews one neighbourhood at the chosen size (the
    one nearest the middle of the field) and reports how many members
    neighbourhoods get;
  - **Display** sets the outline colour, fill pattern (none, solid, hatch,
    cross-hatch, dots), fill colour and opacity, and whether to draw gated
    neighbourhoods or all of them. All of it is saved with the definition;
  - **Carry the gated selection** follows what is gated on the level being
    left: up to the neighbourhoods those entities are in (drawn as
    outlines), or down to their pieces (highlighted on the label image);
  - **Remove this level** removes it and every level built on it.

### Not done yet

- **Hulls are per-slice 2D outlines.** A 3D view shows them as stacked
  outlines, not a surface.
- **Clicking an outline does not select its neighbourhood.** Selection
  goes through the Object Explorer's gates.
- **The context runs in memory only.** A blocked run's cells (a membership
  table rather than `Cell` objects) cannot seed a cellular level yet.

### C5 design notes

### C5 notes

- **Hulls.** The outline is the convex hull of the members' footprints in
  the displayed plane, drawn as a napari Shapes polygon on every slice the
  neighbourhood spans. A 3D view gets a surface later if it is wanted.
- **Fill patterns.** napari has no patterned fill, so hatch, cross-hatch and
  dots are drawn as line segments or points clipped to the hull, in a
  second layer beneath the outline. Solid is the polygon's own translucent
  face.
- **The slider switches:** the explorer's table, gates and gallery (crop
  size follows the level's scale); which level's layers are shown on the
  viewer; and the definition pane, which becomes "define / type this
  level".
