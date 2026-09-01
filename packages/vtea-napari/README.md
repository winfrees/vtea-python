# vtea-napari

napari plugin GUI for VTEA (Volumetric Tissue Exploration and Analysis) — a
thin UI layer over [`vtea-core`](../vtea-core). napari is the closest Python
analog to the ImageJ/Fiji viewer the Java application plugs into today: it's
Qt-based, has native 3D volume rendering, and an active plugin ecosystem.

See [`/docs/PORT_PLAN.md`](../../docs/PORT_PLAN.md) in the repo root for the
full porting plan, including the "Protocol builder: Option A" and "Object
Explorer" sections explaining these widgets' design.

## Status

Phase 4 done. Implemented and tested (608 tests, including real
`napari.Viewer` integration tests that load the plugin the way an end user
would, and end-to-end tests that build a pipeline purely through the
widget and run it):

- **`ProtocolBuilderWidget`** — the protocol builder, registered as a napari
  dock widget (`napari.yaml`). Add steps from a category/function picker
  (populated from `vtea_core.workflow.STEP_REGISTRY`), see them as an
  ordered stack of cards (position, name, parameter summary, Edit/Delete),
  edit parameters through a form, delete steps - the same operations
  `vtea.protocol`'s Java UI exposed. The resulting `vtea_core.workflow.Pipeline`
  runs the same whether triggered from this widget or a script.
  Every step card has its own Run button - there is no pane-level Run, since
  the analysis steps are a graph rather than a chain - and each run drives
  that card's thumbnail. Two vertically-split panes (processing, analysis)
  capped at 30% of the screen width; plotting and gating are in the Object
  Explorer, which this pane publishes its results to.
  Steps added from the menu wire themselves up via `Step.for_function()` /
  `vtea_core.workflow.wiring`: each step's data arguments are resolved from
  the run context by name and its result stored under a semantic key
  (`threshold_mask -> mask`, feeding `label_components(mask)`), so a
  pipeline built entirely by clicking runs without anyone hand-editing
  `input_keys`.
- **Named results** — every step gets a unique default name from the
  function that produced it (`watershed_split_1`, `watershed_split_2`),
  editable in its Edit dialog and shown on its card. `output_key` stays
  semantic and shared - which is what makes a chain wire itself up - and
  each result is *additionally* published under the step's name, so a
  protocol with two segmentations can say which one a later step means.
  The Edit dialog lists the named producers of each data input, so a
  measurement step picks a segmentation by name instead of taking whichever
  ran last; renaming a step re-points everything that referred to it.
- **One flat "data" table** — a measurement step
  (`extract_measurements_by_channel`, the default for the measurements
  category) measures its chosen segmentation against *every* channel and
  collapses the channel dimension into the column names: `mean_ch0`,
  `mean_ch2`, ... Geometry columns (`object_id`, `count`, `centroid-*`)
  describe the object rather than its brightness and appear once. Per-object
  analysis results join the same table under the producing step's name
  (`kmeans_1`, `pca_1_1`/`pca_1_2`), so measured and derived features are
  plottable against each other from the plot's X/Y menus. The clustering and
  reduction steps' `data` input is the table itself - nothing in a protocol
  produces a `data` key, so without the widget seeding it those steps could
  not be run from the GUI at all - and each step narrows it to its own
  chosen features.
- **Every segmentation is measured** — a protocol rarely has one. A nucleus
  is segmented, a cytosol ring is derived from it, a second channel gives
  lysosomes; each is a population of objects with its own features, and
  before this the ring was silently never measured unless someone
  remembered to add a second measurement step and re-point it. Now each
  named segmentation gets a measurement step of its own, named after it
  (`measure_nuclei`, `measure_nuclei_ring`) - the segmentation's own name,
  default or typed in. Rename the segmentation and its measurement follows,
  through the GUI, along with everything wired to either and the feature
  catalog's record of what was measured on what; delete the segmentation and
  the measurement goes with it; measure one by hand and nothing is raised
  for it. Each measurement travels to the Object Explorer as its own table,
  because a ring's rows are rings - joining them onto the nuclei would be a
  claim about which ring belongs to which nucleus that only an association
  step is entitled to make. "Measure every segmentation" in the top row
  turns the whole thing off for a protocol that segments an intermediate
  mask it does not care about.
- **A changed setting recalculates** — editing a step's parameters means the
  result on its card was computed from settings that are no longer that
  step's. Rather than leave it there looking authoritative, the builder
  re-runs that step and everything downstream of it that had already run.
  Steps that have never run are left alone: a click on "Edit" should not
  turn into an hour of watershed. A *rename* recalculates nothing - it
  changes what a result is called, not what it is.
- **Per-step progress bars, off the GUI thread** — every card carries one,
  under its buttons and no wider or taller than they are. A step whose
  duration follows from the size of the work (voxels in the tile, objects in
  the table) gets a real fraction and a countdown, from
  `vtea_core.workflow.estimate_seconds`, calibrated against what the steps
  actually took on this machine so the estimates stop being wrong in the
  same direction every time. A step whose runtime genuinely cannot be
  predicted - t-SNE, UMAP, a Leiden partition, agglomerative clustering -
  gets a continuous bar instead of a fraction that would have to be
  invented, and a tiled run reports the real one it has (tiles done over
  tiles planned). The steps themselves run on a worker thread while the
  event loop keeps being pumped, so clicks and repaints keep working during
  a run; everything that touches a layer happens back on the GUI thread, and
  progress crosses between them through a relay rather than by a worker
  thread setting a widget's text.
- **Reductions and clusterings are data, not layers** — a t-SNE embedding of
  nine thousand nuclei is a (9000, 2) array, so it used to land in the layer
  list as a two-pixel-wide stripe. Every reduction and clustering result now
  goes only where it means something: joined onto the data table as features
  (`umap_1_1`, `leiden_1`) and onto the Object Explorer's axes. The log says
  which columns it added, and asking to show one explicitly says why there
  is nothing to show.
- **UMAP, Louvain and Leiden** — `umap` joins the reduction menu (it keeps
  the global structure t-SNE discards and can project new objects into an
  existing embedding), and `louvain`/`leiden` join the clustering menu.
  Those two are the only clustering steps that decide how many populations
  there are rather than being told: they detect communities in a
  shared-nearest-neighbour graph, with `resolution` as the dial instead of
  `k`. Their backends are optional extras of `vtea-core`; the steps stay in
  the menu regardless and say what to install if picked without one.
- **Classes instead of gate steps** — the analysis menu's `gates` category
  is now `classes`, and the polygon and rectangle *steps* are gone from it.
  They never worked as protocol steps: both needed vertices no step
  produces, so they could only be configured by typing polygon coordinates
  into a form. Drawing a polygon is an Object Explorer gesture and stays
  there, unchanged. What a protocol can carry is the rule - a range
  (`mean_ch2` from 50 to 150), a single gate or ROI or cluster id, or any
  boolean combination of them (AND / OR / NOT / XOR / XNOR / NAND / NOR) -
  which re-runs on the next acquisition without anyone drawing anything
  again. A gate drawn in the explorer reaches a class step as a column of
  the table it is handed (`gate_bright`), and a napari ROI as another
  (`roi_tubules`), so `gate_bright AND roi_tubules == 2 AND NOT kmeans_1 in
  [3, 7]` is a step in a protocol. `label_set` groups classes into the set
  an object's labels live in - an object carries as many as apply, and the
  log says how many carry none and how many carry several rather than
  quietly picking a winner - and `combine_labels` crosses two sets into the
  hierarchy that makes "immune > CD3+" a population you can count, colour
  and map. Each label set lands in the data as one boolean column per label
  plus a code column, so the next level of the hierarchy is written in terms
  of the last.
- **Image gates** — a region painted on a napari Labels layer, read as
  "which objects are in there". Pick the layer in the Object Explorer's
  header: the objects inside are ringed on the plot, the region each object
  is in becomes a column of the data (usable in a class definition), and the
  count is reported. The rings are the plot's spare encoding on purpose -
  fill is the LUT's, so an image gate and a colour-by can be read at the
  same time. With one region they take the colour set by "Ring colour…";
  with several, each ring takes that region's own colour in napari, so a
  tubule and the nuclei inside it are the same colour in both windows. The
  membership lives on the shared session rather than in the table, so a
  re-run of the protocol does not discard a region somebody painted.
- **Gate highlights are the volume, not a section** — a highlight is put
  back on the source image's axes before it goes into the viewer. A
  channel-sliced segmentation has one axis fewer than the image, and napari
  right-aligns arrays of differing ndim, which mapped a z-stack's leading
  axis onto the *channel* axis: the highlight appeared as a single flat
  section that moved when the channel slider did. It also carries the source
  layer's scale and translate, so on an anisotropic stack in microns it
  lands on the objects rather than near them, and above a few million voxels
  it is computed lazily block by block - a dozen gates should not cost a
  dozen copies of the segmentation.
- **The plot is 4:3, and the gate list is under it** — the scatter keeps its
  proportions whatever the dock does, so two runs are not different shapes
  because the window was; and the gate manager, now a wide strip below the
  plot rather than a column beside it, stops taking a third of the plot's
  width (and, at a fixed ratio, its height with it).
- **Continuous and discrete LUTs** — a measured feature gets a gradient and
  a colorbar; a cluster id, a class, a label-set code or an ROI membership
  gets a distinct colour each and a legend. A gradient over cluster ids puts
  cluster 2 "between" 1 and 3, which is a statement about a numbering rather
  than about the tissue. Which one a column is comes from the feature
  catalog (it records the step that produced each column) and from a
  deliberately cautious reading of the values - whole numbers, few of them,
  starting at 0 or -1, which is what an id looks like and what a voxel count
  does not - and the LUT-mode picker overrides both.
- **Derived segmentations and association** — the segmentation menu carries
  the morphology-only steps (`label_ring` for cytosol, `label_shell` for a
  nuclear envelope, `expand_labels`, `subtract_labels`, `restrict_labels_to`,
  and `watershed_ownership` to divide a shared region between the cells in
  it), and the analysis pane carries an `association` category:
  `associate_by_identity` where the child was built from the parent, and
  `associate_objects` where two channels were segmented independently and the
  link has to be inferred. Because those steps read a label image rather than
  an intensity image, their cards say "no channel" and their Edit dialog
  offers none; their `labels`/`mask`/`child_labels`/`parent_labels` inputs
  are pickable by segmentation *name*, so nucleus → ring → association is
  wired by choosing steps from a menu. An association step's segmentation
  names are **not** form fields: they come from the wiring, so a link records
  the step its input is actually pointed at, and re-pointing the step moves
  the names with it. An association draws nothing, so what the log shows is
  its summary — how many children were linked, how many were left
  unassigned, how many were close calls — which is the number that says
  whether the parameters were right.
- **Cells** — a `cells` analysis category: `build_cells` follows the
  associations out from a chosen root segmentation (its `root` is the wiring,
  like an association's names), and `cell_features` produces one row per cell
  from one measurement table per segmentation. Those tables cannot be one
  flat frame — a nucleus table and a lysosome table have different rows — so
  the builder seeds them keyed by the segmentation each step measured, the
  way it seeds `data` for clustering. The Object Explorer plots and gates the
  result: a table picker appears once there is more than one table, and each
  table travels with what its rows are — the id column that names a row
  (`cell_id` rather than `object_id`), the segmentation a gate on it should
  light up on the image, and its own gates, since a polygon drawn over cell
  features selects nothing among objects.
- **Probabilistic ownership** — an `ownership` processing category
  (`distance_ownership`) beside the segmentation steps, since it reads a
  label image and a mask and produces something image-shaped. Its result goes
  on as two layers at once — the hard answer as Labels, the confidence as an
  Image — because the argmax alone is indistinguishable from a watershed,
  which is exactly the problem the confidence map exists to solve, and the
  log reports how much of the field was contested. A
  `weighted_measurements_by_channel` step measures it, and knows which
  segmentation its rows are objects of by asking the ownership rather than
  guessing from the step graph.
- **Reviewing an association** — an "Associations" tab in the Object
  Explorer: the run's summary, and the links the method was least sure about
  worst first with their probability, margin and runner-up. Selecting one
  puts both objects on the image in different colours. A correction is a
  choice between the parents actually considered, plus "no parent", recorded
  as `manual` with the answer it replaced — and a settled link stops coming
  back in the list, since a person's decision is not a posterior. Decisions
  are kept on the session and re-applied after every association run, so
  tuning the parameters and re-running corrects the automated answers without
  undoing the settled ones; the log says how many were kept. Save/Open write
  the associations, corrections included, as JSON.
- **Fixed-choice parameters are dropdowns** — a step function annotating a
  parameter `Literal["many_to_one", "one_to_one"]` gets a combo box rather
  than a text field, read from the annotation text (vtea-core's hints are
  strings at runtime). A mode that can only be picked from a list cannot be
  typed wrong.
- **Analysis categories** — measurements, association, cells, clustering,
  reduction and gates.
  Not classification: its steps need `crops`, a `model`, `object_ids` and
  `class_labels`, none of which any protocol step produces, so every one of
  them could only ever fail with "needs context key(s) [...]". The functions
  stay in `vtea-core` and work from a script; putting them back in the menu
  needs a crop-extraction step and a way to label training objects, neither
  of which exists yet.
- **Channel selection** — "Channel axis" on the widget says which axis of
  the loaded image holds channels (a property of the data, listed with each
  axis's size, defaulting to none); each step's Edit dialog then picks which
  channel *that step* runs on, shown on its card. Only inputs still
  carrying the channel axis are sliced, so a step consuming an earlier
  step's output never has a spatial axis sliced by mistake. A channel-aware
  step is the exception: it is handed the whole multi-channel array and its
  channel choice as an argument, since it has to see every channel at once
  to label its output columns by channel. A step that reads the *feature
  table* rather than the image - clustering, reduction, gating - has no
  channel at all (every channel is already a column there), so it gets no
  channel picker, inherits no channel when added, and is never sliced; its
  card names the features it uses instead.
- **`FeatureSelectWidget`** — which of the measured features a clustering or
  reduction step is built from. A protocol measuring seven properties across
  four channels already has 28 features before the derived ones, so a flat
  checklist alone is the wrong shape of control: this pairs it with a filter
  box and All/None/Invert scoped to *what the filter is showing*, which makes
  "use every `_ch2` feature" one gesture rather than twelve. Each row's
  tooltip carries that feature's provenance from the `FeatureCatalog`. An
  all-checked list is stored as an empty selection, so a protocol doesn't pin
  a list that should grow when a later measurement step adds features.
- **Feature provenance** — every column of the data table is catalogued as
  the step producing it runs: what was measured, on which channel and which
  named segmentation, by which step with what parameters, and - for a
  clustering or reduction - exactly which features were fed to it. The
  catalog lives on the shared session, renders as the publication data
  dictionary, and is dropped and rebuilt on a re-measurement so a stale entry
  can't outlive the column it described.
- **`SpacingControl`** — the physical voxel size, in the builder's top row
  beside the axis pickers because it is the same kind of fact: how to read
  the array as a specimen. Read from the image where the file recorded one,
  and asked for where it didn't — napari fills `layer.scale` with ones both
  when a file says "one unit per voxel" and when it says nothing, so
  "unknown" is a state the user has to be able to resolve rather than a
  value to be believed. Everything distance-dependent that follows (dilation
  thicknesses, object-to-object distances) needs it, and gets it wrong in a
  plausible-looking way without it.
- **`ParameterForm`** — builds the Edit-step form from a registered
  function's actual signature, split into data arguments (arrays/dataframes,
  excluded - resolved from the pipeline's run context) and editable
  configuration values (thresholds, cluster counts, ...). Not magicgui-based
  in the end - `vtea_core`'s `from __future__ import annotations` style
  makes its type hints plain strings at runtime, which tripped up magicgui's
  auto-resolution in practice, so this uses plain qtpy widgets instead (see
  `param_form.py`'s docstring).
- **`StepCardWidget`** — one step's card, with an optional thumbnail preview
  of that step's last-run output (only for the steps that produce an image)
  and its own progress bar.
- **`AnalysisSession`** (`session.py`) — the state the two dock widgets
  share: the protocol, the run context, the feature table, and the gates.
  It is keyed by the napari viewer and owned by neither widget, so results
  computed while the explorer was closed are waiting when it opens, gates
  drawn in the explorer survive it being hidden, and a plugin widget napari
  destroys and rebuilds comes back with its steps. It also carries the
  explorer's *view* - axes, colour/size encodings, point style - because
  closing a napari dock destroys the widget, and a pane that reopened on the
  first two columns with default styling meant rebuilding the view by hand. It is also the seam a
  saved session will be written from (see
  [`/docs/SAVING_AND_ARCHIVING.md`](../../docs/SAVING_AND_ARCHIVING.md)).
- **`ExplorerWidget`** — the `MicroExplorer` equivalent, registered as the
  "Object Explorer" napari dock widget, and **floating by default**: a
  scatter plot docked into napari's side panel is unusable at the width it
  gets there, and gating means working between the plot and the image. It
  reads the shared session rather than owning its own copy of the analysis.
  Holds the scatter plot with the gate manager beneath it (3:1) on a "Plot"
  tab and the per-object crop grid on a "Gallery" tab; subgate within a
  selection for real gate hierarchy; each gate's members get their own
  napari `Labels` overlay painted in *that gate's* colour and shown only
  while the gate is visible, so two gates can be read against each other on
  the image without holding a colour mapping in your head, and the Visible
  checkbox that hides a gate's outline hides its highlight too. See PORT_PLAN.md's
  "Object Explorer" section for what was simplified vs. the Java original
  (one gate type, no dead `GateManager` port, real hierarchy where Java had
  none).
- **`ScatterPlotWidget`** — the matplotlib-backed plot: click to add a gate
  vertex, double-click to close it, right-click to cancel; in rectangle mode
  two clicks (opposite corners) make the gate, still stored as a 4-vertex
  polygon so everything downstream handles one kind of gate. Axis pickers;
  "Color by"/"LUT" comboboxes for point coloring by a third feature
  (replaces `vtea.lut`'s point-coloring, not ImageJ's per-channel image
  LUTs, which napari's `Image` layer controls already give you for free);
  and "Size by" to scale each point by a feature. Both encodings get a
  scale: a labelled colorbar for colour, and a size legend whose sample dots
  are labelled in the feature's own units rather than in matplotlib's
  points². Colour and size choices survive a re-run the way the axes do.
- **`PlotStylePanel`** — the style helper pane under the plot: point size,
  the size range a feature is mapped onto, opacity, and marker shape. A
  different kind of choice from the axis pickers - what goes on the axes is
  part of the analysis, while a few thousand overlapping opaque circles hide
  the structure underneath them and the fix is turning transparency down,
  not re-running anything.
- **`GateManagerWidget`** — the gate pane beside the explorer's plot:
  Rectangle/Polygon drawing modes, the gate list, Delete/Clear, Save/Open as
  plain JSON (`vtea_core.gates.io` — a drawn polygon is the one part of an
  analysis that can't be recomputed, so it has to be saveable alongside the
  figure), and a statistics box giving the selected gate's cell count and
  the mean of each plotted axis over the gated cells only. A gate whose axes
  aren't in the current table — reopened from JSON, or from a run that
  measured different features — is listed with blank counts rather than
  taking the pane down. Double-clicking a gate's colour swatch recolours it,
  on the plot overlay and in the saved JSON.
- **`LogView`** — the message strip at the bottom. A QLabel doesn't wrap, so
  a long traceback stretched the whole dock sideways; this wraps, keeps
  history, and caps itself at 10% of the dock's height with its own
  scrollbar.
- **`GateTableWidget`** — the gate list (replaces `TableWindow`, vtea's
  actual "Gate Management" UI - not the dead `GateManager.java`/
  `microGateManager.java` classes despite the similar names).
- **`GalleryWidget`** — per-object thumbnail grid for a gate's members,
  cropped around each object's centroid (replaces `GalleryViewWindow`).
  Clicking a crop outlines it in yellow and highlights that object alone on
  the image; the outline survives a refresh that still shows the object and
  clears silently on one that doesn't.

## Try it

```bash
pip install -e "../vtea-core" -e ".[dev]"
napari
# Plugins menu -> VTEA -> Protocol Builder, or -> Object Explorer
```
