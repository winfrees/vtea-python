# vtea-napari

napari plugin GUI for VTEA (Volumetric Tissue Exploration and Analysis) — a
thin UI layer over [`vtea-core`](../vtea-core). napari is the closest Python
analog to the ImageJ/Fiji viewer the Java application plugs into today: it's
Qt-based, has native 3D volume rendering, and an active plugin ecosystem.

See [`/docs/PORT_PLAN.md`](../../docs/PORT_PLAN.md) in the repo root for the
full porting plan, including the "Protocol builder: Option A" and "Object
Explorer" sections explaining these widgets' design.

## Status

Phase 4 done. Implemented and tested (361 tests, including real
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
  of that step's last-run output.
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
  Holds the scatter plot and gate manager side by side (2:1) on a "Plot"
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
