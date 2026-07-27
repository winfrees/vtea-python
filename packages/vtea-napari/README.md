# vtea-napari

napari plugin GUI for VTEA (Volumetric Tissue Exploration and Analysis) — a
thin UI layer over [`vtea-core`](../vtea-core). napari is the closest Python
analog to the ImageJ/Fiji viewer the Java application plugs into today: it's
Qt-based, has native 3D volume rendering, and an active plugin ecosystem.

See [`/docs/PORT_PLAN.md`](../../docs/PORT_PLAN.md) in the repo root for the
full porting plan, including the "Protocol builder: Option A" and "Object
Explorer" sections explaining these widgets' design.

## Status

Phase 4 done. Implemented and tested (80 tests, including real
`napari.Viewer` integration tests that load the plugin the way an end user
would):

- **`ProtocolBuilderWidget`** — the protocol builder, registered as a napari
  dock widget (`napari.yaml`). Add steps from a category/function picker
  (populated from `vtea_core.workflow.STEP_REGISTRY`), see them as an
  ordered stack of cards (position, name, parameter summary, Edit/Delete),
  edit parameters through a form, delete steps - the same operations
  `vtea.protocol`'s Java UI exposed. The resulting `vtea_core.workflow.Pipeline`
  runs the same whether triggered from this widget or a script.
  `run_pipeline()` (and the "Run pipeline" button, shown when opened as a
  plugin with a napari viewer) also drives step-card thumbnails.
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
- **`ExplorerWidget`** — the `MicroExplorer` equivalent, registered as the
  "Object Explorer" napari dock widget. Owns a `vtea_core.gates.GateSet`
  against a measurement table (typically a napari `Labels` layer's
  `.features`); draw polygon gates on the scatter plot, manage them in a
  table (visibility/color/name/axes/counts), subgate within a selection for
  real gate hierarchy, and highlight a gate's members as a napari `Labels`
  overlay. See PORT_PLAN.md's "Object Explorer" section for what was
  simplified vs. the Java original (one gate type, no dead `GateManager`
  port, real hierarchy where Java had none).
- **`ScatterPlotWidget`** — the matplotlib-backed plot: click to add a gate
  vertex, double-click to close it, right-click to cancel; axis pickers;
  "Color by"/"LUT" comboboxes for point coloring by a third feature
  (replaces `vtea.lut`'s point-coloring, not ImageJ's per-channel image
  LUTs, which napari's `Image` layer controls already give you for free).
- **`GateTableWidget`** — the gate list (replaces `TableWindow`, vtea's
  actual "Gate Management" UI - not the dead `GateManager.java`/
  `microGateManager.java` classes despite the similar names).
- **`GalleryWidget`** — per-object thumbnail grid for a gate's members,
  cropped around each object's centroid (replaces `GalleryViewWindow`).

## Try it

```bash
pip install -e "../vtea-core" -e ".[dev]"
napari
# Plugins menu -> VTEA -> Protocol Builder, or -> Object Explorer
```
