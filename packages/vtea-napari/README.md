# vtea-napari

napari plugin GUI for VTEA (Volumetric Tissue Exploration and Analysis) — a
thin UI layer over [`vtea-core`](../vtea-core). napari is the closest Python
analog to the ImageJ/Fiji viewer the Java application plugs into today: it's
Qt-based, has native 3D volume rendering, and an active plugin ecosystem.

See [`/docs/PORT_PLAN.md`](../../docs/PORT_PLAN.md) in the repo root for the
full porting plan, including the "Protocol builder: Option A" section
explaining this widget's design.

## Status

Phase 4 in progress. Implemented and tested (25 tests, including a real
`napari.Viewer` integration test that loads the plugin the way an end user
would):

- **`ProtocolBuilderWidget`** — the protocol builder, registered as a napari
  dock widget (`napari.yaml`). Add steps from a category/function picker
  (populated from `vtea_core.workflow.STEP_REGISTRY`), see them as an
  ordered stack of cards (position, name, parameter summary, Edit/Delete),
  edit parameters through a form, delete steps - the same operations
  `vtea.protocol`'s Java UI exposed. The resulting `vtea_core.workflow.Pipeline`
  runs the same whether triggered from this widget or a script.
- **`ParameterForm`** — builds the Edit-step form from a registered
  function's actual signature, split into data arguments (arrays/dataframes,
  excluded - resolved from the pipeline's run context) and editable
  configuration values (thresholds, cluster counts, ...). Not magicgui-based
  in the end - `vtea_core`'s `from __future__ import annotations` style
  makes its type hints plain strings at runtime, which tripped up magicgui's
  auto-resolution in practice, so this uses plain qtpy widgets instead (see
  `param_form.py`'s docstring).
- **`StepCardWidget`** — one step's card.

Still open for Phase 4: a `MicroExplorer`-equivalent plot/table view, an
interactive gate manager, LUT controls (largely free via napari's built-in
colormaps), heatmap/violin plot widgets, and thumbnail previews on step
cards (the Java UI had these; this first pass doesn't).

## Try it

```bash
pip install -e "../vtea-core" -e ".[dev]"
napari
# Plugins menu -> VTEA -> Protocol Builder
```
