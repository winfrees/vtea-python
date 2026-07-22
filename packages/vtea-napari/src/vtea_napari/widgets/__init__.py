"""napari dock widgets.

Implemented: the protocol builder (Option A - see PORT_PLAN.md's "Protocol
builder: Option A" section). Still open for Phase 4: a MicroExplorer-
equivalent plot/table view, an interactive gate manager, and LUT/colormap
controls (largely covered by napari's built-ins already).
"""

from vtea_napari.widgets.protocol_builder import EditStepDialog, ProtocolBuilderWidget
from vtea_napari.widgets.step_card import StepCardWidget

__all__ = ["ProtocolBuilderWidget", "EditStepDialog", "StepCardWidget"]
