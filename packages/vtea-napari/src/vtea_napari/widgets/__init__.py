"""napari dock widgets.

Implemented: the protocol builder (Option A - see PORT_PLAN.md's "Protocol
builder: Option A" section) and the Object Explorer (scatter plot + gate
table + gallery view - see explorer.py's docstring for the MicroExplorer
mapping).
"""

from vtea_napari.widgets.explorer import ExplorerWidget
from vtea_napari.widgets.gallery import GalleryWidget
from vtea_napari.widgets.gate_manager import GateManagerWidget
from vtea_napari.widgets.gate_table import GateTableWidget
from vtea_napari.widgets.log_view import LogView
from vtea_napari.widgets.plot import ScatterPlotWidget
from vtea_napari.widgets.protocol_builder import EditStepDialog, ProtocolBuilderWidget
from vtea_napari.widgets.step_card import StepCardWidget

__all__ = [
    "EditStepDialog",
    "ExplorerWidget",
    "GalleryWidget",
    "GateManagerWidget",
    "GateTableWidget",
    "LogView",
    "ProtocolBuilderWidget",
    "ScatterPlotWidget",
    "StepCardWidget",
]
