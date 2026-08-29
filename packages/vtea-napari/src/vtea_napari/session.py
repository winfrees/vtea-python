"""The analysis state the protocol builder and the Object Explorer share.

The two dock widgets are separate napari plugin widgets, but they are two
views of one analysis: the builder produces a label image and a measurement
table, the explorer plots and gates that table and highlights the result
back on the image. Holding that state in whichever widget happened to
compute it means the other can only see it if it is open, and hiding a dock
(the napari Window menu, or the dock's close button) would take the work
with it.

So the state lives here instead, in an object owned by neither widget and
keyed by the napari viewer they are both attached to. Hiding, closing and
reopening a pane then costs nothing: the reopened widget reads the session
back on show. It is also the seam a saved session will be written from and
restored into (see docs/SAVING_AND_ARCHIVING.md).
"""

from __future__ import annotations

import weakref
from typing import Any

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, Signal
from vtea_core.gates import GateSet
from vtea_core.measurements import FeatureCatalog
from vtea_core.workflow import Pipeline


class AnalysisSession(QObject):
    """One analysis: the run context, the table derived from it, and the
    gates drawn on that table.

    `data_changed` fires when the measurement table or the images change -
    a step was run, a different source layer was picked. `gates_changed`
    fires when a gate is added, edited or removed. Views connect to these
    rather than to each other, so neither widget needs to know whether the
    other exists.
    """

    data_changed = Signal()
    gates_changed = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        # The protocol itself, so a rebuilt builder widget gets its steps
        # back rather than opening empty on a viewer that has already run
        # something.
        self.processing_pipeline = Pipeline()
        self.analysis_pipeline = Pipeline()
        self.context: dict[str, Any] = {}
        self.gate_set = GateSet()
        # What each column of the measurement table is and how it was
        # produced. Lives here so both panes can read it, and so it is
        # already assembled when a session is saved.
        self.feature_catalog = FeatureCatalog()
        # How the explorer's plot is set up - which axes, which encodings,
        # how the points are drawn. Closing a napari dock destroys the
        # widget, so without this the pane reopens on the first two columns
        # with default styling and the view has to be rebuilt by hand.
        self.view_state: dict[str, Any] = {}
        # How to read the source image's axes, set by the builder's pickers
        # and needed by the explorer to crop gallery thumbnails correctly.
        self.source_layer_name: str | None = None
        self.channel_axis: int | None = None
        self.z_axis: int | None = None
        self._table: pd.DataFrame | None = None

    # -- data -------------------------------------------------------------

    def set_context(self, context: dict[str, Any], table: pd.DataFrame | None = None) -> None:
        """Publish a new run context, and the flat feature table derived from
        it. The table is passed in rather than recomputed here because only
        the builder knows the step graph that names its extra columns."""
        self.context = context
        self._table = table
        self.data_changed.emit()

    def results_table(self) -> pd.DataFrame | None:
        if self._table is None or self._table.empty:
            return None
        return self._table

    def labels(self) -> np.ndarray | None:
        """The label image the measurements were taken from, for
        highlighting a gate's members back on the viewer."""
        labels = self.context.get("labels")
        return labels if isinstance(labels, np.ndarray) else None

    def intensity(self) -> np.ndarray | None:
        """The untouched source image, for gallery crops."""
        intensity = self.context.get("intensity")
        return intensity if isinstance(intensity, np.ndarray) else None

    def set_axes(
        self,
        *,
        source_layer_name: str | None = None,
        channel_axis: int | None = None,
        z_axis: int | None = None,
    ) -> None:
        self.source_layer_name = source_layer_name
        self.channel_axis = channel_axis
        self.z_axis = z_axis

    # -- gates ------------------------------------------------------------

    def set_gate_set(self, gate_set: GateSet) -> None:
        self.gate_set = gate_set
        self.gates_changed.emit()

    def notify_gates_changed(self) -> None:
        """Announce an in-place edit of the existing GateSet."""
        self.gates_changed.emit()

    # -- view -------------------------------------------------------------

    def remember_view(self, state: dict[str, Any]) -> None:
        """Keep how the plot is currently set up, so a reopened pane comes
        back to it. Merged rather than replaced, so a partial update from
        one control doesn't drop the rest."""
        self.view_state.update(state)


# Keyed weakly so closing a viewer lets its session go. A napari Viewer (and
# the ViewerModel used in headless tests) is a plain object, so it is a
# usable weak key; anything that isn't falls back to its own session.
_SESSIONS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def session_for(viewer) -> AnalysisSession:
    """The session shared by every VTEA widget attached to `viewer`.

    With no viewer - a widget built standalone, in a script or a test -
    each caller gets its own session, since there is nothing to key a shared
    one on and silently sharing global state between unrelated widgets would
    be worse than not sharing at all.
    """
    if viewer is None:
        return AnalysisSession()
    try:
        return _SESSIONS.setdefault(viewer, AnalysisSession())
    except TypeError:  # not weak-referenceable
        return AnalysisSession()
