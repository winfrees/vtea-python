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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, Signal
from vtea_core.data import Spacing
from vtea_core.gates import GateSet
from vtea_core.measurements import FeatureCatalog
from vtea_core.objects import AssociationSet, CellCollection, ObjectRef
from vtea_core.workflow import Pipeline

OBJECT_TABLE = "Objects"


@dataclass
class TableView:
    """One table the explorer can plot, and what its rows are.

    A per-object table and a per-cell table are both rows of features, but
    they are not interchangeable: their rows are different things, their id
    column has a different name, and the label image a gate highlights is a
    different image. Bundling the three together is what lets the explorer
    switch between them without any of its own code knowing which is which.

    `labels_key` is a context key rather than an array, so the image is
    looked up when it is needed and a re-run does not leave a stale copy
    behind.
    """

    frame: pd.DataFrame
    id_column: str = "object_id"
    labels_key: str = "labels"
    # What a row is, for the status line: "objects", "cells".
    noun: str = "objects"
    gate_set: GateSet = field(default_factory=GateSet)


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
        # Every table the explorer can plot, keyed by name. A protocol that
        # builds cells has two - its objects and its cells - and a gate drawn
        # on one means nothing on the other, so each carries its own gates.
        self.tables: dict[str, TableView] = {}
        self.active_table: str = OBJECT_TABLE
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
        # Physical voxel size. Read from the image where the file recorded
        # it, otherwise asked for: every distance and thickness downstream
        # is wrong without it, and wrong in a way that looks plausible.
        self.spacing: Spacing | None = None
        # How each object a tile boundary cut was put back together, when
        # the run was a blocked one - see vtea_core.blocked.reconcile. Kept
        # on the session rather than in the builder so the explorer can
        # review a seam without the builder being open.
        self.ledger = None
        # Links a person reassigned or broke by hand, child -> parent (or
        # None for "no parent"). Held here rather than only on the
        # AssociationSet a run produced, because re-running the association
        # step replaces that set - and a correction that a re-run silently
        # discards is barely a correction at all.
        self.manual_links: dict[ObjectRef, ObjectRef | None] = {}
        self._table: pd.DataFrame | None = None
        # Gates drawn before any table was published - the explorer can be
        # driven directly, without the builder.
        self._loose_gates = GateSet()

    # -- data -------------------------------------------------------------

    def set_context(
        self,
        context: dict[str, Any],
        table: pd.DataFrame | None = None,
        tables: dict[str, TableView] | None = None,
    ) -> None:
        """Publish a new run context, the flat per-object feature table
        derived from it, and any further tables the run produced.

        The tables are passed in rather than recomputed here because only the
        builder knows the step graph that names their columns. Gates survive
        a re-run: a table that was already here keeps the gates drawn on it,
        since they are drawn on features that still exist.
        """
        self.context = context
        self._table = table
        published = dict(tables or {})
        if table is not None:
            published.setdefault(OBJECT_TABLE, TableView(table))
        for name, view in published.items():
            existing = self.tables.get(name)
            if existing is not None:
                view.gate_set = existing.gate_set
        self.tables = published
        if self.active_table not in self.tables:
            self.active_table = OBJECT_TABLE if OBJECT_TABLE in self.tables else (
                next(iter(self.tables), OBJECT_TABLE)
            )
        self.data_changed.emit()

    def table_names(self) -> list[str]:
        """The tables on offer, the per-object one first."""
        names = list(self.tables)
        if OBJECT_TABLE in names:
            names.remove(OBJECT_TABLE)
            names.insert(0, OBJECT_TABLE)
        return names

    def table_view(self, name: str | None = None) -> TableView | None:
        return self.tables.get(self.active_table if name is None else name)

    def set_active_table(self, name: str) -> None:
        """Switch which table the explorer plots. A no-op for a name that
        isn't on offer, so a remembered choice from a previous run cannot
        leave the pane pointing at nothing."""
        if name in self.tables and name != self.active_table:
            self.active_table = name
            self.data_changed.emit()

    def results_table(self, name: str | None = None) -> pd.DataFrame | None:
        view = self.table_view(name)
        frame = self._table if view is None else view.frame
        if frame is None or frame.empty:
            return None
        return frame

    def id_column(self, name: str | None = None) -> str:
        view = self.table_view(name)
        return "object_id" if view is None else view.id_column

    def row_noun(self, name: str | None = None) -> str:
        view = self.table_view(name)
        return "objects" if view is None else view.noun

    def associations(self) -> dict[str, AssociationSet]:
        """Every association result in the run context, by the step that
        produced it."""
        return {
            key: value
            for key, value in self.context.items()
            if isinstance(value, AssociationSet) and key != "associations"
        }

    def cell_sets(self) -> dict[str, CellCollection]:
        """The cell results this run produced, by step name.

        `CellCollection` rather than `CellSet`: a blocked run composes its
        cells as a membership table rather than as an object graph, and
        everything here only asks a cell result how many cells there are and
        which segmentation identifies them - see vtea_core.objects.cells.
        """
        return {
            key: value
            for key, value in self.context.items()
            if isinstance(value, CellCollection) and key != "cells"
        }

    def record_manual_link(self, child: ObjectRef, parent: ObjectRef | None) -> None:
        """Remember a hand-made decision so a re-run does not undo it."""
        self.manual_links[child] = parent

    def apply_manual_links(self, associations: AssociationSet) -> int:
        """Re-apply the hand-made decisions that concern this set.

        Called after an association step runs, so re-running it with
        different parameters corrects the automated answers while leaving the
        ones a person has already settled. Only children this set actually
        contains are touched - the edits for a different segmentation belong
        to a different step.
        """
        applied = 0
        known = {link.child for link in associations} | set(associations.unassigned)
        for child, parent in self.manual_links.items():
            if child not in known:
                continue
            if parent is None:
                associations.unassign(child)
            else:
                associations.set_parent(child, parent)
            applied += 1
        return applied

    def labels(self, name: str | None = None) -> np.ndarray | None:
        """The label image this table's rows are objects of, for highlighting
        a gate's members back on the viewer. A per-cell table points at the
        segmentation its cells are rooted on, so a gate on cells lights up
        the nuclei that identify them."""
        view = self.table_view(name)
        labels = self.context.get("labels" if view is None else view.labels_key)
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

    def set_ledger(self, ledger) -> None:
        """Record how a blocked run reconciled its objects.

        `None` for an in-memory run, which has no seams to account for -
        which is also what tells a review pane there is nothing to review.
        """
        self.ledger = ledger
        self.data_changed.emit()

    def set_spacing(self, spacing: Spacing | None) -> None:
        self.spacing = spacing

    # -- gates ------------------------------------------------------------

    @property
    def gate_set(self) -> GateSet:
        """The gates on the active table. Each table keeps its own: a polygon
        drawn over cell features selects nothing on a per-object table, and
        silently sharing them between the two would show a gate that cannot
        be what it claims."""
        view = self.table_view()
        return self._loose_gates if view is None else view.gate_set

    @gate_set.setter
    def gate_set(self, gate_set: GateSet) -> None:
        view = self.table_view()
        if view is None:
            self._loose_gates = gate_set
        else:
            view.gate_set = gate_set

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
