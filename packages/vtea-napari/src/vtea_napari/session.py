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
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, Signal
from vtea_core.data import Spacing
from vtea_core.gates import GateSet
from vtea_core.measurements import FeatureCatalog
from vtea_core.neighborhoods import NEIGHBORHOOD_ID, NeighborhoodSet
from vtea_core.objects import AssociationSet, CellCollection, ObjectRef
from vtea_core.workflow import Pipeline

OBJECT_TABLE = "Objects"

# What a feature catalog entry's `measurement` says when the column holds a
# category rather than a quantity - the columns a discrete LUT is for.
CATEGORICAL_MEASUREMENTS = frozenset(
    {"cluster assignment", "class", "label set", "neighborhood type"}
)

# The prefix a gate's membership column gets when a gate is handed to a
# protocol step - see gate_columns.
GATE_COLUMN_PREFIX = "gate_"


def gate_column_name(name: str) -> str:
    """A gate's name as a column a class definition can refer to."""
    cleaned = "".join(
        character if (character.isalnum() or character == "_") else "_" for character in str(name)
    )
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return f"{GATE_COLUMN_PREFIX}{cleaned.strip('_') or 'gate'}"


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


@dataclass
class NeighborhoodResult:
    """A neighbourhood analysis made in the Neighborhoods pane.

    Two levels of objects and the link between them: the neighbourhoods and
    their table (rows are neighbourhoods), which table their members are
    rows of (`source_table`), and `reflected` - what each member took on from
    the neighbourhoods it belongs to, keyed by the member's id and merged
    onto `source_table` whenever that table is published.

    `labels_key` is the label image a neighbourhood row highlights. For a
    neighbourhood centred on an object, its id is that object's id, so it is
    the source table's own label image; a grid neighbourhood has no object to
    light up and leaves it empty.
    """

    neighborhoods: NeighborhoodSet
    table: pd.DataFrame
    source_table: str = OBJECT_TABLE
    labels_key: str = ""
    reflected: pd.DataFrame | None = None

    @property
    def member_id_column(self) -> str:
        return self.neighborhoods.id_column

    def view(self) -> TableView:
        return TableView(
            frame=self.table,
            id_column=NEIGHBORHOOD_ID,
            labels_key=self.labels_key,
            noun="neighborhoods",
        )


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
    # A neighbourhood analysis was added, replaced or removed.
    neighborhoods_changed = Signal()

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
        # Image gates: which region of a napari Labels layer each object
        # falls in, keyed by the column it is published under and indexed by
        # object id. Held here rather than written into the table because
        # the builder rebuilds the table on every run, and a region somebody
        # painted has to survive that - it is an input to the analysis, not
        # an output of it.
        self.image_gates: dict[str, pd.Series] = {}
        # Links a person reassigned or broke by hand, child -> parent (or
        # None for "no parent"). Held here rather than only on the
        # AssociationSet a run produced, because re-running the association
        # step replaces that set - and a correction that a re-run silently
        # discards is barely a correction at all.
        self.manual_links: dict[ObjectRef, ObjectRef | None] = {}
        self._table: pd.DataFrame | None = None
        # What the builder last published, before anything the session adds
        # to it (image gates, reflected neighbourhood columns) - kept so those
        # can be re-applied when they change without re-running the builder.
        self._source_table: pd.DataFrame | None = None
        self._source_views: dict[str, TableView] = {}
        # Neighbourhood analyses made in the Neighborhoods pane, by name. Kept
        # here for the same reason image gates are: the builder rebuilds its
        # tables on every run, and an analysis made on them has to survive
        # that - its neighbourhood table is published beside them, and what
        # it reflected onto the objects is merged back onto theirs by id.
        self.neighborhood_results: dict[str, NeighborhoodResult] = {}
        # Gates drawn before any table was published - the explorer can be
        # driven directly, without the builder.
        self._loose_gates = GateSet()
        # Gates read from a saved protocol, keyed by table name, waiting for
        # a run to publish the table they were drawn on.
        self._pending_gates: dict[str, GateSet] = {}

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
        self._source_table = table
        self._source_views = dict(tables or {})
        self._assemble_tables()
        self.data_changed.emit()

    def _assemble_tables(self) -> None:
        """Publish the builder's tables with everything the session adds to
        them: image-gate columns, the neighbourhood tables made in the
        Neighborhoods pane, and the columns those reflected onto their
        members."""
        table = self._decorate(OBJECT_TABLE, "object_id", self._source_table)
        self._table = table
        published = {
            name: replace(view, frame=self._decorate(name, view.id_column, view.frame))
            for name, view in self._source_views.items()
        }
        if table is not None:
            published.setdefault(OBJECT_TABLE, TableView(table))
        for name, result in self.neighborhood_results.items():
            published.setdefault(name, result.view())
        for name, view in published.items():
            existing = self.tables.get(name)
            if existing is not None:
                view.gate_set = existing.gate_set
            elif name in self._pending_gates:
                view.gate_set = self._pending_gates.pop(name)
        self.tables = published
        if self.active_table not in self.tables:
            self.active_table = OBJECT_TABLE if OBJECT_TABLE in self.tables else (
                next(iter(self.tables), OBJECT_TABLE)
            )

    def _decorate(self, name: str, id_column: str, frame: pd.DataFrame | None):
        if frame is None:
            return None
        if id_column == "object_id":
            frame = self.apply_image_gates(frame)
        return self.apply_reflections(frame, name)

    # -- neighbourhoods ---------------------------------------------------

    def source_frame(self, name: str) -> pd.DataFrame | None:
        """A table as the builder published it, before the session's own
        additions - what a neighbourhood analysis should be built from, so
        a second analysis is not built on the first one's reflected
        columns."""
        if name == OBJECT_TABLE and OBJECT_TABLE not in self._source_views:
            return self._source_table
        view = self._source_views.get(name)
        return None if view is None else view.frame

    def apply_reflections(self, frame: pd.DataFrame | None, table: str = OBJECT_TABLE):
        """Merge what neighbourhoods reflected onto `table`'s rows, by id.

        By id rather than by row, like an image gate, so a re-run that adds
        or drops objects leaves the rest with their values and the new ones
        with none - rather than every value shifting a row.
        """
        if frame is None:
            return None
        results = [
            result
            for result in self.neighborhood_results.values()
            if result.source_table == table
            and result.reflected is not None
            and result.member_id_column in frame.columns
        ]
        if not results:
            return frame
        frame = frame.copy()
        for result in results:
            reflected = result.reflected.set_index(result.member_id_column)
            ids = frame[result.member_id_column]
            for column in reflected.columns:
                frame[column] = ids.map(reflected[column]).to_numpy()
        return frame

    def set_neighborhood_result(self, name: str, result: NeighborhoodResult) -> None:
        """Add or replace a neighbourhood analysis and republish the tables."""
        self.neighborhood_results[name] = result
        self._assemble_tables()
        self.data_changed.emit()
        self.neighborhoods_changed.emit()

    def remove_neighborhood_result(self, name: str) -> None:
        if self.neighborhood_results.pop(name, None) is None:
            return
        self._assemble_tables()
        self.data_changed.emit()
        self.neighborhoods_changed.emit()

    # -- image gates ------------------------------------------------------

    def set_image_gate(self, column: str, object_ids, values) -> None:
        """Record which region of a painted layer each object is in.

        Kept as a Series indexed by object id rather than a bare array, so
        it survives a re-run that adds or drops objects: the ones it still
        knows about keep their region, and the rest come back as "in none"
        instead of silently shifting by a row.
        """
        self.image_gates[column] = pd.Series(
            np.asarray(values), index=np.asarray(object_ids), name=column
        )

    def clear_image_gate(self, column: str | None = None) -> None:
        if column is None:
            self.image_gates.clear()
        else:
            self.image_gates.pop(column, None)

    def apply_image_gates(self, frame: pd.DataFrame | None) -> pd.DataFrame | None:
        """Put the image-gate columns back onto a freshly built table."""
        if frame is None or not self.image_gates or "object_id" not in frame.columns:
            return frame
        frame = frame.copy()
        ids = frame["object_id"]
        for column, values in self.image_gates.items():
            frame[column] = ids.map(values).fillna(0).to_numpy()
        return frame

    def gate_columns(self, frame: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
        """Each gate's membership as a named boolean column.

        This is how a gate reaches a *protocol* step: a class definition
        says `gate_bright AND NOT roi_tubule`, and the class step is handed
        a table with those columns in it. The name is sanitised so it can be
        typed into a definition without backticks.
        """
        frame = self.results_table() if frame is None else frame
        if frame is None:
            return {}
        gate_set = self.gate_set
        columns: dict[str, np.ndarray] = {}
        for gate in gate_set:
            if gate.x_axis not in frame.columns or gate.y_axis not in frame.columns:
                continue
            columns[gate_column_name(gate.name)] = gate_set.mask(gate.id, frame)
        return columns

    def categorical_columns(self, frame: pd.DataFrame | None = None) -> set[str]:
        """Columns that hold categories rather than measurements.

        Read from the feature catalog - which records the step that produced
        each column - rather than guessed from the values, so a clustering's
        output is coloured as clusters even when its ids happen to look like
        a measurement.
        """
        names = {
            descriptor.name
            for descriptor in self.feature_catalog
            if descriptor.measurement in CATEGORICAL_MEASUREMENTS
        }
        names |= set(self.image_gates)
        frame = self.results_table() if frame is None else frame
        if frame is not None:
            names |= {
                str(column)
                for column in frame.columns
                if pd.api.types.is_bool_dtype(frame[column])
            }
        return names

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

    def gates_by_table(self) -> dict[str, GateSet]:
        """Every non-empty gate set, keyed by the table it was drawn on -
        what a saved protocol carries. Gates loaded from a protocol but not
        yet placed (no run since) are included, so saving straight after
        opening loses nothing."""
        gates = {name: gate_set for name, gate_set in self._pending_gates.items() if len(gate_set)}
        for name, view in self.tables.items():
            if len(view.gate_set):
                gates[name] = view.gate_set
        if not self.tables and len(self._loose_gates):
            gates.setdefault(OBJECT_TABLE, self._loose_gates)
        return gates

    def restore_gates(self, gates: dict[str, GateSet]) -> None:
        """Put gates read from a protocol back on their tables - now, for a
        table that exists, or when a run first publishes it."""
        self._pending_gates = {}
        for name, gate_set in gates.items():
            view = self.tables.get(name)
            if view is None:
                self._pending_gates[name] = gate_set
            else:
                view.gate_set = gate_set
        self.gates_changed.emit()

    def clear_results(self) -> None:
        """Forget the last run: its context, tables, catalog and ledger.

        For opening a protocol, whose steps the old results no longer
        describe. Painted image gates and hand-corrected links stay - they
        are inputs somebody made, not outputs of the run.
        """
        self.context = {}
        self._table = None
        self._source_table = None
        self._source_views = {}
        # Made from the old results, so they describe tables that are gone.
        self.neighborhood_results = {}
        self.tables = {}
        self._loose_gates = GateSet()
        self.active_table = OBJECT_TABLE
        self.feature_catalog = FeatureCatalog()
        self.ledger = None
        self.data_changed.emit()
        self.neighborhoods_changed.emit()

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
