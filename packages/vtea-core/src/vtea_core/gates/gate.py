"""Named, stateful gates over a measurement table - the model behind the
napari "Object Explorer" widget's plot + gate table (Phase 4).

Ports the *working* parts of vteaexploration's gating UI model
(PolygonGate's name/axes/color/counts, plus the gate list vtea's
"Gate Management" TableWindow shows) onto vtea_core.gates.polygon_gate.
Two things the Java model didn't have are added deliberately, not as
scope creep:

- Real hierarchy (`Gate.parent_id`, a subgate's membership is intersected
  with its parent's). Java only faked "subgating" by opening a whole new
  MicroExplorer window over a pre-filtered dataset - GateManager.java, the
  class meant to show gate hierarchy, is instantiated but never wired up
  or shown anywhere in the codebase.
- One generic set of gate-combination ops. Java's GateMath/AbstractGateMath
  plugin framework only ever got one working implementer (AND); GateSet's
  `.mask()` already gives boolean arrays that plain &/|/~ combine directly,
  same as vtea_core.gates.polygon.

RectangleGate/FreeFormGate aren't ported as separate classes: Java's own
versions are unused stubs whose methods all `throw
UnsupportedOperationException` - every real gate is a polygon (a rectangle
is just a 4-vertex one), matching vtea_core.gates.rectangle_gate's
already-array-based equivalent.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vtea_core.gates.polygon import polygon_gate


@dataclass
class Gate:
    """One polygon gate: a closed shape over two named measurement columns."""

    name: str
    x_axis: str
    y_axis: str
    vertices: np.ndarray
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    color: str = "#1f77b4"
    visible: bool = True


class GateSet:
    """An ordered collection of named Gates, with membership computed against
    a measurement DataFrame and optional parent-gate chaining."""

    def __init__(self) -> None:
        self._gates: dict[str, Gate] = {}
        self._order: list[str] = []

    def add(self, gate: Gate) -> Gate:
        if gate.parent_id is not None and gate.parent_id not in self._gates:
            raise KeyError(f"unknown parent gate id {gate.parent_id!r}")
        self._gates[gate.id] = gate
        self._order.append(gate.id)
        return gate

    def remove(self, gate_id: str) -> None:
        for child in self.children(gate_id):
            self.remove(child.id)
        self._gates.pop(gate_id)
        self._order.remove(gate_id)

    def get(self, gate_id: str) -> Gate:
        return self._gates[gate_id]

    def children(self, gate_id: str) -> list[Gate]:
        return [gate for gate in self._gates.values() if gate.parent_id == gate_id]

    def __iter__(self):
        return (self._gates[gate_id] for gate_id in self._order)

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, gate_id: str) -> bool:
        return gate_id in self._gates

    def mask(self, gate_id: str, frame: pd.DataFrame) -> np.ndarray:
        """Boolean membership mask for one gate against `frame`, restricted
        to its parent's membership if it has one (real hierarchical gating)."""
        gate = self._gates[gate_id]
        x = frame[gate.x_axis].to_numpy()
        y = frame[gate.y_axis].to_numpy()
        mask = polygon_gate(x, y, gate.vertices)
        if gate.parent_id is not None:
            mask = mask & self.mask(gate.parent_id, frame)
        return mask

    def summary(self, gate_id: str, frame: pd.DataFrame) -> dict:
        """{n_gated, n_total, percent} - the counts vtea's gate table shows per row."""
        mask = self.mask(gate_id, frame)
        n_total = len(frame)
        n_gated = int(mask.sum())
        percent = (100.0 * n_gated / n_total) if n_total else 0.0
        return {"n_gated": n_gated, "n_total": n_total, "percent": percent}

    def statistics(
        self, gate_id: str, frame: pd.DataFrame, columns: Iterable[str] | None = None
    ) -> dict:
        """summary(), plus the mean of each named column over the gated
        objects only - "how many cells did I select, and how bright are
        they?", the question a gate is drawn to answer.

        `columns` defaults to the gate's own two axes, which is what the gate
        manager shows for the plot currently on screen. A column missing from
        `frame`, or an empty gate, gives a NaN mean rather than raising: an
        empty selection is a normal thing to draw, not an error.
        """
        stats = self.summary(gate_id, frame)
        gate = self._gates[gate_id]
        if columns is None:
            columns = (gate.x_axis, gate.y_axis)
        mask = self.mask(gate_id, frame)
        means: dict[str, float] = {}
        for column in columns:
            if column not in frame.columns or not mask.any():
                means[column] = float("nan")
                continue
            means[column] = float(frame.loc[mask, column].mean())
        stats["means"] = means
        return stats
