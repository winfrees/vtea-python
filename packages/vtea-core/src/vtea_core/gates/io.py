"""Reading and writing a GateSet as JSON.

Gates are the part of an analysis that can't be recomputed: a threshold or a
cluster count is a number you can write down, but a polygon someone drew
around a population is a judgement call. Saving them as plain, versioned
JSON - rather than a pickle or a database - is what lets a gate set be
re-opened later, diffed in version control, and deposited alongside a
figure, which is what the FAIR principles ask of the data behind a
published plot.

Gate ids are preserved on round-trip, so `parent_id` keeps pointing at the
right gate and a saved hierarchy comes back as a hierarchy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vtea_core.gates.gate import Gate, GateSet

# Bumped only for a breaking layout change; a reader checks it so a file
# from a future version fails clearly instead of half-loading.
GATES_FORMAT_VERSION = 1


def gate_to_dict(gate: Gate) -> dict[str, Any]:
    return {
        "id": gate.id,
        "name": gate.name,
        "x_axis": gate.x_axis,
        "y_axis": gate.y_axis,
        "vertices": np.asarray(gate.vertices, dtype=float).tolist(),
        "parent_id": gate.parent_id,
        "color": gate.color,
        "visible": bool(gate.visible),
    }


def gate_from_dict(data: dict[str, Any]) -> Gate:
    gate = Gate(
        name=data["name"],
        x_axis=data["x_axis"],
        y_axis=data["y_axis"],
        vertices=np.asarray(data["vertices"], dtype=float),
        parent_id=data.get("parent_id"),
        color=data.get("color", "#1f77b4"),
        visible=bool(data.get("visible", True)),
    )
    # Assigned after construction so the saved id survives rather than the
    # default factory's fresh uuid - parent_id references depend on it.
    if data.get("id"):
        gate.id = data["id"]
    return gate


def gate_set_to_dict(gate_set: GateSet) -> dict[str, Any]:
    return {
        "vtea_gates_version": GATES_FORMAT_VERSION,
        "gates": [gate_to_dict(gate) for gate in gate_set],
    }


def gate_set_from_dict(data: dict[str, Any]) -> GateSet:
    version = data.get("vtea_gates_version")
    if version is not None and version > GATES_FORMAT_VERSION:
        raise ValueError(
            f"gate file version {version} is newer than this VTEA understands "
            f"({GATES_FORMAT_VERSION}); upgrade vtea-core to open it"
        )
    gate_set = GateSet()
    # Parents before children: GateSet.add rejects an unknown parent_id, and
    # a saved file need not list them in dependency order.
    pending = list(data.get("gates", []))
    while pending:
        added = False
        for entry in list(pending):
            parent_id = entry.get("parent_id")
            if parent_id is not None and parent_id not in gate_set:
                continue
            gate_set.add(gate_from_dict(entry))
            pending.remove(entry)
            added = True
        if not added:
            names = [entry.get("name", "?") for entry in pending]
            raise ValueError(f"gate file has gates with missing or circular parents: {names}")
    return gate_set


def save_gates(gate_set: GateSet, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(gate_set_to_dict(gate_set), indent=2), encoding="utf-8")
    return path


def load_gates(path: str | Path) -> GateSet:
    return gate_set_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
