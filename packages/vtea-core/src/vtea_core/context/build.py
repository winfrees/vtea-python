"""Turning a protocol's level definitions into levels.

`build_context` reads the `ContextSpec` top to bottom. Subcellular and
cellular levels are taken from what the protocol's steps already produced
(a measurement table per segmentation, the cells a `build_cells` step
composed); neighbourhood levels are built here, from the level named as
their source, measured, optionally typed, and their characteristics handed
down to every level beneath them - through the cells to the pieces of each
cell, however many links away.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from vtea_core.context.graph import MEMBER_OF, PART_OF, ContextGraph, Level
from vtea_core.context.spec import CELLULAR, NEIGHBORHOOD, SUBCELLULAR, ContextSpec, LevelSpec
from vtea_core.data.spacing import Spacing
from vtea_core.neighborhoods import (
    NEIGHBORHOOD_ID,
    NEIGHBORHOOD_TYPE,
    build_neighborhoods,
    classify_neighborhoods,
    composition_columns,
    neighborhood_features,
)


def _names(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part) for part in value]


def _with_cell_positions(table: pd.DataFrame, cells, root_table: pd.DataFrame | None) -> pd.DataFrame:
    """A cell table with `centroid-*` columns - the root object's, since a
    cell's id is its root's id - so neighbourhoods can be built from cells.
    A per-cell table namespaces them (`nuclei_1.centroid-0`); those are used
    where present, the root's measurement table otherwise."""
    if any(str(column).startswith("centroid-") for column in table.columns):
        return table
    table = table.copy()
    root = getattr(cells, "root_segmentation", "")
    namespaced = [c for c in table.columns if str(c).startswith(f"{root}.centroid-")]
    if namespaced:
        for column in namespaced:
            table[str(column).split(".", 1)[1]] = table[column]
        return table
    if root_table is not None and "object_id" in root_table.columns:
        positions = root_table.set_index("object_id")
        for column in [c for c in root_table.columns if str(c).startswith("centroid-")]:
            table[column] = table["cell_id"].map(positions[column]).to_numpy()
    return table


def _cell_membership(cells, role: str) -> pd.DataFrame:
    rows = [
        (ref.object_id, cell.cell_id) for cell in cells for ref in cell.objects(role)
    ]
    return pd.DataFrame(rows, columns=["child", "parent"])


def _check_not_circular(source: Level, columns, rank: int, level: str) -> None:
    circular = sorted(
        column for column in columns if source.reflected.get(column, -1) >= rank
    )
    if circular:
        raise ValueError(
            f"level '{level}' would be defined on {circular}, which were handed down from "
            f"level(s) at its own rank or above - a level cannot be typed on its own type"
        )


def build_context(
    spec: ContextSpec,
    *,
    measurement_tables: dict[str, pd.DataFrame] | None = None,
    cells: dict[str, Any] | None = None,
    cell_tables: dict[str, pd.DataFrame] | None = None,
    spacing: Spacing | None = None,
) -> ContextGraph:
    """The levels `spec` defines, built from a run's results.

    `measurement_tables` maps a segmentation name to its per-object table;
    `cells` maps a `build_cells` step name to its CellSet, and `cell_tables`
    the same name to its per-cell table (optional - without one, a cellular
    level has only ids and positions). A level whose source the run did not
    produce is refused by name rather than skipped.
    """
    measurement_tables = measurement_tables or {}
    cells = cells or {}
    cell_tables = cell_tables or {}
    graph = ContextGraph()
    graph.spec = spec  # display settings, for whatever draws the levels

    for level in spec:
        rank = spec.rank(level.name)
        if level.tier == SUBCELLULAR:
            _add_subcellular(graph, level, rank, measurement_tables)
        elif level.tier == CELLULAR:
            _add_cellular(graph, level, rank, cells, cell_tables, measurement_tables)
        elif level.tier == NEIGHBORHOOD:
            _add_neighborhood(graph, spec, level, rank, spacing)
    return graph


def _add_subcellular(graph, level: LevelSpec, rank: int, tables) -> None:
    table = tables.get(level.source)
    if table is None:
        raise ValueError(
            f"subcellular level '{level.name}' is made from segmentation '{level.source}', "
            f"which was not measured (measured: {sorted(tables)})"
        )
    graph.add_level(
        Level(level.name, SUBCELLULAR, table, "object_id", rank=rank, labels_key=level.source)
    )


def _add_cellular(graph, level: LevelSpec, rank: int, cells, cell_tables, tables) -> None:
    cell_set = cells.get(level.source)
    if cell_set is None:
        raise ValueError(
            f"cellular level '{level.name}' is made from '{level.source}', which is not a "
            f"build_cells result of this run (cells: {sorted(cells)})"
        )
    if not hasattr(cell_set, "__iter__"):
        raise ValueError(
            f"'{level.source}' holds its cells as a membership table (a blocked run); "
            f"contexts over those are not supported yet"
        )
    root = cell_set.root_segmentation
    table = cell_tables.get(level.source)
    if table is None:
        table = pd.DataFrame({"cell_id": [cell.cell_id for cell in cell_set]})
    table = _with_cell_positions(table, cell_set, tables.get(root))
    graph.add_level(Level(level.name, CELLULAR, table, "cell_id", rank=rank, labels_key=root))
    # Every subcellular level that is a role of these cells is a piece of them.
    roles = set(cell_set.roles())
    for other in list(graph.levels.values()):
        if other.tier == SUBCELLULAR and other.labels_key in roles:
            graph.link(other.name, level.name, _cell_membership(cell_set, other.labels_key), PART_OF)


def _add_neighborhood(graph, spec: ContextSpec, level: LevelSpec, rank: int, spacing) -> None:
    source = graph.level(level.source)
    measure = dict(level.measure)
    used = _names(measure.get("features")) + _names(measure.get("class_column"))
    _check_not_circular(source, used, rank, level.name)

    neighborhoods = build_neighborhoods(
        source.table,
        spacing=spacing,
        id_column=source.id_column,
        source=source.name,
        **level.build,
    )
    table = neighborhood_features(neighborhoods, source.table, **measure)
    if level.classify:
        settings = dict(level.classify)
        _check_not_circular(source, _names(settings.get("features")), rank, level.name)
        if len(table) > 1:
            table = classify_neighborhoods(table, **settings)

    labels_key = source.labels_key if neighborhoods.centred else ""
    new = graph.add_level(
        Level(level.name, NEIGHBORHOOD, table, NEIGHBORHOOD_ID, rank=rank, labels_key=labels_key)
    )
    new.neighborhoods = neighborhoods
    membership = neighborhoods.membership().rename(
        columns={neighborhoods.member_column: "child", NEIGHBORHOOD_ID: "parent"}
    )
    graph.link(source.name, level.name, membership, MEMBER_OF)

    reflect = list(level.reflect) or (
        [NEIGHBORHOOD_TYPE] if NEIGHBORHOOD_TYPE in table.columns else composition_columns(table)
    )
    if not reflect:
        return
    for below in graph.stack():
        if below.rank >= rank:
            continue
        try:
            graph.path(below.name, level.name)
        except ValueError:
            continue  # not beneath this level: a sibling branch
        reflected = graph.reflect_down(
            level.name, below.name, features=reflect, categorical=NEIGHBORHOOD_TYPE
        )
        graph.apply_reflection(below.name, level.name, reflected)
