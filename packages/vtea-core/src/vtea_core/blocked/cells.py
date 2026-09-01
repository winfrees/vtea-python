"""Composing cells, and measuring them, without an object graph.

`build_cells` walks the association links and builds a `Cell` per cell with
an `ObjectRef` per part. That is the readable form and the right one for the
tens of thousands of cells a field produces. At ten million it is several
gigabytes of Python objects before a single measurement has been joined to
any of them - and the measurements are the point.

So the same two operations, in the two shapes a database is for:

- **Composing cells is a recursive join.** Following links out from the
  roots until nothing new joins is exactly `WITH RECURSIVE`, and DuckDB
  spills it rather than holding it. The result is a `(cell_id, role,
  object_id)` membership table.
- **Per-cell features are a join and a group-by.** One join from the
  membership to each role's measurement table, one group-by for the roles a
  cell may have several of. That is the operation a database exists to do,
  and it reads a Parquet measurement table without loading it.

Two things had to survive the port exactly, and they are what most of the
tests here are about. `single_roles` decides whether a role's columns are
`nuclei.mean` or `lysosomes.n` plus `lysosomes.mean_mean`, and it comes from
how each link was *made* rather than from whether this field happens to hold
a cell with two of something - so the same protocol gives the same columns
on every field, which is what makes the tables poolable. And a cell missing
a part keeps a row with NaN in those columns and 0 in its count, because
"this cell has no lysosomes" is a finding and a dropped row is not.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from vtea_core.measurements.store import _quote_identifier, _quote_literal
from vtea_core.objects.association import AssociationSet, ObjectRef
from vtea_core.objects.cells import (
    AGGREGATIONS,
    DEFAULT_AGGREGATIONS,
    Cell,
    CellCollection,
    CellSet,
)

MEMBERSHIP_COLUMNS = ("cell_id", "role", "object_id")

# How each reduction is spelled in SQL. The names on the left are the ones a
# protocol records and the columns are named for, so they do not change.
SQL_AGGREGATIONS = {"sum": "sum", "mean": "avg", "median": "median"}


class CellMembership(CellCollection):
    """Which objects belong to which cell, as a table.

    One row per (cell, role, object). A cell with no parts still has its own
    row - the root - so a cell that lost every child is a cell with no
    children rather than a cell that vanished.

    `unclaimed` holds the objects that are linked to something but whose
    chain never reaches a root, kept for the reason the in-memory form keeps
    them: an analysis that quietly drops a tenth of its objects looks
    exactly like one that does not.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        root: str = "",
        single_roles: Iterable[str] = (),
        unclaimed: pd.DataFrame | None = None,
    ):
        missing = [column for column in MEMBERSHIP_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(
                f"a membership table needs {list(MEMBERSHIP_COLUMNS)}; missing {missing}"
            )
        self.frame = frame
        self.root = root
        self._single_roles = frozenset(single_roles)
        self._unclaimed = (
            unclaimed
            if unclaimed is not None
            else pd.DataFrame(columns=["segmentation", "object_id"])
        )

    # -- what any cell result can answer ----------------------------------

    @property
    def root_segmentation(self) -> str:
        return self.root

    @property
    def single_roles(self) -> frozenset[str]:
        """Roles a cell has at most one of - the root always among them,
        since it is what identifies the cell."""
        return self._single_roles | ({self.root} if self.root else frozenset())

    def roles(self) -> tuple[str, ...]:
        return tuple(sorted(self.frame["role"].unique()))

    def part_roles(self) -> tuple[str, ...]:
        return tuple(role for role in self.roles() if role != self.root)

    def summary(self) -> str:
        parts = [f"{len(self)} cells"]
        counts = self.counts_by_role()
        for role in self.part_roles():
            parts.append(f"{role}: {int(counts.get(role, 0))}/{len(self)}")
        if len(self._unclaimed):
            parts.append(f"{len(self._unclaimed)} objects in no cell")
        return ", ".join(parts)

    def __len__(self) -> int:
        return int(self.frame["cell_id"].nunique())

    # -- the table itself --------------------------------------------------

    @property
    def cell_ids(self) -> np.ndarray:
        return np.sort(self.frame["cell_id"].unique())

    @property
    def unclaimed(self) -> pd.DataFrame:
        return self._unclaimed

    def counts_by_role(self) -> dict[str, int]:
        """How many cells have at least one object of each role."""
        grouped = self.frame.groupby("role")["cell_id"].nunique()
        return {str(role): int(count) for role, count in grouped.items()}

    def objects(self, role: str) -> pd.DataFrame:
        return self.frame[self.frame["role"] == role]

    def to_cell_set(self) -> CellSet:
        """The object form, for a result small enough to want it.

        Reading a table is not how anybody inspects one cell, so this exists
        - and it is deliberately explicit, because it is exactly the
        materialization the table form was built to avoid.
        """
        cells: dict[int, dict[str, list[ObjectRef]]] = {}
        for cell_id, role, object_id in self.frame.itertuples(index=False):
            cells.setdefault(int(cell_id), {}).setdefault(str(role), []).append(
                ObjectRef(str(role), int(object_id))
            )
        built = []
        for cell_id, parts in cells.items():
            root_refs = parts.pop(self.root, None)
            root_ref = (
                root_refs[0] if root_refs else ObjectRef(self.root, cell_id)
            )
            built.append(
                Cell(
                    cell_id=cell_id,
                    root=root_ref,
                    parts={role: tuple(sorted(refs)) for role, refs in parts.items()},
                )
            )
        unclaimed = [
            ObjectRef(str(row.segmentation), int(row.object_id))
            for row in self._unclaimed.itertuples(index=False)
        ]
        return CellSet(built, unclaimed, self._single_roles)


def build_cells_blocked(
    associations: AssociationSet | pd.DataFrame,
    root_ids: Any = None,
    *,
    root: str = "",
    connection: duckdb.DuckDBPyConnection | None = None,
) -> CellMembership:
    """Follow the links out from every root object, as a recursive join.

    `associations` is an `AssociationSet` or the table `to_frame` produces -
    the table form is the one that scales, and the set is accepted because
    most callers have one.

    `root_ids` is the root segmentation's object ids (a `LabelLedger`'s
    `object_ids`, a measurement table's, or `None`). Worth passing for the
    same reason as in memory: without it a nucleus nothing was assigned to
    is not a cell at all, and dropping exactly the cells that lost a part
    biases every per-cell statistic that follows.

    A link chain that loops is refused rather than followed. The recursion
    is capped at the number of segmentations involved, which a hierarchy
    cannot exceed and a cycle always does.
    """
    if not root:
        raise ValueError(
            "build_cells_blocked needs the name of the segmentation that identifies a "
            "cell, e.g. root='nuclei_1'; in a protocol it is taken from the step the "
            "root input points at"
        )
    links = associations.to_frame() if isinstance(associations, AssociationSet) else associations
    links = _with_single_flag(links, associations)
    _refuse_cycles(links)
    _refuse_child_only_root(links, root)

    roots = pd.DataFrame({"object_id": _root_id_array(root_ids, links, root)})
    own = connection is None
    connection = duckdb.connect() if own else connection
    try:
        connection.register("links", links)
        connection.register("roots", roots)
        depth_limit = _depth_limit(links)
        membership = connection.execute(
            """
            WITH RECURSIVE member(cell_id, role, object_id, depth) AS (
                    SELECT object_id, ?, object_id, 0 FROM roots
                UNION ALL
                    SELECT m.cell_id, l.child_segmentation, l.child_id, m.depth + 1
                    FROM member m
                    JOIN links l
                      ON l.parent_segmentation = m.role AND l.parent_id = m.object_id
                    WHERE l.child_segmentation <> ? AND m.depth < ?
            )
            SELECT cell_id, role, object_id, max(depth) AS depth
            FROM member
            GROUP BY cell_id, role, object_id
            ORDER BY cell_id, role, object_id
            """,
            [root, root, depth_limit],
        ).fetch_df()
        if len(membership) and int(membership["depth"].max()) >= depth_limit:
            raise ValueError(
                f"association cycle: following the links out from '{root}' was still "
                f"finding new objects {depth_limit} levels down, which is deeper than "
                f"the {depth_limit} segmentations involved allow. A cell hierarchy "
                f"cannot loop back on itself, so this cannot be composed into cells"
            )
        connection.register("membership", membership)
        unclaimed = connection.execute(
            """
            SELECT DISTINCT l.child_segmentation AS segmentation, l.child_id AS object_id
            FROM links l
            WHERE l.child_segmentation <> ?
              AND l.parent_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM membership m
                  WHERE m.role = l.child_segmentation AND m.object_id = l.child_id
              )
            ORDER BY segmentation, object_id
            """,
            [root],
        ).fetch_df()
    finally:
        if own:
            connection.close()

    return CellMembership(
        membership[list(MEMBERSHIP_COLUMNS)],
        root=root,
        single_roles=single_roles_from_links(links),
        unclaimed=unclaimed,
    )


def single_roles_from_links(links: pd.DataFrame) -> frozenset[str]:
    """The roles a cell can only have one of, read off how each link was made.

    A derived part is one per parent by construction and a one-to-one
    assignment says so; everything else may be many. Taken from the links
    rather than from the data on purpose - a field that happens to contain
    no cell with two lysosomes must not produce a different table shape from
    one that does.
    """
    if not len(links) or "at_most_one" not in links.columns:
        return frozenset()
    # Only the rows that are links: a child left unassigned says nothing
    # about whether its role can hold several.
    linked = links[links["parent_id"].notna()]
    if not len(linked):
        return frozenset()
    grouped = linked.groupby("child_segmentation")["at_most_one"].all()
    return frozenset(str(role) for role, single in grouped.items() if bool(single))


def cell_features_blocked(
    membership: CellMembership,
    measurement_tables: dict[str, Any],
    *,
    aggregations: Iterable[str] = DEFAULT_AGGREGATIONS,
    id_column: str = "object_id",
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """One row per cell, as a join and a group-by.

    `measurement_tables` maps a role to that role's measurement table -
    either a DataFrame or the path of a Parquet file, which is what a table
    of ten million objects should be. The SQL is the same for both, so
    nothing here needs to know which it was given.

    The columns are the ones `vtea_core.objects.cells.cell_features`
    produces, named identically: `nuclei.mean` for a role a cell has one of,
    `lysosomes.n` plus `lysosomes.mean_count` for a role it may have several
    of. A cell missing a part keeps its row, with NaN in those columns and 0
    in its count.
    """
    reductions = [name for name in aggregations if name != "n"]
    unknown = set(reductions) - set(AGGREGATIONS)
    if unknown:
        raise ValueError(f"unknown aggregation(s) {sorted(unknown)}, expected {list(AGGREGATIONS)}")

    own = connection is None
    connection = duckdb.connect() if own else connection
    try:
        connection.register("membership", membership.frame)
        blocks, selects = [], []
        for position, role in enumerate(membership.roles()):
            source = measurement_tables.get(role)
            if source is None:
                continue
            table = f"t_{position}"
            _register_table(connection, table, source)
            columns = _numeric_columns(connection, table, id_column)
            if not columns:
                continue
            alias = f"r_{position}"
            single = role in membership.single_roles
            blocks.append(
                (alias, _role_query(role, table, columns, id_column, single, reductions))
            )
            selects.extend(_role_selects(alias, role, columns, single, reductions))

        sql = "WITH cells AS (SELECT DISTINCT cell_id FROM membership)"
        for alias, query in blocks:
            sql += f", {alias} AS ({query})"
        sql += "\nSELECT c.cell_id" + "".join(f",\n       {item}" for item in selects)
        sql += "\nFROM cells c"
        for alias, _query in blocks:
            sql += f"\nLEFT JOIN {alias} ON {alias}.cell_id = c.cell_id"
        sql += "\nORDER BY c.cell_id"
        return _plain_dtypes(connection.execute(sql).fetch_df())
    finally:
        if own:
            connection.close()


def _plain_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """NumPy dtypes rather than pandas' nullable ones.

    A left join gives a missing integer measurement back as `<NA>` in an
    `Int64` column, where the in-memory form gives `NaN` in a `float64` one.
    Same fact, and a column that changes dtype depending on which code path
    produced it cannot be pooled with one that did not - so the two are made
    to agree, on the answer the older path already gave.
    """
    for column in frame.columns:
        if not isinstance(frame[column].dtype, pd.api.extensions.ExtensionDtype):
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        target = "float64" if frame[column].isna().any() else "int64"
        frame[column] = frame[column].astype(target)
    return frame


# -- SQL construction -------------------------------------------------------


def _role_query(
    role: str,
    table: str,
    columns: Sequence[str],
    id_column: str,
    single: bool,
    reductions: Sequence[str],
) -> str:
    """One role's contribution: a join for a single part, a group-by for
    many."""
    member = (
        f"SELECT m.cell_id, t.* FROM membership m JOIN {table} t "
        f"ON t.{_quote_identifier(id_column)} = m.object_id "
        f"WHERE m.role = {_quote_literal(role)}"
    )
    if single:
        # First by object id where a "single" role somehow holds several,
        # which is the same row the in-memory form keeps and beats picking
        # an arbitrary one.
        picked = ", ".join(
            f"j.{_quote_identifier(column)} AS {_quote_identifier(f'{role}.{column}')}"
            for column in columns
        )
        return (
            f"SELECT j.cell_id, {picked} FROM ({member}) j "
            f"QUALIFY row_number() OVER "
            f"(PARTITION BY j.cell_id ORDER BY j.{_quote_identifier(id_column)}) = 1"
        )

    aggregated = [f"count(*) AS {_quote_identifier(f'{role}.n')}"]
    for reduction in reductions:
        function = SQL_AGGREGATIONS[reduction]
        aggregated.extend(
            f"{function}(j.{_quote_identifier(column)}) AS "
            f"{_quote_identifier(f'{role}.{reduction}_{column}')}"
            for column in columns
        )
    return (
        f"SELECT j.cell_id, {', '.join(aggregated)} FROM ({member}) j GROUP BY j.cell_id"
    )


def _role_selects(
    alias: str, role: str, columns: Sequence[str], single: bool, reductions: Sequence[str]
) -> list[str]:
    if single:
        return [
            f"{alias}.{_quote_identifier(f'{role}.{column}')}" for column in columns
        ]
    # A cell with none of this role gets 0 rather than NULL: "no lysosomes"
    # is a count, and a missing count reads as "not measured".
    count_column = _quote_identifier(f"{role}.n")
    selects = [f"COALESCE({alias}.{count_column}, 0) AS {count_column}"]
    for reduction in reductions:
        selects.extend(
            f"{alias}.{_quote_identifier(f'{role}.{reduction}_{column}')}"
            for column in columns
        )
    return selects


def _register_table(connection, name: str, source: Any) -> None:
    """A DataFrame held in memory, or a Parquet file read as the query needs
    it - which is what a measurement table larger than memory has to be."""
    if isinstance(source, pd.DataFrame):
        connection.register(name, source)
        return
    location = _quote_literal(str(Path(os.fspath(source)).resolve()))
    connection.execute(
        f"CREATE OR REPLACE VIEW {_quote_identifier(name)} AS "
        f"SELECT * FROM read_parquet({location})"
    )


def _numeric_columns(connection, table: str, id_column: str) -> list[str]:
    """The columns worth aggregating, from the table's own schema.

    Read from the database rather than from a DataFrame's dtypes, so a
    Parquet table answers without being loaded.
    """
    described = connection.execute(f"DESCRIBE SELECT * FROM {_quote_identifier(table)}").fetch_df()
    numeric = ("TINY", "SMALL", "INT", "BIGINT", "HUGEINT", "FLOAT", "DOUBLE", "DECIMAL")
    return [
        str(name)
        for name, kind in zip(described["column_name"], described["column_type"])
        if name != id_column and str(kind).upper().startswith(numeric)
    ]


# -- input checks -----------------------------------------------------------


def _with_single_flag(links: pd.DataFrame, associations: Any) -> pd.DataFrame:
    """Make sure the table carries how each link was made.

    `AssociationSet.to_frame` puts it there; a table from somewhere else may
    not, and guessing would silently change the shape of the feature table.
    """
    if "at_most_one" in links.columns:
        return links
    links = links.copy()
    links["at_most_one"] = False
    return links


def _refuse_cycles(links: pd.DataFrame) -> None:
    """Refuse a link chain that loops, before anything tries to follow it.

    A child has at most one parent, so the links are a functional graph and
    a chain either reaches a root or comes back to itself. Following every
    child's parent pointer `depth_limit` times at once settles which: a
    hierarchy of these segmentations cannot be deeper than that, so anything
    still climbing is in a loop.

    Vectorized rather than a traversal, and keyed on integers rather than on
    tuples, because this runs on the same ten million links everything else
    here is shaped around.
    """
    linked = links[links["parent_id"].notna()]
    if not len(linked):
        return

    names = pd.Index(
        sorted(set(linked["child_segmentation"]) | set(linked["parent_segmentation"]))
    )
    ids = np.concatenate(
        [
            np.asarray(linked["child_id"], dtype=np.int64),
            np.asarray(linked["parent_id"], dtype=np.int64),
        ]
    )
    span = int(ids.max()) + 1 if len(ids) else 1
    child_key = names.get_indexer(linked["child_segmentation"]) * span + np.asarray(
        linked["child_id"], dtype=np.int64
    )
    parent_key = names.get_indexer(linked["parent_segmentation"]) * span + np.asarray(
        linked["parent_id"], dtype=np.int64
    )

    order = np.argsort(child_key, kind="stable")
    sorted_keys = child_key[order]
    position = np.searchsorted(sorted_keys, parent_key)
    clipped = np.clip(position, 0, len(sorted_keys) - 1)
    # A parent that is nobody's child is a root: the chain ends there.
    found = sorted_keys[clipped] == parent_key
    parent_of = np.where(found, order[clipped], -1)

    climbing = parent_of.copy()
    for _ in range(_depth_limit(links)):
        alive = climbing >= 0
        if not alive.any():
            return
        climbing = np.where(alive, parent_of[np.where(alive, climbing, 0)], -1)

    caught = linked.iloc[int(np.flatnonzero(climbing >= 0)[0])]
    raise ValueError(
        f"association cycle through {caught['child_segmentation']}#{caught['child_id']}: "
        f"a cell hierarchy cannot loop back on itself, so this cannot be composed "
        f"into cells"
    )


def _refuse_child_only_root(links: pd.DataFrame, root: str) -> None:
    if not len(links):
        return
    is_a_parent = bool((links["parent_segmentation"] == root).any())
    is_a_child = bool((links["child_segmentation"] == root).any())
    if is_a_child and not is_a_parent:
        parents = sorted(set(links["parent_segmentation"].dropna()))
        raise ValueError(
            f"'{root}' is only ever a child in these associations, so following the links "
            f"out from it reaches nothing. A cell's root has to be the parent side: "
            f"associate the other segmentation to '{root}', or build cells rooted on "
            f"'{parents[0] if parents else '?'}' instead."
        )


def _root_id_array(root_ids: Any, links: pd.DataFrame, root: str) -> np.ndarray:
    """Every object that is a cell: the ones given, plus every root object
    the links mention on either side."""
    ids: list[np.ndarray] = []
    if root_ids is not None:
        given = np.asarray(root_ids)
        if given.ndim > 1:
            # A label image, which only a small run would pass.
            given = np.unique(given)
        ids.append(given[given != 0].astype(np.int64))
    if len(links):
        as_parent = links.loc[links["parent_segmentation"] == root, "parent_id"].dropna()
        as_child = links.loc[links["child_segmentation"] == root, "child_id"].dropna()
        ids.append(np.asarray(as_parent, dtype=np.int64))
        ids.append(np.asarray(as_child, dtype=np.int64))
    if not ids:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(ids))


def _depth_limit(links: pd.DataFrame) -> int:
    """How deep a hierarchy of these segmentations could legitimately go.

    Each level introduces another segmentation, so a chain cannot be longer
    than there are segmentations; anything deeper is a loop. Cheap, and it
    is the only thing standing between a cycle and a query that never
    returns.
    """
    if not len(links):
        return 1
    names = set(links["child_segmentation"].dropna()) | set(
        links["parent_segmentation"].dropna()
    )
    return max(len(names), 1) + 1
