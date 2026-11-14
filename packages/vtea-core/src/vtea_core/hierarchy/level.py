"""One level of a spatial hierarchy: objects, and what they are made of.

Level 0 is the segmentation: cells or nuclei, with masks. Every level above
is made of the objects of the level below it - its *parent* - and is the
same kind of thing: a table with one row per object (`centroid-*` and
features), and a membership saying which parent objects each one contains.

Membership is a sparse (n objects x n parent objects) 0/1 matrix in *row*
order: row i of this level's table contains row j of the parent's table when
`membership[i, j]` is 1. Rows rather than ids, so that walking down several
levels is a product of matrices (see `Hierarchy.expand`), whatever the ids
are. Many-to-many: windows overlap, so a parent object can be in any number
of them; regions do not, but are held the same way.

Three kinds of level:

| kind | made of | overlap | `seeds` |
| --- | --- | --- | --- |
| `objects` | nothing: the segmentation | - | - |
| `windows` | a window around each seed object of the parent | yes | parent row of each window's seed |
| `regions` | contiguous parent objects of one type, merged | no | - |
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

OBJECTS = "objects"
WINDOWS = "windows"
REGIONS = "regions"
KINDS = (OBJECTS, WINDOWS, REGIONS)

# The id column of every level above 0 (level 0 keeps its table's own).
LEVEL_ID = "object_id"


@dataclass
class ObjectLevel:
    """One level: its table, its membership to its parent, and how it was made.

    `table` has one row per object, an id column (`id_column`), `centroid-*`
    columns in voxel coordinates of the image (so every level can be drawn
    on it) and features. `membership` is None for level 0. `seeds` is, for
    a windows level, the parent row each window is centred on. `provenance`
    records the builder and every parameter, and the fingerprint of the
    parent it was built from (see `Hierarchy.is_stale`).
    """

    name: str
    kind: str
    table: pd.DataFrame
    id_column: str = LEVEL_ID
    parent: str | None = None
    depth: int = 0
    membership: sparse.csr_matrix | None = None
    seeds: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown level kind {self.kind!r}, expected one of {KINDS}")
        if self.id_column not in self.table.columns:
            raise ValueError(f"level {self.name!r}: its table has no id column {self.id_column!r}")
        self.table = self.table.reset_index(drop=True)
        if self.kind == OBJECTS:
            if self.parent is not None or self.membership is not None:
                raise ValueError("a level of segmented objects has no parent and no members")
        else:
            if self.parent is None or self.membership is None:
                raise ValueError(f"a {self.kind} level needs a parent and a membership")
            self.membership = sparse.csr_matrix(self.membership, dtype=np.int8)
            if self.membership.shape[0] != len(self.table):
                raise ValueError(
                    f"level {self.name!r}: membership has {self.membership.shape[0]} rows "
                    f"for {len(self.table)} objects"
                )
        if self.kind == WINDOWS:
            if self.seeds is None or len(self.seeds) != len(self.table):
                raise ValueError("a windows level needs one seed (parent row) per window")
            self.seeds = np.asarray(self.seeds, dtype=np.int64)
        if not self.table[self.id_column].is_unique:
            raise ValueError(f"level {self.name!r}: object ids must be unique")

    def __len__(self) -> int:
        return len(self.table)

    @property
    def ids(self) -> np.ndarray:
        return self.table[self.id_column].to_numpy()

    def rows_of(self, ids) -> np.ndarray:
        """Table rows of these object ids (ValueError for an unknown id)."""
        lookup = pd.Series(np.arange(len(self.table)), index=self.ids)
        ids = np.atleast_1d(np.asarray(ids))
        missing = ~pd.Index(ids).isin(lookup.index)
        if missing.any():
            raise ValueError(f"level {self.name!r} has no object(s) {list(ids[missing][:5])}")
        return lookup.loc[ids].to_numpy(dtype=np.int64)

    def selection(self, which) -> np.ndarray:
        """A boolean row mask from `which`: None (every object), a boolean
        array or Series of one value per row, a column name of booleans, or
        object ids."""
        n = len(self.table)
        if which is None:
            return np.ones(n, dtype=bool)
        if isinstance(which, str):
            if which not in self.table.columns:
                raise ValueError(f"level {self.name!r} has no column {which!r}")
            which = self.table[which]
        values = np.asarray(which)
        if values.dtype == bool:
            if len(values) != n:
                raise ValueError(f"a selection of {len(values)} values for {n} objects")
            return values.copy()
        mask = np.zeros(n, dtype=bool)
        mask[self.rows_of(values)] = True
        return mask

    def sizes(self) -> np.ndarray:
        """How many parent objects each object contains."""
        if self.membership is None:
            return np.ones(len(self), dtype=np.int64)
        return np.asarray(self.membership.sum(axis=1)).ravel().astype(np.int64)

    def memberships(self) -> np.ndarray:
        """For each *parent* object, how many objects of this level contain it."""
        if self.membership is None:
            raise ValueError("a level of segmented objects has no members")
        return np.asarray(self.membership.sum(axis=0)).ravel().astype(np.int64)

    def fingerprint(self, columns=()) -> str:
        """A hash of what a child level is built from: the ids, the
        membership, and the named columns. A child built on a level whose
        fingerprint has since changed is stale."""
        digest = hashlib.sha1()
        digest.update(np.ascontiguousarray(self.ids).tobytes())
        if self.membership is not None:
            m = self.membership.tocsr()
            m.sort_indices()
            digest.update(m.indptr.tobytes())
            digest.update(m.indices.tobytes())
        for column in columns:
            if column in self.table.columns:
                digest.update(column.encode())
                digest.update(
                    pd.util.hash_pandas_object(self.table[column], index=False).values.tobytes()
                )
        return digest.hexdigest()


def one_hot(rows, n_columns: int) -> sparse.csr_matrix:
    """(len(rows), n_columns) matrix with a single 1 per row, at `rows[i]`."""
    rows = np.asarray(rows, dtype=np.int64)
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (np.arange(len(rows)), rows)),
        shape=(len(rows), n_columns),
    )
