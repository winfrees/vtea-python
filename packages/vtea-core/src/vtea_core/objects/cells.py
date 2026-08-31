"""Composing associations into cells, and measuring them as cells.

An `AssociationSet` says which object belongs to which. A cell is what you
get by following those links to their root: a nucleus, the envelope derived
from it, the cytoplasm assigned to it, and the lysosomes inside that
cytoplasm are four objects in four segmentations and one biological thing.

Two decisions shape this module.

**The root segmentation defines what a cell is.** Nothing in the association
model makes nuclei special, and that is deliberate: with a whole-cell
segmentation as the root, nuclei become its children and everything else
still works. It is also how the multinucleate case is expressed - with one
nucleus per cell the nucleus is the root; where a cytoplasm may hold several,
the cytoplasm is the root and `n_nuclei` is a feature. The flag reads as
"multinucleate" to a user and as "which segmentation identifies a cell" here.

**Cell ids come from the root object's id.** Not from a counter, so that a
gate drawn on cell 412 still means the same cell after a re-run that changed
how many cells there are.

The per-cell table is where this becomes useful rather than merely correct:
one row per cell, columns namespaced by the segmentation each came from, and
one-to-many children aggregated - which is how "how many endolysosomes does
this cell have, and how bright are they" finally becomes a number.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vtea_core.objects.association import Association, AssociationSet, ObjectRef

CELL_FORMAT_VERSION = 1

# How a one-to-many role is reduced to one row. `n` (how many children) is
# always included, because "this cell has no lysosomes" and "this cell's
# lysosomes average 12 units" are different facts and the second hides the
# first.
AGGREGATIONS = ("n", "sum", "mean", "median")
DEFAULT_AGGREGATIONS = ("n", "mean", "sum")


@dataclass(frozen=True)
class Cell:
    """One cell: a root object, and the objects belonging to it by role.

    `parts` is keyed by segmentation name - which is the role, since a
    protocol's steps are already named for what they segment. A role may hold
    several objects (a cytoplasm's lysosomes) or one (its nucleus); nothing
    here assumes which, because that is a property of the data.
    """

    cell_id: int
    root: ObjectRef
    parts: dict[str, tuple[ObjectRef, ...]] = field(default_factory=dict)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(sorted(self.parts))

    def objects(self, role: str) -> tuple[ObjectRef, ...]:
        """The objects of `role`, the root included - a cell's own nucleus is
        as much a part of it as anything assigned to that nucleus, and a
        per-cell table that omitted the root's measurements would be missing
        the very features most protocols start from."""
        if role == self.root.segmentation:
            return (self.root,)
        return self.parts.get(role, ())

    def object(self, role: str) -> ObjectRef | None:
        """The single object of `role`, or None. Returns the first where a
        role holds several - use `objects` when that is possible."""
        found = self.objects(role)
        return found[0] if found else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": int(self.cell_id),
            "root": self.root.to_dict(),
            "parts": {
                role: [ref.to_dict() for ref in refs] for role, refs in sorted(self.parts.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cell:
        return cls(
            cell_id=int(data["cell_id"]),
            root=ObjectRef.from_dict(data["root"]),
            parts={
                role: tuple(ObjectRef.from_dict(entry) for entry in refs)
                for role, refs in data.get("parts", {}).items()
            },
        )


class CellCollection:
    """What any form of a cell result can answer.

    There are two forms and they are the same result at two scales. A
    `CellSet` holds a `Cell` object per cell, which is the readable thing
    and the right one for the tens of thousands of cells a field produces.
    `vtea_core.blocked.cells.CellMembership` holds the same facts as a
    `(cell_id, role, object_id)` table, because at ten million cells the
    object graph is several gigabytes before a single measurement is joined
    to it.

    Everything that only wants to *report* on a result - how many cells, how
    many are missing a part, which segmentation identifies them - goes
    through this, so it does not have to know which form it has.
    """

    @property
    def root_segmentation(self) -> str:
        raise NotImplementedError

    @property
    def single_roles(self) -> frozenset[str]:
        raise NotImplementedError

    def roles(self) -> tuple[str, ...]:
        raise NotImplementedError

    def part_roles(self) -> tuple[str, ...]:
        raise NotImplementedError

    def summary(self) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class CellSet(CellCollection):
    """The cells of one field, plus what did not make it into one.

    `unclaimed` holds objects that are linked to something but whose chain
    never reaches the root - a lysosome inside a cytoplasm that was itself
    never assigned a nucleus. They are kept for the same reason unassigned
    children are kept on an AssociationSet: an analysis that quietly drops
    a tenth of its objects looks exactly like one that does not.
    """

    def __init__(
        self,
        cells: Iterable[Cell] = (),
        unclaimed: Iterable[ObjectRef] = (),
        single_roles: Iterable[str] = (),
    ):
        self._cells: dict[int, Cell] = {cell.cell_id: cell for cell in cells}
        self._unclaimed: list[ObjectRef] = list(unclaimed)
        # Roles a cell has at most one of. Taken from how the association was
        # made - a derived part, or an assignment run one-to-one - rather than
        # from whether this particular field happens to contain a cell with
        # two of something. That matters: the per-cell table's columns are
        # named differently for the two cases, and a shape that depended on
        # the data could not be pooled across fields.
        self._single_roles: frozenset[str] = frozenset(single_roles)

    @property
    def root_segmentation(self) -> str:
        roots = {cell.root.segmentation for cell in self}
        return next(iter(roots)) if len(roots) == 1 else ""

    @property
    def unclaimed(self) -> list[ObjectRef]:
        return sorted(self._unclaimed)

    @property
    def single_roles(self) -> frozenset[str]:
        """Roles a cell has at most one of - the root always among them,
        since it is what identifies the cell."""
        return self._single_roles | {cell.root.segmentation for cell in self}

    def roles(self) -> tuple[str, ...]:
        """Every role present on any cell, in a stable order - the root
        segmentation included, since a cell's own nucleus is one of its
        parts as far as measuring it goes."""
        roles = {role for cell in self for role in cell.parts}
        roles |= {cell.root.segmentation for cell in self}
        return tuple(sorted(roles))

    def part_roles(self) -> tuple[str, ...]:
        """The roles other than the root - what was attached to a cell rather
        than what identifies it."""
        return tuple(sorted({role for cell in self for role in cell.parts}))

    def cell(self, cell_id: int) -> Cell | None:
        return self._cells.get(int(cell_id))

    def complete(self, roles: Iterable[str]) -> list[Cell]:
        """The cells that have at least one object in every named role - the
        ones a measurement needing all the parts can be trusted on."""
        wanted = tuple(roles)
        return [cell for cell in self if all(cell.objects(role) for role in wanted)]

    def missing(self, role: str) -> list[Cell]:
        return [cell for cell in self if not cell.objects(role)]

    def summary(self) -> str:
        parts = [f"{len(self)} cells"]
        for role in self.part_roles():
            absent = len(self.missing(role))
            parts.append(f"{role}: {len(self) - absent}/{len(self)}")
        if self._unclaimed:
            parts.append(f"{len(self._unclaimed)} objects in no cell")
        return ", ".join(parts)

    def __iter__(self):
        return iter(sorted(self._cells.values(), key=lambda cell: cell.cell_id))

    def __len__(self) -> int:
        return len(self._cells)

    def __contains__(self, cell_id: int) -> bool:
        return int(cell_id) in self._cells

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "vtea_cell_version": CELL_FORMAT_VERSION,
            "cells": [cell.to_dict() for cell in self],
            "unclaimed": [ref.to_dict() for ref in self.unclaimed],
            "single_roles": sorted(self._single_roles),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CellSet:
        version = data.get("vtea_cell_version")
        if version is not None and version > CELL_FORMAT_VERSION:
            raise ValueError(
                f"cell file version {version} is newer than this VTEA understands "
                f"({CELL_FORMAT_VERSION}); upgrade vtea-core to read it"
            )
        return cls(
            [Cell.from_dict(entry) for entry in data.get("cells", [])],
            [ObjectRef.from_dict(entry) for entry in data.get("unclaimed", [])],
            data.get("single_roles", []),
        )


def save_cells(cells: CellSet, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(cells.to_dict(), indent=2), encoding="utf-8")
    return path


def load_cells(path: str | Path) -> CellSet:
    return CellSet.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def merge_associations(associations: AssociationSet, other: AssociationSet) -> AssociationSet:
    """One set holding both sets' links, for a chain built a step at a time.

    A protocol associates a cytoplasm to a nucleus in one step and lysosomes
    to that cytoplasm in another; a cell spans both, so they have to be one
    set before it can be composed. Where the two disagree about a child's
    parent, `other` wins, which makes re-running a single association step
    and merging it in a correction rather than a contradiction.
    """
    merged = AssociationSet(list(associations), associations.unassigned)
    for link in other:
        merged.add(link)
    for child in other.unassigned:
        if merged.link_for(child) is None:
            merged.add_unassigned(child)
    return merged


def _children_by_parent(associations: AssociationSet) -> dict[ObjectRef, list[Association]]:
    grouped: dict[ObjectRef, list[Association]] = {}
    for link in associations:
        grouped.setdefault(link.parent, []).append(link)
    return grouped


def _root_ids(root_labels) -> set[int]:
    if root_labels is None:
        return set()
    if isinstance(root_labels, np.ndarray):
        return {int(value) for value in np.unique(root_labels) if value != 0}
    return {int(value) for value in root_labels}


def build_cells(
    associations: AssociationSet,
    root_labels,
    *,
    root: str = "",
) -> CellSet:
    """Follow the links out from every object of the root and call each a cell.

    `root` is a segmentation name: whichever segmentation identifies a cell,
    which is the nucleus in an ordinary protocol and the cytoplasm in a
    multinucleate one.

    `root_labels` is that segmentation's label image (or just its object ids,
    or None). It is worth passing: without it a cell is only known from the
    links pointing at it, so a nucleus nothing was assigned to would silently
    not be a cell at all - and dropping exactly the cells that lost a part
    biases every per-cell statistic that follows.

    A link chain that loops is refused rather than followed: it cannot be a
    hierarchy, and following it would not terminate.
    """
    if not root:
        raise ValueError(
            "build_cells needs the name of the segmentation that identifies a cell, e.g. "
            "root='nuclei_1'; in a protocol it is taken from the step the root input points at"
        )
    by_parent = _children_by_parent(associations)

    is_a_parent = any(ref.segmentation == root for ref in by_parent)
    is_a_child = any(link.child.segmentation == root for link in associations)
    if is_a_child and not is_a_parent:
        raise ValueError(
            f"'{root}' is only ever a child in these associations, so following the links out "
            f"from it reaches nothing. A cell's root has to be the parent side: associate the "
            f"other segmentation to '{root}', or build cells rooted on "
            f"'{next(iter({link.parent.segmentation for link in associations}))}' instead."
        )

    roots: set[int] = _root_ids(root_labels)
    roots |= {ref.object_id for ref in by_parent if ref.segmentation == root}
    # A root object may also appear as somebody's child (a nucleus assigned to
    # a cytoplasm when the nucleus is the root) - it is still a cell.
    roots |= {link.child.object_id for link in associations if link.child.segmentation == root}

    claimed: set[ObjectRef] = set()
    cells = []
    for object_id in sorted(roots):
        root_ref = ObjectRef(root, object_id)
        parts: dict[str, list[ObjectRef]] = {}
        seen = {root_ref}
        queue = [root_ref]
        while queue:
            current = queue.pop()
            for link in by_parent.get(current, []):
                child = link.child
                if child.segmentation == root and child != root_ref:
                    # Another cell's root: it is a cell in its own right, so
                    # it does not become part of this one.
                    continue
                if child in seen:
                    raise ValueError(
                        f"association cycle through {child}: a cell hierarchy cannot loop back "
                        f"on itself, so this cannot be composed into cells"
                    )
                seen.add(child)
                claimed.add(child)
                parts.setdefault(child.segmentation, []).append(child)
                queue.append(child)
        cells.append(
            Cell(
                cell_id=object_id,
                root=root_ref,
                parts={role: tuple(sorted(refs)) for role, refs in parts.items()},
            )
        )

    linked = {link.child for link in associations if link.child.segmentation != root}
    return CellSet(cells, sorted(linked - claimed), _single_roles(associations))


def _single_roles(associations: AssociationSet) -> frozenset[str]:
    """The roles a cell can only have one of, read off how each link was made.

    A derived part is one per parent by construction, and a one-to-one
    assignment says so in its recorded parameters. Everything else may be
    many, and is aggregated in the per-cell table rather than assumed unique.
    """
    from vtea_core.objects.assignment import ONE_TO_ONE
    from vtea_core.objects.association import DERIVED

    by_role: dict[str, list[Association]] = {}
    for link in associations:
        by_role.setdefault(link.child.segmentation, []).append(link)
    return frozenset(
        role
        for role, links in by_role.items()
        if all(
            link.relationship == DERIVED or link.params.get("mode") == ONE_TO_ONE for link in links
        )
    )


def _numeric_columns(table: pd.DataFrame, id_column: str) -> list[str]:
    return [
        column
        for column in table.columns
        if column != id_column and pd.api.types.is_numeric_dtype(table[column])
    ]


def cell_features(
    cells: CellSet,
    measurement_tables: dict[str, pd.DataFrame],
    *,
    aggregations: Iterable[str] = DEFAULT_AGGREGATIONS,
    id_column: str = "object_id",
) -> pd.DataFrame:
    """One row per cell, from one per-object measurement table per role.

    `measurement_tables` maps a role - a segmentation name, the same name the
    cells use - to that segmentation's measurement table. Columns are namespaced by role,
    so a nucleus's brightness and its cytoplasm's are `nuclei.mean_ch0` and
    `cytoplasm.mean_ch0` rather than two things called `mean_ch0`.

    A role a cell has several of is aggregated: `lysosomes.n` for how many,
    and one column per requested reduction (`lysosomes.mean_count`,
    `lysosomes.sum_mean_ch2`, ...). Which roles those are comes from the
    association, not from the data - see `CellSet.single_roles` - so the same
    protocol produces the same columns on every field, which is what makes
    the tables poolable.

    A cell missing a part gets NaN for its columns and 0 for its count, so
    "this cell has no lysosomes" survives into the table as a number rather
    than as a dropped row.
    """
    reductions = [name for name in aggregations if name != "n"]
    unknown = set(reductions) - set(AGGREGATIONS)
    if unknown:
        raise ValueError(f"unknown aggregation(s) {sorted(unknown)}, expected {list(AGGREGATIONS)}")

    index = pd.Index([cell.cell_id for cell in cells], name="cell_id")
    frame = pd.DataFrame(index=index)

    for role in cells.roles():
        table = measurement_tables.get(role)
        if table is None or table.empty:
            continue
        if id_column not in table.columns:
            raise ValueError(
                f"the measurement table for '{role}' has no '{id_column}' column, so its rows "
                f"cannot be matched to objects (columns: {list(table.columns)})"
            )

        owner = {
            ref.object_id: cell.cell_id for cell in cells for ref in cell.objects(role)
        }
        rows = table.copy()
        rows["__cell"] = rows[id_column].map(owner)
        # Cast back to an integer once the objects in no cell are out: the
        # map produces NaN for those, which makes the column float, and a
        # float index quietly turns `cell_id` into 1.0 in the joined table.
        rows = rows[rows["__cell"].notna()]
        rows["__cell"] = rows["__cell"].astype("int64")
        columns = _numeric_columns(table, id_column)

        if role in cells.single_roles:
            block = rows.drop_duplicates("__cell").set_index("__cell")[columns]
            block.columns = [f"{role}.{column}" for column in columns]
            frame = frame.join(block)
            continue

        grouped = rows.groupby("__cell")
        frame[f"{role}.n"] = grouped.size().reindex(index).fillna(0).astype(int)
        for reduction in reductions:
            block = grouped[columns].agg(reduction)
            block.columns = [f"{role}.{reduction}_{column}" for column in columns]
            frame = frame.join(block)

    return frame.reset_index()
