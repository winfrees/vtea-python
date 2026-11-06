"""Phenotype hierarchies: a tree of named cell types, each a rule within its parent.

Cell typing is done the way flow and tissue cytometry have always done it -
coarse first, then finer inside each coarse population - and the result is
a tree, not a flat list:

    Cells
    ├── Immune            CD45+
    │   ├── T cell        CD3+                (within Immune)
    │   │   ├── CD8 T     CD8+ AND NOT CD4+   (within T cell)
    │   │   └── CD4 T     CD4+                (within T cell)
    │   └── B cell        CD20+               (within Immune)
    └── Epithelial        panCK+

A `Phenotype` is a name, an optional parent and a rule: a class expression
(`vtea_core.classes.expression`) over the columns of an object table -
gate-membership columns, ranges of a feature, cluster ids, ROI ids, earlier
labels - evaluated *within* the parent's members. An empty rule means "all
of the parent", which is how a container node such as "Immune" with no rule
of its own simply groups its children.

`PhenotypeTree.assign` answers the question an analysis asks next: what is
each object, at each level? It gives every object its deepest phenotype
(the most specific thing that is known about it), the path to it
("Immune > T cell > CD8 T"), and the phenotype at each depth, so a figure can
be made at whichever level of the hierarchy it needs. Where an object
satisfies more than one sibling, the first sibling in the tree's order wins
and the overlap is counted in `PhenotypeTree.counts` - overlapping siblings
are usually a gate drawn too wide, and should be visible rather than
silently resolved.

The tree saves as plain, versioned JSON (rules, not masks): re-running it
on the next acquisition re-types those cells with the same definitions.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vtea_core.classes.expression import evaluate, referenced_columns
from vtea_core.classes.labelset import HIERARCHY_SEPARATOR, UNLABELLED

PHENOTYPES_FORMAT_VERSION = 1


@dataclass
class Phenotype:
    """One node of a phenotype tree."""

    name: str
    rule: str = ""  # class expression, evaluated within the parent; "" = all of the parent
    parent: str | None = None
    color: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "rule": self.rule, "parent": self.parent,
                "color": self.color, "description": self.description}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Phenotype:
        return cls(name=str(data["name"]), rule=str(data.get("rule", "")),
                   parent=data.get("parent"), color=str(data.get("color", "")),
                   description=str(data.get("description", "")))


class PhenotypeTree:
    """An ordered tree of phenotypes. Names are unique; order among siblings is
    the order they were added (and decides which wins an overlap)."""

    def __init__(self, phenotypes: Iterable[Phenotype] = ()) -> None:
        self._nodes: dict[str, Phenotype] = {}
        for phenotype in phenotypes:
            self.add(phenotype)

    # ---- structure -------------------------------------------------------------------------

    def add(self, phenotype: Phenotype) -> Phenotype:
        if not phenotype.name.strip():
            raise ValueError("a phenotype needs a name")
        if phenotype.name in self._nodes:
            raise ValueError(f"there is already a phenotype called {phenotype.name!r}")
        if phenotype.parent is not None and phenotype.parent not in self._nodes:
            raise KeyError(f"unknown parent phenotype {phenotype.parent!r}")
        if HIERARCHY_SEPARATOR.strip() in phenotype.name:
            raise ValueError(f"a phenotype name cannot contain {HIERARCHY_SEPARATOR.strip()!r}")
        self._nodes[phenotype.name] = phenotype
        return phenotype

    def remove(self, name: str) -> list[str]:
        """Remove a phenotype and everything under it; returns the names removed."""
        removed = [name]
        for child in self.children(name):
            removed += self.remove(child.name)
        del self._nodes[name]
        return removed

    def rename(self, old: str, new: str) -> None:
        if new in self._nodes:
            raise ValueError(f"there is already a phenotype called {new!r}")
        if not new.strip() or HIERARCHY_SEPARATOR.strip() in new:
            raise ValueError(f"{new!r} is not a valid phenotype name")
        nodes = {}
        for key, node in self._nodes.items():
            if node.parent == old:
                node.parent = new
            if key == old:
                node.name = new
                key = new
            nodes[key] = node
        self._nodes = nodes

    def get(self, name: str) -> Phenotype:
        return self._nodes[name]

    def __contains__(self, name: str) -> bool:
        return name in self._nodes

    def __iter__(self):
        """Depth-first, children in their order: the reading order of a legend."""
        def walk(parent):
            for node in self.children(parent):
                yield node
                yield from walk(node.name)
        return walk(None)

    def __len__(self) -> int:
        return len(self._nodes)

    def children(self, name: str | None) -> list[Phenotype]:
        return [node for node in self._nodes.values() if node.parent == name]

    def ancestors(self, name: str) -> list[str]:
        """From the root down to, but not including, `name`."""
        out = []
        parent = self._nodes[name].parent
        while parent is not None:
            out.append(parent)
            parent = self._nodes[parent].parent
        return out[::-1]

    def depth(self, name: str) -> int:
        """1 for a root phenotype."""
        return len(self.ancestors(name)) + 1

    def path(self, name: str) -> str:
        return HIERARCHY_SEPARATOR.join([*self.ancestors(name), name])

    # ---- membership --------------------------------------------------------------------------

    def missing_columns(self, frame: pd.DataFrame) -> dict[str, set[str]]:
        """Phenotypes whose rules read columns `frame` does not have."""
        out = {}
        for node in self._nodes.values():
            if node.rule.strip():
                missing = referenced_columns(node.rule) - set(frame.columns)
                if missing:
                    out[node.name] = missing
        return out

    def own_rule_mask(self, name: str, frame: pd.DataFrame) -> np.ndarray:
        rule = self._nodes[name].rule.strip()
        if not rule:
            return np.ones(len(frame), dtype=bool)
        return np.array(evaluate(rule, frame), dtype=bool)  # a writable copy

    def members(self, name: str, frame: pd.DataFrame, exclusive: bool = False) -> np.ndarray:
        """Which rows of `frame` are this phenotype: its rule within its parent.

        With `exclusive`, an object claimed by an earlier sibling (or by an
        earlier sibling of an ancestor) is not counted again - the membership
        `assign` uses.
        """
        mask = self.own_rule_mask(name, frame)
        node = self._nodes[name]
        if node.parent is not None:
            mask &= self.members(node.parent, frame, exclusive)
        if exclusive:
            for sibling in self.children(node.parent):
                if sibling.name == name:
                    break
                mask &= ~self.own_rule_mask(sibling.name, frame)
        return mask

    def assign(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per row of `frame`: its deepest phenotype ("phenotype"), the path to it
        ("phenotype path") and its phenotype at every depth ("phenotype level 1"
        ...). Objects in no phenotype are `UNLABELLED` at every level."""
        n = len(frame)
        depth_max = max((self.depth(name) for name in self._nodes), default=0)
        levels = [np.full(n, UNLABELLED, dtype=object) for _ in range(depth_max)]
        deepest = np.full(n, UNLABELLED, dtype=object)
        path = np.full(n, UNLABELLED, dtype=object)
        # Visiting parents before children means a child overwrites its parent's
        # "deepest" entry, which is exactly "the most specific thing known".
        for node in self:
            mask = self.members(node.name, frame, exclusive=True)
            if not mask.any():
                continue
            d = self.depth(node.name)
            levels[d - 1][mask] = node.name
            deepest[mask] = node.name
            path[mask] = self.path(node.name)
        out = pd.DataFrame({"phenotype": deepest, "phenotype path": path}, index=frame.index)
        for d, values in enumerate(levels, start=1):
            out[f"phenotype level {d}"] = values
        return out

    def counts(self, frame: pd.DataFrame) -> pd.DataFrame:
        """One row per phenotype: depth, count, % of its parent and of all
        objects, and how many objects its rule also matches that an earlier
        sibling claimed ("overlap", resolved to that sibling in `assign`; 0 for a
        rule-less "rest of the parent" phenotype)."""
        rows = []
        total = len(frame)
        for node in self:
            inclusive = self.members(node.name, frame)
            exclusive = self.members(node.name, frame, exclusive=True)
            parent_n = int(self.members(node.parent, frame, exclusive=True).sum()) \
                if node.parent is not None else total
            n = int(exclusive.sum())
            rows.append({
                "phenotype": node.name, "parent": node.parent or "", "depth": self.depth(node.name),
                "path": self.path(node.name), "rule": node.rule, "count": n,
                "% of parent": 100.0 * n / parent_n if parent_n else 0.0,
                "% of all": 100.0 * n / total if total else 0.0,
                # A rule-less phenotype is "the rest of its parent": overlapping its
                # earlier siblings is its definition, not a problem to report.
                "overlap": int((inclusive & ~exclusive).sum()) if node.rule.strip() else 0,
            })
        return pd.DataFrame(rows, columns=["phenotype", "parent", "depth", "path", "rule", "count",
                                           "% of parent", "% of all", "overlap"])

    # ---- files -------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"format": "vtea-phenotypes", "version": PHENOTYPES_FORMAT_VERSION,
                "phenotypes": [node.to_dict() for node in self]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhenotypeTree:
        if data.get("format") != "vtea-phenotypes":
            raise ValueError("not a VTEA phenotype tree")
        if int(data.get("version", 0)) > PHENOTYPES_FORMAT_VERSION:
            raise ValueError(f"phenotype tree version {data.get('version')} is newer than this "
                             f"version of vtea-core reads ({PHENOTYPES_FORMAT_VERSION})")
        return cls(Phenotype.from_dict(d) for d in data["phenotypes"])


def save_phenotypes(tree: PhenotypeTree, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(tree.to_dict(), indent=2))
    return path


def load_phenotypes(path: str | Path) -> PhenotypeTree:
    return PhenotypeTree.from_dict(json.loads(Path(path).read_text()))
