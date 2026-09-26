"""The hierarchy of levels, and the two operators that move features along it.

A **level** is a table of entities - lysosomes, cells, neighbourhoods,
neighbourhoods of neighbourhoods - with an id column. A **link** says which
entity of one level belongs to which of the level above it, as a membership
table; it is many-to-one for the pieces of a cell and many-to-many for
neighbourhoods, which overlap.

Every link is used the same two ways:

- **aggregate up** - a parent's features from its children's: how many, and
  a reduction of each feature (`cell_features` and `neighborhood_features`
  are the special cases this generalises);
- **reflect down** - a child's features from its parents': the value where
  there is one parent, the mean (a quantity) or the most common value (a
  category) where there are several. `reflect_neighborhoods` is the special
  case.

Both compose along a chain of links, so a lysosome can take on the type of
the neighbourhood-of-neighbourhoods its cell lives in without any code that
knows about lysosomes or about depth two. Each link is a sparse matrix M
(children x parents); a chain is their product.

**The circularity rule.** A column reflected down from a level carries that
level's rank. `typing_features(level)` - the features a level may be typed
(clustered, classified) on - leaves out every column reflected down from
that level or one above it. Otherwise a neighbourhood's type, handed down to
its cells and summed back up, becomes a feature of the neighbourhood it came
from, and the next clustering finds exactly what it was given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from vtea_core.context.spec import CELLULAR, NEIGHBORHOOD, SUBCELLULAR, TIERS

PART_OF = "part_of"  # many-to-one: a piece of a cell
MEMBER_OF = "member_of"  # many-to-many: a member of overlapping neighbourhoods
LINK_KINDS = (PART_OF, MEMBER_OF)

MISSING_CATEGORY = -1
AGGREGATIONS = ("n", "mean", "sum", "min", "max")


@dataclass
class Level:
    """One level: its entities, what identifies them, where it sits.

    `rank` orders the stack (subcellular 0, cellular 1, neighbourhoods
    1 + depth). `labels_key` is the label image its entities are drawn from,
    where they have one. `reflected` records, for every column handed down
    to this level, the rank of the level it came from - what the circularity
    rule reads.
    """

    name: str
    tier: str
    table: pd.DataFrame
    id_column: str
    rank: int = 0
    labels_key: str = ""
    reflected: dict[str, int] = field(default_factory=dict)
    # The NeighborhoodSet a neighbourhood level was built as - its members
    # and centres, which is what a hull outline is drawn from.
    neighborhoods: Any = None

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"level '{self.name}': unknown tier {self.tier!r}, expected {TIERS}")
        if self.id_column not in self.table.columns:
            raise ValueError(
                f"level '{self.name}' has no '{self.id_column}' column "
                f"(columns: {list(self.table.columns)})"
            )

    @property
    def ids(self) -> np.ndarray:
        return self.table[self.id_column].to_numpy()

    def __len__(self) -> int:
        return len(self.table)


@dataclass
class Link:
    """Which child belongs to which parent. `membership` has `child`,
    `parent` and optionally `is_seed` (the one neighbourhood centred on a
    child - its "own")."""

    child: str
    parent: str
    membership: pd.DataFrame
    kind: str = PART_OF

    def __post_init__(self) -> None:
        if self.kind not in LINK_KINDS:
            raise ValueError(f"unknown link kind {self.kind!r}, expected {LINK_KINDS}")
        missing = {"child", "parent"} - set(self.membership.columns)
        if missing:
            raise ValueError(f"a link's membership needs columns {sorted(missing)}")

    @property
    def has_seeds(self) -> bool:
        return "is_seed" in self.membership.columns and bool(self.membership["is_seed"].any())


def _names(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part) for part in value]


class ContextGraph:
    """Levels and the links between them."""

    def __init__(self) -> None:
        self.levels: dict[str, Level] = {}
        self.links: list[Link] = []
        # The definitions the levels were built from, when they were - how
        # each is drawn travels with them.
        self.spec = None

    # -- structure --------------------------------------------------------

    def add_level(self, level: Level) -> Level:
        if level.name in self.levels:
            raise ValueError(f"there is already a level called '{level.name}'")
        self.levels[level.name] = level
        return level

    def link(self, child: str, parent: str, membership: pd.DataFrame, kind: str = PART_OF) -> Link:
        for name in (child, parent):
            if name not in self.levels:
                raise ValueError(f"no level called '{name}'")
        if self.levels[parent].rank <= self.levels[child].rank:
            raise ValueError(
                f"'{parent}' does not sit above '{child}': a link runs from a level to "
                f"one of higher rank"
            )
        link = Link(child, parent, membership, kind)
        self.links = [
            existing
            for existing in self.links
            if not (existing.child == child and existing.parent == parent)
        ]
        self.links.append(link)
        return link

    def level(self, name: str) -> Level:
        try:
            return self.levels[name]
        except KeyError as exc:
            raise KeyError(f"no level called '{name}' (levels: {list(self.levels)})") from exc

    def stack(self) -> list[Level]:
        """Every level, lowest first - the order a context slider shows them in."""
        return sorted(self.levels.values(), key=lambda level: (level.rank, level.name))

    def parents_of(self, name: str) -> list[str]:
        return [link.parent for link in self.links if link.child == name]

    def children_of(self, name: str) -> list[str]:
        return [link.child for link in self.links if link.parent == name]

    def path(self, lower: str, upper: str) -> list[Link]:
        """The chain of links from `lower` up to `upper`, shortest first found.
        Raises when `upper` is not above `lower`."""
        self.level(lower)
        self.level(upper)
        if lower == upper:
            return []
        frontier: list[tuple[str, list[Link]]] = [(lower, [])]
        seen = {lower}
        while frontier:
            current, chain = frontier.pop(0)
            for link in self.links:
                if link.child != current or link.parent in seen:
                    continue
                if link.parent == upper:
                    return [*chain, link]
                seen.add(link.parent)
                frontier.append((link.parent, [*chain, link]))
        raise ValueError(f"'{upper}' is not above '{lower}' - no chain of links joins them")

    # -- the matrices -----------------------------------------------------

    def _matrix(self, link: Link, own: bool) -> sparse.csr_matrix:
        child_ids = self.level(link.child).ids
        parent_ids = self.level(link.parent).ids
        rows = pd.Index(child_ids).get_indexer(link.membership["child"].to_numpy())
        cols = pd.Index(parent_ids).get_indexer(link.membership["parent"].to_numpy())
        keep = (rows >= 0) & (cols >= 0)
        if own and link.has_seeds:
            keep &= link.membership["is_seed"].to_numpy(dtype=bool)
        data = np.ones(int(keep.sum()))
        return sparse.csr_matrix(
            (data, (rows[keep], cols[keep])), shape=(len(child_ids), len(parent_ids))
        )

    def membership_matrix(
        self, lower: str, upper: str, *, relation: Literal["auto", "own", "member"] = "member"
    ) -> sparse.csr_matrix:
        """(entities of `lower`) x (entities of `upper`), 1 where the lower one
        belongs to the upper one through the chain of links between them.

        `relation="own"` follows only the neighbourhood each entity is the
        seed of, on every link that has seeds; `"auto"` does the same (a
        centred neighbourhood's own is what "its neighbourhood" means);
        `"member"` follows every neighbourhood an entity is in.
        """
        own = relation in ("own", "auto")
        if relation == "own":
            chain = self.path(lower, upper)
            if not any(link.has_seeds for link in chain if link.kind == MEMBER_OF):
                raise ValueError(
                    f"no link between '{lower}' and '{upper}' has centred neighbourhoods, so "
                    f"nothing has one of its own; use relation='member'"
                )
        matrix = sparse.identity(len(self.level(lower)), format="csr")
        for link in self.path(lower, upper):
            matrix = matrix @ self._matrix(link, own)
            matrix.data[:] = 1.0  # membership, not a count of routes to it
        return sparse.csr_matrix(matrix)

    # -- the operators ----------------------------------------------------

    def aggregate_up(
        self,
        lower: str,
        upper: str,
        *,
        features: str | list[str] = "",
        aggregations: str | list[str] = "n,mean",
        prefix: str | None = None,
    ) -> pd.DataFrame:
        """One row per entity of `upper`: its members' features from `lower`,
        reduced. Columns are `<prefix>.n` and `<prefix>.<reduction>_<feature>`,
        the prefix defaulting to the lower level's name."""
        reductions = _names(aggregations)
        unknown = set(reductions) - set(AGGREGATIONS)
        if unknown:
            raise ValueError(f"unknown aggregation(s) {sorted(unknown)}, expected {AGGREGATIONS}")
        child, parent = self.level(lower), self.level(upper)
        columns = _names(features)
        missing = [name for name in columns if name not in child.table.columns]
        if missing:
            raise ValueError(f"'{lower}' has no column(s) {missing}")
        prefix = lower if prefix is None else prefix
        matrix = self.membership_matrix(lower, upper, relation="member").T.tocsr()
        counts = np.asarray(matrix.sum(axis=1)).ravel()

        result = pd.DataFrame({parent.id_column: parent.ids})
        if "n" in reductions:
            result[f"{prefix}.n"] = counts.astype(np.int64)
        for name in columns:
            values = child.table[name].to_numpy(dtype=float)
            present = np.isfinite(values)
            filled = np.where(present, values, 0.0)
            totals = matrix @ filled
            have = matrix @ present.astype(float)
            for reduction in reductions:
                if reduction == "sum":
                    result[f"{prefix}.sum_{name}"] = totals
                elif reduction == "mean":
                    with np.errstate(invalid="ignore", divide="ignore"):
                        result[f"{prefix}.mean_{name}"] = np.where(have > 0, totals / have, np.nan)
                elif reduction in ("min", "max"):
                    result[f"{prefix}.{reduction}_{name}"] = _extreme(matrix, values, reduction)
        return result

    def reflect_down(
        self,
        upper: str,
        lower: str,
        *,
        features: str | list[str] = "",
        categorical: str | list[str] = "",
        relation: Literal["auto", "own", "member"] = "auto",
        prefix: str | None = None,
    ) -> pd.DataFrame:
        """One row per entity of `lower`: the features of the `upper` entities
        it belongs to. A quantity is averaged over them and a category takes
        the most common value (the smallest, on a tie); an entity that belongs
        to none gets NaN or -1. Columns are `<prefix>.<feature>`, the prefix
        defaulting to the upper level's name, plus `<prefix>.n_memberships`."""
        parent, child = self.level(upper), self.level(lower)
        columns = _names(features) or _descriptive(parent)
        missing = [name for name in columns if name not in parent.table.columns]
        if missing:
            raise ValueError(f"'{upper}' has no column(s) {missing}")
        categories = set(_names(categorical)) & set(columns)
        prefix = upper if prefix is None else prefix
        has_seeds = any(
            link.has_seeds for link in self.path(lower, upper) if link.kind == MEMBER_OF
        )
        chosen = relation if relation != "auto" else ("own" if has_seeds else "member")
        matrix = self.membership_matrix(lower, upper, relation=chosen)
        counts = np.asarray(matrix.sum(axis=1)).ravel()

        result = pd.DataFrame({child.id_column: child.ids})
        result[f"{prefix}.n_memberships"] = counts.astype(np.int64)
        for name in columns:
            values = parent.table[name]
            if name in categories:
                result[f"{prefix}.{name}"] = _vote(matrix, values.to_numpy())
                continue
            numeric = values.to_numpy(dtype=float)
            present = np.isfinite(numeric)
            totals = matrix @ np.where(present, numeric, 0.0)
            have = matrix @ present.astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                result[f"{prefix}.{name}"] = np.where(have > 0, totals / have, np.nan)
        return result

    def apply_reflection(self, lower: str, upper: str, reflected: pd.DataFrame) -> list[str]:
        """Put a `reflect_down` result onto `lower`'s table, recording where
        each column came from for the circularity rule. Returns the columns."""
        child = self.level(lower)
        rank = self.level(upper).rank
        keyed = reflected.set_index(child.id_column)
        table = child.table.copy()
        ids = table[child.id_column]
        added = []
        for column in keyed.columns:
            table[column] = ids.map(keyed[column]).to_numpy()
            child.reflected[column] = rank
            added.append(column)
        child.table = table
        return added

    def typing_features(self, name: str) -> list[str]:
        """The columns `name` may be typed on: its numeric descriptive columns,
        less every column reflected down from it or from a level above it."""
        level = self.level(name)
        return [
            column
            for column in _descriptive(level)
            if level.reflected.get(column, -1) < level.rank
        ]


def _descriptive(level: Level) -> list[str]:
    """Numeric columns that describe an entity - not its id, seed or place."""
    skip = {level.id_column, "seed", "object_id", "cell_id", "neighborhood_id"}
    return [
        column
        for column in level.table.columns
        if column not in skip
        and not str(column).startswith("centroid-")
        and ".centroid-" not in str(column)
        and pd.api.types.is_numeric_dtype(level.table[column])
    ]


def _vote(matrix: sparse.csr_matrix, values: np.ndarray) -> np.ndarray:
    """Most common parent value per row of `matrix`; smallest on a tie; -1
    for a row with no parent (or only parents with no value)."""
    series = pd.Series(values)
    present = series.notna().to_numpy()
    options = sorted(series[present].unique())
    best = np.full(matrix.shape[0], MISSING_CATEGORY, dtype=np.int64)
    best_count = np.zeros(matrix.shape[0])
    for option in options:
        count = matrix @ ((series == option).to_numpy() & present).astype(float)
        better = count > best_count
        best[better] = int(option)
        best_count[better] = count[better]
    return best


def _extreme(matrix: sparse.csr_matrix, values: np.ndarray, which: str) -> np.ndarray:
    """Per-row min or max of `values` over the columns `matrix` marks."""
    result = np.full(matrix.shape[0], np.nan)
    for row in range(matrix.shape[0]):
        members = values[matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]]
        members = members[np.isfinite(members)]
        if members.size:
            result[row] = members.min() if which == "min" else members.max()
    return result


__all__ = [
    "AGGREGATIONS",
    "CELLULAR",
    "LINK_KINDS",
    "MEMBER_OF",
    "MISSING_CATEGORY",
    "NEIGHBORHOOD",
    "PART_OF",
    "SUBCELLULAR",
    "ContextGraph",
    "Level",
    "Link",
]
