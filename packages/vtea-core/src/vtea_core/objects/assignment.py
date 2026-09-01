"""Turning candidate scores into one parent per child, with a posterior.

Two decisions live here, and they are different decisions.

**How confident are we?** `posterior()` normalises a child's affinities into
a distribution over its candidate parents *plus an explicit orphan option*.
Without that option every child is certain of something: a cytoplasm with a
single distant nucleus 30 microns away would come out as p=1.0 for that
nucleus, which is not a description of the evidence. `orphan_score` is the
affinity that "no parent" is worth, so a candidate has to beat it to win.

**Who gets whom?** `assign()`, and the mode matters more than it looks:

- `many_to_one` takes each child's best parent independently. Right for
  organelles - one cytoplasm holds many lysosomes, and nothing is competing.
- `one_to_one` solves the whole thing at once. Right for nucleus and
  cytoplasm, and *not* the same as taking each child's argmax: per-child
  argmax will happily hand the same nucleus to two neighbouring cytoplasms.
  A global optimum is the only way to honour "one and only one", and it
  sometimes gives a child its second choice so that another child can have
  its first - which is the point, and is why the alternatives are kept on the
  link afterwards.

The global solve is `scipy.optimize.linear_sum_assignment`, which is O(n^3)
and so is run per connected block of the candidate graph rather than over
every object at once. Because the scoring functions only propose nearby
parents, those blocks are small: a field of 50,000 objects is an impossible
matrix and thousands of trivial ones.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from vtea_core.objects.scoring import (
    CandidateScores,
    _coo_from_mapping,
    _ids_and_remap,
    _remapped,
    _row_slice,
    _row_starts,
    _sorted_by_child,
)

# One parent may take any number of children (organelles in a cytoplasm).
MANY_TO_ONE = "many_to_one"
# Each parent takes at most one child, decided globally (nucleus/cytoplasm).
ONE_TO_ONE = "one_to_one"

ASSIGNMENT_MODES = (MANY_TO_ONE, ONE_TO_ONE)


class Posterior:
    """Per child, a probability for each candidate parent and for no parent.

    The orphan probability is held separately rather than under a sentinel
    id, so that "belongs to nobody" can never be confused with "belongs to
    object 0".

    Held in the same three-array form as `CandidateScores`, and for the same
    reason: this is one number per candidate pair, and at ten million
    children a dict of dicts is the ceiling rather than the image. The
    accessors are unchanged, so anything reading a child's row reads it the
    way it always did.
    """

    __slots__ = (
        "_row_starts",
        "child_ids",
        "child_index",
        "method",
        "orphan",
        "params",
        "parent_ids",
        "parent_index",
        "probability",
    )

    def __init__(
        self,
        probabilities: Mapping[int, Mapping[int, float]] | None = None,
        orphan: Any = None,
        child_ids: Any = (),
        parent_ids: Any = (),
        method: str = "",
        params: dict[str, Any] | None = None,
        *,
        child_index: Any = None,
        parent_index: Any = None,
        probability: Any = None,
    ):
        self.child_ids, child_remap = _ids_and_remap(child_ids)
        self.parent_ids, parent_remap = _ids_and_remap(parent_ids)
        self.method = method
        self.params = dict(params or {})

        if probabilities:
            child_index, parent_index, probability = _coo_from_mapping(
                probabilities, self.child_ids, self.parent_ids
            )
        elif child_index is None:
            child_index = parent_index = np.empty(0, dtype=np.int32)
            probability = np.empty(0, dtype=float)
        else:
            child_index, parent_index = _remapped(
                child_index, parent_index, child_remap, parent_remap
            )

        self.child_index = np.asarray(child_index, dtype=np.int32)
        self.parent_index = np.asarray(parent_index, dtype=np.int32)
        self.probability = np.asarray(probability, dtype=float)
        self.child_index, self.parent_index, self.probability = _sorted_by_child(
            self.child_index, self.parent_index, self.probability
        )
        self._row_starts = _row_starts(self.child_index, len(self.child_ids))
        # A child with no candidates is certainly an orphan, so the default
        # is 1.0 rather than 0.0 - the absence of evidence is the answer.
        self.orphan = (
            np.ones(len(self.child_ids), dtype=float)
            if orphan is None
            else _orphan_array(orphan, self.child_ids)
        )

    def row(self, child_id: int) -> slice:
        return _row_slice(self.child_ids, self._row_starts, child_id)

    def candidates_for(self, child_id: int) -> tuple[np.ndarray, np.ndarray]:
        rows = self.row(child_id)
        return self.parent_ids[self.parent_index[rows]], self.probability[rows]

    def for_child(self, child_id: int) -> dict[int, float]:
        parents, values = self.candidates_for(child_id)
        return {int(parent): float(value) for parent, value in zip(parents, values)}

    def orphan_probability(self, child_id: int) -> float:
        position = int(np.searchsorted(self.child_ids, child_id))
        if position >= len(self.child_ids) or int(self.child_ids[position]) != int(child_id):
            return 1.0
        return float(self.orphan[position])

    @property
    def probabilities(self) -> dict[int, dict[int, float]]:
        """The nested dict this used to hold, rebuilt on demand."""
        built: dict[int, dict[int, float]] = {}
        for position, parent, value in zip(self.child_index, self.parent_index, self.probability):
            built.setdefault(int(self.child_ids[position]), {})[
                int(self.parent_ids[parent])
            ] = float(value)
        return built

    @property
    def n_candidates(self) -> int:
        return len(self.probability)

    def __len__(self) -> int:
        return len(self.child_ids)

    def __repr__(self) -> str:
        return (
            f"Posterior(method={self.method!r}, {len(self.child_ids)} children, "
            f"{self.n_candidates} candidates)"
        )


def _orphan_array(orphan: Any, child_ids: np.ndarray) -> np.ndarray:
    if isinstance(orphan, Mapping):
        values = np.ones(len(child_ids), dtype=float)
        for child_id, value in orphan.items():
            position = int(np.searchsorted(child_ids, child_id))
            if position < len(child_ids) and int(child_ids[position]) == int(child_id):
                values[position] = float(value)
        return values
    return np.asarray(orphan, dtype=float)


@dataclass(frozen=True)
class Match:
    """One child's outcome: a parent or none, with the rest of the posterior."""

    child_id: int
    parent_id: int | None
    probability: float
    alternatives: tuple[tuple[int, float], ...] = ()


def posterior(candidates: CandidateScores, *, orphan_score: float = 0.05) -> Posterior:
    """Normalise affinities into a distribution over parents plus orphan.

    `orphan_score` is on the same [0, 1] scale as the affinities: at the
    default, a candidate scoring 0.05 is exactly as likely as having no
    parent at all, and anything weaker loses to it. Setting it to zero makes
    every child with any candidate certain, which is occasionally what you
    want and usually is not.

    One pass over the candidate arrays: the per-child totals are a
    `bincount`, and the division is elementwise. There is no per-child
    Python loop here at all, which is what makes ten million children a
    matter of seconds rather than of minutes.
    """
    if orphan_score < 0:
        raise ValueError(f"orphan_score must not be negative, got {orphan_score}")

    n_children = len(candidates.child_ids)
    totals = (
        np.bincount(
            candidates.child_index, weights=candidates.affinity, minlength=n_children
        ).astype(float)
        + orphan_score
    )
    # A child with no candidates and no orphan mass: nothing can be said, so
    # say nothing rather than dividing by zero.
    speakable = totals > 0
    orphan = np.ones(n_children, dtype=float)
    orphan[speakable] = orphan_score / totals[speakable]

    keep = (candidates.affinity > 0) & speakable[candidates.child_index]
    child_index = candidates.child_index[keep]
    probability = candidates.affinity[keep] / totals[child_index]

    return Posterior(
        child_ids=candidates.child_ids,
        parent_ids=candidates.parent_ids,
        child_index=child_index,
        parent_index=candidates.parent_index[keep],
        probability=probability,
        orphan=orphan,
        method=candidates.method,
        params=dict(candidates.params) | {"orphan_score": float(orphan_score)},
    )


def _alternatives(
    parents: np.ndarray, values: np.ndarray, chosen: int | None
) -> tuple[tuple[int, float], ...]:
    order = np.argsort(values, kind="stable")[::-1]
    return tuple(
        (int(parents[index]), float(values[index]))
        for index in order
        if chosen is None or int(parents[index]) != chosen
    )


class _UnionFind:
    """Disjoint sets over a fixed number of nodes, by rank with path
    compression. The same structure the seam ledger uses, and here for the
    same reason: what connects two children is a chain of shared candidates,
    which is a connected component and nothing more."""

    __slots__ = ("parent", "rank")

    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, node: int) -> int:
        parent = self.parent
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return int(root)

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def _blocks(distribution: Posterior) -> list[tuple[list[int], list[int]]]:
    """Split the candidate graph into independent (children, parents) blocks.

    Two children that cannot compete for any parent, directly or through a
    chain of shared candidates, can be assigned separately without changing
    either answer. This is what makes the global solve affordable.

    Union-find over the candidate arrays rather than a traversal of nested
    dictionaries: children and parents share one node space (parents offset
    past the children), each candidate pair is one union, and the components
    fall out of grouping by root. The answer is identical - it is the same
    connected components - and it is one pass over an array rather than a
    graph walk through boxed integers.
    """
    n_children = len(distribution.child_ids)
    if not distribution.n_candidates:
        return []
    union = _UnionFind(n_children + len(distribution.parent_ids))
    for child, parent in zip(distribution.child_index, distribution.parent_index):
        union.union(int(child), n_children + int(parent))

    children: dict[int, list[int]] = {}
    parents: dict[int, list[int]] = {}
    for child in np.unique(distribution.child_index):
        children.setdefault(union.find(int(child)), []).append(
            int(distribution.child_ids[child])
        )
    for parent in np.unique(distribution.parent_index):
        parents.setdefault(union.find(n_children + int(parent)), []).append(
            int(distribution.parent_ids[parent])
        )
    return [
        (sorted(members), sorted(parents.get(root, [])))
        for root, members in children.items()
    ]


def _assign_one_to_one(distribution: Posterior) -> dict[int, int]:
    """The globally best set of pairings, at most one child per parent."""
    chosen: dict[int, int] = {}
    for children, parents in _blocks(distribution):
        rows = [distribution.candidates_for(child) for child in children]
        column_of = {parent: index for index, parent in enumerate(parents)}
        costs = np.empty((len(children), len(parents)))
        # A non-candidate has to be expensive enough that the optimiser never
        # prefers one over any combination of real pairings, but finite -
        # np.inf makes a block with no feasible complete matching unsolvable,
        # and here the rectangular case is normal rather than an error.
        real = [-np.log(value) for _parents, values in rows for value in values]
        forbidden = len(children) * (max(real, default=0.0) + 1.0) + 1.0
        costs.fill(forbidden)
        for row, (row_parents, values) in enumerate(rows):
            for parent, value in zip(row_parents, values):
                if value:
                    costs[row, column_of[int(parent)]] = -np.log(value)

        indices, columns = linear_sum_assignment(costs)
        for row, column in zip(indices, columns):
            if costs[row, column] < forbidden:
                chosen[children[row]] = parents[column]
    return chosen


def assign(
    distribution: Posterior,
    *,
    mode: str = MANY_TO_ONE,
    min_probability: float = 0.0,
) -> list[Match]:
    """One `Match` per child, in id order, parent None where unassigned.

    A child is left unassigned when no parent beats the orphan option, when
    the winner falls below `min_probability`, or - in one-to-one mode - when
    the global solution had no parent to spare for it. All three are ordinary
    results: forcing every cytoplasm onto its nearest nucleus would hide
    exactly the objects worth looking at.
    """
    if mode not in ASSIGNMENT_MODES:
        raise ValueError(f"unknown assignment mode {mode!r}, expected one of {ASSIGNMENT_MODES}")

    if mode == ONE_TO_ONE:
        chosen = _assign_one_to_one(distribution)
    else:
        chosen = {}
        for position, child_id in enumerate(distribution.child_ids):
            rows = slice(
                int(distribution._row_starts[position]),
                int(distribution._row_starts[position + 1]),
            )
            values = distribution.probability[rows]
            if len(values):
                best = int(np.argmax(values))
                chosen[int(child_id)] = int(
                    distribution.parent_ids[distribution.parent_index[rows][best]]
                )

    matches = []
    for child_id in distribution.child_ids:
        child_id = int(child_id)
        parents, values = distribution.candidates_for(child_id)
        parent_id = chosen.get(child_id)
        probability = 0.0
        if parent_id is not None:
            match = parents == parent_id
            probability = float(values[match][0]) if match.any() else 0.0
        # Losing to "no parent" is a decision, not a low score: a child whose
        # best candidate is weaker than the orphan option has no parent even
        # if it was the argmax among the candidates.
        if parent_id is not None and (
            probability < min_probability or probability < distribution.orphan_probability(child_id)
        ):
            parent_id, probability = None, 0.0
        matches.append(
            Match(
                child_id=child_id,
                parent_id=None if parent_id is None else int(parent_id),
                probability=probability,
                alternatives=_alternatives(parents, values, parent_id),
            )
        )
    return matches
