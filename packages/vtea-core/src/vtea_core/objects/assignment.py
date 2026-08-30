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

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from vtea_core.objects.scoring import CandidateScores

# One parent may take any number of children (organelles in a cytoplasm).
MANY_TO_ONE = "many_to_one"
# Each parent takes at most one child, decided globally (nucleus/cytoplasm).
ONE_TO_ONE = "one_to_one"

ASSIGNMENT_MODES = (MANY_TO_ONE, ONE_TO_ONE)


@dataclass(frozen=True)
class Posterior:
    """Per child, a probability for each candidate parent and for no parent.

    The orphan probability is held separately rather than under a sentinel
    id, so that "belongs to nobody" can never be confused with "belongs to
    object 0".
    """

    probabilities: dict[int, dict[int, float]]
    orphan: dict[int, float]
    child_ids: tuple[int, ...]
    parent_ids: tuple[int, ...]
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def for_child(self, child_id: int) -> dict[int, float]:
        return self.probabilities.get(int(child_id), {})

    def orphan_probability(self, child_id: int) -> float:
        return self.orphan.get(int(child_id), 1.0)


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
    """
    if orphan_score < 0:
        raise ValueError(f"orphan_score must not be negative, got {orphan_score}")

    probabilities: dict[int, dict[int, float]] = {}
    orphan: dict[int, float] = {}
    for child_id in candidates.child_ids:
        scores = candidates.for_child(child_id)
        total = sum(scores.values()) + orphan_score
        if total <= 0:
            # No candidates and no orphan mass: nothing can be said, so say
            # nothing rather than dividing by zero.
            orphan[child_id] = 1.0
            continue
        probabilities[child_id] = {
            parent_id: score / total for parent_id, score in scores.items() if score > 0
        }
        orphan[child_id] = orphan_score / total

    return Posterior(
        probabilities=probabilities,
        orphan=orphan,
        child_ids=candidates.child_ids,
        parent_ids=candidates.parent_ids,
        method=candidates.method,
        params=dict(candidates.params) | {"orphan_score": float(orphan_score)},
    )


def _alternatives(scores: dict[int, float], chosen: int | None) -> tuple[tuple[int, float], ...]:
    return tuple(
        sorted(
            ((parent_id, p) for parent_id, p in scores.items() if parent_id != chosen),
            key=lambda entry: entry[1],
            reverse=True,
        )
    )


def _blocks(probabilities: dict[int, dict[int, float]]) -> list[tuple[list[int], list[int]]]:
    """Split the candidate graph into independent (children, parents) blocks.

    Two children that cannot compete for any parent, directly or through a
    chain of shared candidates, can be assigned separately without changing
    either answer. This is what makes the global solve affordable.
    """
    by_parent: dict[int, list[int]] = {}
    for child_id, scores in probabilities.items():
        for parent_id in scores:
            by_parent.setdefault(parent_id, []).append(child_id)

    seen_children: set[int] = set()
    blocks: list[tuple[list[int], list[int]]] = []
    for start in probabilities:
        if start in seen_children:
            continue
        children: set[int] = set()
        parents: set[int] = set()
        queue = deque([("child", start)])
        seen_children.add(start)
        while queue:
            kind, node = queue.popleft()
            if kind == "child":
                children.add(node)
                for parent_id in probabilities[node]:
                    if parent_id not in parents:
                        parents.add(parent_id)
                        queue.append(("parent", parent_id))
            else:
                for child_id in by_parent.get(node, []):
                    if child_id not in seen_children:
                        seen_children.add(child_id)
                        queue.append(("child", child_id))
        blocks.append((sorted(children), sorted(parents)))
    return blocks


def _assign_one_to_one(probabilities: dict[int, dict[int, float]]) -> dict[int, int]:
    """The globally best set of pairings, at most one child per parent."""
    chosen: dict[int, int] = {}
    for children, parents in _blocks(probabilities):
        costs = np.empty((len(children), len(parents)))
        # A non-candidate has to be expensive enough that the optimiser never
        # prefers one over any combination of real pairings, but finite -
        # np.inf makes a block with no feasible complete matching unsolvable,
        # and here the rectangular case is normal rather than an error.
        real = [
            -np.log(probabilities[child][parent])
            for child in children
            for parent in probabilities[child]
            if parent in parents
        ]
        forbidden = len(children) * (max(real, default=0.0) + 1.0) + 1.0
        costs.fill(forbidden)
        for row, child in enumerate(children):
            for column, parent in enumerate(parents):
                probability = probabilities[child].get(parent)
                if probability:
                    costs[row, column] = -np.log(probability)

        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
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
        chosen = _assign_one_to_one(distribution.probabilities)
    else:
        chosen = {}
        for child_id, scores in distribution.probabilities.items():
            if scores:
                chosen[child_id] = max(scores, key=scores.get)

    matches = []
    for child_id in distribution.child_ids:
        scores = distribution.for_child(child_id)
        parent_id = chosen.get(child_id)
        probability = scores.get(parent_id, 0.0) if parent_id is not None else 0.0
        # Losing to "no parent" is a decision, not a low score: a child whose
        # best candidate is weaker than the orphan option has no parent even
        # if it was the argmax among the candidates.
        if parent_id is not None and (
            probability < min_probability or probability < distribution.orphan_probability(child_id)
        ):
            parent_id, probability = None, 0.0
        matches.append(
            Match(
                child_id=int(child_id),
                parent_id=None if parent_id is None else int(parent_id),
                probability=float(probability),
                alternatives=_alternatives(scores, parent_id),
            )
        )
    return matches
