"""Which object belongs to which — the link between two segmentations.

A protocol can produce several segmentations of the same field: nuclei from
DAPI, a nuclear envelope derived from those nuclei, a cytoplasm segmented
from a cytoskeletal channel, organelle puncta from another. Each is its own
label image with its own object ids, and nothing in a label image says that
cytoplasm 12 is the cytoplasm *of* nucleus 7.

That statement is what an `Association` is. They are kept as their own
object rather than as a column on the measurement table for four reasons:

- **The posterior survives.** "Cytoplasm 12 belongs to nucleus 7 (p=0.55) or
  nucleus 9 (p=0.44)" is a materially different claim from "cytoplasm 12
  belongs to nucleus 7", and it is the one that lets a review step surface
  the few percent of cells worth looking at by eye.
- **How the link was made is recorded**, on the same terms as every feature
  in the FeatureCatalog: the method, its parameters, and the score.
- **It is separable from the images**, so it saves, reloads and diffs as
  plain JSON.
- **It says nothing about direction.** Nothing here makes nuclei the root of
  a cell. A whole-cell segmentation parenting everything is the same model
  read the other way, which is what a workflow starting from whole cells
  needs.

See docs/OBJECT_ASSOCIATION.md for the phases this belongs to.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ASSOCIATION_FORMAT_VERSION = 1

# How a link was arrived at, as opposed to by which algorithm.
#
# DERIVED: the child was constructed from the parent (an annulus around a
#   nucleus), so the link is exact by construction rather than inferred.
# CONTAINED: the child sits inside the parent - an organelle in a cytoplasm.
# ASSIGNED: the two were segmented independently and the link was inferred.
DERIVED = "derived"
CONTAINED = "contained"
ASSIGNED = "assigned"

# A link a person made or corrected by hand. Kept distinct from every
# inferred method forever after: an analysis nobody can correct is one
# nobody can publish, and a correction that becomes indistinguishable from
# an inference is worse than no correction at all.
MANUAL = "manual"


@dataclass(frozen=True, order=True)
class ObjectRef:
    """One object, named by the segmentation it came from.

    `segmentation` is a step name (`watershed_split_1`), which is why steps
    carry unique names: it is the only stable way to say *which* label image
    an id belongs to when a protocol has several.
    """

    segmentation: str
    object_id: int

    def __str__(self) -> str:
        return f"{self.segmentation}#{self.object_id}"

    def to_dict(self) -> dict[str, Any]:
        return {"segmentation": self.segmentation, "object_id": int(self.object_id)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectRef:
        return cls(segmentation=data["segmentation"], object_id=int(data["object_id"]))


@dataclass
class Association:
    """`child` belongs to `parent`, with a recorded confidence and reason."""

    child: ObjectRef
    parent: ObjectRef
    relationship: str = ASSIGNED
    probability: float = 1.0
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    # The parents that were considered and rejected, with their posteriors.
    # Empty for a link that was exact by construction.
    alternatives: list[tuple[ObjectRef, float]] = field(default_factory=list)

    @property
    def is_certain(self) -> bool:
        return self.probability >= 1.0 and not self.alternatives

    @property
    def margin(self) -> float:
        """How much better the chosen parent was than the runner-up. Near
        zero is the signal that a link is worth a human's attention."""
        if not self.alternatives:
            return self.probability
        return self.probability - max(score for _ref, score in self.alternatives)

    def to_dict(self) -> dict[str, Any]:
        return {
            "child": self.child.to_dict(),
            "parent": self.parent.to_dict(),
            "relationship": self.relationship,
            "probability": float(self.probability),
            "method": self.method,
            "params": dict(self.params),
            "alternatives": [
                {"parent": ref.to_dict(), "probability": float(score)}
                for ref, score in self.alternatives
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Association:
        return cls(
            child=ObjectRef.from_dict(data["child"]),
            parent=ObjectRef.from_dict(data["parent"]),
            relationship=data.get("relationship", ASSIGNED),
            probability=float(data.get("probability", 1.0)),
            method=data.get("method", ""),
            params=dict(data.get("params", {})),
            alternatives=[
                (ObjectRef.from_dict(entry["parent"]), float(entry["probability"]))
                for entry in data.get("alternatives", [])
            ],
        )


class AssociationSet:
    """Links between two or more segmentations, indexed both ways.

    A child has at most one parent - that is the model, decided rather than
    discovered, with the runners-up kept on the link itself. A parent may
    have any number of children, which is what makes "the lysosomes of this
    cell" a question with an answer.
    """

    def __init__(self, associations: list[Association] | None = None):
        self._by_child: dict[ObjectRef, Association] = {}
        self._by_parent: dict[ObjectRef, list[Association]] = defaultdict(list)
        for association in associations or []:
            self.add(association)

    def add(self, association: Association) -> Association:
        """Add a link, replacing any the child already had. Re-running an
        association step should correct its own earlier answer rather than
        leave two contradictory links behind."""
        existing = self._by_child.get(association.child)
        if existing is not None:
            self._by_parent[existing.parent].remove(existing)
        self._by_child[association.child] = association
        self._by_parent[association.parent].append(association)
        return association

    def remove_child(self, child: ObjectRef) -> None:
        association = self._by_child.pop(child, None)
        if association is not None:
            self._by_parent[association.parent].remove(association)

    def parent_of(self, child: ObjectRef) -> ObjectRef | None:
        association = self._by_child.get(child)
        return None if association is None else association.parent

    def link_for(self, child: ObjectRef) -> Association | None:
        return self._by_child.get(child)

    def children_of(self, parent: ObjectRef, segmentation: str | None = None) -> list[ObjectRef]:
        """The objects belonging to `parent`, optionally only those from one
        segmentation - "the lysosomes of this cell" rather than everything
        attached to it."""
        children = [link.child for link in self._by_parent.get(parent, [])]
        if segmentation is not None:
            children = [child for child in children if child.segmentation == segmentation]
        return sorted(children)

    def segmentations(self) -> tuple[set[str], set[str]]:
        """(child segmentations, parent segmentations) present in this set."""
        children = {ref.segmentation for ref in self._by_child}
        parents = {link.parent.segmentation for link in self._by_child.values()}
        return children, parents

    def uncertain(self, threshold: float = 0.9) -> list[Association]:
        """Links whose winning margin is below `threshold`, worst first -
        the ones worth a human's attention, which is the whole reason the
        alternatives are kept."""
        return sorted(
            (link for link in self if link.margin < threshold),
            key=lambda link: link.margin,
        )

    def __iter__(self):
        return iter(self._by_child.values())

    def __len__(self) -> int:
        return len(self._by_child)

    def __contains__(self, child: ObjectRef) -> bool:
        return child in self._by_child

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "vtea_association_version": ASSOCIATION_FORMAT_VERSION,
            "associations": [link.to_dict() for link in self],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssociationSet:
        version = data.get("vtea_association_version")
        if version is not None and version > ASSOCIATION_FORMAT_VERSION:
            raise ValueError(
                f"association file version {version} is newer than this VTEA understands "
                f"({ASSOCIATION_FORMAT_VERSION}); upgrade vtea-core to read it"
            )
        return cls([Association.from_dict(entry) for entry in data.get("associations", [])])


def save_associations(associations: AssociationSet, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(associations.to_dict(), indent=2), encoding="utf-8")
    return path


def load_associations(path: str | Path) -> AssociationSet:
    return AssociationSet.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
