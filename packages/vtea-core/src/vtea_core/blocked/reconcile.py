"""Putting back together an object that a tile boundary cut in half.

The hard half of processing data larger than memory, and the part with no
library to defer to. Everything else in `vtea_core.blocked` divides work up
and hands the same NumPy functions smaller arrays; this decides what to
believe when two tiles each have an opinion about the same cell.

There are two independent settings, and collapsing them into one list of
rules hides the more consequential:

**Does the tiling overlap?** With a halo, a tile can hold a *complete* copy
of an object that crosses its core boundary, paid for in redundant
computation. Without one, every voxel is segmented exactly once and no tile
has a complete copy of anything a seam crosses - it can only be reassembled
from pieces, never chosen.

**How is a fragment in one tile recognised as the same object as a fragment
in another?** By overlap (IoU over the region both tiles segmented), by
centroid proximity, by voxels touching across the seam plane, or not at all.

`SeamPolicy.overlap_match()` is the default and is what a user who never
opens the setting gets: the most accurate of the four, and the only one
whose correctness does not depend on a property of the specimen.

What is recorded matters as much as what is decided. `LabelLedger` keeps
every object's fragments, which rule joined them, and how strong the
evidence was - the same instinct as `Association.alternatives` and
`Ownership.confidence()`, and for the same reason: the few percent of
objects a seam ran through are exactly the ones worth looking at by eye,
and an analysis that cannot point at them cannot be checked.

See docs/LARGE_IMAGES.md, "Reflection rules".
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

LEDGER_FORMAT_VERSION = 1

# Whether tiles overlap. Not a detail: it decides whether a complete copy of
# a cut object exists anywhere at all.
OVERLAPPING = "overlapping"
ABUTTING = "abutting"
TILINGS = (OVERLAPPING, ABUTTING)

# How a fragment in one tile is recognised as the same object as a fragment
# in another.
#
# OVERLAP: IoU over the region both tiles segmented. Unambiguous, because it
#   is the same voxels being compared, and the only matching that stays
#   correct where objects are packed tightly. Needs overlapping tiles.
# CENTROID: nearest centroid within a distance. Reads a table rather than an
#   image, so it costs a kd-tree query and no I/O - but two distinct cells
#   either side of a seam can have closer centroids than one cut cell's two
#   halves, which is the normal case in packed epithelium, and an elongated
#   object cut across its long axis is under-merged for the mirror reason.
# TOUCHING: voxels face-adjacent across the seam plane. The only
#   correspondence available without an overlap - and it merges anything
#   that touches there, including cells that were merely adjacent.
# NONE: no correspondence. Every fragment is its own object.
OVERLAP = "overlap"
CENTROID = "centroid"
TOUCHING = "touching"
NO_MATCHING = "none"
MATCHINGS = (OVERLAP, CENTROID, TOUCHING, NO_MATCHING)

# What is done with fragments that were matched.
#
# OWN: one tile's complete copy is kept and replaces the others. Needs
#   overlapping tiles, since otherwise no copy is complete.
# MERGE: the fragments are unioned into one object. The only honest
#   resolution when nothing is complete anywhere.
# FLAG: keep them distinct, link them, mark them contested, resolve nothing.
# RESEGMENT: re-run the segmenter on a window centred on the seam, so the
#   object that was cut is interior and no tile boundary influenced it. The
#   only honest answer when the segmenter is not translation-invariant,
#   because then every tile's copy is shaped by a boundary that is not real
#   and choosing between them keeps a wrong mask.
OWN = "own"
MERGE = "merge"
FLAG = "flag"
RESEGMENT = "resegment"
RESOLUTIONS = (OWN, MERGE, FLAG, RESEGMENT)

# What to do about an object that reached the outer edge of its halo, and so
# was demonstrably not contained by any tile.
FLAG_EXCEEDED = "flag"
RAISE_EXCEEDED = "raise"
EXCEEDED_ACTIONS = (FLAG_EXCEEDED, RAISE_EXCEEDED)

# Objects touching the *dataset* border are genuinely truncated specimens,
# and excluding them is a standard cytometry choice. Objects touching a
# *tile* border are not the same thing at all - dropping those is a bug
# under every strategy except "no merge", where it is a requirement.
KEEP = "keep"
DROP = "drop"
BORDER_ACTIONS = (KEEP, DROP)


class SeamPolicyError(ValueError):
    """A combination of settings that cannot mean anything."""


class HaloExceeded(RuntimeError):
    """An object was larger than the halo meant to contain it."""


@dataclass(frozen=True)
class SeamPolicy:
    """How this run reassembles objects that a tile boundary cut.

    Recorded on the result, not only in the configuration: a table computed
    under `overlap_match()` and a table computed under `no_merge()` are
    different measurements of the same specimen, and one that cannot say
    which it is cannot be compared with anything.
    """

    tiles: str = OVERLAPPING
    matching: str = OVERLAP
    resolution: str = OWN

    halo: int | None = None
    max_object_extent: float | None = None
    min_overlap: float = 0.5
    max_centroid_distance: float | None = None
    # How much context a re-segmented window carries around the object.
    # None takes the plan's halo, which is what the segmenter was already
    # judged to need.
    resegment_margin: int | None = None
    border_objects: str = KEEP
    drop_seam_objects: bool = False
    # No synthetic halo by default. A mirrored border invents objects at the
    # edge of the specimen and fuses them with the real ones they reflect -
    # the right boundary condition for a filter, a fabricated cell for a
    # segmenter. See executor.read_block.
    pad_mode: str | None = None
    on_halo_exceeded: str = FLAG_EXCEEDED

    def __post_init__(self) -> None:
        for value, allowed, name in (
            (self.tiles, TILINGS, "tiles"),
            (self.matching, MATCHINGS, "matching"),
            (self.resolution, RESOLUTIONS, "resolution"),
            (self.border_objects, BORDER_ACTIONS, "border_objects"),
            (self.on_halo_exceeded, EXCEEDED_ACTIONS, "on_halo_exceeded"),
        ):
            if value not in allowed:
                raise SeamPolicyError(f"unknown {name} {value!r}, expected one of {allowed}")
        if not 0 < self.min_overlap <= 1:
            raise SeamPolicyError(f"min_overlap must be in (0, 1], got {self.min_overlap}")
        if self.matching == OVERLAP and self.tiles != OVERLAPPING:
            raise SeamPolicyError(
                "overlap matching compares the two tiles' labellings of the region they "
                "both segmented, so it needs overlapping tiles. Use touching matching "
                "with an abutting tiling, or overlapping tiles with this matching."
            )
        if self.matching == TOUCHING and self.tiles != ABUTTING:
            raise SeamPolicyError(
                "touching matching pairs fragments that meet across a seam plane, which "
                "only means something when tiles abut. With overlapping tiles the same "
                "voxels are in both tiles and overlap matching is both available and "
                "unambiguous."
            )
        if self.resolution == OWN and self.tiles != OVERLAPPING:
            raise SeamPolicyError(
                "'own' keeps one tile's complete copy of the object, and without a halo "
                "no tile has one. Use 'merge', which reassembles it from the pieces."
            )
        if self.resolution == OWN and self.matching == NO_MATCHING:
            raise SeamPolicyError(
                "'own' has to know which fragments are the same object before it can "
                "keep one of them. Choose a matching, or use 'flag'."
            )
        if self.resolution == RESEGMENT and self.matching == NO_MATCHING:
            raise SeamPolicyError(
                "'resegment' re-runs the segmenter on the objects a seam cut, so it has "
                "to know which those are. Choose a matching."
            )
        if self.resolution == RESEGMENT and self.tiles != OVERLAPPING:
            raise SeamPolicyError(
                "'resegment' needs context on both sides of the seam to give an answer "
                "the seam did not influence, and an abutting tiling has none to give."
            )

    # -- the four a user picks from ------------------------------------

    @classmethod
    def overlap_match(cls, *, merge: bool = False, **kwargs: Any) -> SeamPolicy:
        """The default. Overlapping tiles, matched by IoU over the region
        both segmented; one complete copy kept, or `merge=True` to union the
        fragments for objects larger than any affordable halo."""
        return cls(
            tiles=OVERLAPPING, matching=OVERLAP, resolution=MERGE if merge else OWN, **kwargs
        )

    @classmethod
    def centroid_match(cls, **kwargs: Any) -> SeamPolicy:
        """Overlapping tiles matched by centroid proximity - a kd-tree over a
        table rather than a pass over voxels. May over-merge; see CENTROID."""
        return cls(tiles=OVERLAPPING, matching=CENTROID, resolution=MERGE, **kwargs)

    @classmethod
    def touching_merge(cls, **kwargs: Any) -> SeamPolicy:
        """Abutting tiles - no redundant computation at all - merging
        fragments that meet across a seam plane."""
        return cls(tiles=ABUTTING, matching=TOUCHING, resolution=MERGE, **kwargs)

    @classmethod
    def resegment(cls, **kwargs: Any) -> SeamPolicy:
        """For a segmenter that is not translation-invariant.

        The strategies that pick between tiles' copies all assume at least
        one copy is worth keeping. A learned segmenter breaks that: near a
        tile edge its answer is computed from truncated context, so every
        copy is shaped by a boundary that has nothing to do with the
        specimen and the better of two wrong masks is still wrong. This
        re-runs the segmenter on a window centred on the object instead, so
        the object is interior and no boundary ran through it.

        The default for Cellpose, and pointless for anything
        translation-invariant, which gives the same answer wherever the
        window is.
        """
        return cls(tiles=OVERLAPPING, matching=OVERLAP, resolution=RESEGMENT, **kwargs)

    @classmethod
    def no_merge(cls, *, drop_seam_objects: bool = True, **kwargs: Any) -> SeamPolicy:
        """Abutting tiles, nothing reconciled. Every fragment is its own
        object, so this is only honest paired with dropping the objects a
        seam touched - the standard cytometry exclusion, applied to seams as
        well as to the specimen edge. That is the default here; turn it off
        deliberately and read the ledger's seam_exposed_fraction."""
        return cls(
            tiles=ABUTTING,
            matching=NO_MATCHING,
            resolution=FLAG,
            drop_seam_objects=drop_seam_objects,
            **kwargs,
        )

    @property
    def is_overlapping(self) -> bool:
        return self.tiles == OVERLAPPING

    @property
    def matches_fragments(self) -> bool:
        return self.matching != NO_MATCHING

    @property
    def needs_tile_labels(self) -> bool:
        """Whether each tile's own labelling of its whole block has to be
        kept. Overlap matching compares two of them; owning a copy writes
        one of them out. Everything else works from the assembled array and
        a table."""
        return self.matching == OVERLAP or self.resolution == OWN

    def describe(self) -> str:
        if not self.matches_fragments:
            summary = f"{self.tiles} tiles, nothing merged"
        else:
            summary = f"{self.tiles} tiles, matched by {self.matching}, resolved by {self.resolution}"
        if self.drop_seam_objects:
            summary += ", seam-touching objects dropped"
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "tiles": self.tiles,
            "matching": self.matching,
            "resolution": self.resolution,
            "resegment_margin": self.resegment_margin,
            "min_overlap": self.min_overlap,
            "max_centroid_distance": self.max_centroid_distance,
            "border_objects": self.border_objects,
            "drop_seam_objects": self.drop_seam_objects,
            "pad_mode": self.pad_mode,
            "on_halo_exceeded": self.on_halo_exceeded,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SeamPolicy:
        known = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        return cls(**known)


DEFAULT_POLICY = SeamPolicy.overlap_match()


@dataclass(frozen=True)
class Fragment:
    """One tile's view of one object.

    An object entirely inside a tile has exactly one of these. An object a
    seam ran through has one per tile that saw it, and the whole of this
    module is about deciding which of those are the same object.

    Coordinates are global - the volume's, not the block's - so a fragment
    can be compared with one from another tile without anybody having to
    remember which tile it came from.
    """

    tile: tuple[int, ...]
    local_id: int
    provisional_id: int
    core_voxels: int
    block_voxels: int
    centroid: tuple[float, ...]
    bbox: tuple[tuple[int, int], ...]
    faces: frozenset[str] = frozenset()
    at_dataset_border: bool = False
    exceeded_halo: bool = False

    @property
    def touches_seam(self) -> bool:
        """Whether this fragment reaches a boundary with another tile - as
        opposed to the edge of the specimen, which is a different fact."""
        return bool(self.faces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile": list(self.tile),
            "local_id": self.local_id,
            "provisional_id": self.provisional_id,
            "core_voxels": self.core_voxels,
            "block_voxels": self.block_voxels,
            "centroid": list(self.centroid),
            "bbox": [list(pair) for pair in self.bbox],
            "faces": sorted(self.faces),
            "at_dataset_border": self.at_dataset_border,
            "exceeded_halo": self.exceeded_halo,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Fragment:
        return cls(
            tile=tuple(data["tile"]),
            local_id=int(data["local_id"]),
            provisional_id=int(data["provisional_id"]),
            core_voxels=int(data["core_voxels"]),
            block_voxels=int(data["block_voxels"]),
            centroid=tuple(float(value) for value in data["centroid"]),
            bbox=tuple((int(a), int(b)) for a, b in data["bbox"]),
            faces=frozenset(data.get("faces", ())),
            at_dataset_border=bool(data.get("at_dataset_border", False)),
            exceeded_halo=bool(data.get("exceeded_halo", False)),
        )


# What joined an object's fragments, recorded per object rather than per
# run, because an uncut object was not decided by anything.
UNCUT = "uncut"


@dataclass
class LabelLedger:
    """How every object in a blocked segmentation was arrived at.

    The audit trail, and the thing that makes a seam-crossing object
    *gateable*: `to_frame()` joins `n_fragments`, `seam_rule` and
    `seam_confidence` onto the measurement table, so drawing a gate on low
    confidence and opening the gallery is the review workflow, with no new
    UI at all.
    """

    policy: SeamPolicy = field(default_factory=lambda: DEFAULT_POLICY)
    fragments: dict[int, list[Fragment]] = field(default_factory=dict)
    decided_by: dict[int, str] = field(default_factory=dict)
    evidence: dict[int, float] = field(default_factory=dict)
    dropped: dict[int, str] = field(default_factory=dict)
    # Objects a matching related but a resolution deliberately left apart -
    # what `flag` produces. The link is the whole value of that strategy:
    # without it "contested" is a label with nothing behind it.
    links: dict[int, list[int]] = field(default_factory=dict)

    def add(
        self,
        object_id: int,
        fragments: Sequence[Fragment],
        *,
        decided_by: str = UNCUT,
        evidence: float = 1.0,
    ) -> None:
        self.fragments[int(object_id)] = list(fragments)
        self.decided_by[int(object_id)] = decided_by
        self.evidence[int(object_id)] = float(evidence)

    @property
    def object_ids(self) -> list[int]:
        return sorted(self.fragments)

    @property
    def n_objects(self) -> int:
        return len(self.fragments)

    def size(self, object_id: int) -> int:
        """Voxels in the whole object, across every tile that holds part of
        it. The number `filter_by_size` needs and that no single tile has."""
        return sum(fragment.core_voxels for fragment in self.fragments[object_id])

    def sizes(self) -> dict[int, int]:
        return {object_id: self.size(object_id) for object_id in self.fragments}

    def n_fragments(self, object_id: int) -> int:
        return len(self.fragments[object_id])

    def touches_seam(self, object_id: int) -> bool:
        return any(fragment.touches_seam for fragment in self.fragments[object_id])

    def at_dataset_border(self, object_id: int) -> bool:
        return any(fragment.at_dataset_border for fragment in self.fragments[object_id])

    def exceeded_halo(self, object_id: int) -> bool:
        """Whether *no* tile managed to contain this object.

        `all`, not `any`, and the difference is the difference between a
        useful warning and a scary useless one. A tile that saw only part of
        an object reports a truncated view, and that is normal - it is what
        a halo is for, and the neighbour holding the rest may well have the
        whole thing. The object is only in trouble when every tile that saw
        it saw it cut off, because then there is no complete copy anywhere.
        """
        return all(fragment.exceeded_halo for fragment in self.fragments[object_id])

    def confidence(self, object_id: int) -> float:
        """How much to believe this object's boundary, in [0, 1].

        An object no seam went near is 1.0 and means it. An object joined
        across a seam carries the strength of the evidence that joined it.
        An object that outgrew its halo is 0.0 whatever else is true of it,
        because the tile that was supposed to contain it did not.
        """
        if self.exceeded_halo(object_id):
            return 0.0
        if not self.touches_seam(object_id):
            return 1.0
        return float(self.evidence.get(object_id, 0.0))

    @property
    def seam_exposed_fraction(self) -> float:
        """The share of objects a tile boundary touched.

        Reported whether or not anyone asked, because under a no-merge
        strategy it is the size of the error: at a laptop's tile size it is
        routinely one object in seven.
        """
        if not self.fragments:
            return 0.0
        touched = sum(1 for object_id in self.fragments if self.touches_seam(object_id))
        return touched / len(self.fragments)

    @property
    def n_reconciled(self) -> int:
        """Objects assembled from more than one tile's fragments."""
        return sum(1 for object_id in self.fragments if self.n_fragments(object_id) > 1)

    def exceeded(self) -> list[int]:
        return [object_id for object_id in self.object_ids if self.exceeded_halo(object_id)]

    def to_frame(self):
        """One row per object, for joining onto the measurement table."""
        import pandas as pd

        return pd.DataFrame(
            {
                "object_id": self.object_ids,
                "n_fragments": [self.n_fragments(i) for i in self.object_ids],
                "seam_rule": [self.decided_by[i] for i in self.object_ids],
                "seam_confidence": [self.confidence(i) for i in self.object_ids],
                "touches_seam": [self.touches_seam(i) for i in self.object_ids],
                "at_dataset_border": [self.at_dataset_border(i) for i in self.object_ids],
                "exceeded_halo": [self.exceeded_halo(i) for i in self.object_ids],
            }
        )

    def describe(self) -> str:
        parts = [f"{self.n_objects:,} objects", self.policy.describe()]
        if self.n_reconciled:
            parts.append(f"{self.n_reconciled:,} reconciled across tiles")
        parts.append(f"{self.seam_exposed_fraction:.1%} seam-exposed")
        if self.dropped:
            parts.append(f"{len(self.dropped):,} dropped")
        exceeded = self.exceeded()
        if exceeded:
            parts.append(f"{len(exceeded):,} exceeded the halo and are flagged")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": LEDGER_FORMAT_VERSION,
            "policy": self.policy.to_dict(),
            "objects": [
                {
                    "id": object_id,
                    "decided_by": self.decided_by[object_id],
                    "evidence": self.evidence[object_id],
                    "fragments": [f.to_dict() for f in self.fragments[object_id]],
                }
                for object_id in self.object_ids
            ],
            "dropped": {str(key): value for key, value in self.dropped.items()},
            "links": {str(key): value for key, value in self.links.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LabelLedger:
        version = int(data.get("version", 1))
        if version > LEDGER_FORMAT_VERSION:
            raise ValueError(
                f"this ledger was written in format version {version} and this VTEA "
                f"reads up to {LEDGER_FORMAT_VERSION}"
            )
        ledger = cls(policy=SeamPolicy.from_dict(data.get("policy", {})))
        for entry in data.get("objects", []):
            ledger.add(
                int(entry["id"]),
                [Fragment.from_dict(item) for item in entry["fragments"]],
                decided_by=entry.get("decided_by", UNCUT),
                evidence=float(entry.get("evidence", 1.0)),
            )
        ledger.dropped = {int(key): value for key, value in data.get("dropped", {}).items()}
        ledger.links = {
            int(key): [int(item) for item in value]
            for key, value in data.get("links", {}).items()
        }
        return ledger


def save_ledger(ledger: LabelLedger, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(ledger.to_dict(), indent=2))
    return path


def load_ledger(path: str | Path) -> LabelLedger:
    return LabelLedger.from_dict(json.loads(Path(path).read_text()))


# -- matching -----------------------------------------------------------


class UnionFind:
    """Disjoint sets over provisional ids.

    A cut object can be in three or four tiles at once - a corner of the
    grid, or anything long - so pairwise matches have to be closed into
    groups rather than applied one at a time. Union-find does that in one
    pass over the pairs and is the whole of the "stitching" the Java
    `ObjectStitcher` did with a kD-tree.
    """

    def __init__(self, items: Iterable[int] = ()):
        self._parent: dict[int, int] = {int(item): int(item) for item in items}

    def add(self, item: int) -> None:
        self._parent.setdefault(int(item), int(item))

    def find(self, item: int) -> int:
        item = int(item)
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # path compression
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: int, right: int) -> int:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return left_root
        # Lowest id wins, so groups are deterministic whatever order the
        # pairs arrived in - which is what keeps a re-run reproducible.
        winner, loser = sorted((left_root, right_root))
        self._parent[loser] = winner
        return winner

    def groups(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for item in self._parent:
            result.setdefault(self.find(item), []).append(item)
        return {root: sorted(members) for root, members in result.items()}


def overlap_pairs(
    left: np.ndarray, right: np.ndarray, *, min_overlap: float = 0.5
) -> list[tuple[int, int, float]]:
    """Fragments in two labellings *of the same voxels* that are one object.

    `left` and `right` are two tiles' answers for the region they both
    segmented, so the comparison is like for like and the IoU means what it
    says. Returns (left id, right id, IoU) for every pair above the
    threshold.

    Computed sparsely - only the pairs that actually co-occur - because a
    dense objects-by-objects table over a busy shared region is a matrix
    nobody needs and everybody pays for.
    """
    if left.shape != right.shape:
        raise ValueError(f"regions differ in shape: {left.shape} vs {right.shape}")
    flat_left, flat_right = left.ravel(), right.ravel()

    ids_left, counts_left = np.unique(flat_left[flat_left > 0], return_counts=True)
    ids_right, counts_right = np.unique(flat_right[flat_right > 0], return_counts=True)
    if not ids_left.size or not ids_right.size:
        return []

    both = (flat_left > 0) & (flat_right > 0)
    if not both.any():
        return []
    index_left = np.searchsorted(ids_left, flat_left[both])
    index_right = np.searchsorted(ids_right, flat_right[both])
    codes, intersection = np.unique(
        index_left.astype(np.int64) * ids_right.size + index_right, return_counts=True
    )
    at_left, at_right = np.divmod(codes, ids_right.size)
    union = counts_left[at_left] + counts_right[at_right] - intersection
    score = intersection / union

    keep = score >= min_overlap
    return [
        (int(ids_left[a]), int(ids_right[b]), float(value))
        for a, b, value in zip(at_left[keep], at_right[keep], score[keep])
    ]


def touching_pairs(
    lower: np.ndarray, upper: np.ndarray
) -> list[tuple[int, int, float]]:
    """Fragments that meet across a seam plane, from the two slabs either
    side of it.

    The only correspondence available when tiles abut, and it cannot tell a
    cut object from two cells that happened to touch there - the evidence it
    reports is how much of the smaller fragment's face is in contact, which
    is the most that can be said.
    """
    if lower.shape != upper.shape:
        raise ValueError(f"slabs differ in shape: {lower.shape} vs {upper.shape}")
    flat_lower, flat_upper = lower.ravel(), upper.ravel()
    contact = (flat_lower > 0) & (flat_upper > 0)
    if not contact.any():
        return []

    ids_lower, face_lower = np.unique(flat_lower[flat_lower > 0], return_counts=True)
    ids_upper, face_upper = np.unique(flat_upper[flat_upper > 0], return_counts=True)
    index_lower = np.searchsorted(ids_lower, flat_lower[contact])
    index_upper = np.searchsorted(ids_upper, flat_upper[contact])
    codes, touching = np.unique(
        index_lower.astype(np.int64) * ids_upper.size + index_upper, return_counts=True
    )
    at_lower, at_upper = np.divmod(codes, ids_upper.size)
    smaller_face = np.minimum(face_lower[at_lower], face_upper[at_upper])
    score = touching / smaller_face
    return [
        (int(ids_lower[a]), int(ids_upper[b]), float(value))
        for a, b, value in zip(at_lower, at_upper, score)
    ]


def centroid_pairs(
    fragments: Sequence[Fragment],
    *,
    max_distance: float,
    spacing: Any = None,
) -> list[tuple[int, int, float]]:
    """Fragments in *different* tiles whose centroids are close enough to be
    one object.

    Reads a table rather than an image. Distances are physical wherever the
    voxel size is known, so a threshold means the same thing along z as in x
    - which matters more here than almost anywhere, since a seam is a plane
    and the fragments either side of it are separated along one axis.

    The over-merge this can produce is not a bug to be tuned away: two
    distinct cells either side of a seam can genuinely have closer centroids
    than one cut cell's two halves. The evidence returned falls off with
    distance so that a review can sort by it.
    """
    from scipy.spatial import cKDTree

    candidates = [fragment for fragment in fragments if fragment.touches_seam]
    if len(candidates) < 2:
        return []
    sizes = _voxel_sizes(spacing, len(candidates[0].centroid))
    points = np.array([fragment.centroid for fragment in candidates]) * np.asarray(sizes)
    tree = cKDTree(points)

    pairs = []
    for left, right in tree.query_pairs(max_distance):
        if candidates[left].tile == candidates[right].tile:
            # Two objects in one tile are two objects; this is only about
            # fragments that a seam separated.
            continue
        distance = float(np.linalg.norm(points[left] - points[right]))
        pairs.append(
            (
                candidates[left].provisional_id,
                candidates[right].provisional_id,
                float(1.0 - distance / max_distance),
            )
        )
    return pairs


def _voxel_sizes(spacing: Any, ndim: int) -> tuple[float, ...]:
    if spacing is None or not getattr(spacing, "is_known", False):
        return (1.0,) * ndim
    return tuple(float(size) for size in spacing.for_ndim(ndim))


def group_fragments(
    provisional_ids: Iterable[int], pairs: Iterable[tuple[int, int, float]]
) -> tuple[dict[int, int], dict[int, float]]:
    """Close pairwise matches into groups.

    Returns provisional id -> group id (the lowest provisional id in the
    group, so the result does not depend on the order the pairs arrived),
    and group id -> the *weakest* evidence holding it together. Weakest
    rather than average on purpose: a chain of three fragments joined by one
    confident match and one marginal one is only as trustworthy as the
    marginal one, and that is what a review should be sorting on.
    """
    union = UnionFind(provisional_ids)
    strength: dict[int, float] = {}
    for left, right, evidence in pairs:
        union.union(left, right)
        for item in (left, right):
            strength[item] = min(strength.get(item, 1.0), float(evidence))

    assignment, weakest = {}, {}
    for root, members in union.groups().items():
        for member in members:
            assignment[member] = root
        scores = [strength[member] for member in members if member in strength]
        weakest[root] = min(scores) if scores else 1.0
    return assignment, weakest
