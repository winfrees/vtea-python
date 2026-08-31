"""Ownership that does not cost more than the image it describes.

`vtea_core.objects.Ownership` keeps, for every voxel, the best *k* owners
and their probabilities. That is already the frugal form - a dense cell x
voxel posterior would be about 10^11 floats for a modest field - and it is
still the worst number in docs/LARGE_IMAGES.md: over the 2048 x 2048 x 2000
volume the plan works from, a dense top-3 ownership is **201 GB**, six times
the image it is about.

The reason is that it is dense over a volume that is mostly background.
Ownership is only defined inside the mask, and a mask that is 5% foreground
means 95% of those 201 GB are three zeros and three zeroes repeated. So this
keeps the same information restricted to the voxels it is about: a sorted
array of flat indices, and the owners and probabilities beside them. The
same 5% field comes to about 13 GB - still large, and now the same order of
magnitude as the image rather than six times it.

Two design points worth stating:

- **Probabilities are float32.** A posterior that is meaningful to seven
  decimal places is not a posterior anybody has; halving the largest arrays
  costs nothing real.
- **The entries are grouped by tile**, with offsets, so a blocked
  measurement can find the voxels belonging to one tile without searching.
  Built tile by tile, it comes out that way for free.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vtea_core.objects.ownership import MANUAL, OWNERSHIP_FORMAT_VERSION, Ownership

INDEX_DTYPE = np.int64
OWNER_DTYPE = np.int32
PROBABILITY_DTYPE = np.float32


@dataclass
class SparseOwnership:
    """Top-k ownership, kept only where the mask says it means something.

    `indices` are flat positions into a volume of `shape`, ascending within
    each tile. `owners` and `probabilities` are (k, n_voxels), slot 0 being
    the winner - so `owners[0]` is the hard label image and
    `probabilities[0]` is the confidence map, exactly as in the dense form.

    `offsets` marks where each tile's entries begin and end, so a pass over
    the image can find the ownership for the block it is holding without
    searching for it.
    """

    shape: tuple[int, ...]
    indices: np.ndarray
    owners: np.ndarray
    probabilities: np.ndarray
    offsets: np.ndarray | None = None
    segmentation: str = ""
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    manual: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.shape = tuple(int(size) for size in self.shape)
        self.indices = np.asarray(self.indices, dtype=INDEX_DTYPE)
        self.owners = np.atleast_2d(np.asarray(self.owners, dtype=OWNER_DTYPE))
        self.probabilities = np.atleast_2d(
            np.asarray(self.probabilities, dtype=PROBABILITY_DTYPE)
        )
        if self.owners.shape != self.probabilities.shape:
            raise ValueError(
                f"owners {self.owners.shape} and probabilities "
                f"{self.probabilities.shape} must have the same shape"
            )
        if self.owners.shape[1] != self.indices.size:
            raise ValueError(
                f"{self.owners.shape[1]} owner entries for {self.indices.size} voxels"
            )
        if self.manual is None:
            self.manual = np.zeros(self.n_voxels, dtype=bool)

    @property
    def n_voxels(self) -> int:
        """Voxels ownership is defined for - not the volume's size."""
        return int(self.indices.size)

    @property
    def top_k(self) -> int:
        return int(self.owners.shape[0])

    @property
    def density(self) -> float:
        total = int(np.prod(self.shape)) if self.shape else 0
        return self.n_voxels / total if total else 0.0

    @property
    def nbytes(self) -> int:
        return int(
            self.indices.nbytes
            + self.owners.nbytes
            + self.probabilities.nbytes
            + (self.manual.nbytes if self.manual is not None else 0)
        )

    @property
    def dense_nbytes(self) -> int:
        """What the dense form would have cost - the number this exists to
        avoid, kept so a run can report the saving rather than assert it."""
        total = int(np.prod(self.shape)) if self.shape else 0
        return total * self.top_k * (
            np.dtype(np.int32).itemsize + np.dtype(np.float64).itemsize
        )

    def coordinates(self, where: slice | np.ndarray | None = None) -> tuple[np.ndarray, ...]:
        """The voxel positions, as an index tuple ready to slice an array."""
        indices = self.indices if where is None else self.indices[where]
        return np.unravel_index(indices, self.shape)

    def hard(self, out: Any = None) -> Any:
        """The winner per voxel, as an ordinary label image.

        `out` is written into rather than allocated when given, which is how
        a label image larger than memory comes out of this - hand it a Zarr
        array and nothing dense is ever held.
        """
        if out is None:
            out = np.zeros(self.shape, dtype=OWNER_DTYPE)
        out[self.coordinates()] = self.owners[0]
        return out

    def confidence(self, out: Any = None) -> Any:
        """The winning probability per voxel - the map that makes a
        contested boundary visible rather than merely present."""
        if out is None:
            out = np.zeros(self.shape, dtype=PROBABILITY_DTYPE)
        out[self.coordinates()] = self.probabilities[0]
        return out

    def margin(self) -> np.ndarray:
        """How much better the winner was than the runner-up, per owned
        voxel. Near zero is a coin toss, which is not the same thing as a
        low absolute probability."""
        if self.top_k < 2:
            return self.probabilities[0].copy()
        return self.probabilities[0] - self.probabilities[1]

    def contested(self, threshold: float = 0.9) -> np.ndarray:
        """Which entries the method was unsure about - positions into the
        sparse arrays, not a dense mask, since a dense mask is the thing
        this class exists to not build."""
        return np.flatnonzero((self.owners[0] != 0) & (self.probabilities[0] < threshold))

    def object_ids(self) -> list[int]:
        found = np.unique(self.owners)
        return sorted(int(value) for value in found if value != 0)

    def weights_for(self, object_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Where this owner has a claim, and how strong - as (positions,
        weights) into the sparse arrays. What a weighted measurement
        integrates over."""
        weights = np.zeros(self.n_voxels, dtype=np.float64)
        for slot in range(self.top_k):
            claimed = self.owners[slot] == object_id
            weights[claimed] = self.probabilities[slot][claimed]
        where = np.flatnonzero(weights > 0)
        return where, weights[where]

    def override(self, where: np.ndarray, object_id: int) -> None:
        """Give these entries to `object_id` outright, and remember that a
        person said so - a correction indistinguishable from an inference is
        worse than no correction at all."""
        where = np.asarray(where)
        self.owners[0][where] = int(object_id)
        self.probabilities[0][where] = 1.0
        for slot in range(1, self.top_k):
            self.owners[slot][where] = 0
            self.probabilities[slot][where] = 0.0
        self.manual[where] = True
        self.method = MANUAL if not self.method else self.method

    def tile_slice(self, tile: int) -> slice:
        """The entries belonging to one tile of the plan that built this."""
        if self.offsets is None:
            raise ValueError("this ownership was not built tile by tile")
        return slice(int(self.offsets[tile]), int(self.offsets[tile + 1]))

    def summary(self, threshold: float = 0.9) -> str:
        owned = int((self.owners[0] != 0).sum())
        contested = self.contested(threshold).size
        parts = [
            (
                f"{len(self.object_ids())} owners over {owned:,} voxels "
                f"({self.density:.1%} of the volume)"
            )
        ]
        if owned:
            parts.append(f"{contested:,} contested ({100 * contested / owned:.1f}%)")
        overridden = int(self.manual.sum())
        if overridden:
            parts.append(f"{overridden:,} set by hand")
        saving = self.dense_nbytes / self.nbytes if self.nbytes else 0
        parts.append(f"{saving:.0f}x smaller than the dense form")
        return ", ".join(parts)

    # -- conversion and persistence ---------------------------------------

    @classmethod
    def from_dense(cls, ownership: Ownership) -> SparseOwnership:
        """The same ownership, minus the background it says nothing about."""
        owned = np.flatnonzero(ownership.owners[0].reshape(-1) != 0)
        owners = ownership.owners.reshape(ownership.top_k, -1)[:, owned]
        probabilities = ownership.probabilities.reshape(ownership.top_k, -1)[:, owned]
        manual = None
        if ownership.manual is not None:
            manual = ownership.manual.reshape(-1)[owned]
        return cls(
            shape=ownership.shape,
            indices=owned,
            owners=owners,
            probabilities=probabilities,
            segmentation=ownership.segmentation,
            method=ownership.method,
            params=dict(ownership.params),
            manual=manual,
        )

    def to_dense(self) -> Ownership:
        """The dense form, for the code that has not been taught the sparse
        one. Allocates the full array, so only for data that fits - the
        `dense_nbytes` property is there to check before calling it."""
        where = self.coordinates()
        owners = np.zeros((self.top_k, *self.shape), dtype=OWNER_DTYPE)
        probabilities = np.zeros((self.top_k, *self.shape), dtype=np.float64)
        for slot in range(self.top_k):
            owners[slot][where] = self.owners[slot]
            probabilities[slot][where] = self.probabilities[slot]
        manual = np.zeros(self.shape, dtype=bool)
        if self.manual is not None:
            manual[where] = self.manual
        return Ownership(
            owners=owners,
            probabilities=probabilities,
            segmentation=self.segmentation,
            method=self.method,
            params=dict(self.params),
            manual=manual,
        )

    @classmethod
    def concatenate(cls, parts: Sequence[SparseOwnership]) -> SparseOwnership:
        """Join per-tile ownerships into one, keeping the tile offsets.

        The tiles' entries stay in tile order rather than being sorted into
        one global order: sorting would cost a pass over everything and buy
        nothing, since what reads this reads it a tile at a time.
        """
        parts = [part for part in parts if part.n_voxels or part.shape]
        if not parts:
            raise ValueError("nothing to concatenate")
        shape = parts[0].shape
        if any(part.shape != shape for part in parts):
            raise ValueError("cannot join ownerships of different volumes")
        top_k = max(part.top_k for part in parts)
        offsets = np.cumsum([0] + [part.n_voxels for part in parts])
        return cls(
            shape=shape,
            indices=np.concatenate([part.indices for part in parts]) if parts else np.empty(0),
            owners=np.concatenate([_pad_slots(part.owners, top_k) for part in parts], axis=1),
            probabilities=np.concatenate(
                [_pad_slots(part.probabilities, top_k) for part in parts], axis=1
            ),
            offsets=offsets,
            segmentation=parts[0].segmentation,
            method=parts[0].method,
            params=dict(parts[0].params),
            manual=np.concatenate(
                [
                    part.manual
                    if part.manual is not None
                    else np.zeros(part.n_voxels, dtype=bool)
                    for part in parts
                ]
            ),
        )

    def save(self, path: str | Path) -> Path:
        """Compressed npz, like the dense form - the arrays are the data and
        JSON would be several times their size."""
        path = Path(path)
        np.savez_compressed(
            path,
            indices=self.indices,
            owners=self.owners,
            probabilities=self.probabilities,
            manual=self.manual,
            offsets=self.offsets if self.offsets is not None else np.empty(0, INDEX_DTYPE),
            meta=np.array(
                json.dumps(
                    {
                        "vtea_ownership_version": OWNERSHIP_FORMAT_VERSION,
                        "sparse": True,
                        "shape": list(self.shape),
                        "segmentation": self.segmentation,
                        "method": self.method,
                        "params": self.params,
                    }
                )
            ),
        )
        return path if path.suffix else path.with_suffix(".npz")


def _pad_slots(values: np.ndarray, top_k: int) -> np.ndarray:
    """Give every part the same number of owner slots.

    One tile can have voxels contested three ways and its neighbour none at
    all, so the parts do not all come back with the same k.
    """
    if values.shape[0] == top_k:
        return values
    padding = np.zeros((top_k - values.shape[0], values.shape[1]), dtype=values.dtype)
    return np.concatenate([values, padding], axis=0)


def load_sparse_ownership(path: str | Path) -> SparseOwnership:
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        version = meta.get("vtea_ownership_version")
        if version is not None and version > OWNERSHIP_FORMAT_VERSION:
            raise ValueError(
                f"this ownership was written in format version {version} and this VTEA "
                f"reads up to {OWNERSHIP_FORMAT_VERSION}"
            )
        offsets = data["offsets"]
        return SparseOwnership(
            shape=tuple(meta["shape"]),
            indices=data["indices"],
            owners=data["owners"],
            probabilities=data["probabilities"],
            offsets=offsets if offsets.size else None,
            segmentation=meta.get("segmentation", ""),
            method=meta.get("method", ""),
            params=meta.get("params", {}),
            manual=data["manual"],
        )


# -- building one, out of core ------------------------------------------


class HaloTooSmallForReach(ValueError):
    """A marker's claim reaches further than the tiles overlap."""


def required_reach(falloff: float, reach: float | None) -> float:
    """How far a marker's claim actually carries.

    `distance_ownership` defaults `reach` to four falloffs, where a claim is
    under 2%. The scaling contract cannot express "four times another
    parameter unless this one is set", so the number is computed here from
    what the caller actually passed and checked against the plan - a check
    beats a guess, and the failure it prevents is a voxel assigned to the
    nearest marker *this tile happened to contain*.
    """
    return float(reach) if reach is not None else 4.0 * float(falloff)


def ownership_blocked(
    labels: Any,
    mask: Any,
    *,
    plan: Any,
    spacing: Any = None,
    falloff: float = 2.0,
    reach: float | None = None,
    top_k: int = 2,
    segmentation: str = "",
    progress: Any = None,
) -> SparseOwnership:
    """`distance_ownership` over a volume too large to hold.

    Each tile computes ownership for its core plus a halo, and keeps the
    core. The halo is what makes this exact rather than approximate: a voxel
    near a tile edge can be claimed by a marker on the other side of it, and
    a tile that could not see that marker would hand the voxel to the
    nearest one it *could* see. So the halo must cover the reach, and this
    refuses to run when it does not rather than producing a plausible answer
    with a seam down it.
    """
    from vtea_core.objects import distance_ownership

    carry = required_reach(falloff, reach)
    _check_halo(plan, carry, spacing)

    parts = []
    for index, tile in enumerate(plan.tiles()):
        block_labels = np.asarray(labels[tile.padded])
        block_mask = np.asarray(mask[tile.padded])
        inner = tile.inner_unpadded
        if not block_mask[inner].any():
            parts.append(_empty_part(plan.shape, top_k))
        else:
            owned = distance_ownership(
                block_labels,
                block_mask,
                spacing=spacing,
                falloff=falloff,
                reach=reach,
                top_k=top_k,
                segmentation=segmentation,
            )
            parts.append(_core_of(owned, tile, plan.shape, inner))
        if progress is not None:
            progress(index + 1, plan.n_tiles)

    result = SparseOwnership.concatenate(parts)
    result.segmentation = segmentation
    result.params = {"falloff": falloff, "reach": reach, "top_k": top_k, "blocked": True}
    return result


def _check_halo(plan: Any, carry: float, spacing: Any) -> None:
    sizes = (
        tuple(float(size) for size in spacing.for_ndim(len(plan.shape)))
        if spacing is not None and getattr(spacing, "is_known", False)
        else (1.0,) * len(plan.shape)
    )
    needed = [int(np.ceil(carry / size)) for size in sizes]
    short = [
        (axis, plan.halo[axis], needed[axis])
        for axis in plan.tiled_axes
        if plan.splits[axis] > 1 and plan.halo[axis] < needed[axis]
    ]
    if short:
        detail = "; ".join(
            f"axis {axis} has a halo of {have} and needs {want}" for axis, have, want in short
        )
        raise HaloTooSmallForReach(
            f"a marker's claim carries {carry:g} units, further than the tiles overlap "
            f"({detail}). A voxel near a seam would be given to the nearest marker its "
            f"own tile could see rather than the nearest one there is. Raise the memory "
            f"budget so the halo grows, or lower `reach`/`falloff`."
        )


def _empty_part(shape: tuple[int, ...], top_k: int) -> SparseOwnership:
    return SparseOwnership(
        shape=shape,
        indices=np.empty(0, dtype=INDEX_DTYPE),
        owners=np.empty((top_k, 0), dtype=OWNER_DTYPE),
        probabilities=np.empty((top_k, 0), dtype=PROBABILITY_DTYPE),
    )


def _core_of(
    owned: Ownership, tile: Any, shape: tuple[int, ...], inner: tuple[slice, ...]
) -> SparseOwnership:
    """One tile's contribution: its core only, in the volume's coordinates.

    The halo is computed and discarded. It was never this tile's to report -
    those voxels belong to a neighbour's core, and keeping both copies would
    count them twice in every weighted sum they touch.
    """
    core_owners = owned.owners[(slice(None), *inner)]
    core_probabilities = owned.probabilities[(slice(None), *inner)]
    keep = np.flatnonzero(core_owners[0].reshape(-1) != 0)

    core_shape = core_owners.shape[1:]
    local = np.unravel_index(keep, core_shape)
    global_index = np.ravel_multi_index(
        tuple(axis + part.start for axis, part in zip(local, tile.core)), shape
    )
    order = np.argsort(global_index, kind="stable")
    return SparseOwnership(
        shape=shape,
        indices=global_index[order],
        owners=core_owners.reshape(core_owners.shape[0], -1)[:, keep][:, order],
        probabilities=core_probabilities.reshape(core_probabilities.shape[0], -1)[:, keep][
            :, order
        ],
    )
