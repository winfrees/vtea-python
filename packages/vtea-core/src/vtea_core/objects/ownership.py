"""Which cell owns a voxel, when the honest answer is "probably that one".

A plasma membrane that the stain barely resolves, two cells pressed
together, two masks that overlap: in all three the label image asserts an
answer it does not have. `watershed_ownership` gives a usable one and says
nothing about how close the call was. This module keeps the doubt.

**The representation is the design decision here.** A dense cell x voxel
posterior is out of the question - 2,000 cells over a 2048x2048x24 volume is
about 10^11 floats - so what is stored per voxel is the best *k* owners and
their probabilities, k=2 or 3. That is a few times the size of the label
image, and it captures essentially all the real ambiguity: a voxel genuinely
contested by four cells is rare, and is not usefully resolved by knowing the
fourth-place probability. A voxel's remaining mass is simply not
represented, which is the trade being made and is worth stating plainly.

What comes out of it:

- `hard()` - the argmax, an ordinary label image, so everything that already
  takes labels keeps working.
- `confidence()` - the winning probability per voxel, which is the map that
  makes contested regions visible instead of merely present.
- the weights that `vtea_core.measurements.weighted` measures with, where a
  mean becomes a probability-weighted mean and a count becomes an expected
  volume.
- `override()`, because every automated assignment is wrong somewhere and an
  analysis nobody can correct is one nobody can publish.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from vtea_core.data.spacing import Spacing

OWNERSHIP_FORMAT_VERSION = 1

# How the ownership was arrived at, recorded so a result can say whether a
# voxel was decided by an algorithm or by a person.
DISTANCE = "distance"
WATERSHED = "watershed"
MANUAL = "manual"


@dataclass
class Ownership:
    """The best `k` owners of every voxel, and how sure each is.

    `owners[0]` is the winner everywhere, so `owners[0]` *is* the hard label
    image and `probabilities[0]` is the confidence map. Slots are ordered by
    probability, and a slot with owner 0 is empty rather than owned by
    object 0.

    `manual` marks the voxels a person set by hand, so a correction stays
    distinguishable from an inference forever after.
    """

    owners: np.ndarray  # (k, *shape) integer label ids, 0 = unowned
    probabilities: np.ndarray  # (k, *shape) float in [0, 1]
    segmentation: str = ""
    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    manual: np.ndarray | None = None

    def __post_init__(self):
        if self.owners.shape != self.probabilities.shape:
            raise ValueError(
                f"owners {self.owners.shape} and probabilities "
                f"{self.probabilities.shape} must have the same shape"
            )
        if self.manual is None:
            self.manual = np.zeros(self.shape, dtype=bool)

    @classmethod
    def from_labels(cls, labels: np.ndarray, *, segmentation: str = "", method: str = "") -> Ownership:
        """A hard label image as a certain ownership.

        So that a watershed result and a probabilistic one can be measured by
        the same code, and the difference between them is visible in the
        numbers rather than in which function was called.
        """
        labels = np.asarray(labels)
        return cls(
            owners=labels[np.newaxis].astype(np.int32),
            probabilities=(labels != 0)[np.newaxis].astype(float),
            segmentation=segmentation,
            method=method or WATERSHED,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.owners.shape[1:]

    @property
    def top_k(self) -> int:
        return self.owners.shape[0]

    def hard(self) -> np.ndarray:
        """The winner per voxel, as an ordinary label image."""
        return self.owners[0].copy()

    def confidence(self) -> np.ndarray:
        """The winning probability per voxel - the map that makes a contested
        boundary visible rather than merely present."""
        return self.probabilities[0].copy()

    def margin(self) -> np.ndarray:
        """How much better the winner was than the runner-up. Near zero is a
        coin toss, which is a different thing from a low absolute
        probability (a voxel weakly owned by one cell and nobody else)."""
        if self.top_k < 2:
            return self.confidence()
        return self.probabilities[0] - self.probabilities[1]

    def contested(self, threshold: float = 0.9) -> np.ndarray:
        """Owned voxels whose winner did not clear `threshold` - what to show
        a person, and what to count when reporting how much of a field the
        method was unsure about."""
        return (self.owners[0] != 0) & (self.probabilities[0] < threshold)

    def object_ids(self) -> list[int]:
        return sorted({int(value) for value in np.unique(self.owners) if value != 0})

    def weights(self, object_id: int) -> np.ndarray:
        """This owner's probability at every voxel - what a weighted
        measurement integrates over."""
        weights = np.zeros(self.shape, dtype=float)
        for slot in range(self.top_k):
            claimed = self.owners[slot] == object_id
            weights[claimed] = self.probabilities[slot][claimed]
        return weights

    def override(self, region: np.ndarray, object_id: int) -> None:
        """Give `region` to `object_id` outright, and remember that a person
        said so.

        Recorded rather than merely applied: a correction that becomes
        indistinguishable from an inference is worse than no correction at
        all, because nobody reading the result later can tell which voxels
        were reviewed.
        """
        region = np.asarray(region).astype(bool)
        if region.shape != self.shape:
            raise ValueError(f"shapes differ: {region.shape} != {self.shape}")
        self.owners[0][region] = int(object_id)
        self.probabilities[0][region] = 1.0
        for slot in range(1, self.top_k):
            self.owners[slot][region] = 0
            self.probabilities[slot][region] = 0.0
        self.manual[region] = True

    def summary(self, threshold: float = 0.9) -> str:
        owned = int((self.owners[0] != 0).sum())
        contested = int(self.contested(threshold).sum())
        overridden = int(self.manual.sum())
        parts = [f"{len(self.object_ids())} owners over {owned} voxels"]
        if owned:
            parts.append(f"{contested} contested ({100 * contested / owned:.1f}%)")
        if overridden:
            parts.append(f"{overridden} set by hand")
        return ", ".join(parts)

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Arrays as compressed npz, with the provenance beside them. Not
        JSON: this is image-sized data, and the point of the top-k form is
        that it is small enough to keep."""
        path = Path(path)
        np.savez_compressed(
            path,
            owners=self.owners,
            probabilities=self.probabilities,
            manual=self.manual,
            meta=np.array(
                json.dumps(
                    {
                        "vtea_ownership_version": OWNERSHIP_FORMAT_VERSION,
                        "segmentation": self.segmentation,
                        "method": self.method,
                        "params": self.params,
                    }
                )
            ),
        )
        return path if path.suffix else path.with_suffix(".npz")


def load_ownership(path: str | Path) -> Ownership:
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        version = meta.get("vtea_ownership_version")
        if version is not None and version > OWNERSHIP_FORMAT_VERSION:
            raise ValueError(
                f"ownership file version {version} is newer than this VTEA understands "
                f"({OWNERSHIP_FORMAT_VERSION}); upgrade vtea-core to read it"
            )
        return Ownership(
            owners=data["owners"],
            probabilities=data["probabilities"],
            segmentation=meta.get("segmentation", ""),
            method=meta.get("method", ""),
            params=meta.get("params", {}),
            manual=data["manual"],
        )


def _sampling(shape, spacing: Spacing | None) -> tuple[float, ...]:
    if spacing is None or not spacing.is_known:
        return (1.0,) * len(shape)
    return spacing.for_ndim(len(shape))


def distance_ownership(
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    spacing: Spacing | None = None,
    falloff: float = 2.0,
    reach: float | None = None,
    top_k: int = 2,
    segmentation: str = "",
) -> Ownership:
    """A posterior over nearby cells for every voxel of `mask`.

    Each marker's claim on a voxel falls off as `exp(-d / falloff)` with the
    distance from that marker's own surface, physical wherever the voxel size
    is known. `falloff` is therefore the distance over which a cell's claim
    drops to about a third: small values give a near-hard split at the
    midline, large ones a broad zone of genuine uncertainty. `reach` is where
    a marker stops being a candidate at all, four falloffs by default, where
    its claim is under 2%.

    Unlike `watershed_ownership` this says how close each call was, at the
    cost of not following the region's shape - a claim reaches through a thin
    wall as easily as around it. The two are complementary, which is why both
    are here: watershed for a clean split of a well-defined region, this for
    a boundary the stain never really resolved.

    Cost is proportional to the objects rather than to the volume times the
    number of markers: each marker is evaluated inside its own bounding box
    grown by `reach`.
    """
    markers = np.asarray(labels)
    region = np.asarray(mask) != 0
    if markers.shape != region.shape:
        raise ValueError(f"shapes differ: {markers.shape} != {region.shape}")
    if falloff <= 0:
        raise ValueError(f"falloff must be positive, got {falloff}")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    limit = 4.0 * falloff if reach is None else float(reach)
    if limit <= 0:
        raise ValueError(f"reach must be positive, got {reach}")

    sampling = _sampling(markers.shape, spacing)
    margin = [int(np.ceil(limit / size)) for size in sampling]

    owners = np.zeros((top_k, *markers.shape), dtype=np.int32)
    scores = np.zeros((top_k, *markers.shape), dtype=float)
    total = np.zeros(markers.shape, dtype=float)

    for index, box in enumerate(ndi.find_objects(markers)):
        if box is None:
            continue
        marker_id = index + 1
        window = tuple(
            slice(max(0, axis.start - pad), min(extent, axis.stop + pad))
            for axis, pad, extent in zip(box, margin, markers.shape)
        )
        local_region = region[window]
        if not local_region.any():
            continue

        distance = ndi.distance_transform_edt(markers[window] != marker_id, sampling=sampling)
        claim = np.where(local_region & (distance <= limit), np.exp(-distance / falloff), 0.0)
        if not claim.any():
            continue

        total[window] += claim
        _insert_claim(owners, scores, window, marker_id, claim)

    with np.errstate(invalid="ignore", divide="ignore"):
        probabilities = np.where(total > 0, scores / total, 0.0)
    owners[probabilities <= 0] = 0

    return Ownership(
        owners=owners,
        probabilities=probabilities,
        segmentation=segmentation,
        method=DISTANCE,
        params={"falloff": float(falloff), "reach": limit, "top_k": int(top_k)},
    )


def _insert_claim(owners, scores, window, marker_id: int, claim: np.ndarray) -> None:
    """Slot one marker's claim into the running top-k, keeping slots ordered.

    Each voxel is placed exactly once, at the first slot it beats; the slots
    below it shift down. Doing this as the markers are visited is what avoids
    ever holding a cell x voxel array.
    """
    top_k = owners.shape[0]
    remaining = claim > 0
    for slot in range(top_k):
        place = remaining & (claim > scores[slot][window])
        if not place.any():
            continue
        for lower in range(top_k - 1, slot, -1):
            scores[lower][window][place] = scores[lower - 1][window][place]
            owners[lower][window][place] = owners[lower - 1][window][place]
        scores[slot][window][place] = claim[place]
        owners[slot][window][place] = marker_id
        remaining = remaining & ~place
        if not remaining.any():
            return
