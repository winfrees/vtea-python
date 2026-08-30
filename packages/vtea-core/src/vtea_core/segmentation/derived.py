"""Segmentations built from another segmentation by morphology.

A nuclear envelope is a shell straddling the nuclear boundary; a cytosol
compartment is a band outside it. Neither needs any intensity information -
only the nuclear mask and a thickness - and because each derived object is
grown from exactly one parent, the association between them is exact by
construction rather than inferred. Every function here preserves label
identity for that reason: derived object *k* came from parent *k*, so
`vtea_core.objects.associate_by_identity` can state the relationship
without guessing.

**Thicknesses are physical when a `spacing` is given, and in voxels when it
is not.** This is the whole reason `Spacing` exists: confocal z-steps are
routinely several times the lateral pixel size, so a band of "5" is a
sphere in index space and a flattened disc in the specimen. scipy's distance
transform takes a `sampling` argument, which is what makes the physical
reading possible; skimage's own `expand_labels` does not, which is why these
are implemented here rather than called through to it.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries, watershed

from vtea_core.data.spacing import Spacing


def _sampling(labels: np.ndarray, spacing: Spacing | None):
    """Physical voxel sizes for the distance transform, or None to work in
    voxels. An unknown spacing is None rather than ones, so a caller that
    forgot to set one gets voxel behaviour rather than a silent claim that
    the volume is isotropic."""
    if spacing is None or not spacing.is_known:
        return None
    return spacing.for_ndim(labels.ndim)


def _require_labels(labels: np.ndarray, name: str = "labels") -> np.ndarray:
    array = np.asarray(labels)
    if not np.issubdtype(array.dtype, np.integer) and array.dtype != bool:
        raise TypeError(f"{name} must be a label image (integer or boolean), got {array.dtype}")
    return array.astype(np.int32, copy=False) if array.dtype == bool else array


def expand_labels(
    labels: np.ndarray, distance: float = 1.0, *, spacing: Spacing | None = None
) -> np.ndarray:
    """Grow every label outward by `distance`, stopping where two meet.

    The original objects are kept, so this is the object plus a band around
    it. Where two objects would claim the same voxel, the nearer one takes
    it, which is what stops neighbouring nuclei from growing through each
    other.
    """
    labels = _require_labels(labels)
    if distance < 0:
        raise ValueError(f"distance must not be negative, got {distance}")
    if distance == 0:
        return labels.copy()

    distances, nearest = ndi.distance_transform_edt(
        labels == 0, sampling=_sampling(labels, spacing), return_indices=True
    )
    grown = labels.copy()
    reachable = (labels == 0) & (distances <= distance)
    grown[reachable] = labels[tuple(index[reachable] for index in nearest)]
    return grown


def label_ring(
    labels: np.ndarray, thickness: float = 1.0, *, spacing: Spacing | None = None
) -> np.ndarray:
    """The band *outside* each object, and nothing else - a cytosol
    compartment around a nucleus.

    Each ring carries its parent's label, so ring *k* is the cytosol of
    nucleus *k*. Rings stop where they meet, so two nuclei do not share
    cytosol.
    """
    grown = expand_labels(labels, thickness, spacing=spacing)
    return np.where(_require_labels(labels) == 0, grown, 0)


def label_shell(
    labels: np.ndarray,
    inward: float = 0.0,
    outward: float = 1.0,
    *,
    spacing: Spacing | None = None,
) -> np.ndarray:
    """A shell straddling each object's boundary - a nuclear envelope.

    `inward` reaches into the object from its edge, `outward` out of it.
    Either may be zero: `inward=0, outward=1` is a band just outside the
    nucleus, `inward=1, outward=0` a rim just inside it.

    The inward part is measured from the boundary a label has with *anything
    that is not itself* - background or a different label - so two nuclei
    that touch each still get an envelope along the face they share, which
    a plain distance-to-background would miss.
    """
    labels = _require_labels(labels)
    if inward < 0 or outward < 0:
        raise ValueError(f"thicknesses must not be negative, got {inward=}, {outward=}")

    shell = np.zeros_like(labels)
    if outward > 0:
        shell = label_ring(labels, outward, spacing=spacing)

    if inward > 0 or outward == 0:
        boundary = find_boundaries(labels, mode="inner")
        to_boundary = ndi.distance_transform_edt(~boundary, sampling=_sampling(labels, spacing))
        rim = (labels != 0) & (to_boundary <= inward)
        shell = np.where(rim, labels, shell)
    return shell


def watershed_ownership(
    labels: np.ndarray, mask: np.ndarray, *, spacing: Spacing | None = None
) -> np.ndarray:
    """Divide a region among the objects inside it, one owner per voxel.

    The standard answer to "split this cytoplasm between those two nuclei",
    and the deterministic baseline for the whole question of contested area:
    every voxel of `mask` goes to exactly one object of `labels`, so a
    territory carries its owner's id and `associate_by_identity` links the
    two exactly.

    The division follows the region's own shape - the flood is over the
    negated distance transform, so territories meet at the narrowest waist
    between them rather than halfway along a straight line, which is what
    makes it a reasonable answer for touching cells. That distance is
    physical wherever the voxel size is known, so a thin neck in z is not
    mistaken for a wide one.

    `mask` may be a boolean mask or another label image; only whether each
    voxel belongs to the region is used. A marker lying outside the region
    gets no territory, and a region with no marker in it is left as
    background - both are results worth seeing rather than errors.

    An answer without a posterior: it says which cell owns a voxel and never
    that it was a close call. The probabilistic version, and the confidence
    map that goes with it, are Phase 4 in docs/OBJECT_ASSOCIATION.md.
    """
    labels = _require_labels(labels)
    region = np.asarray(mask) != 0
    if labels.shape != region.shape:
        raise ValueError(f"shapes differ: {labels.shape} != {region.shape}")

    elevation = -ndi.distance_transform_edt(region, sampling=_sampling(labels, spacing))
    markers = np.where(region, labels, 0)
    return watershed(elevation, markers=markers, mask=region).astype(labels.dtype, copy=False)


def subtract_labels(labels: np.ndarray, other: np.ndarray) -> np.ndarray:
    """`labels` with everything `other` occupies removed, ids preserved.

    A cytoplasm segmented across a whole cell minus the nuclei inside it is
    the cytosol, without needing to derive it from the nuclei.
    """
    labels = _require_labels(labels)
    other = _require_labels(other, "other")
    if labels.shape != other.shape:
        raise ValueError(f"shapes differ: {labels.shape} != {other.shape}")
    return np.where(other != 0, 0, labels)


def restrict_labels_to(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """`labels` kept only where `mask` is set, ids preserved.

    The counterpart of subtract: organelle puncta restricted to inside a
    cytoplasm, dropping anything detected outside a cell.
    """
    labels = _require_labels(labels)
    mask = np.asarray(mask)
    if labels.shape != mask.shape:
        raise ValueError(f"shapes differ: {labels.shape} != {mask.shape}")
    return np.where(mask != 0, labels, 0)
