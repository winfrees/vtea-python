"""Image gates: which objects lie inside a region drawn on the image.

Every other gate in VTEA is drawn on the plot, over two measured features.
This one is drawn on the *image* - a napari Labels layer, painted by hand
around a tubule, a glomerulus, a region of interest - and asks the question
the plot cannot: not "which objects are bright" but "which objects are in
*there*".

It answers per object with the id of the region containing it (0 for none),
rather than with a boolean, for two reasons. A painted layer usually holds
several regions, and which one an object is in is the interesting part -
three tubules are three populations, not one. And an id joins onto the
napari layer's own colours, so a region and the objects inside it can be
drawn in the same colour without anyone maintaining a mapping.

Two ways to decide, because they answer slightly different questions:

- `CENTROID` (the default) - where the object's centre is. Cheap: it reads
  one voxel per object, so it costs nothing on a volume of any size, and it
  is the answer that matches how a person reads the picture.
- `MAJORITY` - which region the largest part of the object lies in, of the
  regions it touches at all, provided at least `minimum_fraction` of it is
  in there. Exact about objects straddling a boundary, at the cost of one
  pass over the labelled voxels. Background is not a candidate: the
  question an ROI asks is which region an object is in, and "mostly not in
  any" is what `minimum_fraction` is for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CENTROID = "centroid"
MAJORITY = "majority"
MODES = (CENTROID, MAJORITY)

# The column an image gate contributes to the measurement table, prefixed by
# the layer it came from (`roi_tubules`). A name rather than a bare number
# because it has to be readable in a class definition afterwards.
COLUMN_PREFIX = "roi_"

# Outside every region. Zero because that is what a label image already
# means by "background", so the two agree.
OUTSIDE = 0


def column_name(layer_name: str) -> str:
    """The table column an ROI layer's gate is stored under.

    Sanitised into something a class expression can name without backticks:
    `roi_tubules`, not `roi_Tubules (hand drawn)`.
    """
    cleaned = "".join(
        character if (character.isalnum() or character == "_") else "_"
        for character in str(layer_name).strip()
    )
    # Runs collapse: "Tubules (hand drawn)" is roi_Tubules_hand_drawn, not
    # roi_Tubules__hand_drawn__ - the name goes into class definitions, so
    # it has to be typeable from memory.
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return f"{COLUMN_PREFIX}{cleaned.strip('_') or 'layer'}"


def _match_shape(rois: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Line an ROI layer up with the segmentation it is being read against.

    A layer painted over a multi-channel image can carry a singleton channel
    axis the label image does not have (or the other way round), which is a
    shape mismatch that is not a disagreement about the data. Squeezing the
    singletons is safe and fixes the common case; anything else is a real
    mismatch and says so.
    """
    if rois.shape == shape:
        return rois
    squeezed = np.squeeze(rois)
    if squeezed.shape == shape:
        return squeezed
    if squeezed.ndim == len(shape) and all(
        size in (1, expected) for size, expected in zip(squeezed.shape, shape)
    ):
        return np.broadcast_to(squeezed, shape)
    raise ValueError(
        f"the ROI layer's shape {rois.shape} does not match the segmentation's {shape}; "
        f"they must be drawn on the same image"
    )


def objects_in_rois(
    labels: np.ndarray,
    rois: np.ndarray,
    *,
    object_ids=None,
    centroids=None,
    mode: str = CENTROID,
    minimum_fraction: float = 0.5,
) -> pd.DataFrame:
    """Which region each object is in.

    Returns a table of `object_id`, `roi` (0 = in none) and `fraction` (how
    much of the object is in that region - 1.0 for a centroid answer, which
    has nothing else to say).

    `minimum_fraction` applies to `MAJORITY` only, and defaults to half: an
    object is in the region that most of it is in. Lower it to catch objects
    that only reach into a region, raise it to insist on objects wholly
    inside one.

    `centroids` skips computing the object centres when the caller already
    measured them, which is the normal case: a measurement step has already
    produced `centroid-0`, `centroid-1`, ... and re-deriving them from the
    label image would be a pass over the volume to recover a number that is
    sitting in the table.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    labels = np.asarray(labels)
    rois = _match_shape(np.asarray(rois), labels.shape)

    if object_ids is None:
        object_ids = np.unique(labels)
        object_ids = object_ids[object_ids != 0]
    object_ids = np.asarray(object_ids)

    if mode == CENTROID:
        return _by_centroid(labels, rois, object_ids, centroids)
    return _by_majority(labels, rois, object_ids, minimum_fraction)


def _by_centroid(labels, rois, object_ids, centroids) -> pd.DataFrame:
    if centroids is None:
        centroids = _measure_centroids(labels, object_ids)
    centroids = np.asarray(centroids, dtype=float)
    if centroids.shape[0] != object_ids.size:
        raise ValueError(
            f"{centroids.shape[0]} centroids for {object_ids.size} objects; they must line up"
        )
    if centroids.shape[1] != rois.ndim:
        raise ValueError(
            f"centroids have {centroids.shape[1]} coordinates but the ROI layer is "
            f"{rois.ndim}-dimensional"
        )
    coordinates = np.rint(centroids).astype(int)
    for axis, size in enumerate(rois.shape):
        coordinates[:, axis] = np.clip(coordinates[:, axis], 0, size - 1)
    found = rois[tuple(coordinates[:, axis] for axis in range(rois.ndim))]
    return pd.DataFrame(
        {
            "object_id": object_ids,
            "roi": np.asarray(found).astype(int),
            "fraction": np.where(np.asarray(found) != OUTSIDE, 1.0, 0.0),
        }
    )


def _measure_centroids(labels: np.ndarray, object_ids: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    if object_ids.size == 0:
        return np.zeros((0, labels.ndim))
    centres = ndimage.center_of_mass(
        np.ones_like(labels, dtype=bool), labels=labels, index=object_ids
    )
    return np.asarray(centres, dtype=float).reshape(object_ids.size, labels.ndim)


def _by_majority(labels, rois, object_ids, minimum_fraction: float) -> pd.DataFrame:
    """The region the largest part of each object lies in.

    One pass over the labelled voxels, counted with a groupby rather than a
    loop over objects: a per-object loop is a pass over the volume per
    object, which at ten thousand nuclei is the difference between a second
    and an afternoon.

    The fraction is of the whole object, background included, so "half of
    this nucleus is in the tubule" is a statement about the nucleus. Only
    real regions compete for it - an object 40% inside a tubule is 40%
    inside a tubule whether or not the other 60% is inside anything - and
    `minimum_fraction` is what decides whether that counts.
    """
    inside = labels != 0
    pairs = pd.DataFrame(
        {"object_id": np.asarray(labels[inside]).ravel(), "roi": np.asarray(rois[inside]).ravel()}
    )
    empty = pd.DataFrame(
        {
            "object_id": object_ids,
            "roi": np.zeros(object_ids.size, dtype=int),
            "fraction": np.zeros(object_ids.size),
        }
    )
    if pairs.empty:
        return empty
    counts = pairs.value_counts().rename("voxels").reset_index()
    sizes = counts.groupby("object_id")["voxels"].transform("sum")
    counts["fraction"] = counts["voxels"] / sizes
    counts = counts[counts["roi"] != OUTSIDE]
    if counts.empty:
        return empty
    # The largest overlap per object; ties go to the lower region id so the
    # answer does not depend on row order.
    counts = counts.sort_values(["object_id", "voxels", "roi"], ascending=[True, False, True])
    best = counts.drop_duplicates("object_id").set_index("object_id")
    roi = best["roi"].reindex(object_ids, fill_value=OUTSIDE).to_numpy().astype(int)
    fraction = best["fraction"].reindex(object_ids, fill_value=0.0).to_numpy()
    if minimum_fraction > 0:
        roi = np.where(fraction >= minimum_fraction, roi, OUTSIDE)
    return pd.DataFrame({"object_id": object_ids, "roi": roi, "fraction": fraction})


def image_gate(
    labels: np.ndarray,
    rois: np.ndarray,
    *,
    object_ids=None,
    centroids=None,
    mode: str = CENTROID,
    minimum_fraction: float = 0.5,
) -> np.ndarray:
    """`objects_in_rois` as one array: the region id per object, in the
    order of `object_ids`.

    This is the column an image gate contributes to the measurement table -
    non-zero means "in a region", and the value says which one, so a class
    can be `roi_tubules == 2` as easily as `roi_tubules`.
    """
    table = objects_in_rois(
        labels,
        rois,
        object_ids=object_ids,
        centroids=centroids,
        mode=mode,
        minimum_fraction=minimum_fraction,
    )
    return table["roi"].to_numpy()


def centroids_from_frame(frame: pd.DataFrame, ndim: int, prefix: str = "") -> np.ndarray | None:
    """The centroid columns a measurement step already produced, as an
    (n_objects, ndim) array - or None when the table has no centroids.

    `prefix` is for a per-cell table, whose centroids are namespaced by the
    segmentation they were measured on.
    """
    names = [f"{prefix}centroid-{axis}" for axis in range(ndim)]
    if not all(name in frame.columns for name in names):
        return None
    return frame[names].to_numpy(dtype=float)
