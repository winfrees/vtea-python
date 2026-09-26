"""LayerCake3D: 2D objects found slice by slice, linked across z.

Ports `vtea.objects.Segmentation.LayerCake3DSingleThreshold` - the Java
VTEA's default segmentation, so existing protocols and published numbers
depend on it. It is kept for that reason (historical continuity), not
because it is the better segmentation: `label_components` + `watershed_split`
label in true 3D and are usually what a new protocol should use. The two
split touching nuclei differently, which is exactly why a Java analysis
cannot be reproduced without this one.

What the Java does, and what this does:

1. **Threshold** - every voxel at or above `low_threshold` is foreground
   (Java's "Low Threshold").
2. **Watershed, per slice** - Java runs ImageJ's binary "Watershed" on the
   stack, which works one 2D slice at a time: a Euclidean distance map, its
   maxima found with a tolerance of 0.5, and a one-pixel line drawn between
   the basins. Here: `ndimage.distance_transform_edt`, `h_maxima(edm, 0.5)`
   as markers, and `skimage.segmentation.watershed` with
   `watershed_line=True`. The lines are placed by a different flood, so an
   individual split can land a pixel away from ImageJ's.
3. **Regions** - each slice is labelled 8-connected (Java's recursive
   flood fill steps to all eight neighbours). Each region is located by the
   centre of its bounding box, `min + (max - min) // 2`, in integer pixels -
   `microRegion.calculateCenter`, which is the bounding-box centre and *not*
   the centroid.
4. **Linking** - regions are visited in z order. An unclaimed region starts
   a new object; from it, every unclaimed region in the next slice whose
   centre lies within `centroid_offset` pixels (Java's "Centroid Offset",
   `minConstants[2]`) of the current position joins it, and the search
   continues from there with the position moved to the midpoint of the two
   centres. This is a depth-first walk and can branch: two regions in the
   slice below that are both close enough both join. A single-slice image
   links regions *within* the slice the same way.
5. **Size filter** - objects outside `[min_size, max_size]` voxels are
   dropped (Java's "Min Vol" / "Max Vol"), and the rest numbered 1..n in
   the order they were created.

Deliberate differences from the Java, all in what the Java intended rather
than in what it computes when it works:

- The Java removes a claimed region from the list it is iterating over and
  then advances its index anyway, so the region that slid into that slot is
  never examined - neither as a link candidate nor, in the outer loop, as
  the start of an object. It is a loop bug, not a design, and it drops
  objects silently. Here every region is examined.
- The Java's region order within a slice comes from threads finishing in
  whatever order they finish, and from sort comparators that are not
  transitive, so it is not reproducible even between two Java runs. Here it
  is fixed: by z, then by the region's bounding-box centre (y, then x).
- The Java's 2D branch compares the start region against itself with a
  strict `< 10 * offset`, which excludes every region when the offset is 0.
  Here an offset of 0 links nothing and every region is its own object.

The consequence for parity is that objects are compared by centroid, not by
id (see tests/golden) - which is how they would have to be compared anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# ImageJ's binary watershed finds the distance map's maxima with a
# tolerance of 0.5 (EDM.java, MAXFINDER_TOLERANCE).
IMAGEJ_WATERSHED_TOLERANCE = 0.5

_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class SliceRegion:
    """One 2D region: which slice it is on, its label in that slice, and the
    centre of its bounding box, (y, x) in integer pixels."""

    z: int
    label: int
    center: tuple[int, int]


def watershed_slice(mask: np.ndarray, *, tolerance: float = IMAGEJ_WATERSHED_TOLERANCE) -> np.ndarray:
    """ImageJ-style binary watershed of one 2D mask, as a label image.

    Basins are separated by a one-pixel background line, as ImageJ draws
    them, so that the pieces are separate regions however they are later
    labelled.
    """
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)
    distance = ndi.distance_transform_edt(mask)
    peaks = h_maxima(distance, tolerance).astype(bool) & mask
    markers, _ = ndi.label(peaks, structure=_EIGHT_CONNECTED)
    if markers.max() <= 1:
        # One basin: nothing to split, and the flood would only draw lines
        # around the holes it cannot reach.
        labels, _ = ndi.label(mask, structure=_EIGHT_CONNECTED)
        return labels.astype(np.int32)
    return watershed(-distance, markers, mask=mask, watershed_line=True).astype(np.int32)


def slice_regions(mask_slice: np.ndarray, *, watershed: bool = True) -> np.ndarray:
    """The 2D regions of one slice, as a label image.

    With `watershed`, each basin is a region; the watershed line keeps them
    apart. Without, the slice's 8-connected components are.
    """
    if watershed:
        return watershed_slice(mask_slice)
    labels, _ = ndi.label(np.asarray(mask_slice, dtype=bool), structure=_EIGHT_CONNECTED)
    return labels.astype(np.int32)


def _bound_centres(labels_2d: np.ndarray) -> list[tuple[int, tuple[int, int]]]:
    """(label, (y, x) bounding-box centre) for every region of a 2D label
    image - `microRegion.calculateCenter`, integer division included."""
    found = []
    for index, box in enumerate(ndi.find_objects(labels_2d), start=1):
        if box is None:
            continue
        ys, xs = box
        # A slice's stop is one past the last pixel, so (max - min) is
        # (stop - 1) - start.
        cy = ys.start + (ys.stop - 1 - ys.start) // 2
        cx = xs.start + (xs.stop - 1 - xs.start) // 2
        found.append((index, (cy, cx)))
    return found


def _link(regions: list[SliceRegion], offset: float, *, planar: bool) -> list[list[int]]:
    """Group regions into objects, as lists of region indices.

    `regions` must be in visiting order. Across slices (`planar=False`) a
    region links to regions in the slice immediately after it; within a
    single slice (`planar=True`) to regions in the same slice. Candidates
    are looked up in a kd-tree per slice rather than by scanning every
    region, which is what keeps this linear-ish where the Java is quadratic.
    """
    by_slice: dict[int, list[int]] = {}
    for index, region in enumerate(regions):
        by_slice.setdefault(region.z, []).append(index)
    trees = {
        z: cKDTree(np.array([regions[i].center for i in members], dtype=float))
        for z, members in by_slice.items()
    }

    claimed = np.zeros(len(regions), dtype=bool)
    objects: list[list[int]] = []
    for start, region in enumerate(regions):
        if claimed[start]:
            continue
        claimed[start] = True
        members = [start]
        # Depth-first, as the Java's recursion is: (position, slice) pairs,
        # newest first. Each frame remembers which candidates it still has to
        # try, so that after one branch is followed to the end the others at
        # the same depth are still considered against the position they were
        # found from - which is what lets an object branch.
        stack = [(np.asarray(region.center, dtype=float), region.z)]
        while stack:
            position, z = stack.pop()
            target = z if planar else z + 1
            if target not in trees or offset <= 0:
                continue
            candidates = trees[target].query_ball_point(position, r=offset)
            pool = by_slice[target]
            for local in sorted(candidates):
                index = pool[local]
                if claimed[index]:
                    continue
                claimed[index] = True
                members.append(index)
                centre = np.asarray(regions[index].center, dtype=float)
                midpoint = (centre + position) / 2
                # Push the rest of this frame back first, so the branch just
                # found is explored before its siblings - the Java recursion's
                # order.
                stack.append((position, z))
                stack.append((midpoint, target))
                break
        objects.append(members)
    return objects


def layercake_3d(
    volume: np.ndarray,
    *,
    low_threshold: float = 0.0,
    centroid_offset: float = 5.0,
    min_size: int = 20,
    max_size: int = 1000,
    watershed: bool = True,
) -> np.ndarray:
    """LayerCake3D segmentation of a 2D or 3D image (z, y, x) into labels.

    The Java VTEA default. Parameters carry the Java names' meaning:
    `low_threshold` ("Low Threshold") in the image's own intensity units,
    `centroid_offset` ("Centroid Offset") in pixels, `min_size`/`max_size`
    ("Min Vol (vox)"/"Max Vol (vox)") in voxels, `watershed` ("Watershed").
    See the module docstring for exactly how each is used and where this
    differs from the Java.

    Returns an int32 label image the shape of `volume`, 0 for background and
    1..n for the objects in the order they were built.
    """
    volume = np.asarray(volume)
    if volume.ndim not in (2, 3):
        raise ValueError(
            f"layercake_3d segments a single-channel 2D or 3D image, got shape {volume.shape}; "
            f"pick a channel first"
        )
    if min_size is not None and max_size is not None and max_size < min_size:
        raise ValueError(f"max_size ({max_size}) is smaller than min_size ({min_size})")

    stack = volume[np.newaxis] if volume.ndim == 2 else volume
    mask = stack >= low_threshold

    slice_labels = np.zeros(stack.shape, dtype=np.int32)
    regions: list[SliceRegion] = []
    region_sizes: list[int] = []
    for z in range(stack.shape[0]):
        labels_2d = slice_regions(mask[z], watershed=watershed)
        slice_labels[z] = labels_2d
        counts = np.bincount(labels_2d.ravel())
        # Visiting order within a slice: by bounding-box centre, y then x -
        # fixed, unlike the Java's (see the module docstring).
        for label, centre in sorted(_bound_centres(labels_2d), key=lambda item: item[1]):
            regions.append(SliceRegion(z=z, label=label, center=centre))
            region_sizes.append(int(counts[label]))

    objects = _link(regions, float(centroid_offset), planar=stack.shape[0] == 1)
    sizes = np.asarray(region_sizes, dtype=np.int64)

    # One lookup table per slice, mapping region label -> object id, built
    # after the size filter so the surviving objects number 1..n.

    tables = [np.zeros(int(slice_labels[z].max()) + 1, dtype=np.int32) for z in range(stack.shape[0])]
    next_id = 0
    for members in objects:
        size = int(sizes[members].sum())
        if min_size is not None and size < min_size:
            continue
        if max_size is not None and size > max_size:
            continue
        next_id += 1
        for index in members:
            region = regions[index]
            tables[region.z][region.label] = next_id

    result = np.zeros(stack.shape, dtype=np.int32)
    for z in range(stack.shape[0]):
        result[z] = tables[z][slice_labels[z]]
    return result[0] if volume.ndim == 2 else result
