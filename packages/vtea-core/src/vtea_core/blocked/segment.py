"""Segmenting a volume one tile at a time, and getting one object list out.

The step that Phase L2 refused. `label_components` run per tile gives every
tile its own object 1; what makes a tiled segmentation usable is everything
that happens after the function returns - provisional ids that cannot
collide, a correspondence between the tiles' views of the seam, a decision
about each object a seam ran through, and a record of which decision was
made and on what evidence.

The shape of it, in four passes over the data and one over a table:

1. **Segment and catalogue.** Each tile segments its core plus its halo and
   writes its labelling *of its own core* into a provisional array. Every
   voxel therefore has exactly one provisional id, from the tile responsible
   for it, and no two tiles can collide. Each fragment is catalogued at the
   same time: how many voxels it holds, where its bounding box is, which
   tile faces it reaches, and - the check that the whole scheme rests on -
   whether it reached the outer edge of its halo, which proves the tile did
   not contain it.
2. **Match.** By overlap, centroid or contact, per the policy. See
   `vtea_core.blocked.reconcile`.
3. **Group and decide.** Pairwise matches close into groups by union-find,
   and each group becomes one object - or, under `flag`, stays several that
   know about each other.
4. **Write.** A lookup table maps every provisional id to its final one, one
   tile at a time. Under `own`, the winning tile's complete copy is then
   written over the group's extent, which is the difference between keeping
   one tile's coherent object and stitching two half-objects together.

Nothing here holds more than one tile at a time, and the label array it
produces can be larger than memory - it is a Zarr array in scratch, written
region by region, like every other blocked output.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from vtea_core.blocked.executor import read_block
from vtea_core.blocked.plan import Tile, TilePlan
from vtea_core.blocked.reconcile import (
    CENTROID,
    DEFAULT_POLICY,
    DROP,
    FLAG,
    NO_MATCHING,
    OVERLAP,
    OWN,
    RAISE_EXCEEDED,
    RESEGMENT,
    TOUCHING,
    UNCUT,
    Fragment,
    HaloExceeded,
    LabelLedger,
    SeamPolicy,
    centroid_pairs,
    group_fragments,
    overlap_pairs,
    touching_pairs,
)
from vtea_core.blocked.store import ZarrScratch

# The provisional id space. int32 is 2.1 billion objects, which is far more
# than any real dataset and half the storage of int64 for an array the size
# of the image.
LABEL_DTYPE = np.int32
MAX_LABEL = np.iinfo(LABEL_DTYPE).max


class TooManyObjects(RuntimeError):
    """More objects than the label dtype can number."""


@dataclass
class BlockedLabels:
    """A label array larger than memory, and the account of how it was made."""

    array: Any
    ledger: LabelLedger
    plan: TilePlan
    policy: SeamPolicy

    @property
    def n_objects(self) -> int:
        return self.ledger.n_objects

    def describe(self) -> str:
        return f"{self.ledger.describe()}; {self.plan.n_tiles:,} tiles"


AXIS_NAMES = "zyx"


def _face_name(axis: int, ndim: int, upper: bool) -> str:
    """A tile face, as "-z" / "+x". Named rather than numbered so a ledger
    entry can be read without counting axes."""
    letter = AXIS_NAMES[axis - (ndim - len(AXIS_NAMES))] if ndim <= len(AXIS_NAMES) else str(axis)
    return f"{'+' if upper else '-'}{letter}"


def segment_blocked(
    function: Callable[..., np.ndarray],
    sources: Mapping[str, Any],
    *,
    plan: TilePlan,
    scratch: ZarrScratch,
    policy: SeamPolicy = DEFAULT_POLICY,
    params: Mapping[str, Any] | None = None,
    spacing: Any = None,
    name: str = "labels",
    manifest: Any = None,
    progress: Callable[[int, int], None] | None = None,
) -> BlockedLabels:
    """Segment a volume tile by tile and reconcile the objects across seams.

    `function` is an ordinary segmentation function - `label_components`,
    `watershed_split`, anything that takes arrays and returns an integer
    label image of the same shape. It is not told that it is being tiled and
    does not need to be.

    `manifest` is a `vtea_core.blocked.resume.RunManifest` - or a path to
    one - and makes the segmentation pass resumable: tiles already recorded
    there are skipped, and each new one is recorded as it finishes. It needs
    a scratch store that outlives the process (`ZarrScratch(keep=True)`),
    since the manifest records what was done and the store holds it. Worth
    the bookkeeping for an inference run measured in hours, and pointless
    for one measured in seconds.
    """
    params = dict(params or {})
    effective_plan = _plan_for_policy(plan, policy)

    manifest = _open_manifest(manifest, effective_plan, policy, function)
    provisional = _scratch_array(
        scratch, f"{name}__provisional", effective_plan, resuming=manifest is not None
    )
    fragments, tile_arrays = _segment_pass(
        function,
        sources,
        effective_plan,
        policy,
        params,
        provisional,
        scratch,
        name,
        progress,
        manifest,
    )
    pairs = _match(
        scratch, fragments, tile_arrays, provisional, effective_plan, policy, spacing
    )
    ledger, assignment = _resolve(fragments, pairs, policy)
    array = _write_final(
        provisional, assignment, ledger, tile_arrays, effective_plan, policy, scratch, name
    )
    if policy.resolution == RESEGMENT:
        _resegment_cut_objects(
            array, ledger, function, sources, params, effective_plan, policy, progress
        )
    _check_exceeded(ledger, policy)
    if manifest is not None:
        # A resumable run keeps its working arrays. The process can die
        # between the segmentation pass and the matching that reads those
        # arrays, and a resume that had thrown them away would have to
        # re-run every inference to get them back - which is the cost the
        # manifest exists to avoid. The caller owns the scratch store in
        # that case (it had to pass `keep=True` to get here) and owns
        # cleaning it up once the result is somewhere durable.
        manifest.close()
    else:
        for tile_name in tile_arrays.values():
            scratch.drop(tile_name)
        scratch.drop(f"{name}__provisional")
    return BlockedLabels(array=array, ledger=ledger, plan=effective_plan, policy=policy)


def _plan_for_policy(plan: TilePlan, policy: SeamPolicy) -> TilePlan:
    """The plan this policy actually runs on.

    An abutting strategy removes the halo, and that is the whole of what
    "no tile overlap" means here. Note what it does *not* do: a filter's
    halo is a boundary condition of the filter and belongs to the step, not
    to the seam policy - only a segmentation's plan comes through this.
    """
    from dataclasses import replace as _replace

    if policy.is_overlapping:
        if policy.halo is None:
            return plan
        return _replace(plan, halo=tuple(policy.halo if axis in plan.tiled_axes else 0
                                         for axis in range(plan.ndim)))
    return _replace(plan, halo=(0,) * plan.ndim)


def _open_manifest(manifest: Any, plan: TilePlan, policy: SeamPolicy, function) -> Any:
    """A manifest for this run, from one already open or from a path."""
    if manifest is None:
        return None
    from vtea_core.blocked.resume import RunManifest, plan_signature

    signature = plan_signature(plan, policy, getattr(function, "__name__", str(function)))
    if isinstance(manifest, RunManifest):
        if manifest.signature != signature:
            from vtea_core.blocked.resume import ManifestMismatch, _describe_difference

            raise ManifestMismatch(
                f"this manifest records a different run: "
                f"{_describe_difference(manifest.signature, signature)}"
            )
        return manifest
    return RunManifest.start(manifest, signature)


def _scratch_array(scratch: ZarrScratch, key: str, plan: TilePlan, *, resuming: bool) -> Any:
    """The provisional label array, reused when resuming.

    Creating it afresh would zero what a previous run had already written,
    which is the one thing a resume must not do.
    """
    if resuming and key in scratch:
        return scratch.open(key)
    return scratch.create(key, shape=plan.shape, dtype=LABEL_DTYPE)


def _segment_pass(
    function, sources, plan, policy, params, provisional, scratch, name, progress, manifest=None
) -> tuple[list[Fragment], dict[tuple[int, ...], str]]:
    """Pass 1: segment every tile, number the objects so they cannot
    collide, and catalogue what each tile saw."""
    fragments: list[Fragment] = []
    tile_arrays: dict[tuple[int, ...], str] = {}
    next_id = 1
    if manifest is not None:
        fragments.extend(manifest.fragments())
        next_id = manifest.next_id

    for index, tile in enumerate(plan.tiles()):
        if policy.needs_tile_labels:
            tile_arrays[tile.index] = f"{name}__tile{index}"
        if manifest is not None and manifest.is_done(tile.index):
            # Already segmented, its labels already in the provisional array
            # and its fragments already read back from the manifest.
            if progress is not None:
                progress(index + 1, plan.n_tiles)
            continue
        blocks = {
            key: read_block(array, tile, policy.pad_mode) for key, array in sources.items()
        }
        labels = np.asarray(function(**blocks, **params))
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError(
                f"{getattr(function, '__name__', function)} returned {labels.dtype}; a "
                f"segmentation step must return an integer label image"
            )
        local_max = int(labels.max()) if labels.size else 0
        if next_id + local_max > MAX_LABEL:
            raise TooManyObjects(
                f"this dataset has more than {MAX_LABEL:,} objects, which is more than "
                f"the label array can number. Filter by size during segmentation, or "
                f"segment a region at a time."
            )

        base = next_id - 1
        offset = np.where(labels > 0, labels.astype(np.int64) + base, 0).astype(LABEL_DTYPE)
        inner = tile.inner if policy.pad_mode is not None else tile.inner_unpadded
        provisional[tile.core] = offset[inner]
        catalogued = _catalogue(offset, tile, plan, base, local_max, inner)
        fragments.extend(catalogued)
        if policy.needs_tile_labels:
            _store_tile(scratch, name, index, offset)

        next_id += local_max
        if manifest is not None:
            # After the data is written, never before: a manifest entry for
            # a tile whose labels are not in the store would make a resumed
            # run skip work it has not done.
            manifest.record(tile.index, catalogued, next_id)
        if progress is not None:
            progress(index + 1, plan.n_tiles)
    return fragments, tile_arrays


def _store_tile(scratch: ZarrScratch, name: str, index: int, offset: np.ndarray) -> str:
    """Keep this tile's own labelling of its whole block.

    Only overlap matching and the `own` resolution need it - one to compare
    two tiles' answers for the same voxels, the other to write the winner's
    complete copy out. It is the storage cost of an overlapping tiling, and
    it is the same factor as the read amplification the plan reports.
    """
    key = f"{name}__tile{index}"
    scratch.put(key, offset, axes="ZYX"[-offset.ndim :] if offset.ndim <= 3 else "CZYX")
    return key


def _catalogue(
    offset: np.ndarray,
    tile: Tile,
    plan: TilePlan,
    base: int,
    local_max: int,
    inner: tuple[slice, ...],
) -> list[Fragment]:
    """Every object this tile saw, in the volume's coordinates.

    Two economies worth stating, because a naive version of this is what
    makes a tiled segmentation slower than the segmentation. Bounding boxes
    come from one `find_objects` call over the block rather than a pass per
    object. And a true centroid - which needs the object's own voxels - is
    computed only for the fragments that could possibly matter to a seam,
    identified first by the cheap and conservative test of whether their
    bounding box reaches a tile face at all.
    """
    if local_max <= 0:
        return []
    core = offset[inner]
    block_counts = np.bincount(offset.reshape(-1), minlength=base + local_max + 1)
    core_counts = np.bincount(core.reshape(-1), minlength=base + local_max + 1)
    boxes = ndi.find_objects(offset, max_label=base + local_max)
    origin = tile.origin

    fragments = []
    for local_id in range(1, local_max + 1):
        provisional_id = base + local_id
        box = boxes[provisional_id - 1]
        if box is None or core_counts[provisional_id] == 0:
            # Either nothing, or an object that lies wholly in this tile's
            # halo - which means it belongs to a neighbour's core and that
            # neighbour is cataloguing it.
            continue
        bbox = tuple(
            (int(part.start + start), int(part.stop + start))
            for part, start in zip(box, origin)
        )
        fragments.append(
            Fragment(
                tile=tile.index,
                local_id=local_id,
                provisional_id=int(provisional_id),
                core_voxels=int(core_counts[provisional_id]),
                block_voxels=int(block_counts[provisional_id]),
                centroid=_centroid(offset, box, provisional_id, origin),
                bbox=bbox,
                faces=_faces(bbox, tile, plan),
                at_dataset_border=_at_dataset_border(bbox, plan),
                exceeded_halo=_exceeded_halo(bbox, tile, plan),
            )
        )
    return fragments


def _centroid(
    offset: np.ndarray, box: tuple[slice, ...], provisional_id: int, origin: Sequence[int]
) -> tuple[float, ...]:
    """The object's centre of mass, in the volume's coordinates.

    Computed from the bounding box rather than the block, so the cost is the
    object's size and not the tile's.
    """
    mask = offset[box] == provisional_id
    local = ndi.center_of_mass(mask)
    return tuple(
        float(value) + float(part.start) + float(start)
        for value, part, start in zip(local, box, origin)
    )


def _faces(bbox, tile: Tile, plan: TilePlan) -> frozenset[str]:
    """Which tile faces this fragment reaches - and only faces that are
    genuinely shared with another tile. The edge of the specimen is not a
    seam, and confusing the two is how "exclude edge objects" turns from a
    cytometry choice into a bug."""
    faces = set()
    for axis, ((start, stop), core) in enumerate(zip(bbox, tile.core)):
        if start <= core.start and core.start > 0:
            faces.add(_face_name(axis, plan.ndim, upper=False))
        if stop >= core.stop and core.stop < plan.shape[axis]:
            faces.add(_face_name(axis, plan.ndim, upper=True))
    return frozenset(faces)


def _at_dataset_border(bbox, plan: TilePlan) -> bool:
    return any(
        start <= 0 or stop >= plan.shape[axis] for axis, (start, stop) in enumerate(bbox)
    )


def _exceeded_halo(bbox, tile: Tile, plan: TilePlan) -> bool:
    """Whether this object reached the outer edge of the block it was
    segmented in, which proves the tile did not contain it.

    A measurement, not an estimate, and the reason "exact given a sufficient
    halo" is worth saying at all. A block edge that is also the edge of the
    specimen does not count: there is nothing beyond it to have missed.
    """
    for axis, ((start, stop), padded) in enumerate(zip(bbox, tile.padded)):
        if start <= padded.start and padded.start > 0:
            return True
        if stop >= padded.stop and padded.stop < plan.shape[axis]:
            return True
    return False


def _match(
    scratch: ZarrScratch,
    fragments: Sequence[Fragment],
    tile_arrays: Mapping[tuple[int, ...], str],
    provisional: Any,
    plan: TilePlan,
    policy: SeamPolicy,
    spacing: Any,
) -> list[tuple[int, int, float]]:
    """Pass 2: which fragments in different tiles are the same object."""
    if policy.matching == NO_MATCHING:
        return []
    if policy.matching == OVERLAP:
        return _overlap_matches(scratch, tile_arrays, plan, policy)
    if policy.matching == TOUCHING:
        return _touching_matches(provisional, plan)
    if policy.matching == CENTROID:
        distance = policy.max_centroid_distance
        if distance is None:
            # Half the halo is the defensible default: two fragments of one
            # object cannot be further apart than the object, and the halo
            # was sized to hold the object.
            distance = max(max(plan.halo), 1) / 2
        return centroid_pairs(fragments, max_distance=distance, spacing=spacing)
    raise ValueError(f"unknown matching {policy.matching!r}")


def _overlap_matches(
    scratch: ZarrScratch,
    tile_arrays: Mapping[tuple[int, ...], str],
    plan: TilePlan,
    policy: SeamPolicy,
) -> list[tuple[int, int, float]]:
    """IoU between neighbouring tiles' labellings of the voxels they share.

    Every pair of tiles whose *padded* regions intersect is compared, not
    only face neighbours: with a halo wider than half a tile, diagonal
    neighbours overlap too, and an object at a grid corner is in four tiles
    at once.
    """
    pairs: list[tuple[int, int, float]] = []
    tiles = {tile.index: tile for tile in plan.tiles()}
    for index, tile in tiles.items():
        for offset in _forward_neighbours(plan.ndim):
            other_index = tuple(a + b for a, b in zip(index, offset))
            other = tiles.get(other_index)
            if other is None:
                continue
            region = _intersection(tile.padded, other.padded)
            if region is None:
                continue
            left = _read_region(scratch, tile_arrays, tile, region)
            right = _read_region(scratch, tile_arrays, other, region)
            pairs.extend(overlap_pairs(left, right, min_overlap=policy.min_overlap))
    return pairs


def _forward_neighbours(ndim: int) -> Iterable[tuple[int, ...]]:
    """Each neighbouring offset once rather than twice - the pair (A, B) and
    the pair (B, A) are the same comparison."""
    for offset in product((-1, 0, 1), repeat=ndim):
        if any(offset) and next(value for value in offset if value) > 0:
            yield offset


def _intersection(
    left: Sequence[slice], right: Sequence[slice]
) -> tuple[slice, ...] | None:
    region = tuple(
        slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(left, right)
    )
    if any(part.stop <= part.start for part in region):
        return None
    return region


def _read_region(
    scratch: ZarrScratch,
    tile_arrays: Mapping[tuple[int, ...], str],
    tile: Tile,
    region: Sequence[slice],
) -> np.ndarray:
    """A global region out of one tile's own labelling of its block."""
    array = scratch.open(tile_arrays[tile.index])
    local = tuple(
        slice(part.start - start, part.stop - start)
        for part, start in zip(region, tile.origin)
    )
    return np.asarray(array[local])


def _touching_matches(provisional: Any, plan: TilePlan) -> list[tuple[int, int, float]]:
    """Fragments meeting across a seam plane, read two planes at a time.

    Cheap in a way the others are not: it touches only the seam planes
    themselves, which is a vanishing fraction of the volume, and it needs no
    per-tile labelling kept anywhere.
    """
    pairs: list[tuple[int, int, float]] = []
    for axis in plan.tiled_axes:
        for step in range(1, plan.splits[axis]):
            position = step * plan.tile[axis]
            if position >= plan.shape[axis]:
                break
            lower = np.asarray(_plane(provisional, axis, position - 1))
            upper = np.asarray(_plane(provisional, axis, position))
            pairs.extend(touching_pairs(lower, upper))
    return pairs


def _plane(array: Any, axis: int, position: int) -> np.ndarray:
    index = tuple(
        slice(position, position + 1) if a == axis else slice(None)
        for a in range(len(array.shape))
    )
    return np.squeeze(np.asarray(array[index]), axis=axis)


def _resolve(
    fragments: Sequence[Fragment],
    pairs: Sequence[tuple[int, int, float]],
    policy: SeamPolicy,
) -> tuple[LabelLedger, dict[int, int]]:
    """Pass 3: close the matches into objects, and record how.

    Returns the ledger and the provisional-to-final id map. Final ids are
    handed out in order of the groups' lowest provisional id, so they are
    1..N with no gaps and do not depend on the order anything was matched -
    a re-run of the same data gives the same numbers.
    """
    by_provisional = {fragment.provisional_id: fragment for fragment in fragments}
    # A tile also segments objects that lie wholly inside its halo - they
    # belong to a neighbour's core and that neighbour catalogued them, so
    # this tile's copy was deliberately not recorded. Overlap matching still
    # sees those copies and pairs them, so the pairs are filtered back to
    # fragments that exist. Nothing is lost: the pair being dropped relates
    # a discarded duplicate to the real record of the same object.
    pairs = [
        pair for pair in pairs if pair[0] in by_provisional and pair[1] in by_provisional
    ]
    assignment, weakest = group_fragments(by_provisional, pairs)

    ledger = LabelLedger(policy=policy)
    final_of_root: dict[int, int] = {}
    final_of_provisional: dict[int, int] = {}

    if policy.resolution == FLAG:
        # Matched, but deliberately left apart: each fragment stays its own
        # object and learns which others it was related to.
        for order, provisional_id in enumerate(sorted(by_provisional), start=1):
            final_of_provisional[provisional_id] = order
        for provisional_id, root in assignment.items():
            others = sorted(
                final_of_provisional[other]
                for other, other_root in assignment.items()
                if other_root == root and other != provisional_id
            )
            if others:
                ledger.links[final_of_provisional[provisional_id]] = others
    else:
        for order, root in enumerate(sorted(set(assignment.values())), start=1):
            final_of_root[root] = order
        final_of_provisional = {
            provisional_id: final_of_root[root] for provisional_id, root in assignment.items()
        }

    grouped: dict[int, list[Fragment]] = {}
    for provisional_id, final_id in final_of_provisional.items():
        grouped.setdefault(final_id, []).append(by_provisional[provisional_id])

    for final_id, members in grouped.items():
        root = assignment[members[0].provisional_id]
        cut = len(members) > 1 or (policy.resolution == FLAG and final_id in ledger.links)
        ledger.add(
            final_id,
            sorted(members, key=lambda fragment: fragment.provisional_id),
            decided_by=policy.resolution if cut else UNCUT,
            evidence=weakest.get(root, 1.0) if cut else 1.0,
        )

    _mark_dropped(ledger, policy)
    return ledger, final_of_provisional


def _mark_dropped(ledger: LabelLedger, policy: SeamPolicy) -> None:
    """Objects the policy excludes, and why.

    Two different exclusions that must not be confused. Dropping objects at
    the *dataset* border is the standard cytometry choice - they are
    genuinely truncated specimens. Dropping objects at a *tile* border is a
    bug under every strategy but "no merge", where it is the thing that
    makes the count honest rather than inflated.
    """
    for object_id in ledger.object_ids:
        if policy.border_objects == DROP and ledger.at_dataset_border(object_id):
            ledger.dropped[object_id] = "dataset border"
        elif policy.drop_seam_objects and ledger.touches_seam(object_id):
            ledger.dropped[object_id] = "tile seam"


def _check_exceeded(ledger: LabelLedger, policy: SeamPolicy) -> None:
    exceeded = ledger.exceeded()
    if exceeded and policy.on_halo_exceeded == RAISE_EXCEEDED:
        raise HaloExceeded(
            f"{len(exceeded)} object(s) reached the outer edge of their halo, so no tile "
            f"contained them and their shapes are truncated (first few: {exceeded[:5]}). "
            f"Raise the memory budget so the tiles are larger, set "
            f"max_object_extent to the real object size, or use the 'merge' resolution, "
            f"which reassembles objects no tile can hold."
        )


def _write_final(
    provisional: Any,
    assignment: Mapping[int, int],
    ledger: LabelLedger,
    tile_arrays: Mapping[tuple[int, ...], str],
    plan: TilePlan,
    policy: SeamPolicy,
    scratch: ZarrScratch,
    name: str,
) -> Any:
    """Pass 4: the final label array.

    One lookup-table pass turns provisional ids into final ones, tile by
    tile, which is enough for every resolution but `own`. `own` then writes
    the winning tile's complete copy over each reconciled object - the
    difference between keeping one tile's coherent object and stitching two
    half-objects with a step where they meet.
    """
    lookup = np.zeros(_lookup_size(assignment), dtype=LABEL_DTYPE)
    for provisional_id, final_id in assignment.items():
        if final_id not in ledger.dropped:
            lookup[provisional_id] = final_id

    final = scratch.create(f"{name}", shape=plan.shape, dtype=LABEL_DTYPE)
    for tile in plan.tiles():
        final[tile.core] = lookup[np.asarray(provisional[tile.core])]

    if policy.resolution == OWN:
        _write_owned_copies(final, ledger, tile_arrays, plan, scratch)
    return final


def _lookup_size(assignment: Mapping[int, int]) -> int:
    return (max(assignment) + 1) if assignment else 1


def _write_owned_copies(
    final: Any,
    ledger: LabelLedger,
    tile_arrays: Mapping[tuple[int, ...], str],
    plan: TilePlan,
    scratch: ZarrScratch,
) -> None:
    """Replace each reconciled object with the winning tile's own copy.

    The winner is the tile holding the most of the object in its core - the
    overlap ownership the plan settles on, rather than the tile whose core
    happens to contain a centroid. For a concave or elongated object those
    are different tiles, and the centroid can lie outside the object
    entirely.

    Under a sufficient halo and a translation-invariant segmenter every
    tile's copy is the same copy, so this changes nothing and costs a read.
    It earns itself where the copies differ, which is exactly where the
    stitched alternative would show a step at the seam.
    """
    for object_id in ledger.object_ids:
        fragments = ledger.fragments[object_id]
        if len(fragments) < 2 or object_id in ledger.dropped:
            continue
        winner = max(fragments, key=lambda fragment: fragment.core_voxels)
        tile_name = tile_arrays.get(winner.tile)
        if tile_name is None:
            continue
        region = _union_bbox(fragments, plan)
        block = scratch.open(tile_name)
        window = tuple(slice(start, stop) for start, stop in region)

        overlap = _intersection(window, _block_extent(block, winner, plan))
        if overlap is None:
            continue
        local = tuple(
            slice(part.start - origin, part.stop - origin)
            for part, origin in zip(overlap, _origin_of(winner, plan))
        )
        inside = tuple(
            slice(part.start - start, part.stop - start)
            for part, (start, _stop) in zip(overlap, region)
        )
        patch = np.asarray(final[window])
        # Clear and rewrite only where the winner actually has an opinion.
        # Clearing the whole extent first would delete any part of the
        # object lying beyond the winner's own block - which is exactly the
        # case where the halo was too small, and turning a stitched object
        # into a truncated one is the worst possible response to that.
        # Outside this region the assembled labelling stands.
        window_patch = patch[inside]
        window_patch[window_patch == object_id] = 0
        window_patch[np.asarray(block[local]) == winner.provisional_id] = object_id
        final[window] = patch


def _union_bbox(fragments: Sequence[Fragment], plan: TilePlan) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            max(0, min(fragment.bbox[axis][0] for fragment in fragments)),
            min(plan.shape[axis], max(fragment.bbox[axis][1] for fragment in fragments)),
        )
        for axis in range(plan.ndim)
    )


def _origin_of(fragment: Fragment, plan: TilePlan) -> tuple[int, ...]:
    return plan.tile_at(fragment.tile).origin


def _block_extent(block: Any, fragment: Fragment, plan: TilePlan) -> tuple[slice, ...]:
    origin = _origin_of(fragment, plan)
    return tuple(
        slice(start, start + size) for start, size in zip(origin, block.shape)
    )


def _resegment_cut_objects(
    final: Any,
    ledger: LabelLedger,
    function: Callable[..., np.ndarray],
    sources: Mapping[str, Any],
    params: Mapping[str, Any],
    plan: TilePlan,
    policy: SeamPolicy,
    progress: Callable[[int, int], None] | None,
) -> None:
    """Re-run the segmenter on every object a seam ran through.

    The other resolutions choose between the tiles' copies of a cut object.
    That is the right thing when the copies are the same copy - a
    translation-invariant segmenter gives the same answer wherever the
    window falls, so the choice is between duplicates. It is the wrong thing
    for a learned segmenter, whose answer near a tile edge is computed from
    truncated context: every copy is shaped by a boundary that has nothing
    to do with the specimen, and picking the better of two wrong masks
    leaves a wrong mask.

    So this removes the thing being chosen between. Each cut object gets a
    window centred on it, wide enough that the object is interior with
    context all round, and whatever the segmenter says there replaces the
    stitched version. One extra inference per cut object - which is why the
    other strategies exist, and why this is the default only where nothing
    else is honest.
    """
    margin = policy.resegment_margin
    if margin is None:
        margin = max(plan.halo) or 1
    # A window larger than a tile would not have fitted in the budget the
    # tiles were sized for, so it is the natural ceiling.
    ceiling = math.prod(plan.padded_tile)

    # Largest first, so that where two fragments turn out to be one object
    # the substantial one claims the voxels and the sliver is the one left
    # with none. Processing by id instead would hand precedence to whichever
    # tile happened to be numbered first, which is not a reason.
    candidates = sorted(
        (
            object_id
            for object_id in ledger.object_ids
            if object_id not in ledger.dropped and ledger.touches_seam(object_id)
        ),
        key=lambda object_id: (-ledger.size(object_id), object_id),
    )
    absorbed: list[int] = []
    for index, object_id in enumerate(candidates):
        window = _resegment_window(ledger.fragments[object_id], plan, margin)
        if math.prod(_extent(window)) > ceiling:
            # Bigger than a tile: re-segmenting it would need more memory
            # than the plan was built for. Keep the stitched version and say
            # the evidence is what it is.
            ledger.decided_by[object_id] = f"{RESEGMENT}:too-large"
            continue
        remaining, swallowed = _resegment_one(
            final, ledger, object_id, window, function, sources, params
        )
        absorbed.extend(swallowed)
        if remaining == 0:
            # Two fragments that failed to match can turn out to be one
            # object once the segmenter sees the whole of it - the first to
            # be re-segmented claims the voxels and the second is left with
            # none. That is the reconciliation working, so the empty one is
            # removed rather than reported as an object of size zero.
            absorbed.append(object_id)
        if progress is not None:
            progress(index + 1, len(candidates))

    for object_id in dict.fromkeys(absorbed):
        ledger.fragments.pop(object_id, None)
        ledger.decided_by.pop(object_id, None)
        ledger.evidence.pop(object_id, None)
        ledger.dropped[object_id] = "absorbed by a neighbour when re-segmented"


def _resegment_one(
    final: Any,
    ledger: LabelLedger,
    object_id: int,
    window: tuple[slice, ...],
    function: Callable[..., np.ndarray],
    sources: Mapping[str, Any],
    params: Mapping[str, Any],
) -> tuple[int, list[int]]:
    """Re-segment one object in its own window.

    Returns how many voxels it has afterwards - zero meaning a neighbour
    took them - and the ids it absorbed.
    """
    patch = np.asarray(final[window])
    before = patch == object_id
    if not before.any():
        return 0, []

    fresh = np.asarray(function(**{k: np.asarray(a[window]) for k, a in sources.items()}, **params))
    chosen = _best_overlap(fresh, before)
    if chosen == 0:
        # The segmenter found nothing here the second time. That is a real
        # disagreement rather than an error, and the stitched object stands
        # - flagged, so a review can see it happened.
        ledger.decided_by[object_id] = f"{RESEGMENT}:no-match"
        ledger.evidence[object_id] = 0.0
        return int(before.sum()), []

    after = fresh == chosen
    swallowed = _wholly_inside(patch, after, ledger, exclude=object_id)
    patch[before] = 0
    for other in swallowed:
        patch[patch == other] = 0
    # Only into space this object, one it absorbed, or nothing already held.
    # A re-segmented window reaches over its neighbours, and overwriting a
    # neighbour to improve this object would trade an error for an error.
    patch[after & (patch == 0)] = object_id
    final[window] = patch

    kept = patch == object_id
    remaining = int(kept.sum())
    if remaining == 0:
        return 0

    ledger.decided_by[object_id] = RESEGMENT
    ledger.evidence[object_id] = _iou(after, before)
    ledger.fragments[object_id] = [
        _fragment_for(kept, window, ledger.fragments[object_id][0])
    ]
    return remaining, swallowed


def _wholly_inside(
    patch: np.ndarray, mask: np.ndarray, ledger: LabelLedger, *, exclude: int
) -> list[int]:
    """Objects that lie entirely within a re-segmented object.

    The re-segmentation is the better evidence: looking at the whole object
    with context on every side, the segmenter says these voxels are one
    thing. An id sitting wholly inside that answer was a piece of it that
    the tiled pass failed to match - typically a sliver a tile boundary left
    behind - and keeping it would both inflate the object count and hold
    voxels the object should own.

    Two conditions, and the second is what keeps this from swallowing
    neighbours: the id must be entirely inside the new mask *within this
    window*, and the window must contain all of it, so that "entirely
    inside" is a statement about the object and not about the part of it
    that happens to be visible here.
    """
    present = np.unique(patch[mask])
    swallowed = []
    for other in present:
        other = int(other)
        if other in (0, exclude) or other not in ledger.fragments:
            continue
        here = patch == other
        visible = int(here.sum())
        if visible == ledger.size(other) and not (here & ~mask).any():
            swallowed.append(other)
    return swallowed


def _best_overlap(labels: np.ndarray, mask: np.ndarray) -> int:
    """The new object that best corresponds to the old one.

    A window centred on one object contains its neighbours too, so the
    result has to be attributed rather than assumed.
    """
    inside = labels[mask]
    inside = inside[inside > 0]
    if not inside.size:
        return 0
    values, counts = np.unique(inside, return_counts=True)
    return int(values[np.argmax(counts)])


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 0.0


def _resegment_window(
    fragments: Sequence[Fragment], plan: TilePlan, margin: int
) -> tuple[slice, ...]:
    """The object's extent, grown by `margin` and clipped to the volume."""
    return tuple(
        slice(
            max(0, min(f.bbox[axis][0] for f in fragments) - margin),
            min(plan.shape[axis], max(f.bbox[axis][1] for f in fragments) + margin),
        )
        for axis in range(plan.ndim)
    )


def _extent(window: Sequence[slice]) -> tuple[int, ...]:
    return tuple(part.stop - part.start for part in window)


def _fragment_for(
    mask: np.ndarray, window: Sequence[slice], like: Fragment
) -> Fragment:
    """One fragment describing the re-segmented object.

    The fragments it replaces described tiles' partial views of something
    that no longer exists; leaving them would make `filter_by_size` filter
    on a size the object no longer has.
    """
    positions = np.nonzero(mask)
    bbox = tuple(
        (int(axis.min()) + part.start, int(axis.max()) + 1 + part.start)
        for axis, part in zip(positions, window)
    )
    voxels = int(mask.sum())
    return Fragment(
        tile=like.tile,
        local_id=like.local_id,
        provisional_id=like.provisional_id,
        core_voxels=voxels,
        block_voxels=voxels,
        centroid=tuple(float(axis.mean()) + part.start for axis, part in zip(positions, window)),
        bbox=bbox,
        faces=like.faces,
        at_dataset_border=like.at_dataset_border,
        exceeded_halo=False,
    )


def filter_by_size_blocked(
    labels: BlockedLabels,
    *,
    scratch: ZarrScratch,
    min_size: int | None = None,
    max_size: int | None = None,
    name: str = "filtered",
) -> BlockedLabels:
    """Remove objects outside [min_size, max_size], out of core.

    The step Phase L2 had to refuse, and the clearest example of why. The
    remap itself is per voxel and trivially tiled; the *size* it filters on
    belongs to a whole object, and no tile has it - an object split across
    four tiles has four partial counts and not one of them is the answer.
    The ledger has been carrying that total since the objects were
    catalogued, so the filter is a lookup table and one pass.
    """
    if min_size is None and max_size is None:
        return labels

    sizes = labels.ledger.sizes()
    keep = {
        object_id: size
        for object_id, size in sizes.items()
        if object_id not in labels.ledger.dropped
        and (min_size is None or size >= min_size)
        and (max_size is None or size <= max_size)
    }

    lookup = np.zeros(max(sizes, default=0) + 1, dtype=LABEL_DTYPE)
    ledger = LabelLedger(policy=labels.policy)
    for new_id, object_id in enumerate(sorted(keep), start=1):
        lookup[object_id] = new_id
        ledger.add(
            new_id,
            labels.ledger.fragments[object_id],
            decided_by=labels.ledger.decided_by[object_id],
            evidence=labels.ledger.evidence[object_id],
        )

    filtered = scratch.create(name, shape=labels.plan.shape, dtype=LABEL_DTYPE)
    for tile in labels.plan.tiles():
        filtered[tile.core] = lookup[np.asarray(labels.array[tile.core])]
    return BlockedLabels(
        array=filtered, ledger=ledger, plan=labels.plan, policy=labels.policy
    )
