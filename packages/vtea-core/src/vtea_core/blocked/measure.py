"""Measuring objects when neither the image nor the label array fits.

`regionprops_table` needs the whole label array at once, so the measurement
step is the one that survives tiling least well. What replaces it is not a
different calculation - the numbers must be the same numbers - but a
different way of arriving at them.

**Most features compose.** Count, sum, sum of squares, minimum, maximum and
the coordinate sums are all things that can be accumulated a tile at a time
and added up at the end, and mean, standard deviation and centroid fall out
of them with no error term. About a hundred bytes per object, so ten million
objects is a gigabyte, and it never touches the image twice.

**`threshold_mean` does not compose**, and it is worth being precise about
why, because the answer shapes the design. It is the mean of the values in
the top quartile of an object's intensity *range* - so it needs the object's
global minimum and maximum before it can decide which voxels count, and no
tile knows those until every tile has been seen. The plan originally
proposed re-reading each cut object's bounding box for features like this.
It turns out a second streaming pass is both simpler and cheaper: once the
minima and maxima are known, the cutoff is known per object, and the
selection becomes a lookup - fully vectorized, exact, and no random access
at all.

So: two passes over the data, no bounding-box reads, and a result identical
to the whole-image call. Random access will earn its place when shape
features arrive - a surface area or a sphericity cannot be reconstructed
from any accumulator - and the ledger already knows which objects would
need it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from vtea_core.blocked.plan import TilePlan
from vtea_core.data.spacing import Spacing
from vtea_core.measurements.regionprops import VOLUME_COLUMN

# The measurement table's column order, matching extract_measurements so a
# blocked table and an in-memory one are the same table.
GEOMETRY = ("object_id",)
INTENSITY_COLUMNS = ("mean", "min", "max", "sum", "stddev", "threshold_mean")

# The quartile ThresholdMean works on - the top quarter of an object's
# intensity *range*, not of its values. Ported from the Java original via
# measurements.regionprops.threshold_mean, and repeated here because the
# blocked form computes it from a cutoff rather than from a sorted region.
THRESHOLD_FRACTION = 4.0


@dataclass
class ObjectStats:
    """Per-object accumulators, filled a tile at a time.

    Everything here is additive across tiles: two tiles' partial counts add,
    their sums add, their minima take a minimum. That is the whole reason
    measuring out of core is possible at all, and the reason the features
    that are *not* additive are handled separately rather than approximated.
    """

    n_objects: int
    ndim: int
    count: np.ndarray
    total: np.ndarray
    total_squares: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    coordinate_sum: np.ndarray
    bbox_low: np.ndarray
    bbox_high: np.ndarray

    @classmethod
    def empty(cls, n_objects: int, ndim: int) -> ObjectStats:
        size = n_objects + 1
        return cls(
            n_objects=n_objects,
            ndim=ndim,
            count=np.zeros(size, dtype=np.int64),
            total=np.zeros(size, dtype=np.float64),
            total_squares=np.zeros(size, dtype=np.float64),
            minimum=np.full(size, np.inf),
            maximum=np.full(size, -np.inf),
            coordinate_sum=np.zeros((ndim, size), dtype=np.float64),
            bbox_low=np.full((ndim, size), np.iinfo(np.int64).max, dtype=np.int64),
            bbox_high=np.full((ndim, size), np.iinfo(np.int64).min, dtype=np.int64),
        )

    @property
    def object_ids(self) -> np.ndarray:
        """The ids that actually have voxels. An object the ledger knows
        about but that the array does not is not measured into existence."""
        return np.nonzero(self.count[1:])[0] + 1

    def add_tile(
        self, labels: np.ndarray, intensity: np.ndarray, origin: Sequence[int]
    ) -> None:
        """Fold one tile's core into the accumulators.

        The tile's *core*, never its halo: a halo voxel belongs to the
        neighbouring tile and counting it twice would inflate every sum it
        touches.
        """
        flat = labels.reshape(-1)
        if not flat.size:
            return
        values = np.asarray(intensity, dtype=np.float64).reshape(-1)
        size = self.n_objects + 1

        self.count += np.bincount(flat, minlength=size)[:size]
        self.total += np.bincount(flat, weights=values, minlength=size)[:size]
        self.total_squares += np.bincount(flat, weights=values * values, minlength=size)[:size]

        present = np.nonzero(np.bincount(flat, minlength=size)[:size])[0]
        present = present[present > 0]
        if not present.size:
            return

        # ndi's reductions are a C loop over the labelled regions, so they
        # cost the array once rather than once per object.
        self.minimum[present] = np.minimum(
            self.minimum[present], ndi.minimum(intensity, labels, index=present)
        )
        self.maximum[present] = np.maximum(
            self.maximum[present], ndi.maximum(intensity, labels, index=present)
        )

        for axis in range(self.ndim):
            coordinates = _axis_coordinates(labels.shape, axis, origin[axis])
            self.coordinate_sum[axis] += np.bincount(
                flat, weights=coordinates.reshape(-1), minlength=size
            )[:size]
            self.bbox_low[axis, present] = np.minimum(
                self.bbox_low[axis, present],
                ndi.minimum(coordinates, labels, index=present),
            )
            self.bbox_high[axis, present] = np.maximum(
                self.bbox_high[axis, present],
                ndi.maximum(coordinates, labels, index=present),
            )

    def centroids(self) -> np.ndarray:
        """(n_objects + 1, ndim), the mean coordinate of each object."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.coordinate_sum / np.maximum(self.count, 1)).T

    def means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.total / np.maximum(self.count, 1)

    def stddevs(self) -> np.ndarray:
        """Population standard deviation, as `np.std` gives it.

        From the sum and the sum of squares rather than a second pass. The
        subtraction can cancel badly in principle; at the magnitudes
        fluorescence data actually has it is accurate to about ten
        significant figures, and clamping at zero keeps a rounding error
        from becoming a NaN under the square root.
        """
        mean = self.means()
        with np.errstate(invalid="ignore", divide="ignore"):
            variance = self.total_squares / np.maximum(self.count, 1) - mean * mean
        return np.sqrt(np.maximum(variance, 0.0))

    def cutoffs(self) -> np.ndarray:
        """The intensity above which a voxel counts towards
        `threshold_mean` - the top quarter of the object's range.

        An id the array never used still holds its initial infinities, and
        `inf - inf` is a NaN with a warning attached. Left as NaN
        deliberately - nothing selects against it, so the object measures as
        empty, which it is - but computed quietly.
        """
        with np.errstate(invalid="ignore"):
            return self.maximum - (self.maximum - self.minimum) / THRESHOLD_FRACTION

    def bboxes(self) -> np.ndarray:
        """(n_objects + 1, ndim, 2), inclusive-exclusive in global
        coordinates. What a later phase's random-access second pass will
        read, and what a gallery crop is cut from."""
        return np.stack([self.bbox_low, self.bbox_high + 1], axis=-1).transpose(1, 0, 2)


def _axis_coordinates(shape: Sequence[int], axis: int, origin: int) -> np.ndarray:
    """Global coordinates along one axis, broadcast over a block.

    One axis at a time so the peak is one temporary rather than `ndim` of
    them - which is worth the extra passes, since the temporary is the size
    of the tile.
    """
    values = np.arange(shape[axis], dtype=np.float64) + origin
    view = values.reshape([-1 if a == axis else 1 for a in range(len(shape))])
    return np.ascontiguousarray(np.broadcast_to(view, shape))


class _ChannelView:
    """One channel of a multi-channel array, without reading the others.

    `intensity[channel]` on a Zarr array would pull the whole channel into
    memory, which is the thing this module exists to avoid. This inserts the
    channel index into each tile's own read instead, so a four-channel
    volume costs the same per tile as a one-channel one.
    """

    def __init__(self, array: Any, channel: int, channel_axis: int):
        self._array = array
        self._channel = channel
        self._axis = channel_axis % array.ndim
        self.shape = tuple(
            size for axis, size in enumerate(array.shape) if axis != self._axis
        )
        self.dtype = array.dtype
        self.ndim = len(self.shape)

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        full = list(index)
        full.insert(self._axis, self._channel)
        return self._array[tuple(full)]


def accumulate(
    labels: Any,
    intensity: Any,
    *,
    plan: TilePlan,
    n_objects: int,
    progress: Callable[[int, int], None] | None = None,
) -> ObjectStats:
    """Pass one: everything that adds up."""
    stats = ObjectStats.empty(n_objects, plan.ndim)
    for index, tile in enumerate(plan.tiles()):
        block = np.asarray(labels[tile.core])
        if block.any():
            stats.add_tile(
                block,
                np.asarray(intensity[tile.core]),
                [part.start for part in tile.core],
            )
        if progress is not None:
            progress(index + 1, plan.n_tiles * 2)
    return stats


def threshold_means(
    labels: Any,
    intensity: Any,
    *,
    plan: TilePlan,
    stats: ObjectStats,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Pass two: the one feature that needed the first pass to finish.

    With every object's cutoff known, a voxel counts if its value clears the
    cutoff of the object it belongs to - one lookup per voxel, one masked
    bincount per tile. No sorting, no per-object loop, no bounding-box
    reads, and the same answer as `measurements.threshold_mean` computed
    over the whole region at once.
    """
    size = stats.n_objects + 1
    cutoff = stats.cutoffs()
    selected_total = np.zeros(size, dtype=np.float64)
    selected_count = np.zeros(size, dtype=np.int64)

    for index, tile in enumerate(plan.tiles()):
        block = np.asarray(labels[tile.core])
        if block.any():
            flat = block.reshape(-1)
            values = np.asarray(intensity[tile.core], dtype=np.float64).reshape(-1)
            chosen = (flat > 0) & (values >= cutoff[flat])
            if chosen.any():
                picked = flat[chosen]
                selected_total += np.bincount(
                    picked, weights=values[chosen], minlength=size
                )[:size]
                selected_count += np.bincount(picked, minlength=size)[:size]
        if progress is not None:
            progress(plan.n_tiles + index + 1, plan.n_tiles * 2)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = selected_total / selected_count
    result[selected_count == 0] = np.nan
    return result


def _table(
    stats: ObjectStats,
    thresholds: np.ndarray,
    *,
    spacing: Spacing | None,
    suffix: str = "",
    geometry: bool = True,
) -> pd.DataFrame:
    """One tile-free measurement table, in `extract_measurements`' own column
    order so a blocked table and an in-memory one are interchangeable."""
    ids = stats.object_ids
    columns: dict[str, np.ndarray] = {}
    if geometry:
        columns["object_id"] = ids.astype(np.int64)
        centroids = stats.centroids()
        for axis in range(stats.ndim):
            columns[f"centroid-{axis}"] = centroids[ids, axis]
        columns["count"] = stats.count[ids].astype(np.float64)
        if spacing is not None and spacing.is_known:
            voxel = float(np.prod(spacing.for_ndim(stats.ndim)))
            columns[VOLUME_COLUMN] = stats.count[ids] * voxel

    columns[f"mean{suffix}"] = stats.means()[ids]
    columns[f"min{suffix}"] = stats.minimum[ids]
    columns[f"max{suffix}"] = stats.maximum[ids]
    columns[f"sum{suffix}"] = stats.total[ids]
    columns[f"stddev{suffix}"] = stats.stddevs()[ids]
    columns[f"threshold_mean{suffix}"] = thresholds[ids]
    return pd.DataFrame(columns)


def measure_blocked(
    labels: Any,
    intensity: Any,
    *,
    plan: TilePlan,
    n_objects: int,
    spacing: Spacing | None = None,
    suffix: str = "",
    geometry: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """`extract_measurements`, over arrays too large to hold.

    Same columns, same names, same numbers. `labels` and `intensity` need
    only support slicing, so Zarr, Dask and NumPy all work.
    """
    stats = accumulate(
        labels, intensity, plan=plan, n_objects=n_objects, progress=progress
    )
    thresholds = threshold_means(
        labels, intensity, plan=plan, stats=stats, progress=progress
    )
    return _table(stats, thresholds, spacing=spacing, suffix=suffix, geometry=geometry)


def measure_blocked_by_channel(
    labels: Any,
    intensity: Any,
    *,
    plan: TilePlan,
    n_objects: int,
    channel_axis: int | None = None,
    channel: int | None = None,
    spacing: Spacing | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """`extract_measurements_by_channel`, over arrays too large to hold.

    Geometry - the object's id, size and centroid - describes the object
    itself and appears once, unsuffixed. Every intensity column carries the
    channel it was measured on, so features from different channels coexist
    in one table and can be told apart when picking a plot axis.
    """
    if channel_axis is None or len(intensity.shape) == plan.ndim:
        return measure_blocked(
            labels,
            intensity,
            plan=plan,
            n_objects=n_objects,
            spacing=spacing,
            progress=progress,
        )

    axis = channel_axis % len(intensity.shape)
    n_channels = intensity.shape[axis]
    wanted = range(n_channels) if channel is None else [channel]
    for index in wanted:
        if not 0 <= index < n_channels:
            raise ValueError(
                f"channel {index} is out of range - axis {channel_axis} has "
                f"{n_channels} channel(s)"
            )

    tables = []
    for position, index in enumerate(wanted):
        tables.append(
            measure_blocked(
                labels,
                _ChannelView(intensity, index, axis),
                plan=plan,
                n_objects=n_objects,
                spacing=spacing,
                suffix=f"_ch{index}",
                geometry=position == 0,
                progress=progress,
            )
        )
    return pd.concat(tables, axis=1)


def with_seam_columns(frame: pd.DataFrame, ledger: Any) -> pd.DataFrame:
    """Join the ledger's account of each object onto its measurements.

    This is what makes a seam-crossing object *gateable*: `n_fragments`,
    `seam_rule` and `seam_confidence` become ordinary columns, so drawing a
    gate on low confidence and opening the gallery is the review workflow,
    with no new interface at all.
    """
    if ledger is None or "object_id" not in frame.columns:
        return frame
    return frame.merge(ledger.to_frame(), on="object_id", how="left")
