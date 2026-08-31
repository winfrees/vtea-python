"""The statistics a step needs before it can look at a single voxel.

Otsu's threshold is not a property of a tile. Neither is "the 99th
percentile" or "rescale to the image's own range". Run per tile they give a
different answer in every tile, and the seams are visible in the result -
which is the failure this module exists to prevent, and the reason those
steps are marked GLOBAL_STAT in the scaling contract rather than being
quietly treated as elementwise.

The fix is a streaming pass: accumulate the statistic block by block, then
run the ordinary step with the answer baked in as a fixed parameter. Two
things make that worth doing properly rather than sampling:

- **For integer data the histogram is exact.** One bin per value, summed
  over blocks, is the same histogram `skimage` would have built from the
  whole array - so the threshold is the threshold, not an estimate of it.
  Fluorescence data is integer essentially always.
- **The statistic is recorded**, so a result can say what threshold it
  actually used rather than leaving it implicit in the pixels.

Float data gets a binned histogram instead, and says so.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from vtea_core.blocked.plan import TilePlan

# Bins for float data, where there is no exact answer to be had. 2^16 is
# enough that the error in a threshold is far below the noise in the data,
# and small enough to hold and merge without thought.
FLOAT_BINS = 1 << 16

# Above this many distinct integer values, fall back to binning: a per-value
# histogram of int32 data would be four billion counters.
MAX_EXACT_BINS = 1 << 21


@dataclass(frozen=True)
class ImageStats:
    """What one streaming pass over an image learned about it.

    `exact` is the fact worth carrying: it says whether the histogram is the
    one a whole-image call would have built, or a binned approximation of
    it. A threshold derived from the second is still good; it is just not
    the same number, and a result that cannot say which is which cannot be
    compared with one computed the other way.
    """

    minimum: float
    maximum: float
    count: int
    total: float
    counts: np.ndarray
    centers: np.ndarray
    exact: bool
    dtype: np.dtype

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    def describe(self) -> str:
        kind = "exact" if self.exact else f"{len(self.counts)}-bin"
        return (
            f"{self.count:,} voxels in [{self.minimum:g}, {self.maximum:g}], "
            f"mean {self.mean:g}, {kind} histogram"
        )


def scan(source: Any, plan: TilePlan, *, progress=None) -> ImageStats:
    """One streaming pass - well, two - over an array too large to hold.

    The first finds the range, the second fills the histogram, because a
    histogram needs its bin edges before it can count anything and the
    range is only known once every block has been seen. Two reads of the
    data is a real cost, and it is small beside the segmentation the
    threshold is for.
    """
    minimum, maximum, count, total = _range_pass(source, plan, progress)
    dtype = np.dtype(source.dtype)
    edges, exact = _bin_edges(minimum, maximum, dtype)
    counts = _histogram_pass(source, plan, edges, progress)
    centers = (edges[:-1] + edges[1:]) / 2.0 if not exact else edges[:-1]
    return ImageStats(
        minimum=float(minimum),
        maximum=float(maximum),
        count=int(count),
        total=float(total),
        counts=counts,
        centers=centers,
        exact=exact,
        dtype=dtype,
    )


def _range_pass(source: Any, plan: TilePlan, progress) -> tuple[float, float, int, float]:
    minimum, maximum = np.inf, -np.inf
    count, total = 0, 0.0
    for index, tile in enumerate(plan.tiles()):
        # The core, not the padded block: a halo voxel belongs to the
        # neighbouring tile and counting it twice would tilt the histogram.
        block = np.asarray(source[tile.core])
        if block.size:
            minimum = min(minimum, float(block.min()))
            maximum = max(maximum, float(block.max()))
            count += block.size
            total += float(block.sum(dtype=np.float64))
        _report(progress, index + 1, plan.n_tiles * 2)
    if not count:
        raise ValueError("cannot compute statistics for an empty array")
    return minimum, maximum, count, total


def _histogram_pass(source: Any, plan: TilePlan, edges: np.ndarray, progress) -> np.ndarray:
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for index, tile in enumerate(plan.tiles()):
        block = np.asarray(source[tile.core])
        if block.size:
            counts += np.histogram(block.ravel(), bins=edges)[0]
        _report(progress, plan.n_tiles + index + 1, plan.n_tiles * 2)
    return counts


def _bin_edges(minimum: float, maximum: float, dtype: np.dtype) -> tuple[np.ndarray, bool]:
    """Bin edges, and whether they make the histogram exact.

    For integer data with a manageable range, one bin per value - which is
    what `skimage.exposure.histogram` does for an integer image, so a
    threshold computed from it matches the whole-image call exactly.
    """
    if np.issubdtype(dtype, np.integer):
        low, high = int(np.floor(minimum)), int(np.ceil(maximum))
        span = high - low + 1
        if span <= MAX_EXACT_BINS:
            return np.arange(low, high + 2, dtype=np.int64), True
    if maximum <= minimum:
        # A constant image: one bin wide enough to hold it.
        return np.array([minimum, minimum + 1.0]), False
    return np.linspace(minimum, maximum, FLOAT_BINS + 1), False


def _report(progress, done: int, total: int) -> None:
    if progress is not None:
        progress(done, total)


# -- what the statistics are for ----------------------------------------


def otsu_threshold(stats: ImageStats) -> float:
    """Otsu's threshold from an accumulated histogram.

    Handed to `skimage.filters.threshold_otsu` as a precomputed histogram,
    so this is the library's implementation rather than a reimplementation
    of it - which is what makes the blocked answer and the whole-image
    answer the same number rather than two close ones.
    """
    from skimage.filters import threshold_otsu

    if len(stats.counts) < 2 or np.count_nonzero(stats.counts) < 2:
        # A constant image has no two classes to separate. skimage raises
        # here; a threshold at the value itself is the usable answer.
        return float(stats.minimum)
    return float(threshold_otsu(hist=(stats.counts, stats.centers)))


def percentile_threshold(stats: ImageStats, percentile: float) -> float:
    """The value below which `percentile`% of voxels fall.

    Matches `np.percentile`'s default linear interpolation between order
    statistics, which is reconstructible exactly from an exact histogram
    and approximately from a binned one.
    """
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    cumulative = np.cumsum(stats.counts)
    position = percentile / 100.0 * (stats.count - 1)
    lower_index = int(np.floor(position))
    fraction = position - lower_index
    lower = float(stats.centers[int(np.searchsorted(cumulative, lower_index + 1))])
    if fraction == 0:
        return lower
    upper = float(stats.centers[int(np.searchsorted(cumulative, lower_index + 2))])
    return lower + fraction * (upper - lower)


# Which parameter of which step a global statistic fills in. Small and
# explicit, in the spirit of the wiring table: two steps need this, and a
# general mechanism for two cases would be harder to read than the two.
def global_params(
    category: str, function_name: str, params: Mapping[str, Any], stats: ImageStats
) -> dict[str, Any]:
    """The step's parameters, with its global statistic resolved to a fixed
    value - so the step itself runs per tile, unchanged, and gives the
    whole-image answer.
    """
    resolved = dict(params)
    if (category, function_name) == ("segmentation", "threshold_mask"):
        method = resolved.get("method", "fixed")
        if method == "otsu":
            resolved.update(method="fixed", value=otsu_threshold(stats))
        elif method == "percentile":
            resolved.update(
                method="fixed",
                value=percentile_threshold(stats, resolved.pop("percentile")),
            )
    elif (category, function_name) == ("imageprocessing", "enhance_contrast"):
        if resolved.get("method", "normalize") == "normalize":
            resolved["in_range"] = (stats.minimum, stats.maximum)
    return resolved


def needs_global_stats(category: str, function_name: str, params: Mapping[str, Any]) -> bool:
    """Whether this step, with these parameters, has to see everything
    first."""
    from vtea_core.blocked.contract import GLOBAL_STAT
    from vtea_core.workflow.wiring import scaling_for

    return scaling_for(category, function_name).resolve(params).mode == GLOBAL_STAT
