"""Which axis is which, and the one that is not implemented yet.

Every array in vtea-core is (C, Z, Y, X), and until now that has been an
agreement between docstrings rather than a thing in the code:
`io.tiff._to_czyx` reorders by hand, `Step._select_channel` finds the
channel axis by being told its index, and nothing anywhere can say what a
five-dimensional array's axes are.

That stops working the moment data is stored rather than read whole.
OME-NGFF's canonical order is `TCZYX`, and a store written without a time
axis has to be converted the day one is added - so VTEA writes five axes
from the start, with `T` of length one, and squeezes it on the way in. The
store is already a valid time-series store; the analysis is not, and this
module is where that line is drawn rather than left to be discovered.

**Time-series analysis is deliberately not implemented.** A `T` longer than
one raises `TimeSeriesNotSupported`, which names the axis and its length,
rather than failing somewhere inside a transpose. And a caveat worth
stating where someone will read it: linking an object at *t* to the same
object at *t+1* is tracking, a different problem with different algorithms.
A time axis does not bring it along. See docs/LARGE_IMAGES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# OME-NGFF's order, and ours. Everything is a subsequence of it.
CANONICAL = "TCZYX"

# What vtea-core works on in memory: one timepoint, every channel.
VOLUME = "CZYX"

# The axes that have a physical extent, and that a tile plan may divide.
SPATIAL = "ZYX"

# What each axis is, in the vocabulary OME-NGFF uses for its `axes` metadata.
AXIS_TYPES = {"T": "time", "C": "channel", "Z": "space", "Y": "space", "X": "space"}


class TimeSeriesNotSupported(NotImplementedError):
    """A dataset has more than one timepoint, which VTEA cannot yet analyse.

    Its own exception type rather than a bare NotImplementedError so a
    caller - a reader, a GUI, a batch script - can catch exactly this and
    say something useful about it.
    """


@dataclass(frozen=True)
class Axes:
    """An axis order, as a string of `TCZYX` letters.

    `Axes("CZYX")` describes a 4D volume; `Axes("YX")` a single plane.

    Any order of those letters is allowed, not just the canonical one: an
    ImageJ hyperstack really is `ZCYX` on disk, and a reader has to be able
    to say so before it can transpose it. `CANONICAL` and `VOLUME` are
    targets to convert *to*, not a constraint on what can be described.
    """

    order: str

    def __post_init__(self) -> None:
        order = self.order.upper()
        object.__setattr__(self, "order", order)
        if not order:
            raise ValueError("an axis order needs at least one axis")
        unknown = set(order) - set(CANONICAL)
        if unknown:
            raise ValueError(
                f"unknown axis {sorted(unknown)} in {self.order!r}; "
                f"expected some ordered subset of {CANONICAL!r}"
            )
        if len(set(order)) != len(order):
            raise ValueError(f"axis order {self.order!r} repeats an axis")

    def __len__(self) -> int:
        return len(self.order)

    def __contains__(self, axis: str) -> bool:
        return axis.upper() in self.order

    def __iter__(self):
        return iter(self.order)

    @property
    def ndim(self) -> int:
        return len(self.order)

    @property
    def has_time(self) -> bool:
        return "T" in self.order

    @property
    def has_channel(self) -> bool:
        return "C" in self.order

    @property
    def spatial(self) -> str:
        return "".join(axis for axis in self.order if axis in SPATIAL)

    @property
    def is_canonical(self) -> bool:
        """Whether these axes are already in `TCZYX` order."""
        positions = [CANONICAL.index(axis) for axis in self.order]
        return positions == sorted(positions)

    @property
    def spatial_indices(self) -> tuple[int, ...]:
        """Which axes a tile plan may divide - the ones with a physical
        extent. A channel or time axis is taken whole."""
        return tuple(index for index, axis in enumerate(self.order) if axis in SPATIAL)

    def index_of(self, axis: str) -> int:
        """The position of one axis, or -1 when it isn't there.

        -1 rather than an exception: "is there a channel axis, and where?"
        is one question, and callers should not have to ask it twice.
        """
        return self.order.find(axis.upper())

    def types(self) -> list[str]:
        """The OME-NGFF `type` for each axis, in order."""
        return [AXIS_TYPES[axis] for axis in self.order]

    def transpose_to(self, target: str | Axes) -> tuple[int, ...]:
        """The `np.transpose` order that turns this layout into `target`.

        Both must name the same set of axes; adding or dropping one is
        `to_canonical`'s job, not a transpose's.
        """
        target = target.order if isinstance(target, Axes) else target.upper()
        if set(target) != set(self.order):
            raise ValueError(
                f"cannot transpose {self.order!r} to {target!r} - they describe "
                f"different axes"
            )
        return tuple(self.order.index(axis) for axis in target)

    def without(self, axis: str) -> Axes:
        return Axes(self.order.replace(axis.upper(), ""))

    def describe(self, shape: tuple[int, ...] | None = None) -> str:
        if shape is None:
            return self.order
        return ", ".join(f"{axis}={size}" for axis, size in zip(self.order, shape))


def canonical_for(ndim: int) -> Axes:
    """The axis order a plain array of `ndim` dimensions is assumed to have.

    The last axes of `TCZYX`, which is what an unlabelled 3D array almost
    always is (a z-stack, not three channels of a plane). A caller who knows
    better should say so rather than rely on this.
    """
    if not 1 <= ndim <= len(CANONICAL):
        raise ValueError(f"no canonical axis order for {ndim} dimensions")
    return Axes(CANONICAL[-ndim:])


def to_canonical(
    array: np.ndarray,
    axes: str | Axes,
    *,
    target: str | Axes = VOLUME,
    squeeze_time: bool = True,
) -> np.ndarray:
    """Reorder and pad `array` from its own axes into `target`.

    Three things happen, in this order:

    1. **Time is dropped**, if `target` has no time axis. A single timepoint
       squeezes away silently; more than one raises
       `TimeSeriesNotSupported`, because quietly analysing the first frame
       of a movie is worse than refusing to.
    2. **Missing axes are inserted** with length one, so a 2D plane becomes
       a one-channel, one-slice volume and everything downstream can assume
       four dimensions.
    3. **The rest is transposed** into `target`'s order.

    Sample-interleaved axes (`S`, an RGB TIFF) are not handled here; a
    reader has to resolve those before it gets this far.
    """
    axes = axes if isinstance(axes, Axes) else Axes(axes)
    target = target if isinstance(target, Axes) else Axes(target)
    if array.ndim != axes.ndim:
        raise ValueError(
            f"an array of shape {array.shape} does not have the {axes.ndim} axes "
            f"{axes.order!r} describes"
        )

    if axes.has_time and not target.has_time:
        if squeeze_time:
            array, axes = _drop_time(array, axes)
        else:
            raise TimeSeriesNotSupported(
                f"the target layout {target.order!r} has no time axis"
            )

    extra = set(axes.order) - set(target.order)
    if extra:
        raise ValueError(
            f"axes {sorted(extra)} are present in the data but not in the target "
            f"layout {target.order!r}"
        )

    for axis in target.order:
        if axis not in axes:
            # Position is irrelevant - the transpose below puts everything
            # where it belongs - so the front is as good as anywhere.
            array = np.expand_dims(array, axis=0)
            axes = Axes(axis + axes.order)

    return np.transpose(array, axes.transpose_to(target))


def _drop_time(array: np.ndarray, axes: Axes) -> tuple[np.ndarray, Axes]:
    index = axes.index_of("T")
    length = array.shape[index]
    if length != 1:
        raise TimeSeriesNotSupported(
            f"this dataset has {length} timepoints (axis {index} of {array.shape}), and "
            f"analysing a time series is not implemented. VTEA reads and writes a time "
            f"axis so that stores do not need converting when it is - see "
            f"docs/LARGE_IMAGES.md - but for now select a single timepoint before loading."
        )
    return np.take(array, 0, axis=index), axes.without("T")


def n_timepoints(shape: tuple[int, ...], axes: str | Axes) -> int:
    """How many timepoints a dataset has, whether or not it can be analysed.

    A reader records this even when it is 1, so that a store's metadata says
    what it is rather than what this version of VTEA happens to support.
    """
    axes = axes if isinstance(axes, Axes) else Axes(axes)
    index = axes.index_of("T")
    return 1 if index < 0 else int(shape[index])
