"""Physical voxel size, and the difference between knowing it and assuming it.

Every measurement in VTEA has been in voxels until now, which was fine
while nothing compared distances. It stops being fine the moment anything
dilates a mask by a thickness or measures how far one object is from
another: confocal z-steps are routinely 3-10x the lateral pixel size, so a
"5 voxel" dilation is a sphere in index space and a flattened disc in the
specimen - wrong in a way that looks entirely plausible on screen.

The awkward part is that "isotropic, one unit per voxel" and "nobody
recorded the voxel size" are the same array of ones. napari fills
`layer.scale` with ones when a file carries no scale, so a reader cannot
tell the two apart from the value. `Spacing` therefore carries where the
number came from, and `is_known` is what callers check before doing
anything that depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Where a spacing came from. UNKNOWN is not "1x1x1" - it is "we have not
# been told", which is a different thing to report to the user.
FROM_METADATA = "metadata"
FROM_USER = "user"
UNKNOWN = "unknown"

DEFAULT_UNIT = "µm"


@dataclass(frozen=True)
class Spacing:
    """Physical size of one voxel along each axis, in `unit`.

    `values` is in array-axis order, matching the image it describes, so a
    (z, y, x) volume gets (z_size, y_size, x_size).
    """

    values: tuple[float, ...]
    unit: str = DEFAULT_UNIT
    source: str = FROM_USER

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("spacing needs at least one axis")
        if any(not np.isfinite(value) or value <= 0 for value in self.values):
            raise ValueError(f"voxel sizes must be finite and positive, got {self.values}")

    @property
    def is_known(self) -> bool:
        """Whether this describes a real measurement rather than a
        placeholder. Anything derived from a distance should check this and
        ask rather than quietly running in voxels."""
        return self.source != UNKNOWN

    @property
    def is_isotropic(self) -> bool:
        return len(set(self.values)) == 1

    @property
    def voxel_volume(self) -> float:
        return float(np.prod(self.values))

    def for_ndim(self, ndim: int) -> tuple[float, ...]:
        """This spacing trimmed or padded to `ndim` axes.

        A spacing describes the *spatial* axes; an array may carry extra
        leading axes (a channel axis, say). Extra axes are taken from the
        front as 1.0, and a longer spacing is trimmed from the front, so the
        trailing (y, x) sizes always line up with the trailing image axes -
        which is the pairing that is never ambiguous.
        """
        if ndim <= 0:
            raise ValueError(f"ndim must be positive, got {ndim}")
        values = self.values[-ndim:]
        if len(values) < ndim:
            values = (1.0,) * (ndim - len(values)) + values
        return tuple(float(value) for value in values)

    def describe(self) -> str:
        if not self.is_known:
            return "voxel size unknown"
        sizes = " × ".join(_format_size(value) for value in self.values)
        return f"{sizes} {self.unit}"

    def to_dict(self) -> dict[str, Any]:
        return {"values": list(self.values), "unit": self.unit, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Spacing:
        return cls(
            values=tuple(float(value) for value in data["values"]),
            unit=data.get("unit", DEFAULT_UNIT),
            source=data.get("source", FROM_USER),
        )

    @classmethod
    def unknown(cls, ndim: int = 3, unit: str = DEFAULT_UNIT) -> Spacing:
        """A placeholder that measures in voxels and says so."""
        return cls(values=(1.0,) * ndim, unit=unit, source=UNKNOWN)


def _format_size(value: float) -> str:
    text = f"{value:.4g}"
    return text


def spacing_from_scale(scale, unit: str = DEFAULT_UNIT) -> Spacing:
    """Read a napari layer's `.scale` as a Spacing.

    An all-ones scale is treated as unknown rather than as one micron per
    voxel: napari fills it with ones when the file carries no scale, so the
    two cases are indistinguishable from the value alone and assuming the
    generous reading is how anisotropy goes unnoticed.
    """
    values = tuple(float(value) for value in np.atleast_1d(np.asarray(scale, dtype=float)))
    if not values:
        return Spacing.unknown()
    if any(not np.isfinite(value) or value <= 0 for value in values):
        return Spacing.unknown(len(values), unit=unit)
    if all(value == 1.0 for value in values):
        return Spacing(values=values, unit=unit, source=UNKNOWN)
    return Spacing(values=values, unit=unit, source=FROM_METADATA)


def physical_volume(voxel_count, spacing: Spacing | None) -> float | None:
    """A voxel count as a physical volume, or None when the spacing is not
    known - which is the honest answer, and lets a caller leave the column
    out rather than filling it with a number that means voxels."""
    if spacing is None or not spacing.is_known:
        return None
    return float(np.asarray(voxel_count) * spacing.voxel_volume)
