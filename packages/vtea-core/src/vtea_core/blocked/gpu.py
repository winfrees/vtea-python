"""What fits on the GPU, measured rather than computed.

Every other budget in this package is arithmetic: bytes per voxel times
voxels, and the answer is right. The GPU is not like that. What a model can
process in one go depends on the model's architecture, the version of it,
the driver, what else is resident on the card, and how the framework happens
to be caching allocations that day. A formula would be wrong on the second
GPU it met.

So this measures instead: grow a tile until the device refuses, and remember
the largest that worked. Once per (device, model) pair, cached in the user's
config, because the measurement costs a minute and the answer does not
change. The Java codebase carried bespoke GPU-OOM detection and restart
logic for exactly this problem; measuring once beats catching the failure
forever.

Nothing here imports `torch` at module level - the probe takes a callable,
so it is testable without a GPU and usable with any framework.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CALIBRATION_ENV = "VTEA_GPU_CALIBRATION"
CALIBRATION_VERSION = 1

# Where a measurement is remembered. Honours XDG, since a shared cluster
# home is exactly where not having to re-probe matters.
def default_cache_path() -> Path:
    override = os.environ.get(CALIBRATION_ENV)
    if override:
        return Path(override)
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "vtea" / "gpu_calibration.json"


# Where the search starts, and where it gives up. A 64^3 tile is small
# enough that anything with a GPU at all can do it, and 1024^3 is past the
# point where the CPU-side budget binds first anyway.
START_VOXELS = 64**3
CEILING_VOXELS = 1024**3

# How the framework spells running out of device memory. Matched on the
# exception's name and text rather than its type, so this works whether or
# not torch is installed and across the versions that renamed it.
_OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "alloc failed",
)


def is_out_of_memory(error: BaseException) -> bool:
    """Whether an exception is the device saying "not that big".

    Worth distinguishing from every other failure: an out-of-memory result
    means try something smaller, and anything else means stop and report,
    because a probe that treats a broken driver as "too big" will shrink the
    tile to nothing and blame the data.
    """
    text = f"{type(error).__name__} {error}".lower()
    return any(marker in text for marker in _OOM_MARKERS)


@dataclass(frozen=True)
class Calibration:
    """The largest tile one model was seen to process on one device."""

    device: str
    model: str
    max_voxels: int
    # How much of the card was free when this was measured. A calibration
    # taken on an empty card is optimistic on a busy one, and without this
    # there is no way to tell the two apart. 0 means nobody recorded it, in
    # which case the measurement is used as-is.
    free_bytes: int = 0
    measured_at: str = ""
    version: int = CALIBRATION_VERSION

    @property
    def key(self) -> str:
        return f"{self.device}|{self.model}"

    def tile_for(self, ndim: int = 3) -> tuple[int, ...]:
        """A cube-ish tile of this many voxels - the shape to plan with when
        nothing else constrains it."""
        edge = max(1, int(self.max_voxels ** (1 / ndim)))
        return (edge,) * ndim

    def describe(self) -> str:
        return (
            f"{self.model} on {self.device}: up to {self.max_voxels:,} voxels "
            f"({'x'.join(str(size) for size in self.tile_for())})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "model": self.model,
            "max_voxels": self.max_voxels,
            "free_bytes": self.free_bytes,
            "measured_at": self.measured_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Calibration:
        return cls(
            device=str(data["device"]),
            model=str(data["model"]),
            max_voxels=int(data["max_voxels"]),
            free_bytes=int(data.get("free_bytes", 0)),
            measured_at=str(data.get("measured_at", "")),
            version=int(data.get("version", 1)),
        )


def device_name() -> str | None:
    """The CUDA device this run would use, or None without one."""
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 - a driver mismatch is not a reason to abort
        return None


def calibrate(
    attempt: Callable[[tuple[int, ...]], Any],
    *,
    device: str,
    model: str,
    ndim: int = 3,
    start_voxels: int = START_VOXELS,
    ceiling_voxels: int = CEILING_VOXELS,
    refine: bool = True,
    free_bytes: int | None = None,
) -> Calibration:
    """Find the largest tile `attempt` can process, by trying.

    `attempt(shape)` runs the model on a synthetic tile of that shape and is
    expected either to return or to raise something `is_out_of_memory`
    recognises. Anything else propagates: a broken install should be
    reported, not measured around.

    Doubles until it fails or reaches the ceiling, then bisects once between
    the last success and the first failure - which turns a factor-of-two
    answer into a few-percent one for one extra attempt, and a factor of two
    in tile volume is a factor of two in the number of inferences.
    """
    largest = 0
    voxels = max(1, start_voxels)
    smallest_failure = None

    while voxels <= ceiling_voxels:
        if not _try(attempt, voxels, ndim):
            smallest_failure = voxels
            break
        largest = voxels
        voxels *= 2

    if largest == 0:
        raise RuntimeError(
            f"{model} could not process even {start_voxels:,} voxels on {device}. "
            f"The device is too small for this model, or something else is resident "
            f"on it - free the card, or run on the CPU."
        )

    if refine and smallest_failure is not None:
        low, high = largest, smallest_failure
        while high - low > max(low // 10, 1):
            middle = (low + high) // 2
            if _try(attempt, middle, ndim):
                low = middle
            else:
                high = middle
        largest = low

    return Calibration(
        device=device,
        model=model,
        max_voxels=int(largest),
        free_bytes=int(free_bytes or 0),
        measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _try(attempt: Callable[[tuple[int, ...]], Any], voxels: int, ndim: int) -> bool:
    edge = max(1, round(voxels ** (1 / ndim)))
    try:
        attempt((edge,) * ndim)
    except Exception as error:
        if is_out_of_memory(error):
            return False
        raise
    return True


def load_calibration(
    device: str, model: str, *, path: str | os.PathLike | None = None
) -> Calibration | None:
    """A measurement taken earlier, if there is one for this pair."""
    location = Path(path) if path is not None else default_cache_path()
    try:
        entries = json.loads(location.read_text())
    except (OSError, ValueError):
        return None
    entry = entries.get(f"{device}|{model}")
    if not entry:
        return None
    try:
        calibration = Calibration.from_dict(entry)
    except (KeyError, TypeError, ValueError):
        return None
    # A cache written by a newer VTEA may mean something else by these
    # numbers; re-measuring is cheap and being wrong is not.
    return calibration if calibration.version <= CALIBRATION_VERSION else None


def save_calibration(
    calibration: Calibration, *, path: str | os.PathLike | None = None
) -> Path:
    location = Path(path) if path is not None else default_cache_path()
    location.parent.mkdir(parents=True, exist_ok=True)
    try:
        entries = json.loads(location.read_text())
    except (OSError, ValueError):
        entries = {}
    entries[calibration.key] = calibration.to_dict()
    location.write_text(json.dumps(entries, indent=2, sort_keys=True))
    return location


def calibrated_voxels(
    attempt: Callable[[tuple[int, ...]], Any] | None = None,
    *,
    device: str | None = None,
    model: str = "unknown",
    ndim: int = 3,
    path: str | os.PathLike | None = None,
    remeasure: bool = False,
    **kwargs: Any,
) -> Calibration | None:
    """The cached measurement for this device and model, taking it if
    needed.

    Returns None when there is no GPU and no way to measure one, which is
    the caller's cue to use the CPU budget instead of inventing a number.
    """
    device = device or device_name()
    if device is None:
        return None
    if not remeasure:
        cached = load_calibration(device, model, path=path)
        if cached is not None:
            return cached
    if attempt is None:
        return None
    calibration = calibrate(attempt, device=device, model=model, ndim=ndim, **kwargs)
    save_calibration(calibration, path=path)
    return calibration


def gpu_tile_voxels(budget: Any, calibration: Calibration | None) -> int | None:
    """How many voxels a GPU step may take at once, here and now.

    A measurement is about a device in a state, not about a device. A
    calibration taken on an empty card describes an empty card; run the same
    model with a viewer holding two gigabytes of textures and that number is
    a crash. So where the calibration recorded how much was free at the
    time, this scales it by how much is free now - never upwards, since a
    card with more free memory than when it was measured does not prove the
    model would use it.

    Without a recorded free figure there is nothing to scale by and the
    measurement is used as it stands.
    """
    if calibration is None:
        return None
    measured = calibration.max_voxels
    now = budget.gpu_usable_bytes() if budget is not None else None
    if not now or not calibration.free_bytes:
        return measured
    return max(1, int(measured * min(1.0, now / calibration.free_bytes)))


def gpu_plan(
    shape: Sequence[int],
    *,
    voxels: int,
    halo: int | Sequence[int] = 0,
    chunks: Sequence[int] | None = None,
    tiled_axes: Sequence[int] | None = None,
    bound_by: str = "",
) -> Any:
    """A tile plan sized by the device rather than by main memory.

    Expressed as a `MemoryBudget` of exactly the measured capacity at one
    byte per voxel, so the ordinary planner does the work - the halo, the
    chunk snapping, the thin-axis handling and the human summary are all the
    same, and only the number that bounds it comes from somewhere else.
    """
    from vtea_core.blocked.budget import USER, MemoryBudget
    from vtea_core.blocked.plan import plan_tiles

    return plan_tiles(
        shape,
        budget=MemoryBudget(max(1, int(voxels)), fraction=1.0, source=USER),
        bytes_per_voxel=1,
        halo=halo,
        chunks=chunks,
        tiled_axes=tiled_axes,
        bound_by=bound_by or "the GPU",
    )
