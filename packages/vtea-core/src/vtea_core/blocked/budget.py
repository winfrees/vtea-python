"""How much memory this run may use, and where that number came from.

`VolumeDataset.fits_in_memory()` has been comparing against a hardcoded
4 GiB since Phase 1, with a comment in the source calling it a placeholder.
This is the replacement, and the reason it is more than one call to
`psutil` is the second-guessing that gets a job killed:

- **A container lies.** Inside a cgroup - a Docker container, a SLURM job,
  a CI runner - `psutil.virtual_memory()` reports the *host's* memory, which
  can be an order of magnitude more than the process is allowed. A tile plan
  built on 256 GB inside an 8 GB container is not a slow run, it is an
  OOM kill with no traceback. The cgroup limit is therefore checked first
  and wins where it is lower.
- **Free is not available.** What matters is what can still be allocated,
  not what is unused right now, and not the total installed.
- **"We could not tell" is a fact worth keeping**, the same way
  `Spacing.source` keeps it. A budget guessed from a fallback constant and a
  budget the user typed should not be equally believed, and the difference
  is exactly what should be shown next to a tile plan.

The GPU is a separate, smaller and stricter budget, because for Cellpose it
is the binding one. It is deliberately not derived from the CPU budget.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, replace

# Where a budget's number came from, in decreasing order of how much it
# should be believed.
USER = "user"  # passed in explicitly, by a caller or the GUI
ENV = "env"  # VTEA_MEMORY_BUDGET
CGROUP = "cgroup"  # a container/job limit - the real ceiling when present
DETECTED = "detected"  # psutil, on a machine with no cgroup limit
FALLBACK = "fallback"  # nothing could be determined; a constant, and say so

ENV_VAR = "VTEA_MEMORY_BUDGET"

# What fraction of the budget a run may actually spend on tiles. The rest is
# the interpreter, Qt, napari's own copies of what is on screen, the
# allocator's fragmentation, and the fact that peak usage always exceeds the
# sum of what you meant to allocate.
DEFAULT_FRACTION = 0.6

# Used only when nothing at all could be detected. Matches the placeholder
# it replaces (data.volume.DEFAULT_MEMORY_BUDGET_BYTES), so behaviour on an
# undetectable system is unchanged rather than newly surprising.
FALLBACK_TOTAL_BYTES = 4 * 1024**3

# cgroup v1 spells "no limit" as a number near 2**63. Anything above this is
# not a limit anyone set.
_CGROUP_UNLIMITED = 2**62

_CGROUP_V2_PATHS = ("/sys/fs/cgroup/memory.max",)
_CGROUP_V1_PATHS = (
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    "/sys/fs/cgroup/memory.limit_in_bytes",
)

_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "kib": 1024,
    "m": 1000**2,
    "mb": 1000**2,
    "mib": 1024**2,
    "g": 1000**3,
    "gb": 1000**3,
    "gib": 1024**3,
    "t": 1000**4,
    "tb": 1000**4,
    "tib": 1024**4,
}


class BudgetTooSmall(RuntimeError):
    """The budget cannot hold even a minimal tile of this data."""


@dataclass(frozen=True)
class MemoryBudget:
    """The memory a run may use, and the provenance of that figure.

    `total_bytes` is the ceiling. `usable_bytes` is what a single tile may
    actually occupy once the safety fraction is taken and the budget is
    shared between workers - it is the number the tile planner divides by.
    """

    total_bytes: int
    fraction: float = DEFAULT_FRACTION
    workers: int = 1
    gpu_bytes: int | None = None
    source: str = USER

    def __post_init__(self) -> None:
        if self.total_bytes <= 0:
            raise ValueError(f"a memory budget must be positive, got {self.total_bytes}")
        if not 0 < self.fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")
        if self.workers < 1:
            raise ValueError(f"workers must be at least 1, got {self.workers}")

    @property
    def usable_bytes(self) -> int:
        return int(self.total_bytes * self.fraction / self.workers)

    @property
    def is_measured(self) -> bool:
        """Whether this reflects something we actually found out.

        A plan built on a fallback budget is a guess, and should be
        presented as one - the same distinction `Spacing.is_known` draws.
        """
        return self.source != FALLBACK

    @property
    def has_gpu(self) -> bool:
        return self.gpu_bytes is not None and self.gpu_bytes > 0

    def gpu_usable_bytes(self) -> int | None:
        """The GPU equivalent of `usable_bytes`.

        The fraction is applied the same way but the workers division is
        not: GPU work is serialised through the device regardless of how
        many CPU workers a run has.
        """
        if not self.has_gpu:
            return None
        return int(self.gpu_bytes * self.fraction)

    def with_workers(self, workers: int) -> MemoryBudget:
        return replace(self, workers=workers)

    def with_total(self, total_bytes: int, *, source: str = USER) -> MemoryBudget:
        return replace(self, total_bytes=total_bytes, source=source)

    def describe(self) -> str:
        parts = [f"{format_bytes(self.usable_bytes)} usable of {format_bytes(self.total_bytes)}"]
        if self.workers > 1:
            parts.append(f"{self.workers} workers")
        parts.append(_SOURCE_PHRASES.get(self.source, self.source))
        if self.has_gpu:
            parts.append(f"GPU {format_bytes(self.gpu_bytes)}")
        return ", ".join(parts)


_SOURCE_PHRASES = {
    USER: "set by you",
    ENV: f"from ${ENV_VAR}",
    CGROUP: "from this container's limit",
    DETECTED: "detected",
    FALLBACK: "a fallback - nothing could be detected",
}


def detect_memory_budget(
    *,
    total_bytes: int | None = None,
    fraction: float = DEFAULT_FRACTION,
    workers: int = 1,
    gpu: bool = False,
) -> MemoryBudget:
    """The budget for this machine, from the most trustworthy source that
    answers.

    An explicit `total_bytes` wins, then `$VTEA_MEMORY_BUDGET`, then the
    lower of the cgroup limit and what the OS says is available, then a
    constant. `gpu=True` additionally probes the CUDA device, which is only
    worth the import when a step will actually use it.
    """
    gpu_bytes = gpu_free_bytes() if gpu else None

    if total_bytes is not None:
        return MemoryBudget(total_bytes, fraction, workers, gpu_bytes, USER)

    from_env = _env_budget()
    if from_env is not None:
        return MemoryBudget(from_env, fraction, workers, gpu_bytes, ENV)

    limit = cgroup_limit_bytes()
    available = available_bytes()

    if limit is not None and (available is None or limit <= available):
        return MemoryBudget(limit, fraction, workers, gpu_bytes, CGROUP)
    if available is not None:
        return MemoryBudget(available, fraction, workers, gpu_bytes, DETECTED)
    return MemoryBudget(FALLBACK_TOTAL_BYTES, fraction, workers, gpu_bytes, FALLBACK)


def _env_budget() -> int | None:
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return None
    try:
        return parse_size(raw)
    except ValueError:
        # A typo in an environment variable should not silently hand the
        # run the whole machine; nor should it abort a long job at the last
        # moment. Fall through to detection, which is the safe answer.
        return None


def cgroup_limit_bytes() -> int | None:
    """This process's cgroup memory limit, or None if there isn't one.

    Checked before `psutil` because inside a container it is both lower and
    correct, and `psutil` reports the host.
    """
    for path in _CGROUP_V2_PATHS:
        value = _read_cgroup_value(path)
        if value is not None:
            return value
    for path in _CGROUP_V1_PATHS:
        value = _read_cgroup_value(path)
        if value is not None:
            return value
    return None


def _read_cgroup_value(path: str) -> int | None:
    try:
        with open(path) as handle:
            raw = handle.read().strip()
    except (OSError, ValueError):
        return None
    if raw == "max":  # cgroup v2's spelling of "no limit"
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 0 or value >= _CGROUP_UNLIMITED:
        return None
    return value


def available_bytes() -> int | None:
    """What the OS says can still be allocated, or None if we cannot tell.

    `available` rather than `free`: page cache and reclaimable memory are
    free for practical purposes, and a plan built on `free` alone would tile
    a busy workstation into uselessly small pieces.
    """
    try:
        import psutil
    except ImportError:
        return _available_from_meminfo()
    try:
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - a platform without the counter, not our problem to fix
        return _available_from_meminfo()


def _available_from_meminfo() -> int | None:
    """Linux's own answer, for when psutil is not installed."""
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def gpu_free_bytes() -> int | None:
    """Free VRAM on the current CUDA device, or None without one.

    This is the *starting* number, not the answer: what a model can actually
    process in one tile depends on the model and the driver, and is measured
    by a calibration probe rather than computed (see docs/LARGE_IMAGES.md).
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:  # noqa: BLE001 - a driver mismatch is not a reason to abort planning
        return None


def parse_size(text: str) -> int:
    """Bytes from a human-written size: "8G", "512MiB", "1.5 gb", "2048"."""
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*", str(text))
    if match is None:
        raise ValueError(f"could not read {text!r} as a size in bytes")
    number, unit = match.groups()
    try:
        multiplier = _SIZE_UNITS[unit.lower()]
    except KeyError:
        raise ValueError(
            f"unknown size unit {unit!r} in {text!r} - expected one of "
            f"{sorted(u for u in _SIZE_UNITS if u)}"
        ) from None
    value = int(float(number) * multiplier)
    if value <= 0:
        raise ValueError(f"a size must be positive, got {text!r}")
    return value


def format_bytes(value: float | None) -> str:
    """A size a person can read: 8.0 GiB, 512.0 MiB."""
    if value is None:
        return "unknown"
    if value < 1024:
        return f"{int(value)} B"
    exponent = min(int(math.log(value, 1024)), 4)
    unit = ("B", "KiB", "MiB", "GiB", "TiB")[exponent]
    return f"{value / 1024**exponent:.1f} {unit}"
