"""Running a protocol over data that does not fit in memory.

Three things, in the order a run needs them: how much memory there is
(`budget`), what each step does with it (`contract`), and how the data is
therefore divided up (`plan`). The executor that acts on a plan, the
reconciliation of objects a tile boundary cuts in half, and the
out-of-core measurement table are later phases - see docs/LARGE_IMAGES.md.

Nothing in here changes what an algorithm computes. `label_components` is
still a function that takes a NumPy mask and returns a NumPy label array;
this package decides which NumPy arrays it gets handed.
"""

from vtea_core.blocked.budget import (
    CGROUP,
    DETECTED,
    ENV,
    ENV_VAR,
    FALLBACK,
    USER,
    BudgetTooSmall,
    MemoryBudget,
    available_bytes,
    cgroup_limit_bytes,
    detect_memory_budget,
    format_bytes,
    gpu_free_bytes,
    parse_size,
)
from vtea_core.blocked.contract import (
    ACCUMULATE,
    APPROXIMATE,
    BLOCK_MODES,
    DEFAULT_SCALING,
    ELEMENTWISE,
    EXACT,
    EXACT_WITH_HALO,
    GLOBAL_STAT,
    NEIGHBORHOOD,
    OBJECT_LOCAL,
    TABLE,
    HaloSpec,
    HaloTooLarge,
    Scaling,
)
from vtea_core.blocked.executor import (
    BlockedPipeline,
    BlockedResult,
    NotBlockableYet,
    apply_blocked,
    numpy_pad_mode,
    read_block,
    run_step_blocked,
)
from vtea_core.blocked.plan import (
    StepCost,
    Tile,
    TilePlan,
    plan_for_steps,
    plan_tiles,
    step_costs,
)
from vtea_core.blocked.stats import ImageStats, otsu_threshold, percentile_threshold, scan
from vtea_core.blocked.store import ZarrScratch

__all__ = [
    "ACCUMULATE",
    "APPROXIMATE",
    "BLOCK_MODES",
    "CGROUP",
    "DEFAULT_SCALING",
    "DETECTED",
    "ELEMENTWISE",
    "ENV",
    "ENV_VAR",
    "EXACT",
    "EXACT_WITH_HALO",
    "FALLBACK",
    "GLOBAL_STAT",
    "NEIGHBORHOOD",
    "OBJECT_LOCAL",
    "TABLE",
    "USER",
    "BlockedPipeline",
    "BlockedResult",
    "BudgetTooSmall",
    "HaloSpec",
    "HaloTooLarge",
    "ImageStats",
    "MemoryBudget",
    "NotBlockableYet",
    "Scaling",
    "StepCost",
    "Tile",
    "TilePlan",
    "ZarrScratch",
    "apply_blocked",
    "available_bytes",
    "cgroup_limit_bytes",
    "detect_memory_budget",
    "format_bytes",
    "gpu_free_bytes",
    "numpy_pad_mode",
    "otsu_threshold",
    "parse_size",
    "percentile_threshold",
    "plan_for_steps",
    "plan_tiles",
    "read_block",
    "run_step_blocked",
    "scan",
    "step_costs",
]
