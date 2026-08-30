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
from vtea_core.blocked.plan import (
    StepCost,
    Tile,
    TilePlan,
    plan_for_steps,
    plan_tiles,
    step_costs,
)

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
    "BudgetTooSmall",
    "HaloSpec",
    "HaloTooLarge",
    "MemoryBudget",
    "Scaling",
    "StepCost",
    "Tile",
    "TilePlan",
    "available_bytes",
    "cgroup_limit_bytes",
    "detect_memory_budget",
    "format_bytes",
    "gpu_free_bytes",
    "parse_size",
    "plan_for_steps",
    "plan_tiles",
    "step_costs",
]
