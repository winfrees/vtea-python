"""Running an ordinary step over a dataset that does not fit in memory.

The rule this module exists to keep: **the algorithm functions do not
change.** `gaussian_blur` is still a function that takes a NumPy array and
returns one. What changes is which array it is handed - a tile grown by a
halo, padded where the halo fell off the edge of the data - and where the
answer is put.

Three things have to be right for a tiled result to be the same result:

- **The halo.** A voxel at a tile's edge is computed from inputs beyond it,
  so the block handed to the function is the core plus the reach declared in
  the step's scaling contract, and the halo is trimmed off afterwards.
- **The padding at the dataset border**, where there is no data to grow
  into. Synthesizing it the way the function would have is what makes a
  tiled run identical to a whole-image one instead of merely close - and
  the names are a trap, see `PAD_MODES` below.
- **The global statistics**, for the steps that need to see everything
  before they can classify anything. Those run a streaming pass first and
  then run per tile with the answer fixed - see `vtea_core.blocked.stats`.

What is not here yet: steps that produce objects rather than images
(segmentation's labelling, measurements) need objects reconciled across
seams, which is Phase L3 and the hard half of the problem. They are refused
by name rather than run wrongly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vtea_core.blocked.contract import (
    ELEMENTWISE,
    GLOBAL_STAT,
    NEIGHBORHOOD,
    Scaling,
)
from vtea_core.blocked.plan import Tile, TilePlan
from vtea_core.blocked.stats import ImageStats, global_params, scan
from vtea_core.blocked.store import ZarrScratch

# scipy.ndimage and numpy.pad use the same words for different things, and
# getting it wrong puts a one-voxel error along every face of the volume -
# small, plausible-looking, and wrong.
#
#   scipy "reflect"  (d c b a | a b c d)  is numpy "symmetric"
#   scipy "mirror"   (d c b   | a b c d)  is numpy "reflect"
#
# So the default here is "symmetric", because scipy.ndimage's own default is
# "reflect" and this has to reproduce it.
PAD_MODES = {
    "reflect": "symmetric",
    "mirror": "reflect",
    "nearest": "edge",
    "wrap": "wrap",
    "constant": "constant",
}

DEFAULT_PAD_MODE = "symmetric"

# The block modes this phase can run. The rest are named in the error.
SUPPORTED_MODES = (ELEMENTWISE, NEIGHBORHOOD, GLOBAL_STAT)


class NotBlockableYet(NotImplementedError):
    """A step whose blocked form is a later phase."""


def numpy_pad_mode(scipy_mode: str) -> str:
    """The numpy.pad name for a scipy.ndimage boundary mode."""
    try:
        return PAD_MODES[scipy_mode]
    except KeyError:
        raise ValueError(
            f"unknown boundary mode {scipy_mode!r}, expected one of {sorted(PAD_MODES)}"
        ) from None


@dataclass
class BlockedResult:
    """What a blocked run produced, and what it had to assume to do it."""

    array: Any
    plan: TilePlan
    stats: ImageStats | None = None
    resolved_params: dict[str, Any] = field(default_factory=dict)
    exceeded_halo: bool = False

    def describe(self) -> str:
        parts = [f"{self.plan.n_tiles:,} tiles"]
        if self.stats is not None:
            parts.append(self.stats.describe())
        if self.resolved_params:
            fixed = ", ".join(f"{key}={value!r}" for key, value in self.resolved_params.items())
            parts.append(f"resolved {fixed}")
        return "; ".join(parts)


def apply_blocked(
    function: Callable[..., np.ndarray],
    sources: Mapping[str, Any],
    *,
    plan: TilePlan,
    params: Mapping[str, Any] | None = None,
    target: Any = None,
    dtype: Any = None,
    pad_mode: str = DEFAULT_PAD_MODE,
    progress: Callable[[int, int], None] | None = None,
) -> Any:
    """Run a shape-preserving function over `plan`'s tiles.

    `sources` maps the function's array arguments to arrays - anything that
    slices like one, so NumPy, Dask and Zarr all work. They must all have
    the plan's shape; a step that mixes shapes is not something this can
    tile blindly. Everything else goes in `params`.

    `target` is written region by region and returned. Passing a Zarr array
    is what keeps an output larger than memory out of memory; passing
    nothing allocates a NumPy array, which is the right thing for a test and
    the wrong thing for a real volume.

    With a single tile this reduces to calling the function on the whole
    array, with no padding and no trimming, which is why "one tile equals
    the whole image" is a real test of the machinery and not a tautology.
    """
    params = dict(params or {})
    if not sources:
        raise ValueError("apply_blocked needs at least one array input")
    _check_shapes(sources, plan)

    total = plan.n_tiles
    for index, tile in enumerate(plan.tiles()):
        blocks = {
            name: read_block(array, tile, pad_mode) for name, array in sources.items()
        }
        result = np.asarray(function(**blocks, **params))
        if result.shape != tile.padded_shape:
            raise ValueError(
                f"{getattr(function, '__name__', function)} returned shape {result.shape} "
                f"for a {tile.padded_shape} block. Only shape-preserving steps can be run "
                f"this way; a step that changes shape needs its own blocked form."
            )
        core = result[tile.inner]
        if target is None:
            target = np.empty(plan.shape, dtype=np.dtype(dtype) if dtype else core.dtype)
        target[tile.core] = core
        if progress is not None:
            progress(index + 1, total)
    return target


def read_block(array: Any, tile: Tile, pad_mode: str = DEFAULT_PAD_MODE) -> np.ndarray:
    """One tile's input: the padded region, with the part that fell off the
    edge of the dataset synthesized.

    Reading and padding are one operation on purpose. Every caller needs
    both, and a caller that forgets the padding gets a result that is right
    everywhere except the outside of the volume.
    """
    block = np.asarray(array[tile.padded])
    if any(before or after for before, after in tile.pad_width):
        block = np.pad(block, tile.pad_width, mode=pad_mode)
    return block


def _check_shapes(sources: Mapping[str, Any], plan: TilePlan) -> None:
    shapes = {name: tuple(array.shape) for name, array in sources.items()}
    wrong = {name: shape for name, shape in shapes.items() if shape != plan.shape}
    if wrong:
        raise ValueError(
            f"every input must have the plan's shape {plan.shape}; got {wrong}"
        )


def run_step_blocked(
    step: Any,
    sources: Mapping[str, Any],
    *,
    plan: TilePlan,
    target: Any = None,
    params: Mapping[str, Any] | None = None,
    pad_mode: str = DEFAULT_PAD_MODE,
    progress: Callable[[int, int], None] | None = None,
) -> BlockedResult:
    """Run one `vtea_core.workflow.Step` over a plan.

    The step's scaling contract decides what happens: an elementwise or
    neighbourhood step runs straight over the tiles, and a global-statistic
    one gets a streaming pass first and then runs with the statistic fixed -
    an Otsu threshold computed once over everything, not once per tile.

    `params` overrides the step's own, and exists for the caller that has
    already resolved a non-array input out of the run context - a `Spacing`
    travels beside the arrays and is a parameter, not something to tile.
    """
    scaling = _scaling_of(step)
    _require_supported(step, scaling)

    params = dict(getattr(step, "params", {}) or {}) if params is None else dict(params)
    stats = None
    if scaling.mode == GLOBAL_STAT:
        stats = scan(next(iter(sources.values())), plan, progress=progress)
        params = global_params(step.category, step.function_name, params, stats)

    # What the global pass changed - a key that was not there before, or
    # one whose value it replaced. Comparing keys alone would miss the most
    # important case: method="otsu" becoming method="fixed".
    original = dict(getattr(step, "params", {}) or {})
    resolved = {
        key: value
        for key, value in params.items()
        if key not in original or original[key] != value
    }
    array = apply_blocked(
        step.function,
        sources,
        plan=plan,
        params=params,
        target=target,
        pad_mode=pad_mode,
        progress=progress,
    )
    return BlockedResult(array=array, plan=plan, stats=stats, resolved_params=resolved)


def _scaling_of(step: Any) -> Scaling:
    from vtea_core.workflow.wiring import scaling_for

    return scaling_for(step.category, step.function_name).resolve(step.params or {})


def _require_supported(step: Any, scaling: Scaling) -> None:
    """Refuse, by name, anything a later phase owns.

    Two separate reasons to refuse, and the second is the one that would
    otherwise be missed. A block mode this phase cannot schedule is obvious.
    A step that *can* be scheduled but whose objects have to be reconciled
    across seams is not: `label_components` is a one-voxel neighbourhood
    step by every mechanical measure, and tiling it would give every tile
    its own object 1. Running it and returning a plausible-looking label
    image would be the worst outcome available.
    """
    if scaling.needs_reconciliation:
        raise NotBlockableYet(
            f"'{step.category}.{step.function_name}' assigns object identities, or needs "
            f"facts about whole objects, so tiling it means reconciling objects across "
            f"tile boundaries - Phase L3, see docs/LARGE_IMAGES.md. Without that, each "
            f"tile would number its objects independently and the result would look "
            f"right and be wrong. Run this step in memory, on data that fits."
        )
    if scaling.mode in SUPPORTED_MODES:
        return
    raise NotBlockableYet(
        f"'{step.category}.{step.function_name}' is a {scaling.mode} step, and only "
        f"{', '.join(SUPPORTED_MODES)} steps can be run out-of-core so far. Steps that "
        f"work per object are Phase L3 and steps that produce measurements need "
        f"per-object accumulators, Phase L4 - see docs/LARGE_IMAGES.md. Run this step "
        f"in memory, on data that fits."
    )


class BlockedPipeline:
    """A pipeline run out-of-core, with intermediates in a scratch store.

    The in-memory `Pipeline.run` threads results through a dict, which holds
    every intermediate for the life of the run. Here each step's output goes
    into `ZarrScratch` instead and the next step reads it back a tile at a
    time, so what is held is one tile of one array rather than a whole
    protocol's worth of volumes.

    Same steps, same functions, same parameters. The only thing that differs
    is where the arrays live - which is what makes the "one tile equals the
    whole image" test meaningful: it compares this against
    `Pipeline.run` on the same steps and expects the same numbers.

    **The results are only valid inside the context.** They are arrays in a
    scratch directory that closing deletes, and a Zarr array whose files
    have gone reads as zeros rather than raising. Read what is needed, or
    write it somewhere durable with `io.write_ome_zarr`, before leaving the
    block - or pass a `scratch` store of your own and manage its lifetime.
    """

    def __init__(
        self,
        pipeline: Any,
        *,
        plan: TilePlan,
        scratch: ZarrScratch | None = None,
        spacing: Any = None,
        pad_mode: str = DEFAULT_PAD_MODE,
    ):
        self.pipeline = pipeline
        self.plan = plan
        self.scratch = scratch
        self._owns_scratch = scratch is None
        self.spacing = spacing
        self.pad_mode = pad_mode
        self.results: dict[str, BlockedResult] = {}

    def __enter__(self) -> BlockedPipeline:  # noqa: PYI034 - typing.Self needs Python 3.11+, this package supports 3.10
        if self.scratch is None:
            self.scratch = ZarrScratch()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._owns_scratch and self.scratch is not None:
            self.scratch.close()
            self.scratch = None

    def run(
        self,
        context: Mapping[str, Any],
        *,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Run every step, threading stored arrays through a context.

        The context holds arrays that are read a tile at a time rather than
        held, so it looks exactly like `Pipeline.run`'s and costs a
        different amount.
        """
        if self.scratch is None:
            raise RuntimeError("use BlockedPipeline as a context manager, or pass a scratch store")
        working = dict(context)
        for step in self.pipeline.steps:
            step_progress = (
                (lambda done, total, name=step.name: progress(name, done, total))
                if progress is not None
                else None
            )
            result = self._run_one(step, working, progress=step_progress)
            self.results[step.name or step.function_name] = result
            working[step.output_key] = result.array
            if step.name:
                working[step.name] = result.array
        return working

    def _run_one(self, step: Any, context: Mapping[str, Any], *, progress) -> BlockedResult:
        sources = {}
        params = dict(step.params or {})
        for argument, key in step.input_keys.items():
            if key not in context:
                raise KeyError(
                    f"step '{step.category}.{step.function_name}' needs context key "
                    f"'{key}', available: {sorted(context)}"
                )
            value = context[key]
            if _is_arraylike(value):
                sources[argument] = value
            else:
                # A Spacing, a channel axis - configuration that travels in
                # the context but is not something to tile.
                params[argument] = value

        scaling = _scaling_of(step)
        _require_supported(step, scaling)
        probe = _probe_dtype(step, sources, params)
        target = self.scratch.create(
            _scratch_name(step),
            shape=self.plan.shape,
            dtype=probe,
            chunks=_chunks_for(self.plan),
        )
        return run_step_blocked(
            step,
            sources,
            plan=self.plan,
            target=target,
            params=params,
            pad_mode=self.pad_mode,
            progress=progress,
        )


def _scratch_name(step: Any) -> str:
    return step.name or f"{step.category}.{step.function_name}"


def _chunks_for(plan: TilePlan) -> tuple[int, ...]:
    """Store a blocked result in chunks the plan's tiles line up with.

    A tile is written as a whole number of chunks, so no chunk is
    read-modify-written by two tiles - which is both faster and, with
    unlocked parallel writes, the difference between a correct store and a
    partly zeroed one.
    """
    from vtea_core.io.store import DEFAULT_CHUNK

    return tuple(
        min(size, DEFAULT_CHUNK) if axis in plan.tiled_axes else size
        for axis, size in enumerate(plan.tile)
    )


def _probe_dtype(step: Any, sources: Mapping[str, Any], params: Mapping[str, Any]) -> np.dtype:
    """What dtype this step returns, found by running it on a tiny corner.

    Guessing from the inputs would get `threshold_mask` wrong (uint16 in,
    bool out) and `label_components` wrong the other way. A 2-voxel probe
    costs nothing and is never wrong.
    """
    corner = tuple(slice(0, min(2, extent)) for extent in next(iter(sources.values())).shape)
    blocks = {name: np.asarray(array[corner]) for name, array in sources.items()}
    try:
        return np.asarray(step.function(**blocks, **params)).dtype
    except Exception:  # noqa: BLE001 - a step that cannot run on a corner still has to be stored
        return np.dtype(next(iter(blocks.values())).dtype)


def _is_arraylike(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "__getitem__")
