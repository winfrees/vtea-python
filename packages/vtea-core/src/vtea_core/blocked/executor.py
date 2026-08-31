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
    ACCUMULATE,
    APPROXIMATE,
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


class Cancelled(RuntimeError):
    """A run a caller asked to stop.

    Its own exception type rather than a quiet return, because a cancelled
    run has a *partial* result in the scratch store and nothing downstream
    should mistake it for a finished one. What a caller does with the
    partial result is its business - paired with a manifest it is the start
    of a resume, and paired with nothing it is something to throw away -
    but it has to be told the difference.
    """


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
    # Present when this step produced rows rather than voxels.
    table: Any = None
    stats: ImageStats | None = None
    resolved_params: dict[str, Any] = field(default_factory=dict)
    # Present when this step produced objects rather than pixels: how every
    # object a tile boundary cut was put back together.
    ledger: Any = None

    def describe(self) -> str:
        parts = [f"{self.plan.n_tiles:,} tiles"]
        if self.stats is not None:
            parts.append(self.stats.describe())
        if self.resolved_params:
            fixed = ", ".join(f"{key}={value!r}" for key, value in self.resolved_params.items())
            parts.append(f"resolved {fixed}")
        if self.ledger is not None:
            parts.append(self.ledger.describe())
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
    should_stop: Callable[[], bool] | None = None,
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
        # Between tiles, not inside one: a tile is the smallest unit of work
        # that leaves the output in a consistent state, and interrupting one
        # halfway would leave a partly-written region that looks finished.
        _check_cancelled(should_stop)
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


def _check_cancelled(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise Cancelled("the run was cancelled")


def read_block(array: Any, tile: Tile, pad_mode: str | None = DEFAULT_PAD_MODE) -> np.ndarray:
    """One tile's input: the padded region, with the part that fell off the
    edge of the dataset synthesized.

    Reading and padding are one operation on purpose. Every caller needs
    both, and a caller that forgets the padding gets a result that is right
    everywhere except the outside of the volume.

    `pad_mode=None` returns the region as it is, unpadded - which is what a
    *segmentation* wants. A filter's halo is a boundary condition and
    mirroring is the right one; a segmenter handed a mirrored halo finds
    objects in it and fuses them with the real objects they are reflections
    of. At the specimen's own edge the honest answer is that there is no
    more data, and the block is simply smaller. Pair it with
    `Tile.inner_unpadded`.
    """
    block = np.asarray(array[tile.padded])
    if pad_mode is not None and any(before or after for before, after in tile.pad_width):
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
    should_stop: Callable[[], bool] | None = None,
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
        should_stop=should_stop,
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
        policy: Any = None,
    ):
        from vtea_core.blocked.reconcile import DEFAULT_POLICY

        self.pipeline = pipeline
        self.plan = plan
        self.scratch = scratch
        self._owns_scratch = scratch is None
        self.spacing = spacing
        self.pad_mode = pad_mode
        # How objects a tile boundary cut are put back together. Overlap
        # matching by default - see vtea_core.blocked.reconcile.
        self.policy = DEFAULT_POLICY if policy is None else policy
        self.results: dict[str, BlockedResult] = {}
        # Per step name, and per context key, so that a later step can find
        # the ledger belonging to the labels it was pointed at.
        self.ledgers: dict[str, Any] = {}

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
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Run every step, threading stored arrays through a context.

        The context holds arrays that are read a tile at a time rather than
        held, so it looks exactly like `Pipeline.run`'s and costs a
        different amount.

        `should_stop` is polled between tiles and between steps, and raises
        `Cancelled` when it answers yes. A run measured in hours has to be
        stoppable, and the place to stop is a tile boundary: it is the
        smallest unit that leaves the output consistent. Paired with a
        manifest (see `vtea_core.blocked.resume`) a cancelled run is the
        start of a resume rather than wasted work.
        """
        if self.scratch is None:
            raise RuntimeError("use BlockedPipeline as a context manager, or pass a scratch store")
        working = dict(context)
        for step in self.pipeline.steps:
            _check_cancelled(should_stop)
            step_progress = (
                (lambda done, total, name=step.name: progress(name, done, total))
                if progress is not None
                else None
            )
            result = self._run_one(
                step, working, progress=step_progress, should_stop=should_stop
            )
            self.results[step.name or step.function_name] = result
            produced = result.table if result.table is not None else result.array
            working[step.output_key] = produced
            if step.name:
                working[step.name] = produced
            if result.ledger is not None:
                self.ledgers[step.output_key] = result.ledger
                if step.name:
                    self.ledgers[step.name] = result.ledger
        return working

    def _run_one(
        self, step: Any, context: Mapping[str, Any], *, progress, should_stop=None
    ) -> BlockedResult:
        scaling = _scaling_of(step)
        if step.category == "ownership":
            return self._run_ownership(step, context, progress=progress)
        if scaling.needs_reconciliation:
            return self._run_reconciled(
                step, context, progress=progress, should_stop=should_stop
            )
        if scaling.mode == ACCUMULATE:
            return self._run_measurement(step, context, progress=progress)
        sources, params = self._split_inputs(step, context)

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
            should_stop=should_stop,
        )


    def _run_reconciled(
        self, step: Any, context: Mapping[str, Any], *, progress, should_stop=None
    ) -> BlockedResult:
        """A step whose objects have to be joined across tile boundaries.

        Two shapes of it. A segmentation assigns identities, so it runs per
        tile and is reconciled by the seam policy. `filter_by_size` assigns
        nothing but needs a whole object's size, which no tile has and the
        ledger from the segmentation it filters does - so it is a lookup
        table over that.
        """
        from vtea_core.blocked.reconcile import RESEGMENT
        from vtea_core.blocked.segment import segment_blocked

        sources, params = self._split_inputs(step, context)
        scaling = _scaling_of(step)

        if step.function_name == "filter_by_size":
            return self._filter_by_size(step, sources, params)

        if scaling.exactness == APPROXIMATE and self.policy.resolution != RESEGMENT:
            raise NotBlockableYet(
                f"'{step.category}.{step.function_name}' does not give the same answer "
                f"depending on where a tile edge falls, so every tile's copy of a "
                f"seam-crossing object is shaped by a boundary that has nothing to do "
                f"with the specimen. Choosing between them keeps a wrong mask. The "
                f"resolution for that is 'resegment', which re-runs the segmenter on a "
                f"window centred on the seam - Phase L5, with the rest of the "
                f"deep-learning work. See docs/LARGE_IMAGES.md."
            )

        labels = segment_blocked(
            step.function,
            sources,
            plan=self.plan,
            scratch=self.scratch,
            policy=self.policy,
            params=params,
            spacing=self.spacing,
            name=_scratch_name(step),
            progress=progress,
            should_stop=should_stop,
        )
        return BlockedResult(
            array=labels.array, plan=labels.plan, ledger=labels.ledger
        )

    def _filter_by_size(
        self, step: Any, sources: Mapping[str, Any], params: Mapping[str, Any]
    ) -> BlockedResult:
        from vtea_core.blocked.segment import BlockedLabels, filter_by_size_blocked

        source_key = step.input_keys.get("labels", "labels")
        ledger = self.ledgers.get(source_key)
        if ledger is None:
            raise KeyError(
                f"'{step.name or step.function_name}' filters on whole-object sizes, "
                f"which come from the ledger of the segmentation that produced "
                f"'{source_key}'. No blocked segmentation in this pipeline produced it - "
                f"available: {sorted(self.ledgers)}."
            )
        current = BlockedLabels(
            array=sources["labels"],
            ledger=ledger,
            plan=self.plan,
            policy=self.policy,
        )
        filtered = filter_by_size_blocked(
            current,
            scratch=self.scratch,
            min_size=params.get("min_size"),
            max_size=params.get("max_size"),
            name=_scratch_name(step),
        )
        return BlockedResult(
            array=filtered.array, plan=filtered.plan, ledger=filtered.ledger
        )

    def _run_measurement(
        self, step: Any, context: Mapping[str, Any], *, progress
    ) -> BlockedResult:
        """A step that turns voxels into one row per object.

        Streamed rather than tiled: the per-object accumulators are the
        output, and they are the same size whether the image was one tile or
        four thousand. See vtea_core.blocked.measure.
        """
        from vtea_core.blocked.measure import (
            measure_blocked,
            measure_blocked_by_channel,
            with_seam_columns,
        )

        sources, params = self._split_inputs(step, context)
        if step.function_name == "weighted_measurements_by_channel":
            return self._run_weighted(step, context, params, progress=progress)
        labels = sources["labels"]
        intensity = sources.get("intensity", labels)
        ledger = self.ledgers.get(step.input_keys.get("labels", "labels"))
        n_objects = ledger.n_objects if ledger is not None else int(np.asarray(labels).max())

        common = {
            "plan": self.plan,
            "n_objects": n_objects,
            "spacing": params.get("spacing", self.spacing),
            "progress": progress,
        }
        if step.function_name == "extract_measurements_by_channel":
            frame = measure_blocked_by_channel(
                labels,
                intensity,
                channel_axis=params.get("channel_axis"),
                channel=params.get("channel"),
                **common,
            )
        else:
            frame = measure_blocked(labels, intensity, **common)

        return BlockedResult(
            array=None, plan=self.plan, table=with_seam_columns(frame, ledger), ledger=ledger
        )

    def _run_ownership(
        self, step: Any, context: Mapping[str, Any], *, progress
    ) -> BlockedResult:
        """A posterior over owners per voxel, kept only where it means
        something.

        Dense over the volume, this is the largest thing a protocol
        produces - six times the image it describes. Restricted to the mask
        it is about, it is the same order as the image. See
        vtea_core.blocked.ownership.
        """
        from vtea_core.blocked.ownership import ownership_blocked

        sources, params = self._split_inputs(step, context)
        owned = ownership_blocked(
            sources["labels"],
            sources["mask"],
            plan=self.plan,
            spacing=params.get("spacing", self.spacing),
            falloff=params.get("falloff", 2.0),
            reach=params.get("reach"),
            top_k=params.get("top_k", 2),
            segmentation=params.get("segmentation", step.input_keys.get("labels", "")),
            progress=progress,
        )
        return BlockedResult(array=owned, plan=self.plan)

    def _run_weighted(
        self, step: Any, context: Mapping[str, Any], params: Mapping[str, Any], *, progress
    ) -> BlockedResult:
        """Measurements over a probabilistic ownership rather than a label
        image - a count becomes an expected volume and a mean a
        probability-weighted mean."""
        from vtea_core.blocked.measure import weighted_measure_blocked_by_channel

        owned = context[step.input_keys.get("ownership", "ownership")]
        intensity = context[step.input_keys.get("intensity", "intensity")]
        frame = weighted_measure_blocked_by_channel(
            owned,
            intensity,
            plan=self.plan,
            channel_axis=params.get("channel_axis"),
            channel=params.get("channel"),
            spacing=params.get("spacing", self.spacing),
            progress=progress,
        )
        return BlockedResult(array=None, plan=self.plan, table=frame)

    def _split_inputs(
        self, step: Any, context: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """This step's array inputs and its parameters, from the context.

        A Spacing or a channel axis travels in the context beside the
        arrays and is configuration, not something to divide into tiles.
        """
        sources, params = {}, dict(step.params or {})
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
                params[argument] = value
        return sources, params


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
