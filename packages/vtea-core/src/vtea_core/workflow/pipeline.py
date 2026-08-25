"""Step/Pipeline: the headless engine behind the napari protocol builder.

Replaces vtea.protocol/vtea.workflow's Java classes (ProtocolManagerMulti,
the datastructure/Protocol hierarchy, Workflow/AbstractWorkflow). Runs the
same whether driven from the GUI or a script/notebook, matching vtea-core's
headless-usable design goal - the napari widget (Phase 4) is a thin layer
that renders a Pipeline's steps as cards and edits them via magicgui forms,
it doesn't own execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vtea_core.workflow.registry import get_step_function


@dataclass
class Step:
    """One configured pipeline step.

    `input_keys` maps the target function's argument names to keys in the
    Pipeline's run() context (e.g. {"labels": "labels", "intensity":
    "volume"} for extract_measurements, which needs both a prior step's
    output and the original image). `params` are additional fixed keyword
    arguments (thresholds, n_clusters, ...). The function's return value is
    stored in the context under `output_key`.

    Rather than filling these in by hand, build a step with
    `Step.for_function(...)`, which derives them from
    vtea_core.workflow.wiring.

    `channel` picks one channel out of a multi-channel image for this step
    to work on (None = use the array as-is). See run() for exactly which
    inputs that applies to.
    """

    category: str
    function_name: str
    params: dict[str, Any] = field(default_factory=dict)
    input_keys: dict[str, str] = field(default_factory=dict)
    output_key: str = "result"
    comment: str = ""
    channel: int | None = None

    @classmethod
    def for_function(
        cls,
        category: str,
        function_name: str,
        *,
        available: set[str] | None = None,
        **kwargs: Any,
    ) -> Step:
        """A Step with input_keys/output_key derived from the function's
        declared I/O (see vtea_core.workflow.wiring), so it can actually run
        without the caller wiring the data arguments themselves."""
        from vtea_core.workflow.wiring import default_wiring

        input_keys, output_key = default_wiring(category, function_name, available)
        kwargs.setdefault("input_keys", input_keys)
        kwargs.setdefault("output_key", output_key)
        return cls(category=category, function_name=function_name, **kwargs)

    @property
    def function(self):
        return get_step_function(self.category, self.function_name)

    def run(
        self,
        context: dict[str, Any],
        *,
        channel_axis: int | None = None,
        full_ndim: int | None = None,
    ) -> Any:
        """Calls this step's function with its data inputs resolved from
        `context`.

        Channel selection applies only to inputs that still carry the
        channel axis, identified by having the same ndim as the image the
        pipeline started with (`full_ndim`). Anything produced by an earlier
        step has already been reduced to a single channel and has one fewer
        dimension, so it is passed through untouched - which is what keeps a
        second channel-selecting step from slicing a spatial axis by
        mistake.
        """
        missing = [key for key in self.input_keys.values() if key not in context]
        if missing:
            raise KeyError(
                f"step '{self.category}.{self.function_name}' needs context key(s) {missing}, "
                f"available: {list(context)}"
            )
        kwargs = {
            arg: self._select_channel(context[key], channel_axis, full_ndim)
            for arg, key in self.input_keys.items()
        }
        kwargs.update(self.params)
        return self.function(**kwargs)

    def _select_channel(self, value: Any, channel_axis: int | None, full_ndim: int | None) -> Any:
        if self.channel is None or channel_axis is None:
            return value
        if not isinstance(value, np.ndarray) or value.ndim != full_ndim:
            return value
        if not -value.ndim <= channel_axis < value.ndim:
            raise IndexError(
                f"step '{self.category}.{self.function_name}': channel axis {channel_axis} "
                f"is out of range for an array of shape {value.shape}"
            )
        n_channels = value.shape[channel_axis]
        if not 0 <= self.channel < n_channels:
            raise IndexError(
                f"step '{self.category}.{self.function_name}': channel {self.channel} is out of "
                f"range - axis {channel_axis} of a {value.shape} array has {n_channels} channel(s)"
            )
        return np.take(value, self.channel, axis=channel_axis)


class Pipeline:
    """An ordered, editable sequence of Steps.

    `channel_axis` says which axis of the input image is the channel axis
    (None = the image isn't multi-channel). It's a property of the data, so
    it lives here rather than on each Step; which *channel* to use is a
    per-step choice and lives on Step.channel.
    """

    def __init__(self, steps: list[Step] | None = None, channel_axis: int | None = None):
        self.steps: list[Step] = list(steps) if steps else []
        self.channel_axis = channel_axis

    def available_keys(self, seed_keys: set[str] | None = None) -> set[str]:
        """Context keys that will exist once every current step has run -
        i.e. what a step appended next could wire itself to."""
        keys = set() if seed_keys is None else set(seed_keys)
        for step in self.steps:
            keys.add(step.output_key)
        return keys

    def add_step(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    def insert_step(self, index: int, step: Step) -> Step:
        self.steps.insert(index, step)
        return step

    def remove_step(self, index: int) -> Step:
        return self.steps.pop(index)

    def move_step(self, from_index: int, to_index: int) -> None:
        step = self.steps.pop(from_index)
        self.steps.insert(to_index, step)

    def run(
        self, context: dict[str, Any] | None = None, *, channel_axis: int | None = None
    ) -> dict[str, Any]:
        """Runs every step in order, threading results through a shared context.

        Returns the final context (the input context plus every step's
        output_key); the input dict itself isn't mutated.

        `channel_axis` overrides the pipeline's own setting for this run.
        The widest array in the starting context defines "still has a
        channel axis" for per-step channel selection - see Step.run.
        """
        context = dict(context) if context else {}
        axis = self.channel_axis if channel_axis is None else channel_axis
        seeded = [value for value in context.values() if isinstance(value, np.ndarray)]
        full_ndim = max((value.ndim for value in seeded), default=None)
        for step in self.steps:
            context[step.output_key] = step.run(
                context, channel_axis=axis, full_ndim=full_ndim
            )
        return context

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)
