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
    """

    category: str
    function_name: str
    params: dict[str, Any] = field(default_factory=dict)
    input_keys: dict[str, str] = field(default_factory=dict)
    output_key: str = "result"
    comment: str = ""

    @property
    def function(self):
        return get_step_function(self.category, self.function_name)

    def run(self, context: dict[str, Any]) -> Any:
        missing = [key for key in self.input_keys.values() if key not in context]
        if missing:
            raise KeyError(
                f"step '{self.category}.{self.function_name}' needs context key(s) {missing}, "
                f"available: {list(context)}"
            )
        kwargs = {arg: context[key] for arg, key in self.input_keys.items()}
        kwargs.update(self.params)
        return self.function(**kwargs)


class Pipeline:
    """An ordered, editable sequence of Steps."""

    def __init__(self, steps: list[Step] | None = None):
        self.steps: list[Step] = list(steps) if steps else []

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

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Runs every step in order, threading results through a shared context.

        Returns the final context (the input context plus every step's
        output_key); the input dict itself isn't mutated.
        """
        context = dict(context) if context else {}
        for step in self.steps:
            context[step.output_key] = step.run(context)
        return context

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)
