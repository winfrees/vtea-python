"""Step/Pipeline: the headless engine behind the napari protocol builder.

Replaces vtea.protocol and vtea.workflow from the Java codebase. See
pipeline.py and registry.py for the design rationale.
"""

from vtea_core.workflow.pipeline import Pipeline, Step
from vtea_core.workflow.registry import STEP_REGISTRY, available_steps, get_step_function
from vtea_core.workflow.wiring import DATA_PARAMETERS, STEP_IO, StepIO, default_wiring, step_io

__all__ = [
    "DATA_PARAMETERS",
    "STEP_IO",
    "STEP_REGISTRY",
    "Pipeline",
    "Step",
    "StepIO",
    "available_steps",
    "default_wiring",
    "get_step_function",
    "step_io",
]
