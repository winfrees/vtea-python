"""Step/Pipeline: the headless engine behind the napari protocol builder.

Replaces vtea.protocol and vtea.workflow from the Java codebase. See
pipeline.py and registry.py for the design rationale; measure.py for how
every segmentation in a protocol comes to be measured, and cost.py for how
long a step is expected to take before it is run.
"""

from vtea_core.workflow.cost import (
    STEP_COSTS,
    Calibration,
    StepCost,
    cost_for,
    estimate_seconds,
    format_duration,
)
from vtea_core.workflow.measure import (
    DEFAULT_MEASUREMENT,
    MEASUREMENT_PREFIX,
    measured_segmentations,
    measurement_name_for,
    rename_segmentation,
    segmentation_names,
    sync_measurement_steps,
)
from vtea_core.workflow.pipeline import Pipeline, Step, unique_step_name
from vtea_core.workflow.registry import STEP_REGISTRY, available_steps, get_step_function
from vtea_core.workflow.wiring import (
    CHANNEL_ARGUMENT,
    CHANNEL_NONE,
    CHANNEL_SLICE,
    DATA_PARAMETERS,
    IMAGE_OUTPUTS,
    STEP_IO,
    StepIO,
    default_wiring,
    produces_image,
    step_io,
)

__all__ = [
    "CHANNEL_ARGUMENT",
    "CHANNEL_NONE",
    "CHANNEL_SLICE",
    "DATA_PARAMETERS",
    "DEFAULT_MEASUREMENT",
    "IMAGE_OUTPUTS",
    "MEASUREMENT_PREFIX",
    "STEP_COSTS",
    "STEP_IO",
    "STEP_REGISTRY",
    "Calibration",
    "Pipeline",
    "Step",
    "StepCost",
    "StepIO",
    "available_steps",
    "cost_for",
    "default_wiring",
    "estimate_seconds",
    "format_duration",
    "get_step_function",
    "measured_segmentations",
    "measurement_name_for",
    "produces_image",
    "rename_segmentation",
    "segmentation_names",
    "step_io",
    "sync_measurement_steps",
    "unique_step_name",
]
