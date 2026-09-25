"""Step/Pipeline: the headless engine behind the napari protocol builder.

Replaces vtea.protocol and vtea.workflow from the Java codebase. See
pipeline.py and registry.py for the design rationale; io.py for saving a
protocol as JSON; measure.py for how every segmentation in a protocol comes
to be measured, and cost.py for how long a step is expected to take before
it is run.
"""

from vtea_core.workflow.cost import (
    STEP_COSTS,
    Calibration,
    StepCost,
    cost_for,
    estimate_seconds,
    format_duration,
)
from vtea_core.workflow.io import (
    PROTOCOL_FORMAT_VERSION,
    PROTOCOL_SUFFIX,
    Protocol,
    ProtocolError,
    capture_environment,
    describe_source,
    load_protocol,
    protocol_from_dict,
    protocol_to_dict,
    save_protocol,
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
    "PROTOCOL_FORMAT_VERSION",
    "PROTOCOL_SUFFIX",
    "STEP_COSTS",
    "STEP_IO",
    "STEP_REGISTRY",
    "Calibration",
    "Pipeline",
    "Protocol",
    "ProtocolError",
    "Step",
    "StepCost",
    "StepIO",
    "available_steps",
    "capture_environment",
    "cost_for",
    "default_wiring",
    "describe_source",
    "estimate_seconds",
    "format_duration",
    "get_step_function",
    "load_protocol",
    "measured_segmentations",
    "measurement_name_for",
    "produces_image",
    "protocol_from_dict",
    "protocol_to_dict",
    "rename_segmentation",
    "save_protocol",
    "segmentation_names",
    "step_io",
    "sync_measurement_steps",
    "unique_step_name",
]
