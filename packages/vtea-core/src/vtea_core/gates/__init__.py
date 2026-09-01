"""Boolean gate math over measurement tables, and gates drawn on the image.

Ports vtea.gates from the Java codebase. A gate is a boolean NumPy array;
combine gates with &, |, ~ directly instead of dedicated AND/OR/NOT classes.

`image.py` adds the kind of gate the Java version had no equivalent of: one
drawn on a napari Labels layer rather than on the plot, answering "which
objects are inside *that* region" with the region's own id, so a hand-drawn
tubule becomes a column of the measurement table like any other feature.
"""

from vtea_core.gates.gate import Gate, GateSet
from vtea_core.gates.image import (
    CENTROID,
    COLUMN_PREFIX,
    MAJORITY,
    OUTSIDE,
    centroids_from_frame,
    column_name,
    image_gate,
    objects_in_rois,
)
from vtea_core.gates.io import (
    GATES_FORMAT_VERSION,
    gate_set_from_dict,
    gate_set_to_dict,
    load_gates,
    save_gates,
)
from vtea_core.gates.polygon import polygon_gate, rectangle_gate, rectangle_vertices
from vtea_core.gates.seam import (
    CONFIDENCE_COLUMN,
    DEFAULT_THRESHOLD,
    FRAGMENTS_COLUMN,
    SEAM_GATE_COLOR,
    SEAM_GATE_NAME,
    has_seam_columns,
    seam_gate,
    seam_table,
)

__all__ = [
    "CENTROID",
    "COLUMN_PREFIX",
    "CONFIDENCE_COLUMN",
    "DEFAULT_THRESHOLD",
    "FRAGMENTS_COLUMN",
    "GATES_FORMAT_VERSION",
    "MAJORITY",
    "OUTSIDE",
    "SEAM_GATE_COLOR",
    "SEAM_GATE_NAME",
    "Gate",
    "GateSet",
    "centroids_from_frame",
    "column_name",
    "gate_set_from_dict",
    "gate_set_to_dict",
    "has_seam_columns",
    "image_gate",
    "load_gates",
    "objects_in_rois",
    "polygon_gate",
    "rectangle_gate",
    "rectangle_vertices",
    "save_gates",
    "seam_gate",
    "seam_table",
]
