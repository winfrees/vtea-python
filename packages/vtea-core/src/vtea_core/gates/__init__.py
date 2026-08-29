"""Boolean gate math over measurement tables.

Ports vtea.gates from the Java codebase. A gate is a boolean NumPy array;
combine gates with &, |, ~ directly instead of dedicated AND/OR/NOT classes.
"""

from vtea_core.gates.gate import Gate, GateSet
from vtea_core.gates.io import (
    GATES_FORMAT_VERSION,
    gate_set_from_dict,
    gate_set_to_dict,
    load_gates,
    save_gates,
)
from vtea_core.gates.polygon import polygon_gate, rectangle_gate, rectangle_vertices

__all__ = [
    "GATES_FORMAT_VERSION",
    "Gate",
    "GateSet",
    "gate_set_from_dict",
    "gate_set_to_dict",
    "load_gates",
    "polygon_gate",
    "rectangle_gate",
    "rectangle_vertices",
    "save_gates",
]
