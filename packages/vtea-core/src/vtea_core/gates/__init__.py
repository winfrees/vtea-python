"""Boolean gate math over measurement tables.

Ports vtea.gates from the Java codebase. A gate is a boolean NumPy array;
combine gates with &, |, ~ directly instead of dedicated AND/OR/NOT classes.
"""

from vtea_core.gates.polygon import polygon_gate, rectangle_gate

__all__ = ["polygon_gate", "rectangle_gate"]
