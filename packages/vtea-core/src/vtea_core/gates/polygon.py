"""Point-in-shape gating over 2D measurement coordinates.

Replaces vtea.gates (GateMath/AbstractGateMath/AND) from the Java codebase.
There, "AND" is a whole plugin class (~300 lines, with five overloaded
process() variants for combinations of ROI and PolygonGate shapes) that
combines two point-in-polygon tests. Here a gate is just a boolean array,
and AND/OR/NOT are exactly `&`/`|`/`~` on top of NumPy - no dedicated
combinator classes are needed:

    gated = polygon_gate(x, y, v1) & polygon_gate(x, y, v2)
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path


def polygon_gate(x: np.ndarray, y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Boolean mask of which (x, y) points fall inside a polygon.

    vertices: (N, 2) array of polygon vertex coordinates in (x, y) order.
    """
    points = np.column_stack([np.asarray(x), np.asarray(y)])
    return Path(vertices).contains_points(points)


def rectangle_gate(
    x: np.ndarray, y: np.ndarray, *, x_min: float, x_max: float, y_min: float, y_max: float
) -> np.ndarray:
    """Boolean mask for an axis-aligned rectangular gate (replaces ROI-based gating)."""
    x = np.asarray(x)
    y = np.asarray(y)
    return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)


def rectangle_vertices(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """The four corners of the rectangle spanned by two opposite corners,
    counter-clockwise, as a polygon.

    A rectangle drawn by dragging on the plot is stored as a 4-vertex
    polygon rather than its own gate type: Gate already carries vertices,
    and every downstream consumer (membership, overlay drawing, JSON) then
    treats the two kinds identically. The corners are ordered rather than
    taken as given so a rectangle dragged right-to-left or bottom-to-top
    still encloses its interior.
    """
    left, right = sorted((float(x0), float(x1)))
    bottom, top = sorted((float(y0), float(y1)))
    return np.array(
        [[left, bottom], [right, bottom], [right, top], [left, top]], dtype=float
    )
