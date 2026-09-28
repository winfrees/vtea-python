"""How a level above the cells is drawn: a hull per entity, filled with a
pattern.

A neighbourhood has no pixels of its own; it is drawn as the outline around
its members, grown by half the typical spacing between them so the outline
encloses the members rather than passing through their centres. The fill is
a pattern rather than a flat colour so the cells underneath stay visible:
napari has no patterned fill, so hatching is drawn as line segments clipped
to the outline and dots as points inside it.

All in plain numpy (and scipy for the hull), in the image's own (row,
column) coordinates, so it is testable without a viewer and reusable by
anything that draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.path import Path
from scipy.spatial import ConvexHull, QhullError, cKDTree

CIRCLE_SEGMENTS = 24


def typical_spacing(points: np.ndarray) -> float:
    """The median distance from each point to its nearest neighbour - the
    scale an outline is padded by. 1.0 when there is nothing to measure."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 1.0
    distances, _ = cKDTree(points).query(points, k=2)
    spacing = float(np.median(distances[:, 1]))
    return spacing if spacing > 0 else 1.0


def hull_polygon(points: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """The convex hull of 2D `points` (n, 2), grown outward by `pad`, as a
    closed-by-implication polygon (m, 2). One or two points - or points on a
    line - give a padded disc or capsule rather than nothing."""
    points = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(points) == 0:
        return np.empty((0, 2))
    pad = max(float(pad), 0.5)
    angles = np.linspace(0, 2 * np.pi, CIRCLE_SEGMENTS, endpoint=False)
    circle = np.stack([np.sin(angles), np.cos(angles)], axis=1) * pad
    # A hull of discs around every point is the padded hull; building it from
    # the discs handles the one-point and collinear cases the same way.
    cloud = (points[:, None, :] + circle[None, :, :]).reshape(-1, 2)
    try:
        hull = ConvexHull(cloud)
    except QhullError:
        return cloud
    return cloud[hull.vertices]


def contains(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    return Path(np.asarray(polygon)).contains_points(np.asarray(points))


def _clip_to_convex(polygon: np.ndarray, start: np.ndarray, end: np.ndarray):
    """The part of the segment start-end inside a convex polygon
    (Cyrus-Beck), or None."""
    direction = end - start
    low, high = 0.0, 1.0
    count = len(polygon)
    centre = polygon.mean(axis=0)
    for index in range(count):
        a, b = polygon[index], polygon[(index + 1) % count]
        edge = b - a
        normal = np.array([-edge[1], edge[0]])
        if np.dot(normal, centre - a) < 0:
            normal = -normal  # point it inward
        denominator = np.dot(normal, direction)
        numerator = np.dot(normal, start - a)
        if abs(denominator) < 1e-12:
            if numerator < 0:
                return None
            continue
        t = -numerator / denominator
        if denominator > 0:
            low = max(low, t)
        else:
            high = min(high, t)
        if low > high:
            return None
    return np.stack([start + low * direction, start + high * direction])


def hatch_lines(polygon: np.ndarray, spacing: float, angle_degrees: float = 45.0) -> list[np.ndarray]:
    """Parallel line segments `spacing` apart at `angle_degrees`, clipped to a
    convex polygon - one (2, 2) array per segment."""
    polygon = np.asarray(polygon, dtype=float)
    if len(polygon) < 3 or spacing <= 0:
        return []
    angle = np.deg2rad(angle_degrees)
    along = np.array([np.sin(angle), np.cos(angle)])
    across = np.array([-along[1], along[0]])
    centre = polygon.mean(axis=0)
    reach = float(np.max(np.linalg.norm(polygon - centre, axis=1))) + spacing
    segments = []
    for offset in np.arange(-reach, reach + spacing, spacing):
        base = centre + offset * across
        clipped = _clip_to_convex(polygon, base - reach * along, base + reach * along)
        if clipped is not None and np.linalg.norm(clipped[1] - clipped[0]) > 1e-9:
            segments.append(clipped)
    return segments


def dot_grid(polygon: np.ndarray, spacing: float) -> np.ndarray:
    """Points on a square grid `spacing` apart, inside the polygon."""
    polygon = np.asarray(polygon, dtype=float)
    if len(polygon) < 3 or spacing <= 0:
        return np.empty((0, 2))
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    rows = np.arange(low[0] + spacing / 2, high[0], spacing)
    columns = np.arange(low[1] + spacing / 2, high[1], spacing)
    grid = np.stack(np.meshgrid(rows, columns, indexing="ij"), axis=-1).reshape(-1, 2)
    if len(grid) == 0:
        return grid
    return grid[contains(polygon, grid)]


def pattern_for(polygon: np.ndarray, pattern: str, spacing: float):
    """The fill of one outline: ("lines", [segments]), ("points", array), or
    (None, None) for "none" and "solid" (solid is the outline's own face)."""
    if pattern == "hatch":
        return "lines", hatch_lines(polygon, spacing, 45.0)
    if pattern == "crosshatch":
        return "lines", hatch_lines(polygon, spacing, 45.0) + hatch_lines(polygon, spacing, 135.0)
    if pattern == "dots":
        return "points", dot_grid(polygon, spacing)
    return None, None


def level_outlines(graph, level: str, ids=None) -> list[tuple[int, int | None, np.ndarray]]:
    """Outlines of the entities of a neighbourhood level: one
    (entity id, z slice or None, polygon) per slice each one spans.

    Drawn from where the members sit at the level below, in that level's
    `centroid-*` coordinates (the image's own voxels): the (row, column)
    hull, grown by half the typical spacing between members so it encloses
    them, repeated on every z slice between the members' lowest and
    highest. `ids` restricts it to some entities - the gated ones.
    """
    target = graph.level(level)
    neighborhoods = getattr(target, "neighborhoods", None)
    if neighborhoods is None:
        return []
    below = next((link.child for link in graph.links if link.parent == level), None)
    if below is None:
        return []
    source = graph.level(below)
    columns = sorted(
        (c for c in source.table.columns if str(c).startswith("centroid-")),
        key=lambda name: int(str(name).rsplit("-", 1)[1]),
    )
    if len(columns) < 2:
        return []
    positions = pd.DataFrame(
        source.table[columns].to_numpy(dtype=float), index=source.table[source.id_column].to_numpy()
    )
    planar = positions.to_numpy()[:, -2:]
    pad = 0.5 * typical_spacing(planar)
    wanted = None if ids is None else {int(one) for one in np.asarray(ids).ravel()}

    outlines = []
    for neighborhood in neighborhoods:
        if wanted is not None and neighborhood.neighborhood_id not in wanted:
            continue
        members = positions.reindex(list(neighborhood.members)).dropna().to_numpy()
        if len(members) == 0:
            continue
        polygon = hull_polygon(members[:, -2:], pad)
        if members.shape[1] == 2:
            outlines.append((neighborhood.neighborhood_id, None, polygon))
            continue
        low, high = int(np.floor(members[:, 0].min())), int(np.ceil(members[:, 0].max()))
        for z in range(low, high + 1):
            outlines.append((neighborhood.neighborhood_id, z, polygon))
    return outlines
