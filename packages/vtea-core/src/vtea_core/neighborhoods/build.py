"""Drawing neighbourhoods: which objects are near which.

Ports the three methods of the Java Object Explorer's "Neighborhood
analysis" (`XYExplorationPanel.addNeighborhoodFromGate`):

| Java | here | a neighbourhood is |
| --- | --- | --- |
| Nearest-k | `method="nearest"` | an object and its `k` nearest others |
| Spatial by cell | `method="radius"` | an object and everything within `radius` of it |
| Spatial by point | `method="grid"` | everything within `radius` of a point on a grid `interval` apart |

and the Java's "Randomize" option, as `randomize=True`.

Distances are measured between object centroids, in physical units when a
voxel size is known - the Java took a "Voxel Z scale" factor for this
instead, which is the special case of a spacing whose x and y are 1. With
no spacing, distances are in voxels, and `radius`/`interval` are too.

Differences from the Java, made deliberately:

- **Randomize** permutes which object sits at which position, rather than
  drawing new positions. Both destroy the spatial arrangement, which is the
  point of a null model; a permutation also keeps the tissue's density -
  crowded where the tissue is crowded - so what it removes is who is next to
  whom and nothing else. The Java drew fresh positions, which also removes
  the density, and so answers a different question.
- **The grid** spans the objects' own extent (or `extent`, when given),
  starting half an interval in, rather than the image's width from pixel
  `interval`: the table does not carry the image size, and a grid that
  starts at the corner of an empty margin samples the margin.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from vtea_core.data.spacing import Spacing
from vtea_core.neighborhoods.model import (
    GRID,
    METHODS,
    NEAREST,
    RADIUS,
    Neighborhood,
    NeighborhoodSet,
    centroid_columns,
)


def object_positions(
    data: pd.DataFrame, spacing: Spacing | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(voxel positions, physical positions) of every row's centroid.

    The two differ only by the voxel size: voxel positions are what a
    neighbourhood's centre is drawn at on the image, physical ones what its
    distances are measured in.
    """
    columns = centroid_columns(data)
    if not columns:
        raise ValueError(
            "neighbourhoods need object positions, and this table has no centroid-* columns "
            f"(columns: {list(data.columns)})"
        )
    voxels = data[columns].to_numpy(dtype=float)
    scale = _scale(spacing, len(columns))
    return voxels, voxels * scale


def _scale(spacing: Spacing | None, ndim: int) -> np.ndarray:
    if spacing is None or not spacing.is_known:
        return np.ones(ndim)
    return np.asarray(spacing.for_ndim(ndim), dtype=float)


def _grid_points(low: np.ndarray, high: np.ndarray, interval: float) -> np.ndarray:
    """A regular grid over [low, high] with `interval` between points, in
    physical units, starting half an interval in. An axis shorter than two
    intervals gets one point, at its middle - as the Java does."""
    axes = []
    for start, stop in zip(low, high):
        length = stop - start
        if length <= 2 * interval:
            axes.append(np.array([(start + stop) / 2]))
        else:
            axes.append(np.arange(start + interval / 2, stop, interval))
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([axis.ravel() for axis in mesh], axis=1)


def build_neighborhoods(
    data: pd.DataFrame,
    *,
    method: Literal["radius", "nearest", "grid"] = "radius",
    radius: float = 50.0,
    k: int = 10,
    interval: float = 0.0,
    spacing: Spacing | None = None,
    randomize: bool = False,
    random_state: int | None = 0,
    id_column: str = "object_id",
    source: str = "",
    extent: tuple[float, ...] | None = None,
) -> NeighborhoodSet:
    """The neighbourhoods of every object in `data`, a per-object (or per-cell)
    table with `centroid-*` columns and an id column.

    `method`: `"radius"` (Java "Spatial by cell") - each object and every
    object within `radius` of it; `"nearest"` (Java "Nearest-k") - each
    object and its `k` nearest; `"grid"` (Java "Spatial by point") - every
    object within `radius` of each point of a grid `interval` apart (0 means
    `interval = radius`, the Java default), empty points dropped.

    `radius` and `interval` are in the spacing's unit when a voxel size is
    known, voxels otherwise. `randomize` permutes positions among the
    objects before anything is measured, as a null model - see the module
    docstring. `extent` bounds the grid (voxels, one value per axis); by
    default it is the objects' own extent.

    A centred neighbourhood takes its seed object's id, so the neighbourhood
    table joins back to the object table on id.
    """
    if method not in METHODS:
        raise ValueError(f"unknown neighbourhood method {method!r}, expected one of {METHODS}")
    if id_column not in data.columns:
        raise ValueError(
            f"the table has no '{id_column}' column to identify its objects by "
            f"(columns: {list(data.columns)})"
        )
    if method == NEAREST and k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if method in (RADIUS, GRID) and radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    ids = data[id_column].to_numpy(dtype=np.int64)
    voxels, physical = object_positions(data, spacing)
    scale = _scale(spacing, voxels.shape[1])
    if randomize:
        order = np.random.default_rng(random_state).permutation(len(ids))
        voxels, physical = voxels[order], physical[order]

    params = {
        "method": method,
        "radius": float(radius),
        "k": int(k),
        "interval": float(interval),
        "randomize": bool(randomize),
        "random_state": random_state,
        "unit": spacing.unit if spacing is not None and spacing.is_known else "voxel",
    }
    result = NeighborhoodSet([], method, source, id_column, params)
    if len(ids) == 0:
        return result

    tree = cKDTree(physical)
    neighborhoods: list[Neighborhood] = []

    def centred(row: int, hits) -> Neighborhood:
        rows = np.union1d(np.asarray(hits, dtype=np.int64), [row])
        spread = np.linalg.norm(physical[rows] - physical[row], axis=1).mean()
        return Neighborhood(
            int(ids[row]),
            tuple(sorted(int(ids[index]) for index in rows)),
            tuple(float(value) for value in voxels[row]),
            seed=int(ids[row]),
            mean_distance=float(spread),
        )

    if method == NEAREST:
        wanted = min(int(k) + 1, len(ids))  # the object itself comes back first
        _distances, found = tree.query(physical, k=wanted)
        found = np.asarray(found).reshape(len(ids), wanted)
        neighborhoods = [centred(row, found[row]) for row in range(len(ids))]
    elif method == RADIUS:
        hits = tree.query_ball_point(physical, r=radius)
        neighborhoods = [centred(row, hits[row]) for row in range(len(ids))]
    else:
        step = float(interval) if interval and interval > 0 else float(radius)
        if extent is not None:
            low = np.zeros(voxels.shape[1])
            high = np.asarray(extent, dtype=float)[-voxels.shape[1]:] * scale
        else:
            low, high = physical.min(axis=0), physical.max(axis=0)
        points = _grid_points(low, high, step)
        for point, hits in zip(points, tree.query_ball_point(points, r=radius)):
            if not hits:
                continue
            spread = np.linalg.norm(physical[hits] - point, axis=1).mean()
            neighborhoods.append(
                Neighborhood(
                    len(neighborhoods) + 1,
                    tuple(sorted(int(ids[index]) for index in hits)),
                    tuple(float(value) for value in point / scale),
                    mean_distance=float(spread),
                )
            )
        params["interval"] = step
    return NeighborhoodSet(neighborhoods, method, source, id_column, params)
