"""The hierarchy as protocol steps.

A protocol threads one `Hierarchy` through its steps under the context key
`hierarchy`: one step makes it from the object table, then each step adds a
level (or types one) and hands it on, so a protocol re-runs the whole chain
- cells, windows, their types, windows on those, regions - on the next
acquisition. Every argument is a plain value the protocol builder's Edit
form can show: levels by name, populations and types by column name.

A level name that already exists is replaced, with every level built on it,
so running a protocol again gives the same hierarchy rather than an error.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from vtea_core.data.spacing import Spacing
from vtea_core.hierarchy.hierarchy import Hierarchy
from vtea_core.neighborhoods.measure import NEIGHBORHOOD_TYPE, _names, classify_neighborhoods


def hierarchy_from_objects(
    data: pd.DataFrame,
    spacing: Spacing | None = None,
    labels: np.ndarray | None = None,
    *,
    name: str = "Objects",
    id_column: str = "object_id",
    connectivity: int = 1,
) -> Hierarchy:
    """A hierarchy whose root is the object table `data`. With the label
    image (`labels`, whose values are the ids), regions can follow touching
    masks."""
    return Hierarchy(
        data,
        name=name,
        id_column=id_column,
        spacing=spacing,
        labels=labels,
        connectivity=connectivity,
    )


def _level_name(hierarchy: Hierarchy, name: str) -> str:
    """A level by name; empty means the newest."""
    return name or list(hierarchy.levels)[-1]


def _fresh(hierarchy: Hierarchy, name: str) -> None:
    if name in hierarchy.levels and name != hierarchy.root:
        hierarchy.remove(name)


def _selection(column: str):
    return column or None


def add_windows_level(
    hierarchy: Hierarchy,
    *,
    parent: str = "",
    name: str = "Windows",
    method: Literal["radius", "nearest"] = "radius",
    radius: float = 50.0,
    k: int = 10,
    classes: str = "",
    seeds: str = "",
    members: str = "",
    features: str = "",
    aggregations: str = "mean",
) -> Hierarchy:
    """Add a windows level on `parent` (empty: the newest level). `classes`:
    the parent column to measure composition over. `seeds` / `members`: a
    parent column of True/False choosing which objects get a window and
    which can be counted in one (empty: all)."""
    parent = _level_name(hierarchy, parent)
    _fresh(hierarchy, name)
    hierarchy.add_windows(
        parent,
        name,
        method=method,
        radius=radius,
        k=k,
        classes=classes or None,
        seeds=_selection(seeds),
        members=_selection(members),
        features=features,
        aggregations=aggregations,
    )
    return hierarchy


def add_regions_level(
    hierarchy: Hierarchy,
    *,
    parent: str = "",
    name: str = "Regions",
    types: str = NEIGHBORHOOD_TYPE,
    values: str = "",
    touching: bool = True,
    d: float = 0.0,
    sigma: float = 0.0,
    tau: float = 0.5,
    min_size: int = 1,
    classes: str = "",
    features: str = "",
    aggregations: str = "mean",
) -> Hierarchy:
    """Add a regions level on `parent` (empty: the newest level): contiguous
    objects with the same `types` merged. `values`: the types to make regions
    of, comma-separated (empty: all). `sigma` 0 means the median member
    spacing."""
    parent = _level_name(hierarchy, parent)
    _fresh(hierarchy, name)
    hierarchy.add_regions(
        parent,
        name,
        types=types,
        values=_values(values, hierarchy[parent].table[types]) if values else None,
        touching=touching,
        d=d,
        sigma=sigma or None,
        tau=tau,
        min_size=min_size,
        classes=classes or None,
        features=features,
        aggregations=aggregations,
    )
    return hierarchy


def _values(text: str, column: pd.Series) -> list:
    """Comma-separated type values, as numbers when the column holds numbers."""
    names = _names(text)
    if pd.api.types.is_numeric_dtype(column):
        return [float(v) if "." in v else int(v) for v in names]
    return names


def classify_level(
    hierarchy: Hierarchy,
    *,
    level: str = "",
    method: Literal["kmeans", "gaussian_mixture", "hierarchical", "leiden", "louvain"] = "kmeans",
    n_clusters: int = 5,
    features: str = "",
    normalize: Literal["none", "zscore", "robust", "minmax"] = "none",
    random_state: int = 0,
    column: str = NEIGHBORHOOD_TYPE,
) -> Hierarchy:
    """Sort a level's objects into types (by default on their composition),
    in `column` - the classes the next level up is measured over."""
    target = hierarchy[_level_name(hierarchy, level)]
    typed = classify_neighborhoods(
        target.table,
        method=method,
        n_clusters=n_clusters,
        features=features,
        normalize=normalize,
        random_state=random_state,
        column=column,
    )
    target.table[column] = typed[column].to_numpy()
    return hierarchy


def reflect_level(
    hierarchy: Hierarchy,
    *,
    level: str = "",
    onto: str = "",
    columns: str = NEIGHBORHOOD_TYPE,
    how: Literal["member", "own"] = "member",
    categorical: str = NEIGHBORHOOD_TYPE,
    prefix: str = "",
) -> pd.DataFrame:
    """A level's columns handed down to the objects of `onto` (empty: the
    root), one row per object with its id: the mean / most common over the
    objects containing it, or its own window's value."""
    level = _level_name(hierarchy, level)
    onto = onto or hierarchy.root
    out = hierarchy.reflect(level, onto, _names(columns), how=how, categorical=_names(categorical))
    out.columns = [f"{prefix}{c}" for c in out.columns]
    target = hierarchy[onto]
    out.insert(0, target.id_column, target.ids)
    return out


STEPS = {
    "hierarchy_from_objects": hierarchy_from_objects,
    "add_windows_level": add_windows_level,
    "add_regions_level": add_regions_level,
    "classify_level": classify_level,
    "reflect_level": reflect_level,
}
