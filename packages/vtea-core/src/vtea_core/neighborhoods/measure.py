"""Measuring neighbourhoods, and sorting them into types.

A neighbourhood's measurements are about its members: how many there are,
what fraction of them is each class, and - for any per-object feature -
how it is distributed among them. The table this produces has one row per
neighbourhood and is plotted, gated and clustered like any other; clustering
it is what turns "the composition around each cell" into a small number of
neighbourhood *types*, which is what `reflect` then hands back to the cells.

The composition columns keep the Java VTEA's names, so a table exported
here reads like one exported there:

| Java plugin | column | value |
| --- | --- | --- |
| `ClassFraction` | `Class_<c>_ClassFraction` | percent of the members in class c (0-100) |
| `ClassSums` | `Class_<c>_ClassSums` | members in class c |
| `TotalObjects` | `n_objects` | members |

and so do the classes: every integer from 0 (or the smallest class, if it is
negative) up to the largest, whether or not a class in between is ever
seen, as `NeighborhoodMeasurementsProcessor` enumerated them. A class
column that is not integer-valued - a label set's names, say - is
enumerated as its sorted distinct values instead.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from vtea_core.neighborhoods.model import NEIGHBORHOOD_ID, NeighborhoodSet

AGGREGATIONS = ("mean", "median", "sum", "std", "min", "max")
COMPOSITION_MEASUREMENTS = ("ClassFraction", "ClassSums")

# The column `classify_neighborhoods` writes, and which `reflect` hands back
# to every member as the kind of neighbourhood it lives in.
NEIGHBORHOOD_TYPE = "neighborhood_type"


def _names(value) -> list[str]:
    """A comma-separated string (what a protocol's Edit form produces) or a
    list, as a list of names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part) for part in value]


def class_values(values: pd.Series) -> list:
    """The classes a composition is reported over - see the module docstring."""
    present = pd.Series(values).dropna()
    if present.empty:
        return []
    numeric = pd.to_numeric(present, errors="coerce")
    if numeric.notna().all() and np.allclose(numeric, np.round(numeric)):
        low = min(0, int(numeric.min()))
        return list(range(low, int(numeric.max()) + 1))
    return sorted(present.astype(str).unique())


def class_column_name(value, measurement: str) -> str:
    return f"Class_{value}_{measurement}"


def neighborhood_features(
    neighborhoods: NeighborhoodSet,
    data: pd.DataFrame,
    *,
    class_column: str = "",
    features: str = "",
    aggregations: str = "mean",
) -> pd.DataFrame:
    """One row per neighbourhood, measured from its members' rows in `data`.

    Always: `neighborhood_id`, `seed` (the object it is centred on, -1 for a
    grid point), `n_objects`, `centroid-*` (where it is, in voxels - so it
    can be drawn and plotted like any object) and `mean_distance` (how far
    its members sit from its centre, in the unit it was built in: a density
    measure). With a `class_column`, the Java composition columns (see the
    module docstring). With `features` - a comma-separated list of `data`'s
    columns - each reduced over the members by every one of `aggregations`
    (comma-separated, from AGGREGATIONS), as `<aggregation>_<feature>`.
    """
    id_column = neighborhoods.id_column
    if id_column not in data.columns:
        raise ValueError(
            f"these neighbourhoods are of '{id_column}' rows, and the table has no such column "
            f"(columns: {list(data.columns)})"
        )
    reductions = _names(aggregations)
    unknown = set(reductions) - set(AGGREGATIONS)
    if unknown:
        raise ValueError(f"unknown aggregation(s) {sorted(unknown)}, expected {list(AGGREGATIONS)}")
    wanted = _names(features)
    missing = [name for name in wanted if name not in data.columns]
    if missing:
        raise ValueError(f"no such feature(s) in the table: {missing}")
    if class_column and class_column not in data.columns:
        raise ValueError(
            f"no class column '{class_column}' in the table - pick the column holding each "
            f"object's class (a clustering, a class, a gate)"
        )

    ids = neighborhoods.ids()
    frame = pd.DataFrame({NEIGHBORHOOD_ID: ids})
    frame["seed"] = [-1 if n.seed is None else n.seed for n in neighborhoods]
    frame["n_objects"] = neighborhoods.sizes()
    centers = neighborhoods.centers()
    ndim = centers.shape[1] if centers.size else 0
    for axis in range(ndim):
        frame[f"centroid-{axis}"] = centers[:, axis]
    if len(frame) == 0:
        return frame

    membership = neighborhoods.membership()
    member = neighborhoods.member_column
    rows = membership.merge(
        data.rename(columns={id_column: member}), on=member, how="left", suffixes=("", "_member")
    )

    frame["mean_distance"] = [n.mean_distance for n in neighborhoods]

    if class_column:
        classes = class_values(data[class_column])
        key = rows[class_column]
        if classes and isinstance(classes[0], str):
            key = key.astype(str)
        counts = (
            pd.crosstab(rows[NEIGHBORHOOD_ID], key)
            .reindex(index=ids, columns=classes, fill_value=0)
            .fillna(0)
        )
        totals = frame["n_objects"].to_numpy(dtype=float)
        # Java's order: every class's fraction, then every class's sum.
        for value in classes:
            frame[class_column_name(value, "ClassFraction")] = (
                100.0 * counts[value].to_numpy(dtype=float) / np.where(totals > 0, totals, 1)
            )
        for value in classes:
            frame[class_column_name(value, "ClassSums")] = counts[value].to_numpy(dtype=np.int64)

    if wanted:
        grouped = rows.groupby(NEIGHBORHOOD_ID)[wanted]
        for reduction in reductions:
            block = grouped.agg(reduction).reindex(ids)
            for name in wanted:
                frame[f"{reduction}_{name}"] = block[name].to_numpy()
    return frame


def composition_columns(frame: pd.DataFrame, measurement: str = "ClassFraction") -> list[str]:
    """The composition columns of a neighbourhood table, of one measurement."""
    suffix = f"_{measurement}"
    return [
        column
        for column in frame.columns
        if str(column).startswith("Class_") and str(column).endswith(suffix)
    ]


def classify_neighborhoods(
    neighborhood_table: pd.DataFrame,
    *,
    method: Literal["kmeans", "gaussian_mixture", "hierarchical", "leiden", "louvain"] = "kmeans",
    n_clusters: int = 5,
    features: str = "",
    normalize: Literal["none", "zscore", "robust", "minmax"] = "none",
    random_state: int | None = 0,
    column: str = NEIGHBORHOOD_TYPE,
) -> pd.DataFrame:
    """The neighbourhood table with a neighbourhood type per row, in `column`.

    Clusters the neighbourhoods on `features` (comma-separated), by default
    their class composition (`Class_*_ClassFraction`) - which is the usual
    meaning of "a kind of neighbourhood": the same mix of cells. `method`
    is any of the object clusterings; `n_clusters` is ignored by the graph
    methods, which find their own number. `normalize` as for every
    clustering step.
    """
    from vtea_core import clustering

    columns = _names(features) or composition_columns(neighborhood_table)
    if not columns:
        raise ValueError(
            "nothing to classify neighbourhoods on: name the features to use, or measure the "
            "neighbourhoods with a class column so they have a composition"
        )
    missing = [name for name in columns if name not in neighborhood_table.columns]
    if missing:
        raise ValueError(f"no such neighbourhood feature(s): {missing}")
    result = neighborhood_table.copy()
    if len(result) == 0:
        result[column] = np.empty(0, dtype=np.int64)
        return result
    matrix = np.nan_to_num(result[columns].to_numpy(dtype=float))

    if method in ("kmeans", "gaussian_mixture"):
        n = max(1, min(int(n_clusters), len(result)))
        labels = getattr(clustering, method)(
            matrix, n, random_state=random_state, normalize=normalize
        )
    elif method == "hierarchical":
        n = max(1, min(int(n_clusters), len(result)))
        labels = clustering.hierarchical(matrix, n, normalize=normalize)
    elif method in ("leiden", "louvain"):
        labels = getattr(clustering, method)(
            matrix,
            n_neighbors=min(15, max(len(result) - 1, 1)),
            random_state=random_state,
            normalize=normalize,
        )
    else:
        raise ValueError(f"unknown method {method!r}")
    result[column] = np.asarray(labels, dtype=np.int64)
    return result
