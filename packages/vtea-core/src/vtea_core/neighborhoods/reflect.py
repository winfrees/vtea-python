"""Handing a neighbourhood's characteristics back to the objects in it.

Neighbourhoods are objects made of objects, and the relationship runs both
ways: once a neighbourhood has been measured - its composition, its density,
the type it was clustered into - every cell in it *has* that as a property
of where it lives. This is what lets a gate say "CD8 T cells in
tumour-rich neighbourhoods", which is a statement about a cell that no
measurement of the cell itself can make.

There are two ways an object relates to neighbourhoods, and they answer
different questions:

- **its own** neighbourhood - the one centred on it (`nearest` and `radius`
  neighbourhoods have one per object). "What surrounds this cell?"
  Reflected one-to-one: the object takes its own neighbourhood's values.
- **the neighbourhoods it is a member of** - every neighbourhood it falls
  in, its own included. Grid neighbourhoods only have this kind, and it is
  the one that says what a cell is *part of* rather than what is around it.
  Reflected as a mean over those neighbourhoods for a quantity, and as the
  most frequent value for a category (a neighbourhood type), with how many
  neighbourhoods voted.

`relation="auto"` takes the own neighbourhood where there is one and the
membership otherwise.

The result is one row per object of `data`, in `data`'s order, so it lines
up with the table it is joined back onto; an object in no neighbourhood
gets NaN for a quantity and -1 for a category rather than being dropped.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from vtea_core.neighborhoods.measure import NEIGHBORHOOD_TYPE, _names
from vtea_core.neighborhoods.model import NEIGHBORHOOD_ID, NeighborhoodSet

# Columns of a neighbourhood table that locate or identify it rather than
# describe it - reflecting a neighbourhood's centroid onto a cell would give
# the cell a second, wrong, position.
_NOT_REFLECTED = frozenset({NEIGHBORHOOD_ID, "seed"})

OWN = "own"
MEMBER = "member"
RELATIONS = ("auto", OWN, MEMBER)

MISSING_CATEGORY = -1


def reflectable_columns(neighborhood_table: pd.DataFrame) -> list[str]:
    """The neighbourhood columns that describe a neighbourhood - everything
    numeric except its id, seed and position."""
    return [
        column
        for column in neighborhood_table.columns
        if column not in _NOT_REFLECTED
        and not str(column).startswith("centroid-")
        and pd.api.types.is_numeric_dtype(neighborhood_table[column])
    ]


def reflect_neighborhoods(
    neighborhoods: NeighborhoodSet,
    neighborhood_table: pd.DataFrame,
    data: pd.DataFrame,
    *,
    features: str = "",
    categorical: str = NEIGHBORHOOD_TYPE,
    relation: Literal["auto", "own", "member"] = "auto",
    prefix: str = "",
) -> pd.DataFrame:
    """Per-object columns carrying the neighbourhoods' characteristics.

    `features` (comma-separated) names the neighbourhood columns to hand
    back, by default every descriptive one (see `reflectable_columns`).
    `categorical` names which of them are categories - voted on rather
    than averaged - and is the neighbourhood type by default. `prefix` is
    put in front of every output column (`prefix` + name); a protocol step
    leaves it empty and the column takes the step's own name instead.

    Always adds `n_neighborhoods`: how many neighbourhoods the object is a
    member of, which is how crowded its surroundings are.
    """
    if relation not in RELATIONS:
        raise ValueError(f"unknown relation {relation!r}, expected one of {RELATIONS}")
    id_column = neighborhoods.id_column
    if id_column not in data.columns:
        raise ValueError(
            f"these neighbourhoods are of '{id_column}' rows, and the table has no such column"
        )
    if NEIGHBORHOOD_ID not in neighborhood_table.columns:
        raise ValueError("the neighbourhood table has no neighborhood_id column")
    if relation == OWN and not neighborhoods.centred:
        raise ValueError(
            f"'{neighborhoods.method}' neighbourhoods are not centred on objects, so no object "
            f"has one of its own; use relation='member'"
        )
    if relation == "auto":
        relation = OWN if neighborhoods.centred else MEMBER

    columns = _names(features) or reflectable_columns(neighborhood_table)
    missing = [name for name in columns if name not in neighborhood_table.columns]
    if missing:
        raise ValueError(f"no such neighbourhood column(s): {missing}")
    categories = set(_names(categorical)) & set(columns)

    object_ids = data[id_column].to_numpy()
    result = pd.DataFrame({id_column: object_ids})
    membership = neighborhoods.membership()
    counts = membership.groupby(id_column).size()
    result[f"{prefix}n_neighborhoods"] = (
        pd.Series(object_ids).map(counts).fillna(0).astype(np.int64).to_numpy()
    )

    table = neighborhood_table.set_index(NEIGHBORHOOD_ID)
    if relation == OWN:
        # A centred neighbourhood's id is its seed's id, so this is a lookup.
        own = table.reindex(object_ids)
        for name in columns:
            values = own[name].to_numpy()
            if name in categories:
                values = pd.Series(values).fillna(MISSING_CATEGORY).astype(np.int64).to_numpy()
            result[f"{prefix}{name}"] = values
        return result

    joined = membership[[NEIGHBORHOOD_ID, id_column]].join(table[columns], on=NEIGHBORHOOD_ID)
    grouped = joined.groupby(id_column)
    for name in columns:
        if name in categories:
            voted = grouped[name].agg(_most_frequent)
            result[f"{prefix}{name}"] = (
                pd.Series(object_ids).map(voted).fillna(MISSING_CATEGORY).astype(np.int64).to_numpy()
            )
        else:
            result[f"{prefix}{name}"] = pd.Series(object_ids).map(grouped[name].mean()).to_numpy()
    return result


def _most_frequent(values: pd.Series):
    """The commonest value; the smallest of any tie, so the answer does not
    depend on the order neighbourhoods happened to be listed in."""
    present = values.dropna()
    if present.empty:
        return np.nan
    counts = present.value_counts()
    return min(counts.index[counts == counts.max()])
