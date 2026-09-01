"""The protocol steps that make classes and label sets.

These replace the `gates` category of the step registry. A gate is drawn on
a plot with a mouse, which is an Object Explorer gesture and never was a
protocol step: the two registered gate steps needed vertices nothing in a
protocol produces, so they could only ever be configured by typing polygon
coordinates into a form. What a protocol genuinely needs is the *rule* half
- "these objects count as tubule epithelium" - written down so it re-runs on
the next acquisition without anyone drawing anything again.

So the category is `classes`, and a class is one of three things the
request names:

1. a **range** of a measured feature (`mean_ch2` from 50 to 150),
2. a **single gate** or napari ROI or cluster id - anything already in the
   table as a column,
3. a **combination** of those with AND / OR / NOT / XOR / XNOR / NAND / NOR
   (see vtea_core.classes.expression).

`label_set` then groups classes into the set an object's labels come from,
and `combine_label_sets` puts two sets together into the hierarchy that
makes "immune > CD3+" a population you can count.

Every step here takes the measurement *table* rather than a feature matrix:
a class is written in terms of column names, so the names have to survive
into the step. That is why none of these declares a `feature_input` in
vtea_core.workflow.wiring - the matrix conversion that clustering wants
would throw away exactly what these need.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from vtea_core.classes.expression import evaluate, referenced_columns
from vtea_core.classes.labelset import (
    COMBINE_MODES,
    CROSS,
    LabelSet,
    ObjectLabel,
    combine_label_sets,
)

# Columns that are row labels rather than measurements, and so never
# something to build a class out of by accident.
_IDENTIFIER_COLUMNS = ("object_id", "cell_id", "label")


def class_from_range(
    data: pd.DataFrame,
    column: str = "",
    minimum: float | None = None,
    maximum: float | None = None,
    inclusive: bool = True,
) -> np.ndarray:
    """Objects whose `column` lies between `minimum` and `maximum`.

    The plainest kind of class, and the one the request spells out:
    "mean_ch2 from 50 to 150". Either bound may be left unset for an open
    range - `minimum=50` alone is "50 and up".

    Returns a boolean per row, which the builder folds into the table as a
    column named after the step, so later classes can refer to it by name.
    """
    if not column:
        raise ValueError("a range class needs a column to be a range of")
    if column not in data.columns:
        raise ValueError(
            f"no column {column!r} in the table (columns: {', '.join(map(str, data.columns))})"
        )
    values = pd.to_numeric(data[column], errors="coerce")
    mask = np.ones(len(data), dtype=bool)
    if minimum is not None:
        mask &= (values >= minimum).to_numpy() if inclusive else (values > minimum).to_numpy()
    if maximum is not None:
        mask &= (values <= maximum).to_numpy() if inclusive else (values < maximum).to_numpy()
    # NaN compares false against every bound; with both bounds open that
    # would silently include the objects the measurement failed on.
    return mask & values.notna().to_numpy()


def class_from_values(data: pd.DataFrame, column: str = "", values: str = "") -> np.ndarray:
    """Objects whose `column` is one of `values` - a comma-separated list.

    This is how a categorical output becomes a class: cluster 3 and 7 of a
    k-means or Leiden step (`values="3, 7"`), the objects inside ROI 2 of a
    napari Labels layer, or the members of one gate (`values="True"`).
    Values are matched numerically where the column is numeric and as text
    otherwise, so `values="T cell"` works on a column of names.
    """
    if not column:
        raise ValueError("a value class needs a column to read")
    if column not in data.columns:
        raise ValueError(
            f"no column {column!r} in the table (columns: {', '.join(map(str, data.columns))})"
        )
    wanted = [part.strip() for part in str(values).split(",") if part.strip()]
    if not wanted:
        raise ValueError("a value class needs at least one value, e.g. '3, 7'")
    series = data[column]
    if pd.api.types.is_bool_dtype(series):
        booleans = {value.lower() in ("true", "1", "yes") for value in wanted}
        return np.asarray(series.isin(list(booleans)))
    if pd.api.types.is_numeric_dtype(series):
        try:
            numbers = [float(value) for value in wanted]
        except ValueError as exc:
            raise ValueError(
                f"{column!r} holds numbers, so its values must be numbers, not {values!r}"
            ) from exc
        return np.asarray(series.isin(numbers))
    return np.asarray(series.astype(str).isin(wanted))


def class_from_expression(data: pd.DataFrame, expression: str = "") -> np.ndarray:
    """Objects satisfying a boolean expression over the table's columns.

    The general case, and the only one that can express what the request
    asks for in one step: combinations of gates, napari ROI labels and
    clustering outputs with AND / OR / NOT / XOR / XNOR / NAND / NOR, plus
    comparisons and ranges. See vtea_core.classes.expression for the
    grammar; a few examples:

        gate_high AND NOT roi_tubule
        kmeans_1 in [3, 7] OR 50 <= mean_ch2 <= 150
        leiden_1 == 0 XNOR gate_dim
    """
    return evaluate(expression, data)


def class_columns(data: pd.DataFrame) -> list[str]:
    """The boolean columns of a table - what a label set is built from.

    Every class step writes one, so this is "the classes defined so far",
    without the caller having to track them.
    """
    return [
        str(column)
        for column in data.columns
        if column not in _IDENTIFIER_COLUMNS and pd.api.types.is_bool_dtype(data[column])
    ]


def label_set(
    data: pd.DataFrame,
    classes: str = "",
    name: str = "labels",
    parent: str = "",
) -> LabelSet:
    """Group class columns into one label set.

    `classes` is a comma-separated list of the boolean columns to take as
    labels; empty means every boolean column in the table, which is the
    usual case - the classes defined above it in the protocol. `parent`
    names the coarser set this one refines, so a hierarchy can be walked
    back up afterwards.

    An object may be in several of them: that is the point. `LabelSet`
    keeps the whole membership and reports how many objects carry more than
    one label rather than quietly picking a winner.
    """
    wanted = [part.strip() for part in classes.split(",") if part.strip()]
    if not wanted:
        wanted = class_columns(data)
    missing = [column for column in wanted if column not in data.columns]
    if missing:
        raise ValueError(
            f"no column(s) {', '.join(missing)} in the table - a label set is built from "
            f"class columns, which the class steps above it produce"
        )
    if not wanted:
        raise ValueError(
            "no classes to group: add a class step (range, values or expression) first"
        )
    built = LabelSet(name or "labels", n_objects=len(data), parent=parent)
    ids = data["object_id"].to_numpy() if "object_id" in data.columns else None
    if ids is not None:
        built.object_ids = ids
    for column in wanted:
        built.add(
            ObjectLabel(
                name=str(column),
                mask=np.asarray(data[column], dtype=bool),
                definition=str(column),
                source="label_set",
            )
        )
    return built


def combine_labels(
    label_set: LabelSet,
    other: LabelSet,
    mode: Literal["cross", "union", "intersect"] = CROSS,
    name: str = "",
    keep_unmatched: bool = True,
) -> LabelSet:
    """Put two label sets together - the hierarchy step.

    `cross` refines the first set by the second ("immune > CD3+"), which is
    what builds levels of precision; `union` keeps both sets' labels side by
    side; `intersect` keeps only the labels both sets define, where they
    agree. See vtea_core.classes.labelset.combine_label_sets.
    """
    if mode not in COMBINE_MODES:
        raise ValueError(f"unknown combine mode {mode!r}, expected one of {COMBINE_MODES}")
    return combine_label_sets(
        label_set, other, name=name, mode=mode, keep_unmatched=keep_unmatched
    )


def class_definition_columns(expression: str) -> set[str]:
    """Which columns a class definition reads - re-exported here so the GUI
    can check a definition against the table before running it."""
    return referenced_columns(expression)
