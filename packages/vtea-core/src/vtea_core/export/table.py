"""The feature table as a file anything can open, with its data dictionary.

Java VTEA's main output is the per-object table exported as CSV; this is
that, plus the part Java never wrote: one row per column saying what the
column is and how it was produced (`FeatureCatalog.to_dataframe()`), written
beside the table as `<name>.dictionary.csv`. A `mean_ch2` column is
meaningless to a reader without it.

CSV for opening in Excel, R or anything else; Parquet for tables large
enough that CSV is slow, written through DuckDB (already a dependency) so
pyarrow is not needed.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from vtea_core.measurements.catalog import (
    INTENSITY,
    KNOWN_MEASUREMENTS,
    FeatureCatalog,
    FeatureDescriptor,
    classify_column,
)

TABLE_FORMATS = (".csv", ".parquet")
DICTIONARY_SUFFIX = ".dictionary.csv"


def dictionary_path_for(path: str | Path) -> Path:
    """`objects.csv` -> `objects.dictionary.csv`, beside it."""
    path = Path(path)
    return path.with_name(path.stem + DICTIONARY_SUFFIX)


def data_dictionary(frame: pd.DataFrame, catalog: FeatureCatalog | None = None) -> pd.DataFrame:
    """One row per column of `frame`, in the table's own column order.

    Columns the catalog records are described from the step that produced
    them. The rest - columns added outside a protocol, gate memberships - are
    still listed, described only as far as their name says, so the
    dictionary never silently skips a column the table has.
    """
    catalog = catalog or FeatureCatalog()
    described = FeatureCatalog()
    for column in frame.columns:
        name = str(column)
        descriptor = catalog.get(name) or _describe_from_name(name)
        described.add(descriptor)
    return described.to_dataframe()


# What a column is when nothing recorded it and its name is not one the
# measurement steps produce: said plainly rather than guessed at.
UNRECORDED = "unrecorded"


def _describe_from_name(name: str) -> FeatureDescriptor:
    kind, measurement, channel = classify_column(name)
    if kind == INTENSITY and measurement not in KNOWN_MEASUREMENTS:
        return FeatureDescriptor(name=name, kind=UNRECORDED)
    return FeatureDescriptor(name=name, kind=kind, measurement=measurement, channel=channel)


def export_table(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    catalog: FeatureCatalog | None = None,
    dictionary: bool = True,
) -> list[Path]:
    """Write `frame` to `path` (.csv or .parquet, from the suffix) and, by
    default, its data dictionary beside it. Returns the paths written."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in TABLE_FORMATS:
        raise ValueError(f"cannot export a table as {suffix or 'no extension'}; use .csv or .parquet")
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        _write_parquet(frame, path)
    written = [path]
    if dictionary:
        dictionary_path = dictionary_path_for(path)
        data_dictionary(frame, catalog).to_csv(dictionary_path, index=False)
        written.append(dictionary_path)
    return written


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    connection = duckdb.connect()
    try:
        connection.register("exported", frame)
        escaped = str(path).replace("'", "''")
        connection.execute(f"COPY exported TO '{escaped}' (FORMAT PARQUET)")
    finally:
        connection.close()
