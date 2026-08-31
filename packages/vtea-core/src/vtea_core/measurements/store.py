"""DuckDB-backed storage for per-object measurement tables.

Replaces vtea.jdbc.H2DatabaseEngine from the Java codebase: an in-memory
H2 database with MEASUREMENTS/OBJECTS tables accessed via raw JDBC. This
gives the same "session-scoped SQL-queryable table" role with first-class
pandas/Arrow interop instead.

Two ways to put a table in. `register` holds a DataFrame in memory, which is
right for the tens of thousands of objects a normal field produces.
`register_parquet` points at a file instead, which is what a dataset larger
than memory needs: DuckDB reads Parquet lazily, and it spills its own
aggregations and joins to disk, so a table of ten million objects is queried
rather than loaded. The SQL is identical either way, which is the point -
nothing downstream needs to know which kind of table it is looking at.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd


def _quote_literal(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _quote_identifier(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def write_measurements(frame: pd.DataFrame, path: str | os.PathLike) -> Path:
    """Write a measurement table where it can outlive the process.

    Parquet rather than CSV: it keeps dtypes, it compresses a table that is
    mostly float columns to a fraction of its size, and DuckDB can query it
    in place without reading it. A measurement table is also the one part of
    a large run worth keeping - the images can be regenerated from the
    protocol, the objects cannot be re-found without re-running everything.

    Written through DuckDB rather than through pandas, which would need
    `pyarrow` - a hundred megabytes of dependency for a format the database
    already in this package reads and writes natively.
    """
    path = Path(os.fspath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.register("frame", frame)
        connection.sql("SELECT * FROM frame").write_parquet(str(path))
    finally:
        connection.close()
    return path


def read_measurements(path: str | os.PathLike) -> pd.DataFrame:
    """The whole table, in memory. Use `MeasurementStore.register_parquet`
    instead for one that does not fit."""
    connection = duckdb.connect()
    try:
        return connection.execute(
            "SELECT * FROM read_parquet(?)", [str(Path(os.fspath(path)).resolve())]
        ).fetch_df()
    finally:
        connection.close()


class MeasurementStore:
    """A DuckDB in-memory connection with pandas DataFrames registered as tables."""

    def __init__(self):
        self._con = duckdb.connect(database=":memory:")

    def register(self, table_name: str, frame: pd.DataFrame) -> None:
        """(Re-)registers a DataFrame as a queryable table under `table_name`."""
        self._con.register(table_name, frame)

    def register_parquet(self, table_name: str, path: str | os.PathLike) -> None:
        """Point a table at a Parquet file instead of holding it in memory.

        The table is read as the query needs it, so this works on a file
        larger than RAM. Registered as a view rather than copied in, so
        re-running the analysis and rewriting the file updates the table
        without re-registering it.
        """
        # A CREATE VIEW cannot be prepared, so the path and the name go in
        # as literals - quoted the way SQL wants them rather than trusted,
        # since a filesystem path is allowed to contain a quote and a table
        # name comes from a step name.
        location = _quote_literal(str(Path(os.fspath(path)).resolve()))
        name = _quote_identifier(table_name)
        self._con.execute(
            f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({location})"
        )

    def query(self, sql: str) -> pd.DataFrame:
        return self._con.execute(sql).fetch_df()

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> MeasurementStore:  # noqa: PYI034 - typing.Self needs Python 3.11+, this package supports 3.10
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
