"""DuckDB-backed storage for per-object measurement tables.

Replaces vtea.jdbc.H2DatabaseEngine from the Java codebase: an in-memory
H2 database with MEASUREMENTS/OBJECTS tables accessed via raw JDBC. This
gives the same "session-scoped SQL-queryable table" role with first-class
pandas/Arrow interop instead.
"""

from __future__ import annotations

import pandas as pd
import duckdb


class MeasurementStore:
    """A DuckDB in-memory connection with pandas DataFrames registered as tables."""

    def __init__(self):
        self._con = duckdb.connect(database=":memory:")

    def register(self, table_name: str, frame: pd.DataFrame) -> None:
        """(Re-)registers a DataFrame as a queryable table under `table_name`."""
        self._con.register(table_name, frame)

    def query(self, sql: str) -> pd.DataFrame:
        return self._con.execute(sql).fetch_df()

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "MeasurementStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
