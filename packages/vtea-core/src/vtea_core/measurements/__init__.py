"""Per-object feature/measurement extraction, backed by skimage.measure.regionprops_table.

Ports vtea.objects measurement classes and vteaobjects.MicroObject from the Java
codebase. Results are stored as rows in a DuckDB/pandas table rather than a
per-object Java object graph (replaces vtea.jdbc.H2DatabaseEngine).

Measurement *algorithms* (Mean, Sum, StdDev, ...) are Phase 2 - see
PORT_PLAN.md. MeasurementStore (storage) lands in Phase 1 alongside
VolumeDataset since it's part of the core data model, not an algorithm.
"""

from vtea_core.measurements.store import MeasurementStore

__all__ = ["MeasurementStore"]
