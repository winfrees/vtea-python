"""Per-object feature/measurement extraction, backed by skimage.measure.regionprops_table.

Ports vtea.objects measurement classes and vteaobjects.MicroObject from the Java
codebase. Results are stored as rows in a DuckDB/pandas table rather than a
per-object Java object graph (replaces vtea.jdbc.H2DatabaseEngine).
"""
