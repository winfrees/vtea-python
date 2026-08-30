"""Per-object feature/measurement extraction, backed by skimage.measure.regionprops_table.

Ports vtea.objects measurement classes and vteaobjects.MicroObject from the Java
codebase. Results are stored as rows in a DuckDB/pandas table rather than a
per-object Java object graph (replaces vtea.jdbc.H2DatabaseEngine).
"""

from vtea_core.measurements.catalog import (
    DERIVED,
    GEOMETRY,
    IDENTIFIER,
    INTENSITY,
    FeatureCatalog,
    FeatureDescriptor,
    classify_column,
)
from vtea_core.measurements.regionprops import (
    GEOMETRY_COLUMNS,
    extract_measurements,
    extract_measurements_by_channel,
    feature_matrix,
    is_feature_column,
    parse_feature_name,
    threshold_mean,
)
from vtea_core.measurements.store import MeasurementStore
from vtea_core.measurements.weighted import (
    weighted_measurements,
    weighted_measurements_by_channel,
)

__all__ = [
    "DERIVED",
    "GEOMETRY",
    "GEOMETRY_COLUMNS",
    "IDENTIFIER",
    "INTENSITY",
    "FeatureCatalog",
    "FeatureDescriptor",
    "MeasurementStore",
    "classify_column",
    "extract_measurements",
    "extract_measurements_by_channel",
    "feature_matrix",
    "is_feature_column",
    "parse_feature_name",
    "threshold_mean",
    "weighted_measurements",
    "weighted_measurements_by_channel",
]
