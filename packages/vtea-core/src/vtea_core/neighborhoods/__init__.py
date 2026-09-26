"""Neighbourhoods: a second level of objects, made of segmented objects.

Four steps, each usable on its own from a script and each a protocol step:

1. `build_neighborhoods` - which objects are near which (the Java VTEA's
   Nearest-k, Spatial by cell and Spatial by point, and its randomised null
   model), as a `NeighborhoodSet`.
2. `neighborhood_features` - one row per neighbourhood: its size, place,
   density, class composition (the Java `ClassFraction`/`ClassSums`) and
   any member feature aggregated.
3. `classify_neighborhoods` - neighbourhood types, by clustering that table.
4. `reflect_neighborhoods` - the other direction: each object takes on the
   characteristics of the neighbourhoods it belongs to, so a cell can be
   gated on the kind of neighbourhood it lives in.

See docs/NEIGHBORHOODS.md for the model, and `model.py` for why membership
is a many-to-many table rather than an association.
"""

from vtea_core.neighborhoods.build import build_neighborhoods, object_positions
from vtea_core.neighborhoods.measure import (
    AGGREGATIONS,
    COMPOSITION_MEASUREMENTS,
    NEIGHBORHOOD_TYPE,
    class_column_name,
    class_values,
    classify_neighborhoods,
    composition_columns,
    neighborhood_features,
)
from vtea_core.neighborhoods.model import (
    CENTRED_METHODS,
    GRID,
    MEMBER_ID,
    METHODS,
    NEAREST,
    NEIGHBORHOOD_FORMAT_VERSION,
    NEIGHBORHOOD_ID,
    RADIUS,
    Neighborhood,
    NeighborhoodSet,
    centroid_columns,
    load_neighborhoods,
    save_neighborhoods,
)
from vtea_core.neighborhoods.reflect import (
    MEMBER,
    MISSING_CATEGORY,
    OWN,
    RELATIONS,
    reflect_neighborhoods,
    reflectable_columns,
)

__all__ = [
    "AGGREGATIONS",
    "CENTRED_METHODS",
    "COMPOSITION_MEASUREMENTS",
    "GRID",
    "MEMBER",
    "MEMBER_ID",
    "METHODS",
    "MISSING_CATEGORY",
    "NEAREST",
    "NEIGHBORHOOD_FORMAT_VERSION",
    "NEIGHBORHOOD_ID",
    "NEIGHBORHOOD_TYPE",
    "OWN",
    "RADIUS",
    "RELATIONS",
    "Neighborhood",
    "NeighborhoodSet",
    "build_neighborhoods",
    "centroid_columns",
    "class_column_name",
    "class_values",
    "classify_neighborhoods",
    "composition_columns",
    "load_neighborhoods",
    "neighborhood_features",
    "object_positions",
    "reflect_neighborhoods",
    "reflectable_columns",
    "save_neighborhoods",
]
