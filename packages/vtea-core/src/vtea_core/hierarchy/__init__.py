"""A spatial hierarchy: neighbourhoods as objects, built again on themselves.

Level 0 is the segmentation. Each level above is made of the objects of the
one below, and is the same kind of thing - a table of objects with
positions and features - so it can be gated, clustered, classified and
built on again, to any depth. Two builders:

- **windows** (`Hierarchy.add_windows`): a window around each object,
  overlapping, so an object is in many. They describe what surrounds each
  object, and how that changes (an infiltrate's front, its hinterland).
- **regions** (`Hierarchy.add_regions`): contiguous objects of one type
  merged, without overlap - functional tissue units (a tubule, a glomerulus).
  Touching masks at level 0; above it, candidates whose seeds touch or are
  within a distance, joined when their members sit close to each other.

Information moves both ways: `reflect` hands a level's features down to the
objects it contains, `aggregate` summarises them up, and `members_of` /
`containers_of` follow a selection between levels.

See docs/HIERARCHY.md.
"""

from vtea_core.hierarchy.build import WINDOW_METHODS, build_regions, build_windows, member_weights
from vtea_core.hierarchy.features import FAMILIES, transform_composition
from vtea_core.hierarchy.hierarchy import HIERARCHY_FORMAT_VERSION, Hierarchy
from vtea_core.hierarchy.level import KINDS, LEVEL_ID, OBJECTS, REGIONS, WINDOWS, ObjectLevel
from vtea_core.hierarchy.measure import aggregate, composition, shape

__all__ = [
    "FAMILIES",
    "HIERARCHY_FORMAT_VERSION",
    "KINDS",
    "LEVEL_ID",
    "OBJECTS",
    "REGIONS",
    "WINDOWS",
    "WINDOW_METHODS",
    "Hierarchy",
    "ObjectLevel",
    "aggregate",
    "build_regions",
    "build_windows",
    "composition",
    "member_weights",
    "shape",
    "transform_composition",
]
