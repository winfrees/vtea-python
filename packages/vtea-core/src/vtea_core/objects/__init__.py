"""Relationships between objects of different segmentations.

A label image says which voxels belong to object 7; it says nothing about
whether object 7 of one segmentation is the nucleus *of* object 12 of
another. That statement is what this package holds - see
docs/OBJECT_ASSOCIATION.md for the design and the phases it lands in.

Two ways to arrive at it. `associate_by_identity` for a segmentation derived
from another, where the ids already match and every link is certain;
`associate_objects` for two channels segmented independently, where the
evidence is geometric, the answer has a posterior, and some children turn
out to have no parent at all.
"""

from vtea_core.objects.assignment import (
    ASSIGNMENT_MODES,
    MANY_TO_ONE,
    ONE_TO_ONE,
    Match,
    Posterior,
    assign,
    posterior,
)
from vtea_core.objects.associate import associate_objects
from vtea_core.objects.association import (
    ASSIGNED,
    ASSOCIATION_FORMAT_VERSION,
    CONTAINED,
    DERIVED,
    MANUAL,
    Association,
    AssociationSet,
    ObjectRef,
    load_associations,
    save_associations,
)
from vtea_core.objects.cells import (
    AGGREGATIONS,
    CELL_FORMAT_VERSION,
    DEFAULT_AGGREGATIONS,
    Cell,
    CellSet,
    build_cells,
    cell_features,
    load_cells,
    merge_associations,
    save_cells,
)
from vtea_core.objects.identity import associate_by_identity
from vtea_core.objects.ownership import (
    DISTANCE,
    OWNERSHIP_FORMAT_VERSION,
    WATERSHED,
    Ownership,
    distance_ownership,
    load_ownership,
)
from vtea_core.objects.scoring import (
    BOUNDARY_DISTANCE,
    CENTROID_DISTANCE,
    CONTAINMENT,
    SCORING_METHODS,
    CandidateScores,
    boundary_distance,
    centroid_distance,
    containment,
    score_candidates,
)

__all__ = [
    "AGGREGATIONS",
    "ASSIGNED",
    "ASSIGNMENT_MODES",
    "ASSOCIATION_FORMAT_VERSION",
    "BOUNDARY_DISTANCE",
    "CELL_FORMAT_VERSION",
    "CENTROID_DISTANCE",
    "CONTAINED",
    "CONTAINMENT",
    "DEFAULT_AGGREGATIONS",
    "DERIVED",
    "DISTANCE",
    "MANUAL",
    "MANY_TO_ONE",
    "ONE_TO_ONE",
    "OWNERSHIP_FORMAT_VERSION",
    "SCORING_METHODS",
    "WATERSHED",
    "Association",
    "AssociationSet",
    "CandidateScores",
    "Cell",
    "CellSet",
    "Match",
    "ObjectRef",
    "Ownership",
    "Posterior",
    "assign",
    "associate_by_identity",
    "associate_objects",
    "boundary_distance",
    "build_cells",
    "centroid_distance",
    "cell_features",
    "containment",
    "distance_ownership",
    "load_associations",
    "load_cells",
    "load_ownership",
    "merge_associations",
    "posterior",
    "save_associations",
    "save_cells",
    "score_candidates",
]
