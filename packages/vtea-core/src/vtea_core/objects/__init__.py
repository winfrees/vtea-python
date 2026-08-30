"""Relationships between objects of different segmentations.

A label image says which voxels belong to object 7; it says nothing about
whether object 7 of one segmentation is the nucleus *of* object 12 of
another. That statement is what this package holds - see
docs/OBJECT_ASSOCIATION.md for the design and the phases it lands in.
"""

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
from vtea_core.objects.identity import associate_by_identity

__all__ = [
    "ASSIGNED",
    "ASSOCIATION_FORMAT_VERSION",
    "CONTAINED",
    "DERIVED",
    "MANUAL",
    "Association",
    "AssociationSet",
    "ObjectRef",
    "associate_by_identity",
    "load_associations",
    "save_associations",
]
