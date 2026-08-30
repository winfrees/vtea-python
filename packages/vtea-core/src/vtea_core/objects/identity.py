"""Associating a derived segmentation with the one it was derived from.

The easy half of the problem, and worth having as its own function rather
than being implicit: a segmentation made by `vtea_core.segmentation.derived`
keeps its parent's label, so object *k* of the envelope is the envelope of
nucleus *k* by construction. Nothing is inferred, nothing has a posterior,
and every link is certain.

Stating it explicitly is what lets the rest of the system - cell identity,
per-cell features, the saved record - treat a derived relationship and an
inferred one the same way, while the `relationship` field keeps them
distinguishable to anyone reading the result.
"""

from __future__ import annotations

import numpy as np

from vtea_core.objects.association import DERIVED, Association, AssociationSet, ObjectRef


def associate_by_identity(
    child_labels: np.ndarray,
    parent_labels: np.ndarray,
    *,
    child_name: str = "child",
    parent_name: str = "parent",
    require_parent: bool = True,
) -> AssociationSet:
    """Link objects that share a label id.

    For a derived segmentation this is exactly right and exactly certain.
    A child id with no matching parent means the two segmentations are not
    what this function assumes - a genuine mistake rather than an unlucky
    object - so it raises by default rather than quietly dropping it.
    """
    child_ids = set(np.unique(np.asarray(child_labels))) - {0}
    parent_ids = set(np.unique(np.asarray(parent_labels))) - {0}

    orphans = sorted(child_ids - parent_ids)
    if orphans and require_parent:
        raise ValueError(
            f"{len(orphans)} object(s) in '{child_name}' have no object of the same id in "
            f"'{parent_name}' (first few: {orphans[:5]}). Identity association is for a "
            f"segmentation derived from another, which keeps its parent's ids; two "
            f"independently segmented channels need an inferred association instead."
        )

    return AssociationSet(
        [
            Association(
                child=ObjectRef(child_name, int(object_id)),
                parent=ObjectRef(parent_name, int(object_id)),
                relationship=DERIVED,
                probability=1.0,
                method="identity",
            )
            for object_id in sorted(child_ids & parent_ids)
        ]
    )
