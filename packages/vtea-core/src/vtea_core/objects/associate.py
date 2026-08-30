"""Associating two independently segmented channels - the step itself.

`associate_by_identity` handles the case where the child was *built* from
the parent and the ids already match. This is the other case, and the one
people will use every day: nuclei segmented from DAPI, a cytoplasm from a
cytoskeletal marker, organelles from a third channel, with nothing but
geometry to say which belongs to which.

The work is done by `vtea_core.objects.scoring` (what is the evidence?) and
`vtea_core.objects.assignment` (who gets whom, and how sure are we?). This
module is the thin layer that puts the answer into an `AssociationSet`,
carrying with every link the method, its parameters, the posterior, and the
parents that were considered and rejected - so that a result can be argued
with rather than only accepted.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from vtea_core.data.spacing import Spacing
from vtea_core.objects.assignment import MANY_TO_ONE, assign, posterior
from vtea_core.objects.association import (
    ASSIGNED,
    CONTAINED,
    Association,
    AssociationSet,
    ObjectRef,
)
from vtea_core.objects.scoring import CONTAINMENT, score_candidates


def associate_objects(
    child_labels: np.ndarray,
    parent_labels: np.ndarray,
    *,
    spacing: Spacing | None = None,
    child_name: str = "child",
    parent_name: str = "parent",
    method: Literal["containment", "centroid_distance", "boundary_distance"] = CONTAINMENT,
    mode: Literal["many_to_one", "one_to_one"] = MANY_TO_ONE,
    max_distance: float = 10.0,
    orphan_score: float = 0.05,
    min_probability: float = 0.0,
) -> AssociationSet:
    """Link each object of `child_labels` to at most one of `parent_labels`.

    `method` is how the evidence is measured - `containment` (overlap
    fraction, needs no distance), `centroid_distance`, or
    `boundary_distance` (the gap between surfaces). `mode` is how the
    competition is resolved: `many_to_one` for organelles sharing a parent,
    `one_to_one` when a parent may take only one child, which is the
    nucleus/cytoplasm case and is solved globally rather than per child.

    `max_distance` is in physical units where the voxel size is known and in
    voxels otherwise, and applies to the distance methods only.
    `min_probability` refuses a link the evidence doesn't support; children
    left over are recorded as unassigned rather than dropped.
    """
    candidates = score_candidates(
        child_labels,
        parent_labels,
        method=method,
        spacing=spacing,
        max_distance=max_distance,
    )
    matches = assign(
        posterior(candidates, orphan_score=orphan_score),
        mode=mode,
        min_probability=min_probability,
    )

    params = {
        "method": method,
        "mode": mode,
        "orphan_score": float(orphan_score),
        "min_probability": float(min_probability),
    }
    if method != CONTAINMENT:
        params["max_distance"] = float(max_distance)
    # Containment says the child is *inside* the parent, which is a stronger
    # claim than proximity and worth keeping distinguishable in the record.
    relationship = CONTAINED if method == CONTAINMENT else ASSIGNED

    associations = AssociationSet()
    for match in matches:
        child = ObjectRef(child_name, match.child_id)
        if match.parent_id is None:
            associations.add_unassigned(child)
            continue
        associations.add(
            Association(
                child=child,
                parent=ObjectRef(parent_name, match.parent_id),
                relationship=relationship,
                probability=match.probability,
                method=method,
                params=params,
                alternatives=[
                    (ObjectRef(parent_name, parent_id), probability)
                    for parent_id, probability in match.alternatives
                ],
            )
        )
    return associations
