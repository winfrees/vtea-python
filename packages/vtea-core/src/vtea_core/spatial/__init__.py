"""Spatial features of objects: where each object is relative to the others.

Once some objects have identities (classes: phenotypes, clusters, gates), each
object can be described by its surroundings, and those descriptions gated and
clustered like any measurement - to split a type further, e.g. CD8 T cells
touching tumour cells from those that are not.

Per object, for each class of interest:

| function | columns | measured between |
| --- | --- | --- |
| `nearest_distance` | `distance to <class>` | centroids |
| `neighbors_within` | `n <class> within <r>`, `fraction <class> within <r>` | centroids |
| `surface_distance` | `surface distance to <class>` | the masks themselves, voxel by voxel |
| `contacts` | `contacts <class>` | touching masks |

An object is never its own neighbour: the distance from a CD8 T cell to the
nearest CD8 T cell is to *another* one. Distances are in physical units when
the caller gives physical positions / a known `Spacing` (so anisotropic 3D
stacks are measured correctly), voxels otherwise. Objects with no class
(None / NaN) are still measured, and still count as neighbours in the
totals, but get no columns of their own.

Enrichment statistics over these neighbourhoods (cell type x neighbourhood
type, pairwise neighbour enrichment against a permutation null) are in
`vtea_core.spatial.enrichment`.
"""

from vtea_core.spatial.enrichment import neighbor_enrichment, neighbor_graph, type_enrichment
from vtea_core.spatial.features import (
    class_targets,
    contacts,
    nearest_distance,
    neighbors_within,
    surface_distance,
    touching_pairs,
)

__all__ = [
    "class_targets",
    "contacts",
    "nearest_distance",
    "neighbor_enrichment",
    "neighbor_graph",
    "neighbors_within",
    "surface_distance",
    "touching_pairs",
    "type_enrichment",
]
