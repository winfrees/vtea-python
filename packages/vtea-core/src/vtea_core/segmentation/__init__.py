"""Segmentation: threshold -> label -> (optional watershed split) -> size filter.

Composable primitives, not the ~15 overlapping classes in
vtea.objects.Segmentation. Most of those Java classes reduce to "threshold,
connected-component label, optionally watershed-split touching objects,
filter by size" using different libraries/workarounds (MorphoLibJ vs
ImgLib2 vs hand-rolled 2D-slice-linking) for want of one fast native 3D
connected-components implementation - scikit-image/scipy provide that
directly. Large-volume handling is Dask's job (see vtea_core.data), not a
separate algorithm. Cellpose/DeepImageJ-based segmentation is deferred to
Phase 3 (deep learning consolidation); ImageJ ROI import is deferred as an
I/O format concern.
"""

from vtea_core.segmentation.labeling import filter_by_size, label_components, watershed_split
from vtea_core.segmentation.manual import import_labels, labels_from_points
from vtea_core.segmentation.threshold import threshold_mask

__all__ = [
    "threshold_mask",
    "label_components",
    "watershed_split",
    "filter_by_size",
    "labels_from_points",
    "import_labels",
]
