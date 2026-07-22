"""Mapping per-object class predictions back onto a label image.

Pure NumPy - no torch dependency, works with predictions from any
classifier (the CNN in vtea_core.classification.cnn, or anything else).
"""

from __future__ import annotations

import numpy as np


def class_map(labels: np.ndarray, object_ids: np.ndarray, class_labels: np.ndarray) -> np.ndarray:
    """Replaces each object's pixels with its predicted class id.

    `object_ids[i]` -> `class_labels[i]`; pixels belonging to an object not
    listed in `object_ids` (including background) are left at 0. Uses the
    same label-remap pattern as segmentation.filter_by_size().

    Class id 0 is reserved for "unmapped/background", same as label id 0 -
    if 0 is a meaningful class in your classifier's output space, shift
    class ids by +1 before calling this.
    """
    object_ids = np.asarray(object_ids, dtype=np.int64)
    class_labels = np.asarray(class_labels)
    if object_ids.shape != class_labels.shape:
        raise ValueError(f"object_ids shape {object_ids.shape} != class_labels shape {class_labels.shape}")

    remap = np.zeros(int(labels.max()) + 1, dtype=class_labels.dtype)
    remap[object_ids] = class_labels
    return remap[labels]
