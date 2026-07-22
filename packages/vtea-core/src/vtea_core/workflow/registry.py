"""Maps category -> step name -> callable, for both the Pipeline engine and
the napari protocol-builder widget's "Add Step" menu.

Replaces the role vtea.services' 13 SciJava-PluginService-backed classes
played in the Java codebase (populating each dropdown/menu of available
algorithms) - here it's a plain nested dict over the real functions built
in Phases 2-3, rather than reflection-based plugin discovery. A real
entry-points-based registry (for third-party extensions) can replace this
later without changing Pipeline's interface.
"""

from __future__ import annotations

from typing import Callable

from vtea_core.classification import class_map
from vtea_core.clustering import auto_k_kmeans, gaussian_mixture, hierarchical, kmeans
from vtea_core.gates import polygon_gate, rectangle_gate
from vtea_core.imageprocessing import enhance_contrast, gaussian_blur, median_filter, subtract_background
from vtea_core.measurements import extract_measurements
from vtea_core.reduction import isomap, laplacian_eigenmap, pca, tsne
from vtea_core.segmentation import (
    cellpose_segmentation,
    filter_by_size,
    label_components,
    labels_from_points,
    threshold_mask,
    watershed_split,
)

STEP_REGISTRY: dict[str, dict[str, Callable]] = {
    "imageprocessing": {
        "gaussian_blur": gaussian_blur,
        "median_filter": median_filter,
        "enhance_contrast": enhance_contrast,
        "subtract_background": subtract_background,
    },
    "segmentation": {
        "threshold_mask": threshold_mask,
        "label_components": label_components,
        "watershed_split": watershed_split,
        "filter_by_size": filter_by_size,
        "labels_from_points": labels_from_points,
        "cellpose_segmentation": cellpose_segmentation,
    },
    "measurements": {
        "extract_measurements": extract_measurements,
    },
    "clustering": {
        "kmeans": kmeans,
        "gaussian_mixture": gaussian_mixture,
        "hierarchical": hierarchical,
        "auto_k_kmeans": auto_k_kmeans,
    },
    "reduction": {
        "pca": pca,
        "isomap": isomap,
        "laplacian_eigenmap": laplacian_eigenmap,
        "tsne": tsne,
    },
    "gates": {
        "polygon_gate": polygon_gate,
        "rectangle_gate": rectangle_gate,
    },
    "classification": {
        "class_map": class_map,
    },
}

try:
    from vtea_core.classification import predict, train_classifier

    STEP_REGISTRY["classification"]["train_classifier"] = train_classifier
    STEP_REGISTRY["classification"]["predict"] = predict
except ImportError:
    pass  # torch (the deeplearning extra) not installed


def get_step_function(category: str, function_name: str) -> Callable:
    try:
        return STEP_REGISTRY[category][function_name]
    except KeyError as exc:
        raise KeyError(f"unknown step '{category}.{function_name}'") from exc


def available_steps(category: str | None = None) -> dict:
    """The full registry, or one category's {name: function} mapping."""
    if category is None:
        return STEP_REGISTRY
    if category not in STEP_REGISTRY:
        raise KeyError(f"unknown category {category!r}, expected one of {list(STEP_REGISTRY)}")
    return STEP_REGISTRY[category]
