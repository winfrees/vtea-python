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

from collections.abc import Callable

from vtea_core.classification import class_map
from vtea_core.clustering import (
    auto_k_kmeans,
    gaussian_mixture,
    hierarchical,
    kmeans,
    leiden,
    louvain,
)
from vtea_core.gates import polygon_gate, rectangle_gate
from vtea_core.imageprocessing import (
    enhance_contrast,
    gaussian_blur,
    median_filter,
    subtract_background,
)
from vtea_core.measurements import (
    extract_measurements,
    extract_measurements_by_channel,
    weighted_measurements_by_channel,
)
from vtea_core.objects import (
    associate_by_identity,
    associate_objects,
    build_cells,
    cell_features,
    distance_ownership,
    merge_associations,
)
from vtea_core.reduction import isomap, laplacian_eigenmap, pca, tsne, umap
from vtea_core.segmentation import (
    cellpose_segmentation,
    expand_labels,
    filter_by_size,
    label_components,
    label_ring,
    label_shell,
    labels_from_points,
    restrict_labels_to,
    subtract_labels,
    threshold_mask,
    watershed_ownership,
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
        # Derived from another segmentation by morphology, keeping its ids -
        # a nuclear envelope, a cytosol band - so the association between
        # them is exact rather than inferred.
        "expand_labels": expand_labels,
        "label_ring": label_ring,
        "label_shell": label_shell,
        "subtract_labels": subtract_labels,
        "restrict_labels_to": restrict_labels_to,
        # Divides a region among the objects inside it - the
        # deterministic answer to which cell owns a contested voxel.
        "watershed_ownership": watershed_ownership,
    },
    "association": {
        "associate_by_identity": associate_by_identity,
        "associate_objects": associate_objects,
        # Two association steps become one hierarchy: a cell spans both
        # nucleus <- cytoplasm and cytoplasm <- lysosome.
        "merge_associations": merge_associations,
    },
    "ownership": {
        "distance_ownership": distance_ownership,
    },
    "cells": {
        "build_cells": build_cells,
        "cell_features": cell_features,
    },
    "measurements": {
        "extract_measurements": extract_measurements,
        "extract_measurements_by_channel": extract_measurements_by_channel,
        # Measures a probabilistic ownership instead of a hard label
        # image: a count becomes an expected volume and a mean a
        # probability-weighted mean.
        "weighted_measurements_by_channel": weighted_measurements_by_channel,
    },
    "clustering": {
        "kmeans": kmeans,
        "gaussian_mixture": gaussian_mixture,
        "hierarchical": hierarchical,
        "auto_k_kmeans": auto_k_kmeans,
        # Community detection on a shared-neighbour graph: finds how many
        # populations there are rather than being told. Listed even where
        # python-igraph/leidenalg are not installed - unlike the
        # deep-learning steps below, the import that would fail is inside
        # the function, so the step names a missing package when it is run
        # instead of silently vanishing from the menu.
        "louvain": louvain,
        "leiden": leiden,
    },
    "reduction": {
        "pca": pca,
        "isomap": isomap,
        "laplacian_eigenmap": laplacian_eigenmap,
        "tsne": tsne,
        # Same arrangement as the graph clustering above: umap-learn is
        # optional, and the step says so when run rather than not appearing.
        "umap": umap,
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
