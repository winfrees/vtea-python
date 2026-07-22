"""Parity tests: vtea-core output vs. the Java-VTEA golden fixtures.

Skipped entirely until tests/golden/fixtures/ is populated (see README.md).
Individual tests are also skipped until the corresponding vtea-core
functionality exists - each one names the phase it's blocked on so it's
obvious what unblocks it. Bodies just exercise the fixture loaders for now;
the real assertions (using compare.segmentation_iou / feature_table_diff /
cluster_assignment_ari) get filled in alongside each Phase 2 port.
"""

import pytest

from fixtures import (
    fixtures_available,
    load_label_mask,
    load_measurements,
    load_synthetic_clustering_input,
    load_synthetic_kmeans_k3,
    load_synthetic_pca,
)

pytestmark = pytest.mark.skipif(
    not fixtures_available(),
    reason="Golden fixtures not present - see tests/golden/README.md to generate them",
)

DATASETS = ["AQtest_human_crop", "C1-IU_VTEA_ExampleData_001"]


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.skip(reason="vtea_core.segmentation not implemented yet (Phase 2)")
def test_single_threshold_segmentation_matches_java(dataset):
    load_label_mask(dataset)
    # actual_mask = vtea_core.segmentation.single_threshold(...)
    # assert segmentation_iou(actual_mask > 0, expected_mask > 0) > 0.99
    raise NotImplementedError


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.skip(reason="vtea_core.measurements not implemented yet (Phase 2)")
def test_measurements_match_java(dataset):
    load_measurements(dataset)
    # actual = vtea_core.measurements.extract(...)
    # diff = feature_table_diff(actual, expected, key="object_id", rtol=1e-3)
    # assert diff.empty, diff
    raise NotImplementedError


@pytest.mark.skip(reason="vtea_core.clustering not implemented yet (Phase 2)")
def test_kmeans_matches_java_on_synthetic_data():
    load_synthetic_clustering_input()
    load_synthetic_kmeans_k3()
    # actual_labels = vtea_core.clustering.kmeans(data[["x", "y"]].to_numpy(), n_clusters=3)
    # assert cluster_assignment_ari(actual_labels, expected["cluster"].to_numpy()) > 0.95
    raise NotImplementedError


@pytest.mark.skip(reason="vtea_core.reduction not implemented yet (Phase 2)")
def test_pca_matches_java_on_synthetic_data():
    load_synthetic_clustering_input()
    load_synthetic_pca()
    # actual = vtea_core.reduction.pca(data[["x", "y"]].to_numpy(), n_components=2)
    # PCA sign/axis order isn't guaranteed to match; compare explained variance instead
    # of raw components directly.
    raise NotImplementedError
