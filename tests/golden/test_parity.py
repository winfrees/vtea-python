"""Parity tests: vtea-core output vs. the Java-VTEA golden fixtures.

Skipped entirely until tests/golden/fixtures/ is populated (see README.md);
the image-derived tests additionally need the sample TIFFs in
tests/golden/data/. Everything they exercise is ported, so a skip here means
"no fixtures yet", never "not implemented".
"""

import numpy as np
import pytest
from compare import cluster_assignment_ari, feature_table_diff, segmentation_iou
from fixtures import (
    DATA_DIR,
    fixtures_available,
    load_label_mask,
    load_measurements,
    load_metadata,
    load_source_channel,
    load_synthetic_clustering_input,
    load_synthetic_kmeans_k3,
    load_synthetic_pca,
)
from vtea_core.clustering import kmeans
from vtea_core.measurements import extract_measurements
from vtea_core.reduction import pca
from vtea_core.segmentation import threshold_mask

pytestmark = pytest.mark.skipif(
    not fixtures_available(),
    reason="Golden fixtures not present - see tests/golden/README.md to generate them",
)

DATASETS = ["AQtest_human_crop", "C1-IU_VTEA_ExampleData_001"]


def _single_threshold(dataset):
    """What SingleThreshold3D does: channel 0 >= the threshold, truncated to
    an int as the generator's protocol passes it, as one object."""
    if not (DATA_DIR / f"{dataset}.tif").exists():
        pytest.skip(f"{dataset}.tif not in tests/golden/data/ - see README.md")
    volume = load_source_channel(dataset)
    threshold = int(float(load_metadata(dataset)["threshold"].split()[0]))
    labels = threshold_mask(volume, method="fixed", value=threshold).astype(np.int32)
    return volume, labels


@pytest.mark.parametrize("dataset", DATASETS)
def test_single_threshold_segmentation_matches_java(dataset):
    _, labels = _single_threshold(dataset)
    expected = load_label_mask(dataset)
    assert labels.shape == expected.shape
    assert segmentation_iou(labels > 0, expected > 0) > 0.99


@pytest.mark.parametrize("dataset", DATASETS)
def test_measurements_match_java(dataset):
    volume, labels = _single_threshold(dataset)
    expected = load_measurements(dataset)
    actual = extract_measurements(labels, volume)[list(expected.columns)]
    diff = feature_table_diff(actual, expected, key="object_id", rtol=1e-3)
    assert diff.empty, diff


def test_kmeans_matches_java_on_synthetic_data():
    data = load_synthetic_clustering_input()
    expected = load_synthetic_kmeans_k3()
    actual = kmeans(data[["x", "y"]].to_numpy(), n_clusters=3, random_state=0)
    assert cluster_assignment_ari(actual, expected["cluster"].to_numpy()) > 0.95


def test_pca_matches_java_on_synthetic_data():
    data = load_synthetic_clustering_input()
    expected = load_synthetic_pca()
    actual = pca(data[["x", "y"]].to_numpy(), n_components=2)
    # Sign, centring and scale are implementation choices; the axes are not.
    # Each component must be the same line through the data, up to sign.
    for index in range(actual.shape[1]):
        r = np.corrcoef(actual[:, index], expected[f"pc{index + 1}"].to_numpy())[0, 1]
        assert abs(r) > 0.999, f"pc{index + 1}: |r| = {abs(r):.4f}"
