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
    layercake_fixture_path,
    load_label_mask,
    load_layercake_labels,
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
from scipy.spatial import cKDTree
from vtea_core.segmentation import layercake_3d, threshold_mask

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


def _centroids(labels):
    from scipy import ndimage as ndi

    ids = np.unique(labels)
    ids = ids[ids != 0]
    return np.array(ndi.center_of_mass(np.ones(labels.shape), labels, ids))


@pytest.mark.parametrize("dataset", DATASETS)
def test_layercake3d_matches_java(dataset):
    """The Java default segmentation. Objects are matched by centroid, not
    id: the Java's region order is not reproducible even between Java runs
    (see vtea_core.segmentation.layercake), and it skips regions a loop bug
    steps over - so a few per cent of objects are expected to differ, and
    the rest must be the same objects."""
    if not layercake_fixture_path(dataset).exists():
        pytest.skip("no LayerCake3D fixture yet - milestone M2 in docs/PORT_PLAN.md")
    if not (DATA_DIR / f"{dataset}.tif").exists():
        pytest.skip(f"{dataset}.tif not in tests/golden/data/ - see README.md")
    metadata = load_metadata(dataset)
    volume = load_source_channel(dataset)
    labels = layercake_3d(
        volume,
        low_threshold=float(metadata["layercake_threshold"]),
        centroid_offset=float(metadata["layercake_offset"]),
        min_size=int(metadata["layercake_min"]),
        max_size=int(metadata["layercake_max"]),
        watershed=metadata.get("layercake_watershed", "true").lower() == "true",
    )
    expected = load_layercake_labels(dataset)
    assert labels.shape == expected.shape
    assert segmentation_iou(labels > 0, expected > 0) > 0.95

    actual_centres, expected_centres = _centroids(labels), _centroids(expected)
    assert abs(len(actual_centres) - len(expected_centres)) <= 0.05 * len(expected_centres)
    distances, _ = cKDTree(expected_centres).query(actual_centres)
    assert np.mean(distances < 2.0) > 0.9


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
