import numpy as np
import pandas as pd
import pytest

from vtea_core.clustering import cluster_profiles, describe_clusters, top_markers


@pytest.fixture
def data():
    # Cluster 1: CD3+CD8+, cluster 2: CD3+CD4+, cluster 3: neither; one unclustered row.
    rows = ([(100, 90, 5)] * 4) + ([(100, 5, 80)] * 3) + ([(5, 5, 5)] * 5) + [(1000, 1000, 1000)]
    frame = pd.DataFrame(rows, columns=["CD3", "CD8", "CD4"])
    labels = np.array([1] * 4 + [2] * 3 + [3] * 5 + [-1])
    return frame, labels


def test_profiles(data):
    frame, labels = data
    p = cluster_profiles(frame, labels, ["CD3", "CD8", "CD4"])
    assert list(p["mean"].index) == [1, 2, 3]
    assert p["mean"].loc[1, "CD8"] == 90 and p["mean"].loc[2, "CD4"] == 80
    # z against the 12 clustered rows (the unclustered outlier is left out)
    clustered = frame.iloc[:12]
    expected = (90 - clustered["CD8"].mean()) / clustered["CD8"].std(ddof=0)
    assert p["z"].loc[1, "CD8"] == pytest.approx(expected)
    assert list(p["counts"]["objects"]) == [4, 3, 5]
    assert p["counts"]["fraction"].sum() == pytest.approx(1.0)


def test_top_markers_and_description(data):
    frame, labels = data
    z = cluster_profiles(frame, labels, ["CD3", "CD8", "CD4"])["z"]
    assert top_markers(z, 2)[1] == ["CD8", "CD3"]
    assert top_markers(z, 2, low=True)[3][0] in ("CD3", "CD8", "CD4")
    desc = describe_clusters(z)
    assert desc[1].startswith("CD8+ CD3+") and "CD4-" in desc[1]
    assert desc[3].startswith(" ") is False and desc[3].endswith("-")


def test_constant_feature_and_errors(data):
    frame, labels = data
    frame = frame.assign(flat=1.0)
    z = cluster_profiles(frame, labels, ["CD3", "flat"])["z"]
    assert (z["flat"] == 0).all()
    assert describe_clusters(z[["flat"]])[1] == "no distinctive feature"
    with pytest.raises(KeyError):
        cluster_profiles(frame, labels, ["nope"])
    with pytest.raises(ValueError):
        cluster_profiles(frame, labels[:3], ["CD3"])
