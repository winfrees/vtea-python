"""Spatial features (distances, neighbours, contacts by class) and enrichment."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from vtea_core.data import Spacing
from vtea_core.spatial import (
    contacts,
    nearest_distance,
    neighbor_enrichment,
    neighbor_graph,
    neighbors_within,
    surface_distance,
    touching_pairs,
    type_enrichment,
)


def test_nearest_distance_skips_the_object_itself():
    points = np.array([[0, 0], [0, 3], [0, 10], [0, 14]], float)
    classes = ["A", "A", "B", None]
    out = nearest_distance(points, classes)
    assert list(out.columns) == ["distance to A", "distance to B"]
    assert out["distance to A"].tolist() == [3, 3, 7, 11]
    # The only B: no other B for it; everyone else measures to it.
    assert np.isnan(out.loc[2, "distance to B"]) and out["distance to B"].tolist()[:2] == [10, 7]
    assert nearest_distance(points, classes, targets=["C"])["distance to C"].isna().all()


def test_neighbors_within_counts_and_fractions():
    points = np.array([[0, 0], [0, 1], [0, 2], [0, 50]], float)
    out = neighbors_within(points, [1, 1, 2, 2], radius=1.5)
    assert out["n 1 within 1.5"].tolist() == [1, 1, 1, 0]
    assert out["n 2 within 1.5"].tolist() == [0, 1, 0, 0]
    assert out["fraction 2 within 1.5"].tolist()[:3] == [0.0, 0.5, 0.0]
    assert np.isnan(out.loc[3, "fraction 1 within 1.5"])  # nobody near it
    with pytest.raises(ValueError):
        neighbors_within(points, [1, 1, 2, 2], radius=0)


def _brute_surface(labels, a, b_ids, sampling):
    pa = np.argwhere(labels == a) * sampling
    pb = np.argwhere(np.isin(labels, b_ids) & (labels != a)) * sampling
    return cdist(pa, pb).min() if len(pb) else np.nan


def test_surface_distance_is_exact_and_anisotropic():
    rng = np.random.default_rng(1)
    labels = np.zeros((6, 40, 40), np.int32)
    ids, classes = [], []
    for k, (z0, y0, x0) in enumerate(rng.integers([0, 0, 0], [4, 34, 34], size=(12, 3)), start=1):
        region = labels[z0 : z0 + 2, y0 : y0 + 5, x0 : x0 + 4]
        region[region == 0] = k
        ids.append(k)
        classes.append("A" if k % 3 else "B")
    ids, classes = np.array(ids), np.array(classes, dtype=object)
    present = [i for i in ids if (labels == i).any()]
    spacing = Spacing((3.0, 0.5, 0.5))
    out = surface_distance(labels, ids, classes, spacing)
    sampling = np.array([3.0, 0.5, 0.5])
    for row, i in enumerate(ids):
        for target in ("A", "B"):
            members = ids[classes == target]
            expected = _brute_surface(labels, i, members, sampling) if i in present else np.nan
            got = out.loc[row, f"surface distance to {target}"]
            assert (np.isnan(got) and np.isnan(expected)) or got == pytest.approx(expected), (
                i,
                target,
            )


def test_touching_objects_are_one_voxel_apart_and_counted_as_contacts():
    labels = np.zeros((10, 10), np.int32)
    labels[1:4, 1:4] = 1
    labels[1:4, 4:7] = 2  # touches 1 along a face
    labels[4:6, 7:9] = 3  # touches 2 at a corner only
    labels[8:10, 0:2] = 4  # alone
    out = surface_distance(labels, [1, 2, 3, 4], ["T", "T", "U", "U"], Spacing((2.0, 2.0)))
    assert out["surface distance to T"].tolist()[:2] == [2.0, 2.0]
    assert sorted(map(tuple, touching_pairs(labels))) == [(1, 2), (2, 1)]
    assert len(touching_pairs(labels, connectivity=2)) == 4
    c = contacts(labels, [1, 2, 3, 4], ["T", "T", "U", "U"], connectivity=2)
    assert c["contacts"].tolist() == [1, 2, 1, 0]
    assert c["contacts T"].tolist() == [1, 1, 1, 0] and c["contacts U"].tolist() == [0, 1, 0, 0]


def test_type_enrichment():
    classes = ["T"] * 8 + ["B"] * 8
    types = [1] * 7 + [2] + [2] * 7 + [1]
    out = type_enrichment(classes, types).set_index(["class", "type"])
    assert out.loc[("T", 1), "observed"] == 7 and out.loc[("T", 1), "expected"] == 4
    assert out.loc[("T", 1), "log2 enrichment"] > 0 > out.loc[("T", 2), "log2 enrichment"]
    assert out.loc[("T", 1), "fraction of type"] == pytest.approx(7 / 8)


def test_neighbor_graph():
    points = np.array([[0, 0], [0, 1], [0, 3]], float)
    assert sorted(map(tuple, neighbor_graph(points, radius=1.5))) == [(0, 1), (1, 0)]
    knn = neighbor_graph(points, k=1)
    assert sorted(map(tuple, knn)) == [(0, 1), (1, 0), (2, 1)]
    with pytest.raises(ValueError):
        neighbor_graph(points)


def test_neighbor_enrichment_finds_segregated_classes():
    # Two clusters, each of one class: A next to A, never next to B.
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=(60, 2))
    b = rng.normal(0, 1, size=(60, 2)) + [20, 0]
    points = np.vstack([a, b])
    classes = ["A"] * 60 + ["B"] * 60
    out = neighbor_enrichment(points, classes, k=5, n_permutations=100).set_index(
        ["class", "neighbor"]
    )
    assert out.loc[("A", "A"), "observed"] == pytest.approx(5.0) and out.loc[("A", "A"), "z"] > 5
    assert out.loc[("A", "B"), "observed"] == 0 and out.loc[("A", "B"), "z"] < -5
    assert out.loc[("A", "B"), "p"] == pytest.approx(1 / 101)
    # Random labels: nothing stands out.
    mixed = rng.permutation(classes)
    out = neighbor_enrichment(points, mixed, radius=1.5, n_permutations=200)
    assert (out["z"].abs() < 4).all()
    pd.testing.assert_frame_equal(
        neighbor_enrichment(points, mixed, radius=1.5, n_permutations=50, random_state=3),
        neighbor_enrichment(points, mixed, radius=1.5, n_permutations=50, random_state=3),
    )
