"""Trajectories: Slingshot-style lineages, pseudotime, and what changes along them."""

import json

import numpy as np
import pandas as pd
import pytest

from vtea_core.trajectory import (
    association,
    benjamini_hochberg,
    composition_along,
    feature_trends,
    lineage_tree,
    slingshot,
)


def branching(n=200, seed=0, dims=4):
    """A Y of round clusters (as a clustering gives): trunk A -> B, then
    B -> C -> C2 (up) and B -> D (down). Returns points, labels, the true time
    (distance along the Y), and the true branch (0 trunk, 1 up, 2 down)."""
    rng = np.random.default_rng(seed)
    step = 3.0
    up, down = np.array([1.0, 1.0]) / np.sqrt(2), np.array([1.0, -1.0]) / np.sqrt(2)
    trunk_end = np.array([step, 0.0])
    pts, labs, time, branch = [], [], [], []
    for name, t, br in (
        ("A", 0.0, 0),
        ("B", step, 0),
        ("C", 2 * step, 1),
        ("C2", 3 * step, 1),
        ("D", 2 * step, 2),
    ):
        tt = t + rng.normal(0, 0.5, n)
        if br == 0:
            xy = np.column_stack([tt, rng.normal(0, 0.5, n)])
        else:
            d = up if br == 1 else down
            normal = np.array([-d[1], d[0]])
            xy = trunk_end + (tt - step)[:, None] * d + rng.normal(0, 0.5, n)[:, None] * normal
        pts.append(xy)
        labs += [name] * n
        time.append(tt)
        branch += [br] * n
    p = np.vstack(pts)
    if dims > 2:
        p = np.column_stack([p, rng.normal(0, 0.3, (len(p), dims - 2))])
    return p, np.array(labs, dtype=object), np.concatenate(time), np.array(branch)


def test_lineage_tree_finds_both_branches_from_the_start():
    p, labels, _, _ = branching()
    clusters, _, edges, lineages, dist = lineage_tree(p, labels, "A")
    assert sorted(map(tuple, lineages)) == [("A", "B", "C", "C2"), ("A", "B", "D")]
    assert len(edges) == len(clusters) - 1
    assert dist.shape == (5, 5) and np.allclose(dist, dist.T)


def test_forced_end_is_a_leaf():
    p, labels, _, _ = branching()
    *_, lineages, _ = lineage_tree(p, labels, "A", ends=["C"])
    assert any(lin[-1] == "C" for lin in lineages)
    assert not any("C" in lin[:-1] for lin in lineages)


def test_euclidean_distance_gives_the_same_tree_here():
    p, labels, _, _ = branching()
    *_, lineages, _ = lineage_tree(p, labels, "A", distance="euclidean")
    assert sorted(map(tuple, lineages)) == [("A", "B", "C", "C2"), ("A", "B", "D")]


def test_pseudotime_follows_the_true_order():
    p, labels, time, branch = branching()
    traj = slingshot(p, labels, "A")
    assert traj.n_lineages == 2
    for j, lin in enumerate(traj.lineages):
        up = "C" in lin
        own = (branch == 0) | (branch == (1 if up else 2))
        t = traj.pseudotime[:, j]
        ok = own & np.isfinite(t)
        assert ok.mean() > 0.6 * own.mean()
        assert np.corrcoef(t[ok], time[ok])[0, 1] > 0.95
        # The other branch's objects are not on this lineage.
        other = branch == (2 if up else 1)
        assert np.isnan(t[other]).mean() > 0.95
    # Trunk objects are on both lineages, at about the same pseudotime.
    trunk = (labels == "A") & np.isfinite(traj.pseudotime).all(axis=1)
    assert trunk.mean() > 0.15
    diff = np.abs(traj.pseudotime[trunk, 0] - traj.pseudotime[trunk, 1])
    assert np.median(diff) < 1.0
    avg = traj.average_pseudotime()
    assert np.isfinite(avg).all()
    main = traj.main_lineage()
    assert set(np.unique(main)) <= {1, 2}
    json.dumps(traj.to_dict())  # saved with a workspace


def test_curves_start_at_the_start_cluster():
    p, labels, _, _ = branching()
    traj = slingshot(p, labels, "A")
    a = p[labels == "A"].mean(axis=0)
    for curve in traj.curves:
        assert np.linalg.norm(curve[0] - a) < np.linalg.norm(curve[-1] - a)


def test_unlabelled_objects_are_left_out():
    p, labels, _, _ = branching()
    labels = labels.copy()
    labels[:10] = None
    traj = slingshot(p, labels, "A")
    assert np.isnan(traj.pseudotime[:10]).all()


def test_bad_input():
    p, labels, _, _ = branching(n=20)
    with pytest.raises(ValueError):
        slingshot(p, labels, "Z")
    with pytest.raises(ValueError):
        slingshot(p, labels, "A", ends=["Q"])
    with pytest.raises(ValueError):
        slingshot(p[:20], labels[:20], "A")  # one cluster
    q = p.copy()
    q[0, 0] = np.nan
    with pytest.raises(ValueError):
        slingshot(q, labels, "A")


def test_a_marker_rising_along_one_branch_is_found():
    p, labels, time, branch = branching()
    rng = np.random.default_rng(1)
    marker = np.where(branch == 1, time, 0.0) + rng.normal(0, 0.5, len(time))
    flat = rng.normal(0, 1, len(time))
    values = pd.DataFrame({"THY1": marker, "noise": flat})
    traj = slingshot(p, labels, "A")
    names = ["t1", "t2"]
    test = association(traj.pseudotime, values, weights=traj.weights, lineages=names)
    up = names[next(j for j, lin in enumerate(traj.lineages) if "C" in lin)]
    row = test[(test["lineage"] == up) & (test["feature"] == "THY1")].iloc[0]
    assert row["rho"] > 0.5 and row["q"] < 1e-6 and row["start vs end (SD)"] > 1
    noise = test[test["feature"] == "noise"]
    assert (noise["rho"].abs() < 0.2).all()
    trends = feature_trends(
        traj.pseudotime, values, weights=traj.weights, n_bins=10, lineages=names
    )
    rising = trends[(trends["lineage"] == up) & (trends["feature"] == "THY1")]["mean"].to_numpy()
    assert rising[-1] > rising[0] + 5
    assert set(trends.columns) == {
        "lineage",
        "bin",
        "pseudotime",
        "feature",
        "mean",
        "sem",
        "objects",
    }


def test_composition_along_sums_to_one_per_bin():
    p, labels, _, _ = branching()
    traj = slingshot(p, labels, "A")
    comp = composition_along(traj.pseudotime, labels, weights=traj.weights, n_bins=8)
    sums = comp.groupby(["lineage", "bin"])["fraction"].sum().dropna()
    assert np.allclose(sums, 1.0)
    first = comp[(comp["lineage"] == "t1") & (comp["bin"] == 0)].set_index("class")["fraction"]
    assert first["A"] > 0.8


def test_benjamini_hochberg():
    q = benjamini_hochberg([0.01, 0.04, 0.03, np.nan, 0.5])
    assert np.isnan(q[3])
    assert np.allclose(q[[0, 1, 2, 4]], [0.04, 0.16 / 3, 0.16 / 3, 0.5])
