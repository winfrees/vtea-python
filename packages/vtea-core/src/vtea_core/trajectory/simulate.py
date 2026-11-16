"""Simulated trajectories with a known truth, for validating `slingshot` and the tradeSeq tests.

`simulate_trajectory` lays out a topology (linear, bifurcation,
trifurcation, or a tree with two branch points) as straight segments in a
low-dimensional space, places objects along them with noise, cuts them into
clusters of equal length (as a clustering would), and makes features whose
behaviour along each lineage is known:

| kind | truth |
| --- | --- |
| `null` | constant everywhere (no test should find it) |
| `rise` | rises along one lineage only (association, start-vs-end, diffEnd, pattern) |
| `end` | differs only at the end of one lineage (diffEnd, pattern, association) |
| `transient` | a bump in the middle of one lineage's last segment, back to baseline at both ends (association and pattern, *not* start-vs-end or diffEnd) |
| `shared` | the same rise on every lineage (association and start-vs-end, *not* diffEnd or pattern) |

Features are Gaussian (`family="gaussian"`, like asinh-scaled intensities) or
negative-binomial counts (`family="nb"`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOPOLOGIES = {
    # name: [(segment, parent segment or None)], segments of unit length
    "linear": [("s0", None), ("s1", "s0")],
    "bifurcation": [("s0", None), ("a", "s0"), ("b", "s0")],
    "trifurcation": [("s0", None), ("a", "s0"), ("b", "s0"), ("c", "s0")],
    "tree": [("s0", None), ("a", "s0"), ("b", "s0"), ("a1", "a"), ("a2", "a")],
}
KINDS = ("null", "rise", "end", "transient", "shared")


def _layout(topology, dims):
    """Segment start / end points. Siblings leave their parent's end at angles
    100 (two) or 60 (three) degrees apart, in the plane of the parent's direction and an axis that
    changes with depth (so second-level branches spread in another plane)."""
    segs = TOPOLOGIES[topology]
    starts, ends, direction, level = {}, {}, {}, {}
    for name, parent in segs:
        if parent is None:
            starts[name], level[name] = np.zeros(dims), 0
            d = np.zeros(dims)
            d[0] = 1.0
        else:
            level[name] = level[parent] + 1
            siblings = [s for s, p in segs if p == parent]
            k = siblings.index(name)
            step = 100.0 if len(siblings) == 2 else 60.0
            angle = np.deg2rad(step * (k - (len(siblings) - 1) / 2))
            base = direction[parent]
            axis = np.zeros(dims)
            axis[1 + (level[name] - 1) % max(dims - 1, 1)] = 1.0
            ortho = axis - (axis @ base) * base
            ortho /= np.linalg.norm(ortho)
            d = np.cos(angle) * base + np.sin(angle) * ortho
            starts[name] = ends[parent]
        direction[name] = d / np.linalg.norm(d)
        ends[name] = starts[name] + direction[name] * 3.0
    return segs, starts, ends


def simulate_trajectory(
    topology: str = "bifurcation",
    *,
    n_per_segment: int = 300,
    clusters_per_segment: int = 2,
    dims: int = 3,
    noise: float = 0.35,
    branch_overlap: float = 0.2,
    n_features: int = 60,
    fraction_signal: float = 0.5,
    effect: float = 1.0,
    family: str = "gaussian",
    mean_count: float = 20.0,
    seed: int = 0,
) -> dict:
    """A simulated dataset. Returns a dict with `points` (objects x dims),
    `labels` (cluster names; the last cluster of a segment also takes the
    first `branch_overlap` of its children, as a clustering puts one cluster at
    a branch point), `start` (the first cluster), `time` (true
    pseudotime: distance along the tree from the start), `lineages` (true
    lineages as lists of clusters), `membership` (objects x lineages, True on
    the lineage), `values` (objects x features DataFrame) and `truth`
    (feature -> kind, lineage it acts on, and which tests should find it)."""
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {list(TOPOLOGIES)}")
    rng = np.random.default_rng(seed)
    segs, starts, ends = _layout(topology, max(dims, 2))
    parent = dict(segs)
    depth = {}
    for name, par in segs:
        depth[name] = 0.0 if par is None else depth[par] + 3.0
    leaves = [s for s, _ in segs if s not in parent.values()]
    paths = []
    for leaf in leaves:
        path, node = [], leaf
        while node is not None:
            path.append(node)
            node = parent[node]
        paths.append(path[::-1])
    pts, labs, time, seg_of, frac = [], [], [], [], []
    for name, _ in segs:
        u = rng.uniform(0, 1, n_per_segment)
        xy = starts[name] + u[:, None] * (ends[name] - starts[name])
        pts.append(xy + rng.normal(0, noise, xy.shape))
        cl = np.minimum((u * clusters_per_segment).astype(int), clusters_per_segment - 1)
        seg_labels = np.array([f"{name}.{c}" for c in cl], dtype=object)
        par = parent[name]
        if par is not None:  # the branch point's cluster reaches into each branch
            seg_labels[u < branch_overlap] = f"{par}.{clusters_per_segment - 1}"
        labs += list(seg_labels)
        time.append(depth[name] + 3.0 * u)
        seg_of += [name] * n_per_segment
        frac.append(u)
    points = np.vstack(pts)
    labels = np.array(labs, dtype=object)
    time = np.concatenate(time)
    seg_of = np.array(seg_of, dtype=object)
    n = len(points)
    membership = np.column_stack([np.isin(seg_of, p) for p in paths])
    lineages = [[f"{s}.{c}" for s in p for c in range(clusters_per_segment)] for p in paths]
    t_max = max(depth[p[-1]] + 3.0 for p in paths)
    equal_lengths = len({depth[p[-1]] for p in paths}) == 1
    # Features.
    n_signal = round(n_features * fraction_signal)
    kinds = ["null"] * (n_features - n_signal)
    signal_kinds = [k for k in KINDS if k != "null"]
    kinds += [signal_kinds[i % len(signal_kinds)] for i in range(n_signal)]
    rows, columns = [], {}
    branch_depth = 3.0  # where the first branch leaves the trunk
    for i, kind in enumerate(kinds):
        target = int(rng.integers(len(paths)))
        on = membership[:, target]
        own = on & ~np.all(membership, axis=1) if len(paths) > 1 else on  # after the branch point
        signal = np.zeros(n)
        tt = time / t_max
        if kind == "rise":
            signal = np.where(own, effect * (time - branch_depth) / (t_max - branch_depth), 0.0)
            if len(paths) == 1:
                signal = effect * tt
        elif kind == "end":
            last = depth[paths[target][-1]]
            signal = np.where(on & (time >= last + 1.5), effect, 0.0)
        elif kind == "transient":  # a bump mid-way along the lineage's last segment
            last = paths[target][-1]
            mid = depth[last] + 1.5
            bump = effect * np.exp(-0.5 * ((time - mid) / 0.35) ** 2)
            signal = np.where(seg_of == last, bump, 0.0)
        elif kind == "shared":
            signal = effect * tt
        if family == "gaussian":
            y = signal + rng.normal(0, 1.0, n)
        else:
            mu = mean_count * np.exp(signal)
            y = rng.negative_binomial(5, 5 / (5 + mu)).astype(float)
        name = f"f{i:03d}_{kind}"
        columns[name] = y
        expected = {
            "association": kind != "null",
            "start_vs_end": kind in ("rise", "shared", "end"),
            "diff_end": kind in ("rise", "end") and len(paths) > 1,
            "pattern": kind in ("rise", "end", "transient") and len(paths) > 1,
        }
        if kind == "shared" and not equal_lengths:  # its ends and pattern differ with length
            expected["diff_end"] = expected["pattern"] = None
        rows.append({"feature": name, "kind": kind, "lineage": target, **expected})
    return {
        "points": points,
        "labels": labels,
        "start": f"{segs[0][0]}.0",
        "time": time,
        "segment": seg_of,
        "lineages": lineages,
        "membership": membership,
        "values": pd.DataFrame(columns),
        "truth": pd.DataFrame(rows).set_index("feature"),
        "family": family,
    }
