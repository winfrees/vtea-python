"""Lineages and pseudotime from clusters: a Slingshot-style trajectory.

Slingshot (Street et al., BMC Genomics 19:477, 2018) finds trajectories in two
stages, and this is a port of that design to plain numpy / scipy:

1. **Lineages.** Clusters (cell types, clusters, neighbourhood types) are the
   nodes of a graph; the distance between two clusters is the Mahalanobis-like
   distance Slingshot uses, d(i, j)^2 = (m_i - m_j)' (S_i + S_j)^-1 (m_i - m_j),
   with m the cluster means and S their covariances. The minimum spanning tree
   (MST) of that graph, rooted at a **start** cluster chosen by the user (e.g.
   the healthy, high-LRP2 proximal tubule cluster), gives one lineage per leaf:
   the path root -> leaf. **End** clusters, when given, are forced to be leaves
   (the MST is built without them and each is attached to its nearest other
   cluster), as in Slingshot's `end.clus`.
2. **Curves.** Each lineage starts as the polyline through its clusters' means
   and is refined into a principal curve (Hastie & Stuetzle 1989): the objects
   of the lineage's clusters are projected onto the curve (their arc length is
   the pseudotime), the curve is re-fitted as a smooth of the coordinates
   against arc length, and so on. Lineages that share their first clusters are
   **shrunk** towards their mean curve over the shared part, so they share a
   trunk and part where the data part (Slingshot's simultaneous principal
   curves). Objects in clusters shared by several lineages then get a weight
   for each from their distance to each curve, and weights below `min_weight`
   are dropped: an object on the trunk counts for every lineage, an object near
   one branch only for that one.

Pseudotime is the arc length along the curve from the start, in the units of
the space (e.g. PCA of scaled features, or a UMAP map). NaN where an object is
not on a lineage.

What is not ported: Slingshot's artificial "omega" cluster that lets a graph
fall apart into separate trees, and its smoothing spline (a Gaussian smoother
on binned arc length is used, O(n) in the objects).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree


@dataclass
class Trajectory:
    """Lineages, curves and pseudotime of one trajectory analysis.

    - `clusters`: the cluster names, in the order of `centers` and the graph.
    - `centers`: (clusters, dims) cluster means in the space.
    - `edges`: [(cluster, cluster)] of the cluster tree (MST).
    - `start`, `ends`: the start cluster and the forced end clusters.
    - `lineages`: one list of clusters per lineage, start first.
    - `curves`: one (points, dims) array per lineage, start first.
    - `pseudotime`: (objects, lineages), arc length along each curve (NaN = not
      on it).
    - `weights`: (objects, lineages), 0..1, how much each object belongs to
      each lineage.
    """

    clusters: list
    centers: np.ndarray
    edges: list
    start: object
    ends: list
    lineages: list
    curves: list
    pseudotime: np.ndarray
    weights: np.ndarray
    distances: np.ndarray = field(default=None, repr=False)

    @property
    def n_lineages(self) -> int:
        return len(self.lineages)

    def average_pseudotime(self) -> np.ndarray:
        """One pseudotime per object: the weighted mean over its lineages (as
        Slingshot's `averagePseudotime`); NaN for objects on none."""
        t = np.where(np.isfinite(self.pseudotime), self.pseudotime, 0.0)
        w = np.where(np.isfinite(self.pseudotime), self.weights, 0.0)
        total = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (t * w).sum(axis=1) / total
        out[total <= 0] = np.nan
        return out

    def main_lineage(self) -> np.ndarray:
        """Each object's lineage with the largest weight, 1..L (0 = none). Ties
        (objects on a shared trunk) go to the first lineage."""
        w = np.where(np.isfinite(self.pseudotime), self.weights, 0.0)
        out = np.argmax(w, axis=1) + 1
        out[w.max(axis=1, initial=0.0) <= 0] = 0
        return out

    def to_dict(self) -> dict:
        """The lineage structure and curves (not the per-object arrays), JSON-safe."""
        return {
            "clusters": [_plain(c) for c in self.clusters],
            "centers": np.asarray(self.centers, float).tolist(),
            "edges": [[_plain(a), _plain(b)] for a, b in self.edges],
            "start": _plain(self.start),
            "ends": [_plain(c) for c in self.ends],
            "lineages": [[_plain(c) for c in lin] for lin in self.lineages],
            "curves": [np.asarray(c, float).tolist() for c in self.curves],
        }


def cluster_distances(points, labels, clusters, *, distance: str = "slingshot") -> np.ndarray:
    """(clusters, clusters) distances between cluster means: "slingshot" (the
    covariance-scaled distance of Slingshot), or "euclidean"."""
    points = np.asarray(points, float)
    labels = np.asarray(labels, dtype=object)
    centers = np.vstack([points[labels == c].mean(axis=0) for c in clusters])
    k, d = centers.shape
    if distance == "euclidean":
        diff = centers[:, None, :] - centers[None, :, :]
        return np.sqrt((diff**2).sum(axis=2))
    if distance != "slingshot":
        raise ValueError(f"unknown distance {distance!r} (slingshot or euclidean)")
    covs = []
    for c in clusters:
        x = points[labels == c]
        cov = np.cov(x, rowvar=False) if len(x) > 1 else np.zeros((d, d))
        covs.append(np.atleast_2d(cov))
    out = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            diff = centers[i] - centers[j]
            s = covs[i] + covs[j]
            # A tiny ridge keeps flat clusters (one object, or collinear) invertible.
            s = s + np.eye(d) * 1e-9 * max(np.trace(s) / d, 1e-12)
            value = float(diff @ np.linalg.pinv(s) @ diff)
            out[i, j] = out[j, i] = np.sqrt(max(value, 0.0))
    return out


def lineage_tree(points, labels, start, *, ends=None, distance: str = "slingshot"):
    """The cluster tree and its lineages, without the curves.

    Returns (clusters, centers, edges, lineages, distances): `lineages` are the
    paths from `start` to every leaf of the minimum spanning tree (each forced
    end is a leaf), in the order of the leaves' distance from the start along the
    tree (nearest first)."""
    points = np.asarray(points, float)
    labels = np.asarray(labels, dtype=object)
    ok = ~pd.isna(pd.Series(labels, dtype=object)).to_numpy()
    clusters = sorted({lab for lab in labels[ok]}, key=_sort_key)
    if start not in clusters:
        raise ValueError(f"the start cluster {start!r} has no objects")
    ends = [e for e in (ends or []) if e != start]
    missing = [e for e in ends if e not in clusters]
    if missing:
        raise ValueError("these end clusters have no objects: " + ", ".join(map(str, missing)))
    if len(clusters) < 2:
        raise ValueError("a trajectory needs at least two clusters")
    dist = cluster_distances(points[ok], labels[ok], clusters, distance=distance)
    centers = np.vstack([points[ok][labels[ok] == c].mean(axis=0) for c in clusters])
    index = {c: i for i, c in enumerate(clusters)}
    inner = [i for i, c in enumerate(clusters) if c not in ends]
    edges = []
    if len(inner) > 1:
        sub = dist[np.ix_(inner, inner)]
        # Zero distances would be read as "no edge" by csgraph: nudge them.
        tree = minimum_spanning_tree(
            np.where(sub > 0, sub, 1e-12) * (1 - np.eye(len(inner)))
        ).tocoo()
        edges = [(inner[a], inner[b]) for a, b in zip(tree.row, tree.col)]
    for e in ends:  # forced leaves: attached to the nearest cluster that is not an end
        i = index[e]
        j = inner[int(np.argmin(dist[i, inner]))]
        edges.append((j, i))
    adjacency = {i: set() for i in range(len(clusters))}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    root = index[start]
    parent, order, depth = {root: None}, [root], {root: 0.0}
    for node in order:  # breadth-first from the root
        for nxt in sorted(adjacency[node]):
            if nxt not in parent:
                parent[nxt] = node
                depth[nxt] = depth[node] + dist[node, nxt]
                order.append(nxt)
    leaves = [n for n in order if n != root and len(adjacency[n]) == 1]
    leaves.sort(key=lambda n: depth[n])
    lineages = []
    for leaf in leaves:
        path, node = [], leaf
        while node is not None:
            path.append(clusters[node])
            node = parent[node]
        lineages.append(path[::-1])
    if not lineages:  # a single cluster joined to nothing (cannot happen with >= 2)
        raise ValueError("no lineage could be found from the start cluster")
    named_edges = [(clusters[a], clusters[b]) for a, b in edges]
    return clusters, centers, named_edges, lineages, dist


def slingshot(
    points,
    labels,
    start,
    *,
    ends=None,
    distance: str = "slingshot",
    n_iter: int = 10,
    n_points: int = 100,
    span: float = 0.1,
    shrink: bool = True,
    min_weight: float = 0.1,
    tol: float = 1e-3,
) -> Trajectory:
    """Lineages and pseudotime of the objects `points` (objects x dims, e.g. PCA
    of scaled features, or a 2D map) grouped by `labels` (one cluster per
    object; None / NaN for objects in none), starting from cluster `start`.

    `ends`: clusters that must be ends of lineages. `n_iter`: principal-curve
    iterations (stops early when the curves change by less than `tol` of the
    data's spread). `n_points`: points on each curve. `span`: width of the
    smoother, as a fraction of each lineage's length. `shrink`: shrink lineages
    to their common curve where they share clusters. `min_weight`: an object's
    weight on a lineage below this is dropped to 0."""
    points = np.asarray(points, float)
    if points.ndim == 1:
        points = points[:, None]
    labels = np.asarray(labels, dtype=object)
    if len(points) != len(labels):
        raise ValueError("points and labels have different lengths")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite (leave out objects with missing features)")
    clusters, centers, edges, lineages, dist = lineage_tree(
        points, labels, start, ends=ends, distance=distance
    )
    n, n_lin = len(points), len(lineages)
    index = {c: i for i, c in enumerate(clusters)}
    # Initial weights: 1 for every object of the lineage's clusters.
    weights = np.zeros((n, n_lin))
    for j, lin in enumerate(lineages):
        weights[:, j] = np.isin(labels, np.array(lin, dtype=object))
    curves = [_polyline_resample(centers[[index[c] for c in lin]], n_points) for lin in lineages]
    scale = float(np.sqrt(((points - points.mean(axis=0)) ** 2).sum(axis=1).mean())) or 1.0
    shared = _shared_prefixes(lineages)
    lam = np.full((n, n_lin), np.nan)
    dists = np.full((n, n_lin), np.inf)
    for _ in range(max(int(n_iter), 1)):
        new = []
        for j in range(n_lin):
            on = weights[:, j] > 0
            lam_j, _ = _project(points[on], curves[j])
            new.append(_smooth_curve(points[on], lam_j, weights[on, j], n_points, span, curves[j]))
        if shrink and n_lin > 1:
            new = _shrink(new, points, labels, lineages, weights, shared)
        change = max(_curve_change(a, b) for a, b in zip(curves, new)) / scale
        curves = new
        if change < tol:
            break
    for j in range(n_lin):
        on = weights[:, j] > 0
        lam[on, j], dists[on, j] = _project(points[on], curves[j])
    weights = _reweight(weights, dists, min_weight)
    pseudotime = np.where(weights > 0, lam, np.nan)
    return Trajectory(
        clusters=clusters,
        centers=centers,
        edges=edges,
        start=start,
        ends=list(ends or []),
        lineages=lineages,
        curves=curves,
        pseudotime=pseudotime,
        weights=weights,
        distances=dist,
    )


# ---- curves -----------------------------------------------------------------------------


def _polyline_resample(vertices, n_points):
    """`n_points` points evenly spaced by arc length along a polyline."""
    vertices = np.asarray(vertices, float)
    if len(vertices) == 1:
        return np.repeat(vertices, n_points, axis=0)
    seg = np.sqrt(((np.diff(vertices, axis=0)) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.repeat(vertices[:1], n_points, axis=0)
    t = np.linspace(0, s[-1], n_points)
    return np.column_stack([np.interp(t, s, vertices[:, k]) for k in range(vertices.shape[1])])


def _arc(curve):
    seg = np.sqrt((np.diff(curve, axis=0) ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _project(points, curve, dense: int = 2000):
    """(arc length, distance) of each point's nearest point on the polyline
    `curve`. Points beyond either end project onto the end.

    The curve is resampled to `dense` points and searched with a k-d tree; each
    point is then projected exactly onto the two segments either side of its
    nearest dense point (O(n log m), not O(n m))."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, float)
    if len(points) == 0:
        return np.zeros(0), np.zeros(0)
    arc = _arc(curve)
    if arc[-1] > 0:
        t = np.linspace(0, arc[-1], max(dense, len(curve)))
        fine = np.column_stack([np.interp(t, arc, curve[:, k]) for k in range(curve.shape[1])])
    else:
        t, fine = arc[:1], curve[:1]
    if len(fine) == 1:
        return np.zeros(len(points)), np.sqrt(((points - fine[0]) ** 2).sum(axis=1))
    _, near = cKDTree(fine).query(points)
    best_lam = np.full(len(points), np.nan)
    best_d2 = np.full(len(points), np.inf)
    for offset in (-1, 0):  # the segment before and the one after the nearest point
        i = np.clip(near + offset, 0, len(fine) - 2)
        a, b = fine[i], fine[i + 1]
        ab = b - a
        len2 = (ab**2).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where(len2 > 0, ((points - a) * ab).sum(axis=1) / len2, 0.0)
        u = np.clip(u, 0.0, 1.0)
        d2 = ((points - (a + u[:, None] * ab)) ** 2).sum(axis=1)
        lam = t[i] + u * (t[i + 1] - t[i])
        better = d2 < best_d2
        best_d2[better], best_lam[better] = d2[better], lam[better]
    return best_lam, np.sqrt(best_d2)


def _smooth_curve(points, lam, weights, n_points, span, previous):
    """A new curve: each coordinate smoothed against arc length (a Gaussian
    smoother of the weighted means in bins of arc length), at `n_points` arc
    lengths from the smallest to the largest. The previous curve is kept when
    there are too few objects."""
    if len(points) < 3 or not np.ptp(lam) > 0:
        return previous
    lo, hi = float(lam.min()), float(lam.max())
    n_bins = max(n_points, 20)
    edges = np.linspace(lo, hi, n_bins + 1)
    which = np.clip(np.searchsorted(edges, lam, side="right") - 1, 0, n_bins - 1)
    w_bin = np.bincount(which, weights=weights, minlength=n_bins)
    sums = np.column_stack(
        [
            np.bincount(which, weights=weights * points[:, k], minlength=n_bins)
            for k in range(points.shape[1])
        ]
    )
    centers = (edges[:-1] + edges[1:]) / 2
    grid = np.linspace(lo, hi, n_points)
    bandwidth = max(span * (hi - lo), (hi - lo) / n_bins)
    kernel = np.exp(-0.5 * ((grid[:, None] - centers[None, :]) / bandwidth) ** 2)
    # Kernel-weighted mean of the objects: sum K * (weighted sums) / sum K * (weights).
    total = (kernel @ w_bin)[:, None]
    if not (total > 0).all():
        return previous
    return (kernel @ sums) / total


def _curve_change(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(axis=1)).mean())


def _shared_prefixes(lineages):
    """{(i, j): number of clusters lineages i and j share from the start}."""
    out = {}
    for i in range(len(lineages)):
        for j in range(i + 1, len(lineages)):
            k = 0
            while k < min(len(lineages[i]), len(lineages[j])) and lineages[i][k] == lineages[j][k]:
                k += 1
            out[(i, j)] = k
    return out


def _shrink(curves, points, labels, lineages, weights, shared):
    """Shrink each group of lineages towards their mean curve over the part they
    share. The shrinking weight of a curve point is 1 up to the median arc length
    of the shared clusters' objects and falls to 0 (cosine) by the largest, so
    the lineages leave their common trunk where the data leave it."""
    n_lin = len(curves)
    out = [c.copy() for c in curves]
    for j in range(n_lin):
        partners = [k for k in range(n_lin) if k != j and shared[(min(j, k), max(j, k))] > 0]
        if not partners:
            continue
        common = min(shared[(min(j, k), max(j, k))] for k in partners)
        group = [j] + partners
        trunk = np.array(lineages[j][:common], dtype=object)
        on = np.isin(labels, trunk) & (weights[:, j] > 0)
        if on.sum() < 2:
            continue
        lam_trunk, _ = _project(points[on], curves[j])
        start, stop = float(np.median(lam_trunk)), float(np.max(lam_trunk))
        arc_j = _arc(curves[j])
        # The mean of the group's curves at the same arc lengths.
        mean = np.zeros_like(curves[j])
        for k in group:
            arc_k = _arc(curves[k])
            mean += np.column_stack(
                [np.interp(arc_j, arc_k, curves[k][:, d]) for d in range(curves[k].shape[1])]
            )
        mean /= len(group)
        w = np.ones(len(arc_j))
        if stop > start:
            frac = np.clip((arc_j - start) / (stop - start), 0, 1)
            w = 0.5 * (1 + np.cos(np.pi * frac))
        w[arc_j > stop] = 0.0
        out[j] = w[:, None] * mean + (1 - w[:, None]) * curves[j]
    return out


def _reweight(weights, dists, min_weight):
    """Objects on several lineages: weight each by how close it is to that curve
    compared with the closest (1 for the closest, (closest / d)^2 for the others),
    dropping weights below `min_weight`. Objects on a shared trunk are about as
    close to every curve and keep about 1 for each."""
    out = weights.copy()
    multi = (weights > 0).sum(axis=1) > 1
    if multi.any():
        d = np.where(weights[multi] > 0, dists[multi], np.inf)
        best = d.min(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(np.isfinite(d), np.where(d > 0, (best / d) ** 2, 1.0), 0.0)
        w[best[:, 0] == 0] = np.where(np.isfinite(d[best[:, 0] == 0]), 1.0, 0.0)
        w[w < min_weight] = 0.0
        out[multi] = w
    return out


def _sort_key(value):
    return (0, value, "") if isinstance(value, (int, float, np.integer)) else (1, 0, str(value))


def _plain(value):
    return value.item() if hasattr(value, "item") else value
