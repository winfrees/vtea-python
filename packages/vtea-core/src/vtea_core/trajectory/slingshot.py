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
2. **Curves** (`getCurves`). Each lineage starts as the polyline through its
   clusters' means (extended past the end centres, `extend = "y"`) and is
   refined into a principal curve (Hastie & Stuetzle 1989), iterating:
   - each coordinate is smoothed against the objects' arc length with a
     weighted cubic smoothing spline of 5 degrees of freedom (R's
     `smooth.spline(df = 5)`), evaluated at `approx_points` = 150 points;
   - every object is projected onto every curve, with the curve ends
     stretched (`stretch = 2`, as `princurve::project_to_curve`);
   - at each branch point, deepest first, the curves of the lineages through
     it are **shrunk** towards their average where they share objects, by the
     survival function of a cosine kernel over the shared objects' arc
     lengths (between the box-plot whiskers): Slingshot's simultaneous
     principal curves, so lineages share a trunk and part where the data part;
   - **reweight**: an object on several lineages is weighted 1 - Z^2 on each,
     Z its distance to that curve as a quantile of the lineage's distances;
     **reassign**: Z < 0.5 adds it to a lineage, Z > 0.9 with a weight < 0.1
     drops it;
   - until the total weighted squared distance changes by less than
     `thresh` (0.001) of itself, or `maxit` (15) iterations.

Pseudotime is the arc length along the curve, 0 at the lineage's earliest
object, in the units of the space (e.g. PCA of scaled features, or a UMAP
map). NaN where an object is not on a lineage.

Validated on simulations with a known truth and checked against R slingshot
(`validate`, docs/TRAJECTORIES.md). Not ported: the artificial "omega"
cluster that lets the graph fall apart into separate trees, and the "loess"
smoother option. One known failure mode, shared with R slingshot: if no
cluster covers a branch point, the first clusters of the branches are joined
to each other rather than to the trunk.
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
    shrink: float = 1.0,
    extend: str = "y",
    reweight: bool = True,
    reassign: bool = True,
    thresh: float = 0.001,
    maxit: int = 15,
    stretch: float = 2.0,
    approx_points: int = 150,
    smoother_df: float = 5.0,
) -> Trajectory:
    """Lineages and pseudotime of the objects `points` (objects x dims, e.g. PCA
    of scaled features, or a 2D map) grouped by `labels` (one cluster per
    object; None / NaN for objects in none), starting from cluster `start`.

    The arguments and their defaults are those of R slingshot's `slingshot()`
    (`getLineages` + `getCurves`): `ends` (`end.clus`), `distance`
    (`dist.method`: "slingshot" or "euclidean"), `shrink` (0..1), `extend`
    ("y": the initial curve is extended past the first and last cluster
    centres to reach their objects; "n": not), `reweight`, `reassign`,
    `thresh` (relative change of the total squared distance that stops the
    iterations), `maxit`, `stretch` (how far the curve ends are extended when
    projecting, as `princurve::project_to_curve`), `approx_points` (points per
    curve), and `smoother_df` (degrees of freedom of the smoothing spline,
    slingshot's `smooth.spline(..., df = 5)`)."""
    from vtea_core.trajectory.splines import smoothing_spline

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
    weights = np.zeros((n, n_lin))
    for j, lin in enumerate(lineages):
        weights[:, j] = np.isin(labels, np.array(lin, dtype=object))
    # Initial curves: the polylines through the lineages' cluster centres.
    curves = []
    for lin in lineages:
        vertices = centers[[index[c] for c in lin]]
        if extend == "y":
            vertices = _extend(vertices, points, labels, lin[0], lin[-1])
        curves.append(_polyline_resample(vertices, approx_points))
    lam = np.zeros((n, n_lin))
    dists = np.zeros((n, n_lin))
    for j in range(n_lin):
        lam[:, j], dists[:, j] = _project(points, curves[j], stretch=stretch)
    total = np.nansum(weights * dists**2)
    plan = _shrink_plan(lineages)
    labelled = weights.max(axis=1) > 0  # objects with a class in the tree
    for _ in range(max(int(maxit), 1)):
        for j in range(n_lin):
            on = weights[:, j] > 0
            if on.sum() < 4:
                continue
            t = lam[on, j]
            smooth = smoothing_spline(t, points[on], weights[on, j], df=smoother_df)
            curves[j] = np.atleast_2d(smooth(np.linspace(t.min(), t.max(), approx_points)))
            if curves[j].shape[0] != approx_points:
                curves[j] = curves[j].T
            lam[:, j], dists[:, j] = _project(points, curves[j], stretch=stretch)
        if shrink > 0 and n_lin > 1:
            curves = _shrink_hierarchical(curves, lam, labels, lineages, weights, plan, shrink)
            for j in range(n_lin):
                lam[:, j], dists[:, j] = _project(points, curves[j], stretch=stretch)
        new_total = np.nansum(weights * dists**2)
        if reweight or reassign:
            weights = _reweight_reassign(weights, dists, reweight, reassign, labelled)
        converged = abs(total - new_total) <= thresh * max(total, 1e-300)
        total = new_total
        if converged:
            break
    pseudotime = np.full((n, n_lin), np.nan)
    for j in range(n_lin):
        on = weights[:, j] > 0
        if on.any():  # pseudotime 0 at the earliest object of the lineage
            pseudotime[on, j] = lam[on, j] - lam[on, j].min()
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


def _extend(vertices, points, labels, first, last):
    """Slingshot's `extend = "y"`: the first and last segments are continued past
    the end centres as far as the end clusters' objects reach along them."""
    v = np.asarray(vertices, float).copy()
    if len(v) < 2:
        return v
    out = [v]
    for end, (a, b) in ((first, (0, 1)), (last, (-1, -2))):
        u = v[a] - v[b]
        norm = np.linalg.norm(u)
        if norm == 0:
            continue
        u = u / norm
        reach = ((points[labels == end] - v[a]) @ u).max(initial=0.0)
        if reach > 0:
            tip = v[a] + reach * u
            out = [tip[None, :], *out] if a == 0 else [*out, tip[None, :]]
    return np.vstack(out)


def _arc(curve):
    seg = np.sqrt((np.diff(curve, axis=0) ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _project(points, curve, dense: int = 2000, stretch: float = 0.0):
    """(arc length, distance) of each point's nearest point on the polyline
    `curve`. With `stretch` (princurve's), the first and last segments are
    extended by `stretch` times their length first, so points beyond the ends
    are not all piled onto the end point; arc length is still measured from the
    curve's own first point (negative before it).

    The curve is resampled to `dense` points and searched with a k-d tree; each
    point is then projected exactly onto the two segments either side of its
    nearest dense point (O(n log m), not O(n m))."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, float)
    curve = np.asarray(curve, float)
    if len(points) == 0:
        return np.zeros(0), np.zeros(0)
    offset = 0.0
    if stretch > 0 and len(curve) > 1:
        head = curve[0] + stretch * (curve[0] - curve[1])
        tail = curve[-1] + stretch * (curve[-1] - curve[-2])
        offset = float(np.linalg.norm(head - curve[0]))
        curve = np.vstack([head, curve, tail])
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
    for shift in (-1, 0):  # the segment before and the one after the nearest point
        i = np.clip(near + shift, 0, len(fine) - 2)
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
    return best_lam - offset, np.sqrt(best_d2)


# ---- shrinkage ------------------------------------------------------------------------------


def _shrink_plan(lineages):
    """Branch points, deepest first: [(node, groups)] where `groups` are the sets
    of lineages (indices) that go through `node` and then on to each child."""
    out = []
    depth_nodes = {}
    for lin in lineages:
        for d, node in enumerate(lin):
            depth_nodes.setdefault((d, node), set())
    for d, node in sorted(depth_nodes, key=lambda x: -x[0]):
        through = [
            j
            for j, lin in enumerate(lineages)
            if len(lin) > d and lin[d] == node and lin[: d + 1] == lineages[j][: d + 1]
        ]
        children = {}
        for j in through:
            nxt = lineages[j][d + 1] if len(lineages[j]) > d + 1 else None
            children.setdefault(nxt, []).append(j)
        if len(children) > 1:
            out.append((d, node, list(children.values())))
    return out


def _avg_curve(curves):
    """The mean of curves at the same arc lengths (each held at its end beyond its
    length), over the longest one's arc length."""
    arcs = [_arc(c) for c in curves]
    grid = np.linspace(0, max(a[-1] for a in arcs), len(curves[0]))
    total = np.zeros((len(grid), curves[0].shape[1]))
    for c, a in zip(curves, arcs):
        total += np.column_stack([np.interp(grid, a, c[:, k]) for k in range(c.shape[1])])
    return total / len(curves)


def _percent_shrinkage(lam_curve, lam_shared):
    """Slingshot's cosine-kernel shrinkage weight at each curve point: 1 before the
    shared objects' lower whisker, 0 after the upper, the survival function of
    the kernel 1 + cos(pi u) in between (R `density(kernel = "cosine")`)."""
    v = np.asarray(lam_shared, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.zeros(len(lam_curve))
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    lo = v[v >= q1 - 1.5 * iqr].min()
    hi = v[v <= q3 + 1.5 * iqr].max()
    if hi <= lo:
        return np.zeros(len(lam_curve))
    u = np.clip(2 * (np.asarray(lam_curve, float) - lo) / (hi - lo) - 1, -1, 1)
    cdf = (u + 1) / 2 + np.sin(np.pi * u) / (2 * np.pi)
    return 1.0 - cdf


def _shrink_hierarchical(curves, lam, labels, lineages, weights, plan, shrink):
    """At each branch point, deepest first: the curves of the lineages through it
    are shrunk towards their average where they share objects (the clusters up
    to the branch point), by `shrink` times the cosine weight."""
    curves = [c.copy() for c in curves]
    for depth, _node, groups in plan:
        members = [j for g in groups for j in g]
        avg = _avg_curve([curves[j] for j in members])
        avg_arc = _arc(avg)
        trunk = np.array(lineages[members[0]][: depth + 1], dtype=object)
        shared = np.isin(labels, trunk)
        for j in members:
            on = shared & (weights[:, j] > 0)
            arc_j = _arc(curves[j])
            pct = _percent_shrinkage(arc_j, lam[on, j]) * shrink
            target = np.column_stack(
                [np.interp(arc_j, avg_arc, avg[:, k]) for k in range(avg.shape[1])]
            )
            curves[j] = (1 - pct)[:, None] * curves[j] + pct[:, None] * target
    return curves


def _reweight_reassign(weights, dists, reweight, reassign, labelled=None):
    """Slingshot's reweighting and reassignment. Z is each object's distance to a
    curve as a quantile of the distances of that lineage's objects. Reweight: an
    object on several lineages gets 1 - Z^2 on each, scaled so its largest weight
    is 1. Reassign: Z < 0.5 adds the object to the lineage (weight 1); an
    object far from a curve (Z > 0.9) with a small weight (< 0.1) is dropped."""
    w = weights.copy()
    n, n_lin = w.shape
    z = np.full((n, n_lin), np.nan)
    for j in range(n_lin):
        ref = np.sort(dists[w[:, j] > 0, j])
        if len(ref):
            z[:, j] = np.searchsorted(ref, dists[:, j], side="right") / len(ref)
    if reweight:
        multi = (w > 0).sum(axis=1) > 1
        if multi.any():
            zz = np.where(w[multi] > 0, z[multi], np.nan)
            ww = 1 - zz**2
            top = np.nanmax(ww, axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                ww = np.where(top > 0, ww / top, 1.0)
            w[multi] = np.where(np.isfinite(ww), np.clip(ww, 0, 1), 0.0)
    if reassign:  # only objects with a class (unclassed ones are outside the tree)
        ok = np.ones(n, bool) if labelled is None else np.asarray(labelled, bool)
        w[np.isfinite(z) & (z < 0.5) & ok[:, None]] = 1.0
        drop = np.isfinite(z) & (z > 0.9) & (w < 0.1)
        rows = drop.any(axis=1)
        w[drop & rows[:, None]] = 0.0
    # Every object keeps at least the lineage it is closest to (relative to that
    # lineage's spread) when reweighting dropped all of them.
    lost = (w.max(axis=1) <= 0) & (weights.max(axis=1) > 0)
    if lost.any():
        best = np.nanargmin(np.where(np.isfinite(z[lost]), z[lost], np.inf), axis=1)
        w[np.flatnonzero(lost), best] = 1.0
    return w


def _sort_key(value):
    return (0, value, "") if isinstance(value, (int, float, np.integer)) else (1, 0, str(value))


def _plain(value):
    return value.item() if hasattr(value, "item") else value
