"""Clustering: KMeans, Gaussian Mixture, hierarchical, and automatic-k selection.

Ports vtea.clustering from the Java codebase. KMeans/GaussianMix/hierarchical
(Ward/Single/Complete) map directly onto scikit-learn. X-Means and G-Means
(vtea.clustering.Xmeans/GMeansClust) are both thin wrappers around Smile's
own XMeans/GMeans implementations that pick k automatically via a model-
selection criterion - consolidated here into one auto_k_kmeans() using BIC,
since that's what both are conceptually doing. DeterministicAnnealingClust
is not ported: its @Plugin registration is commented out in the Java source
("Unstable behaviour, disabled"), so it was never actually available to
users there either.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans as _KMeans
from sklearn.mixture import GaussianMixture as _GaussianMixture

_LINKAGES = ("ward", "single", "complete")


def kmeans(data: np.ndarray, n_clusters: int, *, random_state: int | None = None) -> np.ndarray:
    """Cluster assignments via KMeans."""
    return _KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state).fit_predict(data)


def gaussian_mixture(data: np.ndarray, n_clusters: int, *, random_state: int | None = None) -> np.ndarray:
    """Cluster assignments via Gaussian Mixture Model."""
    return _GaussianMixture(n_components=n_clusters, random_state=random_state).fit_predict(data)


def hierarchical(data: np.ndarray, n_clusters: int, *, linkage: str = "ward") -> np.ndarray:
    """Cluster assignments via agglomerative hierarchical clustering.

    linkage: "ward", "single", or "complete" - replaces
    vtea.clustering.hierarchical's WardCluster/SingleCluster/CompleteCluster.
    """
    if linkage not in _LINKAGES:
        raise ValueError(f"unknown linkage {linkage!r}, expected one of {_LINKAGES}")
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(data)


def _kmeans_bic(data: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Bayesian Information Criterion for a k-means fit (Pelleg & Moore, 2000).

    Higher is better (this is a log-likelihood-based BIC, not the "-2*LL"
    convention) - auto_k_kmeans picks the k that maximizes it.
    """
    n, d = data.shape
    k = centers.shape[0]
    counts = np.bincount(labels, minlength=k)

    ssd = 0.0
    for c in range(k):
        cluster_points = data[labels == c]
        if len(cluster_points):
            ssd += np.sum((cluster_points - centers[c]) ** 2)
    variance = max(ssd / max(n - k, 1), 1e-10)

    log_likelihood = 0.0
    for c in range(k):
        n_c = counts[c]
        if n_c == 0:
            continue
        log_likelihood += n_c * np.log(n_c / n) - (n_c * d / 2) * np.log(2 * np.pi * variance) - (n_c - 1) / 2

    n_params = k * (d + 1)
    return log_likelihood - n_params / 2 * np.log(n)


def auto_k_kmeans(
    data: np.ndarray, *, k_min: int = 2, k_max: int = 10, random_state: int | None = None
) -> tuple[np.ndarray, dict[int, float]]:
    """Runs KMeans for k in [k_min, k_max] and returns the BIC-best labels.

    Also returns the per-k BIC scores, for the caller to plot (replacing
    vtea.clustering.kestimation's model-selection UI). Replaces
    vtea.clustering.Xmeans and GMeansClust, which both wrap Smile's own
    X-Means/G-Means - different algorithms for finding the same thing (an
    automatically-chosen k), consolidated into one BIC-based search.
    """
    if k_max < k_min:
        raise ValueError(f"k_max ({k_max}) must be >= k_min ({k_min})")

    scores: dict[int, float] = {}
    best_labels: np.ndarray | None = None
    best_score = -np.inf
    for k in range(k_min, k_max + 1):
        model = _KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit(data)
        score = _kmeans_bic(data, model.labels_, model.cluster_centers_)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_labels = model.labels_

    assert best_labels is not None  # k_min <= k_max guarantees at least one iteration
    return best_labels, scores
