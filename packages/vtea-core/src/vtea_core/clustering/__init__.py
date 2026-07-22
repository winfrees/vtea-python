"""Clustering algorithms, registered under the vtea_core.clustering entry-point group.

Ports vtea.clustering from the Java codebase. KMeans/GaussianMixture/hierarchical
map onto scikit-learn directly; X-Means and G-Means are consolidated into a
single BIC-based auto_k_kmeans(). Deterministic annealing is not ported - it
was disabled/unregistered in the Java source. See clustering.py for details.
"""

from vtea_core.clustering.clustering import auto_k_kmeans, gaussian_mixture, hierarchical, kmeans

__all__ = ["kmeans", "gaussian_mixture", "hierarchical", "auto_k_kmeans"]
