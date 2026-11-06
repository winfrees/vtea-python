"""Clustering algorithms, registered under the vtea_core.clustering entry-point group.

Ports vtea.clustering from the Java codebase. KMeans/GaussianMixture/hierarchical
map onto scikit-learn directly; X-Means and G-Means are consolidated into a
single BIC-based auto_k_kmeans(). Deterministic annealing is not ported - it
was disabled/unregistered in the Java source. See clustering.py for details.

Louvain and Leiden (graph.py) are additions rather than ports: they find
communities in a shared-nearest-neighbour graph and so decide how many
populations there are instead of being told, which is the question every
algorithm above has to have answered for it first.

profiles.py describes what each cluster is (its mean and z-score over the
features, and its top markers), which is what a cluster is named from.
"""

from vtea_core.clustering.clustering import auto_k_kmeans, gaussian_mixture, hierarchical, kmeans
from vtea_core.clustering.graph import leiden, louvain, shared_neighbor_graph
from vtea_core.clustering.profiles import cluster_profiles, describe_clusters, top_markers

__all__ = [
    "auto_k_kmeans",
    "cluster_profiles",
    "describe_clusters",
    "gaussian_mixture",
    "hierarchical",
    "kmeans",
    "leiden",
    "louvain",
    "shared_neighbor_graph",
    "top_markers",
]
