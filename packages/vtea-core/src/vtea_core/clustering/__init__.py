"""Clustering algorithms, registered under the vtea_core.clustering entry-point group.

Ports vtea.clustering from the Java codebase. KMeans/GaussianMixture/hierarchical
map onto scikit-learn directly; X-Means, G-Means, and deterministic annealing have
no direct sklearn equivalent and need their BIC/AIC model-selection logic ported.
"""
