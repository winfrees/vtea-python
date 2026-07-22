"""Dimensionality reduction, registered under the vtea_core.reduction entry-point group.

Ports vtea.reduction (PCA, Isomap, Laplacian Eigenmap, t-SNE) from the Java
codebase onto scikit-learn (PCA, Isomap, SpectralEmbedding, TSNE).
"""

from vtea_core.reduction.reduction import isomap, laplacian_eigenmap, pca, pca_explained_variance, tsne

__all__ = ["pca", "pca_explained_variance", "isomap", "laplacian_eigenmap", "tsne"]
