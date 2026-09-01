"""Dimensionality reduction, registered under the vtea_core.reduction entry-point group.

Ports vtea.reduction (PCA, Isomap, Laplacian Eigenmap, t-SNE) from the Java
codebase onto scikit-learn (PCA, Isomap, SpectralEmbedding, TSNE), and adds
UMAP - a projection the Java original predates, and the one to reach for
when the distance between two islands of the embedding needs to mean
something (see reduction.umap).
"""

from vtea_core.reduction.reduction import (
    isomap,
    laplacian_eigenmap,
    pca,
    pca_explained_variance,
    tsne,
    umap,
)

__all__ = ["isomap", "laplacian_eigenmap", "pca", "pca_explained_variance", "tsne", "umap"]
