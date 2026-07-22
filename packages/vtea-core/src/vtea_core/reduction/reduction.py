"""Dimensionality reduction: PCA, Isomap, Laplacian Eigenmap, t-SNE.

Ports vtea.reduction (PCAReduction, Isomap, LaplacianEigenMap,
TSNEReductionAdjust) from the Java codebase, all of which wrap bespoke Java
libraries (Smile's PCA/IsoMap/LaplacianEigenmap, a standalone Barnes-Hut
t-SNE library) - scikit-learn ships equivalents directly.
vtea.spatial.IsoMapSmile is an unregistered duplicate of vtea.reduction.Isomap
(both wrap smile.manifold.IsoMap, but IsoMapSmile has no @Plugin annotation)
and isn't ported separately.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA as _PCA
from sklearn.manifold import TSNE as _TSNE
from sklearn.manifold import Isomap as _Isomap
from sklearn.manifold import SpectralEmbedding


def pca(data: np.ndarray, n_components: int) -> np.ndarray:
    """Principal Component Analysis embedding."""
    return _PCA(n_components=n_components).fit_transform(data)


def pca_explained_variance(data: np.ndarray, n_components: int) -> np.ndarray:
    """Cumulative explained-variance ratio for the first `n_components`.

    For picking a dimensionality by desired variance - PCAReduction's
    "Desired Variance" mode, as an alternative to a fixed n_components.
    """
    model = _PCA(n_components=n_components).fit(data)
    return np.cumsum(model.explained_variance_ratio_)


def isomap(data: np.ndarray, n_components: int, *, n_neighbors: int = 5) -> np.ndarray:
    """Isomap manifold embedding."""
    return _Isomap(n_components=n_components, n_neighbors=n_neighbors).fit_transform(data)


def laplacian_eigenmap(data: np.ndarray, n_components: int, *, n_neighbors: int = 5) -> np.ndarray:
    """Laplacian Eigenmap embedding (scikit-learn calls this SpectralEmbedding)."""
    return SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors).fit_transform(data)


def tsne(
    data: np.ndarray, n_components: int = 2, *, perplexity: float = 30.0, random_state: int | None = None
) -> np.ndarray:
    """t-distributed Stochastic Neighbor Embedding."""
    return _TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state).fit_transform(data)
