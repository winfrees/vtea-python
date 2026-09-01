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


def umap(
    data: np.ndarray,
    n_components: int = 2,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = None,
) -> np.ndarray:
    """Uniform Manifold Approximation and Projection embedding.

    The projection the Java VTEA never had: t-SNE preserves who is near
    whom and throws away everything about how far apart the groups are, so
    the distance between two t-SNE islands means nothing. UMAP keeps more of
    that global structure, runs an order of magnitude faster on the object
    counts a tissue volume produces, and - unlike scikit-learn's t-SNE - can
    project new objects into an existing embedding, which is what makes a
    second acquisition comparable to the first.

    `n_neighbors` trades local detail (low) against global structure (high);
    `min_dist` is how tightly points may be packed in the embedding, and is
    cosmetic rather than structural. `random_state` makes a run
    reproducible, at the cost of UMAP's parallelism.

    umap-learn is an optional dependency (`pip install "vtea-core[umap]"`),
    imported here rather than at module import so the rest of the reduction
    module works without it.
    """
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "umap-learn is needed for the UMAP projection and is not installed. "
            'Install it with `pip install "vtea-core[umap]"` (or `pip install umap-learn`).'
        ) from exc

    return UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    ).fit_transform(data)
