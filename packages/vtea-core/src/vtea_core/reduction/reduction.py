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

from vtea_core.measurements.normalize import Normalization, normalize_features


def pca(data: np.ndarray, n_components: int, *, normalize: Normalization = "none") -> np.ndarray:
    """Principal Component Analysis embedding.

    `normalize` rescales each feature first - Java's "Z-scale all data" is
    `"zscore"`, and for PCA it is the difference between components of the
    correlation and of the covariance matrix. Every reduction step takes it.
    """
    data = normalize_features(data, normalize)
    return _PCA(n_components=n_components).fit_transform(data)


def pca_explained_variance(data: np.ndarray, n_components: int) -> np.ndarray:
    """Cumulative explained-variance ratio for the first `n_components`.

    For picking a dimensionality by desired variance - PCAReduction's
    "Desired Variance" mode, as an alternative to a fixed n_components.
    """
    model = _PCA(n_components=n_components).fit(data)
    return np.cumsum(model.explained_variance_ratio_)


def isomap(
    data: np.ndarray,
    n_components: int,
    *,
    n_neighbors: int = 5,
    normalize: Normalization = "none",
) -> np.ndarray:
    """Isomap manifold embedding."""
    data = normalize_features(data, normalize)
    return _Isomap(n_components=n_components, n_neighbors=n_neighbors).fit_transform(data)


def laplacian_eigenmap(
    data: np.ndarray,
    n_components: int,
    *,
    n_neighbors: int = 5,
    normalize: Normalization = "none",
) -> np.ndarray:
    """Laplacian Eigenmap embedding (scikit-learn calls this SpectralEmbedding)."""
    data = normalize_features(data, normalize)
    return SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors).fit_transform(data)


def tsne(
    data: np.ndarray,
    n_components: int = 2,
    *,
    perplexity: float = 30.0,
    random_state: int | None = None,
    normalize: Normalization = "none",
    n_jobs: int | None = -1,
    method: str = "sklearn",
) -> np.ndarray:
    """t-distributed Stochastic Neighbor Embedding.

    `method`: "sklearn" (the default) is scikit-learn's Barnes-Hut t-SNE;
    "opentsne" is openTSNE's FFT-accelerated t-SNE (Linderman et al. 2019),
    which runs its gradient on every core and is the faster of the two on
    tens of thousands of objects and more. openTSNE is optional
    (`pip install openTSNE`). The two optimise the same objective but are
    different implementations, so a run is reproducible within a method,
    not across them. `n_jobs` is the number of cores (-1: all).
    """
    if method not in ("sklearn", "opentsne"):
        raise ValueError(f"method must be 'sklearn' or 'opentsne', not {method!r}")
    data = normalize_features(data, normalize)
    if method == "opentsne":
        try:
            from openTSNE import TSNE as _OpenTSNE
        except ImportError as exc:
            raise ImportError(
                "openTSNE is needed for method='opentsne' and is not installed. "
                "Install it with `pip install openTSNE`."
            ) from exc
        model = _OpenTSNE(n_components=n_components, perplexity=perplexity,
                          random_state=random_state, n_jobs=n_jobs if n_jobs is not None else 1)
        return np.asarray(model.fit(np.asarray(data, dtype=float)))
    return _TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state,
                 n_jobs=n_jobs).fit_transform(data)


def umap(
    data: np.ndarray,
    n_components: int = 2,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = None,
    normalize: Normalization = "none",
    n_jobs: int = -1,
    knn: tuple[np.ndarray, np.ndarray] | None = None,
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
    cosmetic rather than structural.

    Speed. `random_state` makes a run reproducible, at the cost of UMAP's
    parallelism: umap-learn then runs on one core. With `random_state=None`
    it runs on `n_jobs` cores (-1: all), about 4.4 times faster on 4 cores
    for 50,000 objects (10 s against 46 s), with a layout that differs a
    little from run to run. `knn` is a neighbour search already made on the
    same (normalised) matrix - (indices, distances) with each object left
    out, as `vtea_core.clustering.nearest_neighbors` returns - so that UMAP
    and graph clustering share one search; it needs at least `n_neighbors` - 1
    columns. The map is as good as one from UMAP's own search, but not the
    same one: that search draws on the random state before the layout does.

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

    data = normalize_features(data, normalize)
    n_objects = len(np.asarray(data))
    options = {}
    if knn is not None:
        indices, distances = (np.asarray(a) for a in knn)
        if indices.shape[0] != n_objects:
            raise ValueError(f"knn has {indices.shape[0]} rows for {n_objects} objects")
        k = n_neighbors - 1  # umap-learn counts each object as its own first neighbour
        if indices.shape[1] < k:
            raise ValueError(f"knn has {indices.shape[1]} neighbours; UMAP needs {k}")
        own = np.arange(n_objects)[:, None]
        options["precomputed_knn"] = (
            np.hstack([own, indices[:, :k]]).astype(np.int64),
            np.hstack([np.zeros((n_objects, 1)), distances[:, :k]]).astype(np.float32),
        )
    if random_state is None:
        options["n_jobs"] = n_jobs
    return UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **options,
    ).fit_transform(data)
