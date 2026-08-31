"""Clustering, reduction and plotting when there are ten million objects.

A different problem from the rest of this package, and worth saying why. Up
to here the constraint has been voxels: an image too large to hold, divided
into tiles. Here the image is gone and what is left is a table, and the
numbers are smaller than they look - ten million objects with forty float32
features is 1.6 GB, which mostly fits. What does not fit is 10^8 rows, and
what does not work at *any* size is the O(n^2) methods.

So this is not a tiling problem, it is an estimator problem, and the answers
are the ones scikit-learn already has: fit on batches where an incremental
fit exists, fit on a subsample and extend where it does not, and refuse to
pretend where neither is possible.

**Every choice is returned rather than taken quietly.** `EstimatorChoice`
says which method ran, how many rows it was fitted on, and whether the
result is exact - because "k-means on ten million objects" and "k-means
fitted on fifty thousand of them and applied to the rest" are different
claims, and a table that cannot tell them apart cannot be compared with one
computed the other way. That is the same instinct as `Spacing.source` and
`LabelLedger.decided_by`, applied to the analysis end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Above this many rows, the exact fit stops being the sensible default. Not a
# hard limit - an exact fit still runs, it just takes what it takes - but the
# point where the streaming or subsampled form is worth its approximation.
DEFAULT_EXACT_ROWS = 200_000

# How many rows a subsampled fit uses. Large enough that a cluster holding a
# percent of the objects still gets hundreds of examples, small enough that
# an O(n^2) method finishes.
DEFAULT_SUBSAMPLE = 50_000

# Agglomerative clustering is O(n^2) in time and memory. At 100,000 rows the
# distance matrix alone is 40 GB, so this is the size above which the exact
# method is not slow but impossible.
HIERARCHICAL_CEILING = 20_000

EXACT = "exact"
STREAMED = "streamed"
SUBSAMPLED = "subsampled"


@dataclass(frozen=True)
class EstimatorChoice:
    """Which method actually ran, and what that means for the answer."""

    method: str
    kind: str = EXACT
    n_rows: int = 0
    n_fitted: int = 0
    reason: str = ""

    @property
    def exact(self) -> bool:
        """Whether the whole-table method ran, over the whole table.

        Deliberately strict. `STREAMED` sees every row but runs a different
        algorithm to do it, and how close that lands varies by estimator and
        by data - `IncrementalPCA` is usually indistinguishable and
        `MiniBatchKMeans` is merely good. Neither is the same fit, and a
        flag that said otherwise for one of them would be a flag nobody
        could rely on for either.
        """
        return self.kind == EXACT

    def describe(self) -> str:
        if self.kind == EXACT:
            summary = f"{self.method} over all {self.n_rows:,} objects"
            return f"{summary} ({self.reason})" if self.reason else summary
        return (
            f"{self.method} ({self.kind}) - fitted on {self.n_fitted:,} of "
            f"{self.n_rows:,} objects; {self.reason}"
        )


def _rng(random_state: int | None) -> np.random.Generator:
    return np.random.default_rng(random_state)


def subsample(
    data: np.ndarray, size: int, *, random_state: int | None = None
) -> np.ndarray:
    """Row positions for a fit that cannot afford every row.

    Uniform without replacement. Deliberately not stratified: stratifying
    needs labels, and the labels are what is being computed.
    """
    if len(data) <= size:
        return np.arange(len(data))
    return np.sort(_rng(random_state).choice(len(data), size=size, replace=False))


def kmeans_scaled(
    data: np.ndarray,
    n_clusters: int,
    *,
    random_state: int | None = None,
    max_exact_rows: int = DEFAULT_EXACT_ROWS,
    batch_size: int = 4096,
) -> tuple[np.ndarray, EstimatorChoice]:
    """k-means that keeps working as the table grows.

    Above `max_exact_rows`, `MiniBatchKMeans` - which is a different
    algorithm converging to a similar place rather than the same fit made
    cheaper, so it is reported as approximate however good it is in
    practice.
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans

    rows = len(data)
    if rows <= max_exact_rows:
        model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        return model.fit_predict(data), EstimatorChoice("kmeans", EXACT, rows, rows)

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=min(batch_size, rows),
        n_init="auto",
    )
    labels = model.fit_predict(data)
    return labels, EstimatorChoice(
        "kmeans",
        STREAMED,
        rows,
        rows,
        f"MiniBatchKMeans above {max_exact_rows:,} rows",
    )


def gaussian_mixture_scaled(
    data: np.ndarray,
    n_clusters: int,
    *,
    random_state: int | None = None,
    max_exact_rows: int = DEFAULT_EXACT_ROWS,
    sample_size: int = DEFAULT_SUBSAMPLE,
) -> tuple[np.ndarray, EstimatorChoice]:
    """A mixture fitted on a subsample and applied to every row.

    The assignment is exact given the fitted model - `predict` is a closed
    form - so what is approximate is the model, not which cluster each object
    lands in. That distinction is worth keeping: it means adding objects does
    not move the ones already classified.
    """
    from sklearn.mixture import GaussianMixture

    rows = len(data)
    model = GaussianMixture(n_components=n_clusters, random_state=random_state)
    if rows <= max_exact_rows:
        return model.fit_predict(data), EstimatorChoice("gaussian_mixture", EXACT, rows, rows)

    chosen = subsample(data, sample_size, random_state=random_state)
    model.fit(data[chosen])
    return model.predict(data), EstimatorChoice(
        "gaussian_mixture",
        SUBSAMPLED,
        rows,
        len(chosen),
        "the model is fitted on a sample; each row's assignment is then exact",
    )


def hierarchical_scaled(
    data: np.ndarray,
    n_clusters: int,
    *,
    linkage: str = "ward",
    ceiling: int = HIERARCHICAL_CEILING,
    sample_size: int = HIERARCHICAL_CEILING,
    random_state: int | None = None,
) -> tuple[np.ndarray, EstimatorChoice]:
    """Agglomerative clustering, which has no scalable form.

    O(n^2) in time and memory: at a hundred thousand objects the distance
    matrix alone is 40 GB, so above the ceiling the exact method is not slow
    but impossible. What runs instead is a fit on a sample followed by
    assignment to the nearest cluster centre - a different method wearing the
    same name, which is exactly why it is labelled rather than substituted
    silently.
    """
    from sklearn.cluster import AgglomerativeClustering

    rows = len(data)
    if rows <= ceiling:
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage
        ).fit_predict(data)
        return labels, EstimatorChoice("hierarchical", EXACT, rows, rows)

    chosen = subsample(data, min(sample_size, ceiling), random_state=random_state)
    sample_labels = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage
    ).fit_predict(data[chosen])
    centres = np.stack(
        [data[chosen][sample_labels == cluster].mean(axis=0) for cluster in range(n_clusters)]
    )
    labels = _nearest_centre(data, centres)
    return labels, EstimatorChoice(
        "hierarchical",
        SUBSAMPLED,
        rows,
        len(chosen),
        "agglomerative clustering is O(n^2); the rest are assigned to the nearest "
        "cluster centre, which is not the same algorithm",
    )


def _nearest_centre(data: np.ndarray, centres: np.ndarray, batch: int = 100_000) -> np.ndarray:
    """Nearest centre per row, a batch at a time so the distance matrix is
    bounded by the batch rather than by the table."""
    labels = np.empty(len(data), dtype=np.int32)
    for start in range(0, len(data), batch):
        chunk = data[start : start + batch]
        distances = ((chunk[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels[start : start + batch] = np.argmin(distances, axis=1)
    return labels


def pca_scaled(
    data: np.ndarray,
    n_components: int,
    *,
    max_exact_rows: int = DEFAULT_EXACT_ROWS,
    batch_size: int = 4096,
) -> tuple[np.ndarray, EstimatorChoice]:
    """PCA over a table too large to decompose at once.

    `IncrementalPCA` merges a partial SVD per batch rather than
    decomposing the whole matrix, which is close but not the same
    arithmetic - and how close depends on the data, not just on the batch
    size. Where the eigenvalues are well separated the components come out
    indistinguishable; where two of them are nearly equal the subspace is
    still right but the individual axes inside it can rotate, because with
    equal eigenvalues there is no fact of the matter about which axis is
    which. Measured on three blobs with one dominant direction: the leading
    component correlates with the exact fit at 1 - 2e-9, the second at 0.94.

    So it is reported as streamed rather than exact. It is much closer to
    the whole-table answer than `MiniBatchKMeans` is, and "much closer" is
    not "the same".
    """
    from sklearn.decomposition import PCA, IncrementalPCA

    rows = len(data)
    if rows <= max_exact_rows:
        return (
            PCA(n_components=n_components).fit_transform(data),
            EstimatorChoice("pca", EXACT, rows, rows),
        )

    model = IncrementalPCA(n_components=n_components, batch_size=min(batch_size, rows))
    return model.fit_transform(data), EstimatorChoice(
        "pca",
        STREAMED,
        rows,
        rows,
        "IncrementalPCA merges a partial SVD per batch; components with nearly equal "
        "eigenvalues can rotate within their subspace",
    )


def isomap_scaled(
    data: np.ndarray,
    n_components: int,
    *,
    n_neighbors: int = 5,
    sample_size: int = DEFAULT_SUBSAMPLE,
    max_exact_rows: int = DEFAULT_EXACT_ROWS,
    random_state: int | None = None,
) -> tuple[np.ndarray, EstimatorChoice]:
    """Isomap fitted on a sample and extended to the rest.

    sklearn's `Isomap` has a `transform`, so the extension is the library's
    own out-of-sample method rather than something invented here.
    """
    from sklearn.manifold import Isomap

    rows = len(data)
    model = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    if rows <= max_exact_rows:
        return model.fit_transform(data), EstimatorChoice("isomap", EXACT, rows, rows)

    chosen = subsample(data, sample_size, random_state=random_state)
    model.fit(data[chosen])
    return model.transform(data), EstimatorChoice(
        "isomap", SUBSAMPLED, rows, len(chosen), "extended with Isomap.transform"
    )


def laplacian_eigenmap_scaled(
    data: np.ndarray,
    n_components: int,
    *,
    n_neighbors: int = 5,
    sample_size: int = DEFAULT_SUBSAMPLE,
    max_exact_rows: int = DEFAULT_EXACT_ROWS,
    random_state: int | None = None,
) -> tuple[np.ndarray, EstimatorChoice | None]:
    """Laplacian eigenmaps, which cannot be extended to unseen rows.

    sklearn's `SpectralEmbedding` has no `transform`, and inventing one
    (Nystrom) here would be a research decision wearing an implementation's
    clothes. So above the threshold this embeds a sample and returns NaN for
    the rest, which is what "these rows were not embedded" honestly looks
    like - a plot can drop them and a table can say so, and neither is
    misled into treating a fabricated position as a measurement.
    """
    from sklearn.manifold import SpectralEmbedding

    rows = len(data)
    model = SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    if rows <= max_exact_rows:
        return (
            model.fit_transform(data),
            EstimatorChoice("laplacian_eigenmap", EXACT, rows, rows),
        )

    chosen = subsample(data, sample_size, random_state=random_state)
    embedded = model.fit_transform(data[chosen])
    result = np.full((rows, n_components), np.nan)
    result[chosen] = embedded
    return result, EstimatorChoice(
        "laplacian_eigenmap",
        SUBSAMPLED,
        rows,
        len(chosen),
        "SpectralEmbedding has no out-of-sample transform, so the rows outside the "
        "sample are NaN rather than guessed",
    )


def tsne_scaled(
    data: np.ndarray,
    n_components: int = 2,
    *,
    perplexity: float = 30.0,
    sample_size: int = DEFAULT_SUBSAMPLE,
    max_exact_rows: int = DEFAULT_EXACT_ROWS,
    random_state: int | None = None,
) -> tuple[np.ndarray, EstimatorChoice]:
    """t-SNE, extended where the installed library can and honest where not.

    sklearn's `TSNE` has no `transform`. `openTSNE` does, and if it is
    installed this uses it; otherwise the rows outside the sample come back
    NaN for the same reason as the Laplacian case.
    """
    rows = len(data)
    if rows <= max_exact_rows:
        from sklearn.manifold import TSNE

        embedded = TSNE(
            n_components=n_components, perplexity=perplexity, random_state=random_state
        ).fit_transform(data)
        return embedded, EstimatorChoice("tsne", EXACT, rows, rows)

    chosen = subsample(data, sample_size, random_state=random_state)
    try:
        from openTSNE import TSNE as OpenTSNE
    except ImportError:
        from sklearn.manifold import TSNE

        embedded = TSNE(
            n_components=n_components, perplexity=perplexity, random_state=random_state
        ).fit_transform(data[chosen])
        result = np.full((rows, n_components), np.nan)
        result[chosen] = embedded
        return result, EstimatorChoice(
            "tsne",
            SUBSAMPLED,
            rows,
            len(chosen),
            "sklearn's TSNE has no out-of-sample transform and openTSNE is not "
            "installed, so the rows outside the sample are NaN rather than guessed",
        )

    fitted = OpenTSNE(
        n_components=n_components, perplexity=perplexity, random_state=random_state
    ).fit(data[chosen])
    return np.asarray(fitted.transform(data)), EstimatorChoice(
        "tsne", SUBSAMPLED, rows, len(chosen), "extended with openTSNE.transform"
    )


# -- plotting a table nobody can draw point by point --------------------

# Above this many points a scatter plot is a solid block of ink: the points
# overlap, the density is invisible, and drawing them costs more than looking
# at them. A 2D histogram shows the same data and more of it.
DEFAULT_SCATTER_LIMIT = 50_000


def should_bin(n_rows: int, limit: int = DEFAULT_SCATTER_LIMIT) -> bool:
    return n_rows > limit


def binned_scatter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int = 256,
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A scatter plot's density, as counts on a grid.

    Returns (counts, x edges, y edges), oriented so `counts[i, j]` is the
    x bin i and y bin j - the transpose of `histogram2d`'s convention, and
    the one an image of the plot wants.

    This replaces the *backdrop* of the plot and nothing else. Gate
    membership is still evaluated on every row: a polygon drawn on this
    surface tests the objects, not the bins, so the gate is exact even
    though what it was drawn over is a summary.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y differ in length: {x.shape} vs {y.shape}")
    finite = np.isfinite(x) & np.isfinite(y)
    counts, x_edges, y_edges = np.histogram2d(
        x[finite], y[finite], bins=bins, range=limits
    )
    return counts, x_edges, y_edges
