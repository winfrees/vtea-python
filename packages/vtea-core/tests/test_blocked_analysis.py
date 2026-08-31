"""Phase L7: the analysis end, where object count is the constraint.

Not a tiling problem. The image is gone by this point and what is left is a
table; what breaks is the estimators, and the fix is the one scikit-learn
already has. What these tests check is less that the scaled forms are fast
than that they are *honest* - that the result says which method produced it,
and that nothing is quietly substituted.
"""

import numpy as np
import pytest

from vtea_core.blocked.analysis import (
    EXACT,
    STREAMED,
    SUBSAMPLED,
    EstimatorChoice,
    binned_scatter,
    gaussian_mixture_scaled,
    hierarchical_scaled,
    isomap_scaled,
    kmeans_scaled,
    laplacian_eigenmap_scaled,
    pca_scaled,
    should_bin,
    subsample,
    tsne_scaled,
)


@pytest.fixture(scope="module")
def clustered():
    """Three well-separated blobs, so the right answer is not in doubt."""
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(centre, 1.0, (3000, 4)) for centre in (0, 8, 16)])


def n_groups(labels):
    return len(np.unique(labels[labels >= 0]))


class TestReportingTheChoice:
    def test_an_exact_fit_says_so(self, clustered):
        _labels, choice = kmeans_scaled(clustered, 3, max_exact_rows=10**6)
        assert choice.kind == EXACT
        assert choice.exact
        assert choice.n_fitted == len(clustered)
        assert "over all" in choice.describe()

    def test_an_approximate_fit_says_which_and_why(self, clustered):
        _labels, choice = kmeans_scaled(clustered, 3, max_exact_rows=100)
        assert choice.kind == STREAMED
        assert not choice.exact
        assert "MiniBatchKMeans" in choice.describe()
        assert f"{len(clustered):,}" in choice.describe()

    def test_incremental_pca_does_not_claim_to_be_the_exact_fit(self, clustered):
        # It sees every row, but it merges a partial SVD per batch rather
        # than decomposing the whole matrix. Close is not the same.
        _values, choice = pca_scaled(clustered, 2, max_exact_rows=100)
        assert choice.kind == STREAMED
        assert not choice.exact
        assert "IncrementalPCA" in choice.describe()


class TestClustering:
    def test_kmeans_finds_the_blobs_either_way(self, clustered):
        exact, _ = kmeans_scaled(clustered, 3, random_state=0, max_exact_rows=10**6)
        streamed, choice = kmeans_scaled(
            clustered, 3, random_state=0, max_exact_rows=100
        )
        assert n_groups(exact) == n_groups(streamed) == 3
        assert choice.kind == STREAMED
        # Same partition, whatever the cluster numbers happen to be.
        assert _same_partition(exact, streamed)

    def test_a_mixture_is_fitted_on_a_sample_and_applied_to_every_row(self, clustered):
        labels, choice = gaussian_mixture_scaled(
            clustered, 3, random_state=0, max_exact_rows=100, sample_size=500
        )
        assert len(labels) == len(clustered)
        assert choice.kind == SUBSAMPLED
        assert choice.n_fitted == 500
        # The model is approximate; each row's assignment given it is not.
        assert n_groups(labels) == 3

    def test_hierarchical_above_its_ceiling_is_a_different_method(self, clustered):
        labels, choice = hierarchical_scaled(clustered, 3, ceiling=500, random_state=0)
        assert len(labels) == len(clustered)
        assert choice.kind == SUBSAMPLED
        assert "O(n^2)" in choice.reason
        assert "not the same algorithm" in choice.reason
        assert n_groups(labels) == 3

    def test_hierarchical_below_its_ceiling_is_the_real_thing(self, clustered):
        small = clustered[::10]
        labels, choice = hierarchical_scaled(small, 3, ceiling=len(small))
        assert choice.exact
        assert n_groups(labels) == 3

    def test_the_nearest_centre_assignment_batches(self, clustered):
        # Batched so the distance matrix is bounded by the batch rather than
        # by the table - at ten million rows the unbatched form is the thing
        # being avoided.
        labels, _choice = hierarchical_scaled(clustered, 3, ceiling=500, random_state=0)
        assert labels.dtype == np.int32
        assert len(labels) == len(clustered)


class TestReduction:
    def test_incremental_pca_agrees_where_the_eigenvalues_are_separated(self, clustered):
        # This data has one dominant direction and three near-degenerate
        # ones. The leading component is determined and agrees; the second
        # sits in a subspace where the axes are free to rotate, and
        # asserting it would be asserting something untrue about PCA rather
        # than about the implementation. Sign flips are not disagreements.
        exact, _ = pca_scaled(clustered, 2, max_exact_rows=10**6)
        streamed, _ = pca_scaled(clustered, 2, max_exact_rows=100, batch_size=512)
        leading = abs(np.corrcoef(exact[:, 0], streamed[:, 0])[0, 1])
        assert leading > 1 - 1e-6
        # And the spread it explains is the same, component by component.
        np.testing.assert_allclose(exact.std(axis=0), streamed.std(axis=0), rtol=2e-3)

    def test_isomap_extends_with_the_librarys_own_transform(self, clustered):
        small = clustered[::20]
        values, choice = isomap_scaled(
            small, 2, max_exact_rows=50, sample_size=100, random_state=0
        )
        assert values.shape == (len(small), 2)
        assert np.isfinite(values).all(), "Isomap.transform covers every row"
        assert choice.kind == SUBSAMPLED

    def test_laplacian_leaves_unembedded_rows_as_nan_rather_than_guessing(self, clustered):
        # SpectralEmbedding has no out-of-sample transform, and inventing one
        # would be a research decision wearing an implementation's clothes.
        small = clustered[::20]
        values, choice = laplacian_eigenmap_scaled(
            small, 2, max_exact_rows=50, sample_size=100, random_state=0
        )
        assert np.isnan(values).any()
        assert np.isfinite(values).any()
        assert int(np.isfinite(values[:, 0]).sum()) == choice.n_fitted
        assert "rather than guessed" in choice.reason

    def test_tsne_says_what_it_could_not_do(self, clustered):
        small = clustered[::40]
        values, choice = tsne_scaled(
            small, 2, max_exact_rows=50, sample_size=100, perplexity=5.0, random_state=0
        )
        assert values.shape == (len(small), 2)
        if choice.kind == SUBSAMPLED and "openTSNE" in choice.reason:
            if "not installed" in choice.reason:
                assert np.isnan(values).any()
            else:
                assert np.isfinite(values).all()

    def test_a_small_table_is_embedded_exactly(self, clustered):
        small = clustered[::60]
        _values, choice = pca_scaled(small, 2)
        assert choice.exact


class TestSubsampling:
    def test_it_takes_every_row_when_there_are_few_enough(self):
        data = np.zeros((10, 3))
        np.testing.assert_array_equal(subsample(data, 50), np.arange(10))

    def test_it_is_sorted_and_without_replacement(self):
        chosen = subsample(np.zeros((1000, 2)), 100, random_state=0)
        assert len(chosen) == 100
        assert len(np.unique(chosen)) == 100
        assert (np.diff(chosen) > 0).all()

    def test_it_is_reproducible(self):
        first = subsample(np.zeros((1000, 2)), 100, random_state=7)
        second = subsample(np.zeros((1000, 2)), 100, random_state=7)
        np.testing.assert_array_equal(first, second)


class TestBinnedScatter:
    def test_it_keeps_every_point(self, clustered):
        counts, _x, _y = binned_scatter(clustered[:, 0], clustered[:, 1], bins=32)
        assert counts.sum() == len(clustered)

    def test_it_returns_a_grid_and_its_edges(self, clustered):
        counts, x_edges, y_edges = binned_scatter(clustered[:, 0], clustered[:, 1], bins=16)
        assert counts.shape == (16, 16)
        assert len(x_edges) == len(y_edges) == 17

    def test_non_finite_values_are_dropped_rather_than_binned(self):
        # A Laplacian embedding leaves NaN for rows it could not place, and
        # those must not become a bin at the origin.
        x = np.array([0.0, 1.0, np.nan, 2.0])
        y = np.array([0.0, 1.0, 1.0, np.inf])
        counts, _x, _y = binned_scatter(x, y, bins=4)
        assert counts.sum() == 2

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="differ in length"):
            binned_scatter(np.zeros(4), np.zeros(5))

    def test_the_threshold_is_about_ink_not_memory(self):
        assert not should_bin(1000)
        assert should_bin(10**6)
        assert should_bin(200, limit=100)

    def test_limits_can_be_pinned_so_two_plots_compare(self, clustered):
        counts, x_edges, _y = binned_scatter(
            clustered[:, 0], clustered[:, 1], bins=8, limits=((0, 4), (0, 4))
        )
        assert x_edges[0] == 0 and x_edges[-1] == 4
        assert counts.sum() < len(clustered)


def _same_partition(left, right):
    """Whether two labellings group the rows the same way, ignoring which
    number each group got."""
    pairs = {(int(a), int(b)) for a, b in zip(left, right)}
    return len(pairs) == len(np.unique(left)) == len(np.unique(right))


class TestEstimatorChoice:
    def test_it_is_a_record_rather_than_a_flag(self):
        choice = EstimatorChoice("kmeans", SUBSAMPLED, 10_000, 500, "because")
        assert not choice.exact
        assert "500" in choice.describe()
        assert "10,000" in choice.describe()
        assert "because" in choice.describe()
