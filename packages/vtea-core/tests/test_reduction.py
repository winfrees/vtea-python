import numpy as np

from vtea_core.reduction import isomap, laplacian_eigenmap, pca, pca_explained_variance, tsne


def make_correlated_data(seed=0, n=50):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    # y is a noisy copy of x, z is independent noise -> most variance along one axis
    y = x + rng.normal(scale=0.01, size=n)
    z = rng.normal(scale=0.01, size=n)
    return np.column_stack([x, y, z])


class TestPCA:
    def test_output_shape(self):
        data = make_correlated_data()
        embedding = pca(data, n_components=2)
        assert embedding.shape == (50, 2)

    def test_first_component_captures_most_variance(self):
        data = make_correlated_data()
        cumvar = pca_explained_variance(data, n_components=3)
        assert cumvar[0] > 0.9  # first PC alone explains >90% given the strong x~y correlation

    def test_explained_variance_is_nondecreasing(self):
        data = make_correlated_data()
        cumvar = pca_explained_variance(data, n_components=3)
        assert np.all(np.diff(cumvar) >= -1e-10)


class TestIsomap:
    def test_output_shape(self):
        data = make_correlated_data(n=30)
        embedding = isomap(data, n_components=2, n_neighbors=5)
        assert embedding.shape == (30, 2)


class TestLaplacianEigenmap:
    def test_output_shape(self):
        data = make_correlated_data(n=30)
        embedding = laplacian_eigenmap(data, n_components=2, n_neighbors=10)
        assert embedding.shape == (30, 2)


class TestTSNE:
    def test_output_shape(self):
        data = make_correlated_data(n=30)
        embedding = tsne(data, n_components=2, perplexity=5, random_state=0)
        assert embedding.shape == (30, 2)

    def test_deterministic_with_fixed_random_state(self):
        data = make_correlated_data(n=30)
        a = tsne(data, n_components=2, perplexity=5, random_state=0)
        b = tsne(data, n_components=2, perplexity=5, random_state=0)
        np.testing.assert_array_equal(a, b)
