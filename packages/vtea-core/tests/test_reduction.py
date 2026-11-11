import numpy as np
import pytest

from vtea_core.reduction import isomap, laplacian_eigenmap, pca, pca_explained_variance, tsne
from vtea_core.reduction import umap as umap_projection


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


class TestTSNEMethods:
    def test_opentsne_gives_an_embedding_and_is_reproducible(self):
        pytest.importorskip("openTSNE")
        data = make_correlated_data(n=60)
        a = tsne(data, perplexity=5, random_state=0, method="opentsne")
        b = tsne(data, perplexity=5, random_state=0, method="opentsne")
        assert a.shape == (60, 2)
        np.testing.assert_allclose(a, b)

    def test_an_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="method"):
            tsne(make_correlated_data(n=30), perplexity=5, method="fast")


class TestUMAP:
    def test_unseeded_runs_on_every_core(self):
        pytest.importorskip("umap")
        data = make_correlated_data(n=40)
        embedding = umap_projection(data, n_neighbors=5, random_state=None, n_jobs=-1)
        assert embedding.shape == (40, 2) and np.isfinite(embedding).all()

    def test_a_shared_neighbour_search_gives_as_good_a_map(self):
        """UMAP from a neighbour search made once (and shared with graph
        clustering) keeps neighbourhoods as well as UMAP searching for
        itself, and a seeded run of it is still reproducible. (Not the same
        map: UMAP's own search draws on the random state first.)"""
        pytest.importorskip("umap")
        from sklearn.manifold import trustworthiness

        from vtea_core.clustering import nearest_neighbors

        rng = np.random.default_rng(3)
        centres = rng.normal(0, 4, size=(4, 6))
        data = centres[rng.integers(0, 4, 300)] + rng.normal(size=(300, 6))
        knn = nearest_neighbors(data, n_neighbors=15)
        own = umap_projection(data, random_state=0)
        shared = umap_projection(data, random_state=0, knn=knn)
        np.testing.assert_allclose(shared, umap_projection(data, random_state=0, knn=knn))
        assert abs(trustworthiness(data, own, n_neighbors=10) - trustworthiness(data, shared, n_neighbors=10)) < 0.02

    def test_a_knn_of_the_wrong_size_is_rejected(self):
        pytest.importorskip("umap")
        from vtea_core.clustering import nearest_neighbors

        data = np.random.default_rng(3).normal(size=(40, 3))
        with pytest.raises(ValueError, match="neighbours"):
            umap_projection(data, n_neighbors=10, knn=nearest_neighbors(data, n_neighbors=4))
        with pytest.raises(ValueError, match="rows"):
            umap_projection(data, n_neighbors=5, knn=nearest_neighbors(data[:30], n_neighbors=5))

    def test_output_shape(self):
        umap = pytest.importorskip("umap")  # noqa: F841 - umap-learn is optional
        data = make_correlated_data(n=30)
        embedding = umap_projection(data, n_components=2, n_neighbors=5, random_state=0)
        assert embedding.shape == (30, 2)

    def test_deterministic_with_fixed_random_state(self):
        pytest.importorskip("umap")
        data = make_correlated_data(n=30)
        first = umap_projection(data, n_components=2, n_neighbors=5, random_state=0)
        second = umap_projection(data, n_components=2, n_neighbors=5, random_state=0)
        np.testing.assert_allclose(first, second)

    def test_a_missing_umap_learn_says_how_to_install_it(self, monkeypatch):
        """The step is in the menu whether or not umap-learn is installed,
        so the failure it gives when it is not has to be the instruction."""
        import builtins

        real_import = builtins.__import__

        def without_umap(name, *args, **kwargs):
            if name == "umap":
                raise ImportError("no module named umap")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_umap)
        with pytest.raises(ImportError, match="vtea-core\\[umap\\]"):
            umap_projection(make_correlated_data(n=10))
