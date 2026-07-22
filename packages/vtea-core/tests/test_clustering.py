import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from vtea_core.clustering import auto_k_kmeans, gaussian_mixture, hierarchical, kmeans


def make_three_blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [50, 0], [25, 50]])
    points = np.concatenate([rng.normal(c, 2.0, size=(30, 2)) for c in centers])
    true_labels = np.repeat([0, 1, 2], 30)
    return points, true_labels


class TestKMeans:
    def test_recovers_well_separated_clusters(self):
        data, true_labels = make_three_blobs()
        labels = kmeans(data, n_clusters=3, random_state=0)
        assert adjusted_rand_score(true_labels, labels) > 0.95

    def test_returns_correct_length(self):
        data, _ = make_three_blobs()
        labels = kmeans(data, n_clusters=3, random_state=0)
        assert len(labels) == len(data)


class TestGaussianMixture:
    def test_recovers_well_separated_clusters(self):
        data, true_labels = make_three_blobs()
        labels = gaussian_mixture(data, n_clusters=3, random_state=0)
        assert adjusted_rand_score(true_labels, labels) > 0.95


class TestHierarchical:
    @pytest.mark.parametrize("linkage", ["ward", "single", "complete"])
    def test_recovers_well_separated_clusters(self, linkage):
        data, true_labels = make_three_blobs()
        labels = hierarchical(data, n_clusters=3, linkage=linkage)
        assert adjusted_rand_score(true_labels, labels) > 0.9

    def test_unknown_linkage_raises(self):
        data, _ = make_three_blobs()
        with pytest.raises(ValueError, match="unknown linkage"):
            hierarchical(data, n_clusters=3, linkage="bogus")


class TestAutoKKMeans:
    def test_finds_correct_k_on_well_separated_blobs(self):
        data, true_labels = make_three_blobs()
        labels, scores = auto_k_kmeans(data, k_min=2, k_max=6, random_state=0)
        best_k = max(scores, key=scores.get)
        assert best_k == 3
        assert adjusted_rand_score(true_labels, labels) > 0.95

    def test_scores_cover_full_k_range(self):
        data, _ = make_three_blobs()
        _, scores = auto_k_kmeans(data, k_min=2, k_max=5, random_state=0)
        assert set(scores.keys()) == {2, 3, 4, 5}

    def test_k_max_less_than_k_min_raises(self):
        data, _ = make_three_blobs()
        with pytest.raises(ValueError, match="k_max"):
            auto_k_kmeans(data, k_min=5, k_max=2)
