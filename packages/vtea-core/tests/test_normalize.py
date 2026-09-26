"""Feature normalisation before clustering and reduction - the Java
"Z-scale all data" checkbox, and the two alternatives beside it."""

import numpy as np
import pytest
from vtea_core.clustering import auto_k_kmeans, gaussian_mixture, hierarchical, kmeans
from vtea_core.measurements import NORMALIZATIONS, normalize_features
from vtea_core.reduction import isomap, laplacian_eigenmap, pca, tsne


class TestNormalizeFeatures:
    def test_zscore_matches_the_java_population_formula(self):
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 60.0]])
        result = normalize_features(data, "zscore")
        expected = (data - data.mean(0)) / data.std(0, ddof=0)
        np.testing.assert_allclose(result, expected)

    def test_a_constant_feature_becomes_zero(self):
        data = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        for method in ("zscore", "robust", "minmax"):
            assert np.all(normalize_features(data, method)[:, 1] == 0)

    def test_robust_ignores_an_outlier_that_flattens_a_zscore(self):
        data = np.r_[np.arange(20.0), [1e6]][:, np.newaxis]
        zscored = normalize_features(data, "zscore")[:20, 0]
        robust = normalize_features(data, "robust")[:20, 0]
        assert np.ptp(zscored) < 0.01  # every ordinary object squashed together
        assert np.ptp(robust) > 1.5

    def test_robust_falls_back_when_the_middle_half_is_one_value(self):
        data = np.r_[np.zeros(10), [5.0, 9.0]][:, np.newaxis]
        result = normalize_features(data, "robust")
        assert np.ptp(result) > 0

    def test_minmax_spans_zero_to_one(self):
        data = np.array([[2.0], [4.0], [6.0]])
        np.testing.assert_allclose(normalize_features(data, "minmax")[:, 0], [0, 0.5, 1])

    def test_none_returns_a_float_copy(self):
        data = np.array([[1, 2], [3, 4]])
        result = normalize_features(data, "none")
        assert result.dtype == float
        result[0, 0] = 99
        assert data[0, 0] == 1

    def test_unknown_method_is_refused(self):
        with pytest.raises(ValueError, match="unknown normalization"):
            normalize_features(np.ones((2, 2)), "whiten")

    def test_the_input_is_not_modified(self):
        data = np.array([[1.0, 2.0], [3.0, 5.0]])
        before = data.copy()
        normalize_features(data, "zscore")
        np.testing.assert_array_equal(data, before)

    def test_advertises_its_options(self):
        assert NORMALIZATIONS == ("none", "zscore", "robust", "minmax")


def two_populations_hidden_by_scale():
    """Two groups separated on a small-valued feature, with a large-valued
    feature of pure noise that dominates any unscaled distance."""
    rng = np.random.default_rng(0)
    signal = np.r_[rng.normal(0, 0.1, 50), rng.normal(1, 0.1, 50)]
    noise = rng.normal(0, 1000, 100)
    truth = np.r_[np.zeros(50), np.ones(50)]
    return np.c_[signal, noise], truth


def agreement(labels, truth):
    same = labels == labels[0]
    return max(np.mean(same == (truth == truth[0])), np.mean(same != (truth == truth[0])))


class TestEveryStepTakesIt:
    @pytest.mark.parametrize(
        "cluster",
        [
            lambda data, n: kmeans(data, n, random_state=0, normalize="zscore"),
            lambda data, n: gaussian_mixture(data, n, random_state=0, normalize="zscore"),
            lambda data, n: hierarchical(data, n, normalize="zscore"),
        ],
    )
    def test_zscore_recovers_what_scale_was_hiding(self, cluster):
        data, truth = two_populations_hidden_by_scale()
        assert agreement(np.asarray(cluster(data, 2)), truth) > 0.95

    def test_without_it_the_large_feature_wins(self):
        data, truth = two_populations_hidden_by_scale()
        assert agreement(kmeans(data, 2, random_state=0), truth) < 0.8

    def test_auto_k_takes_it(self):
        data, _truth = two_populations_hidden_by_scale()
        labels, scores = auto_k_kmeans(data, k_min=2, k_max=3, random_state=0, normalize="zscore")
        assert len(labels) == 100 and set(scores) == {2, 3}

    @pytest.mark.parametrize("reduce", [pca, isomap, laplacian_eigenmap])
    def test_reductions_take_it(self, reduce):
        data, _truth = two_populations_hidden_by_scale()
        assert reduce(data, 2, normalize="zscore").shape == (100, 2)

    def test_pca_on_zscored_data_is_pca_of_the_correlation_matrix(self):
        data, _truth = two_populations_hidden_by_scale()
        scaled = pca(data, 2, normalize="zscore")
        # Both features carry equal variance after scaling, so neither
        # component is the noise feature alone.
        assert np.var(scaled[:, 0]) / np.var(scaled[:, 1]) < 50

    def test_tsne_takes_it(self):
        data, _truth = two_populations_hidden_by_scale()
        assert tsne(data, 2, perplexity=10, random_state=0, normalize="zscore").shape == (100, 2)
