import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from vtea_core.clustering import leiden, louvain, nearest_neighbors, shared_neighbor_graph
from vtea_core.clustering.graph import _relabel_by_size

igraph = pytest.importorskip("igraph")


def make_three_blobs(seed=0, n=40):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [50, 0], [25, 50]])
    points = np.concatenate([rng.normal(c, 2.0, size=(n, 2)) for c in centers])
    true_labels = np.repeat([0, 1, 2], n)
    return points, true_labels


class TestSharedNeighborGraph:
    def test_edges_are_listed_once_in_index_order(self):
        data, _ = make_three_blobs(n=10)
        edges, weights = shared_neighbor_graph(data, n_neighbors=5)
        assert (edges[:, 0] < edges[:, 1]).all()
        assert len(edges) == len(weights)

    def test_weights_are_jaccard_overlaps(self):
        data, _ = make_three_blobs(n=10)
        _edges, weights = shared_neighbor_graph(data, n_neighbors=5)
        assert weights.min() > 0
        assert weights.max() <= 1.0

    def test_no_edges_bridge_well_separated_blobs(self):
        """Three blobs 50 apart share no neighbours, so the graph should
        come apart into three pieces on its own - which is what lets
        community detection find three communities without being told."""
        data, true_labels = make_three_blobs(n=15)
        edges, _weights = shared_neighbor_graph(data, n_neighbors=5)
        crossing = [pair for pair in edges if true_labels[pair[0]] != true_labels[pair[1]]]
        assert crossing == []

    def test_more_neighbours_than_objects_is_capped_not_an_error(self):
        data, _ = make_three_blobs(n=3)
        edges, _weights = shared_neighbor_graph(data, n_neighbors=500)
        assert len(edges) > 0

    def test_one_object_is_rejected_with_a_readable_message(self):
        with pytest.raises(ValueError, match="at least two objects"):
            shared_neighbor_graph(np.zeros((1, 3)))

    def test_a_non_matrix_is_rejected(self):
        with pytest.raises(ValueError, match="n_objects, n_features"):
            shared_neighbor_graph(np.zeros((4, 4, 4)))


class TestLouvain:
    def test_recovers_well_separated_clusters_without_being_told_how_many(self):
        data, true_labels = make_three_blobs()
        labels = louvain(data, random_state=0)
        assert adjusted_rand_score(true_labels, labels) > 0.95

    def test_returns_one_label_per_object(self):
        data, _ = make_three_blobs()
        labels = louvain(data, random_state=0)
        assert labels.shape == (len(data),)

    def test_too_few_neighbours_fragments_a_blob(self):
        """Not a defect - the dial doing what it does. A graph built from
        five neighbours per object has structure inside each blob, and
        modularity finds it; the docstring's warning about it is only worth
        anything if it is true."""
        data, true_labels = make_three_blobs()
        fragmented = louvain(data, n_neighbors=5, random_state=0)
        assert len(set(fragmented)) > len(set(true_labels))

    def test_a_higher_resolution_does_not_merge_the_blobs(self):
        data, true_labels = make_three_blobs()
        labels = louvain(data, resolution=2.0, random_state=0)
        assert len(set(labels)) >= len(set(true_labels))


class TestLeiden:
    def test_recovers_well_separated_clusters(self):
        data, true_labels = make_three_blobs()
        labels = leiden(data, random_state=0)
        assert adjusted_rand_score(true_labels, labels) > 0.95

    def test_ids_are_ordered_by_community_size(self):
        """Cluster 0 is the population the sample is mostly made of, in
        every run - otherwise two runs of one dataset colour differently and
        the id itself means nothing."""
        rng = np.random.default_rng(1)
        big = rng.normal([0, 0], 1.0, size=(90, 2))
        small = rng.normal([40, 40], 1.0, size=(20, 2))
        labels = leiden(np.concatenate([big, small]), n_neighbors=8, random_state=0)
        counts = np.bincount(labels)
        assert list(counts) == sorted(counts, reverse=True)

    def test_it_is_deterministic_for_a_fixed_seed(self):
        data, _ = make_three_blobs()
        first = leiden(data, random_state=7)
        second = leiden(data, random_state=7)
        np.testing.assert_array_equal(first, second)


class TestNearestNeighbors:
    def test_each_object_is_left_out_of_its_own_neighbours(self):
        data, _ = make_three_blobs(n=10)
        indices, distances = nearest_neighbors(data, n_neighbors=5)
        assert indices.shape == distances.shape == (30, 5)
        assert not (indices == np.arange(30)[:, None]).any()
        assert (np.diff(distances, axis=1) >= 0).all()

    def test_a_shared_search_gives_the_same_graph(self):
        """The search made once and handed over is the graph the function
        would have built itself - what lets UMAP and Leiden share one."""
        data, _ = make_three_blobs()
        own = shared_neighbor_graph(data, n_neighbors=8)
        shared = shared_neighbor_graph(data, n_neighbors=8, knn=nearest_neighbors(data, n_neighbors=12))
        np.testing.assert_array_equal(own[0], shared[0])
        np.testing.assert_allclose(own[1], shared[1])

    def test_a_search_of_other_objects_is_rejected(self):
        data, _ = make_three_blobs(n=10)
        with pytest.raises(ValueError, match="rows"):
            shared_neighbor_graph(data, knn=nearest_neighbors(data[:20], n_neighbors=5))


class TestLeidenImplementations:
    def test_igraph_and_leidenalg_find_the_same_communities(self):
        pytest.importorskip("leidenalg")
        data, true_labels = make_three_blobs()
        fast = leiden(data, random_state=0, implementation="igraph")
        reference = leiden(data, random_state=0, implementation="leidenalg")
        assert adjusted_rand_score(fast, reference) == 1.0
        assert adjusted_rand_score(true_labels, fast) > 0.95

    def test_both_are_deterministic_for_a_fixed_seed(self):
        pytest.importorskip("leidenalg")
        data, _ = make_three_blobs()
        for implementation in ("igraph", "leidenalg"):
            first = leiden(data, random_state=3, implementation=implementation)
            np.testing.assert_array_equal(first, leiden(data, random_state=3, implementation=implementation))

    def test_a_shared_search_gives_the_same_clusters(self):
        data, _ = make_three_blobs()
        knn = nearest_neighbors(data, n_neighbors=15)
        np.testing.assert_array_equal(leiden(data, random_state=0), leiden(data, random_state=0, knn=knn))
        np.testing.assert_array_equal(louvain(data, random_state=0), louvain(data, random_state=0, knn=knn))

    def test_an_unknown_implementation_is_rejected(self):
        data, _ = make_three_blobs(n=5)
        with pytest.raises(ValueError, match="implementation"):
            leiden(data, implementation="fast")


class TestRelabelBySize:
    def test_largest_community_becomes_zero(self):
        membership = np.array([5, 5, 5, 2, 2, 9])
        assert list(_relabel_by_size(membership)) == [0, 0, 0, 1, 1, 2]

    def test_result_is_integer_typed_so_it_can_be_a_label_column(self):
        assert _relabel_by_size(np.array([3, 3, 1])).dtype == np.int32
