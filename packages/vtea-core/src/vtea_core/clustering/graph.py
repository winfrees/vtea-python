"""Graph-based clustering: Louvain and Leiden over a shared-neighbour graph.

The clustering VTEA shipped with is centroid-based - k-means, a Gaussian
mixture, Ward linkage - and every one of them needs to be told how many
populations the tissue contains before it has looked at it. That is the
wrong question to have to answer first: the number of cell types in a
biopsy is the thing being measured, not an input to measuring it.

Community detection asks it the other way round. Objects are joined into a
graph by how similar their features are, and the partition that maximises
modularity - more edges inside a community than a random graph of the same
degrees would have - decides both the assignment and how many communities
there are. `resolution` moves the granularity (higher splits more finely),
which is a much more natural dial than `k`.

The graph is built the way single-cell pipelines build theirs: a k-nearest
neighbour graph, symmetrised, with each edge weighted by the Jaccard overlap
of the two objects' neighbourhoods (a shared-nearest-neighbour graph). That
weighting is what makes the result robust to the arbitrariness of any single
neighbour link - two objects that merely happen to be each other's 15th
neighbour, with nothing else in common, contribute an edge of almost no
weight.

Leiden (Traag et al. 2019) is Louvain (Blondel et al. 2008) with the defect
that made Louvain occasionally return internally-disconnected communities
fixed, and it is the one to prefer; both are offered because Louvain is what
most published analyses ran and reproducing one means running what it ran.

Neither algorithm is implemented here. `leidenalg`/`python-igraph` (or
`networkx` for Louvain) do that, and are optional dependencies - install
them with `pip install "vtea-core[graph]"`. They are imported inside the
functions rather than at module import so that vtea-core keeps working
without them, the same way the deep-learning steps do.
"""

from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np

# What a neighbourhood is worth keeping: an edge weaker than this is two
# objects that share almost none of their neighbours, which is noise in the
# graph rather than evidence of a community.
MIN_JACCARD = 1.0 / 15.0


@contextmanager
def _seeded(random_state: int | None):
    """Run the block with Python's RNG seeded, then put it back as it was."""
    if random_state is None:
        yield
        return
    state = random.getstate()
    random.seed(random_state)
    try:
        yield
    finally:
        random.setstate(state)


def _install_hint(package: str, extra: str = "graph") -> str:
    return (
        f"{package} is needed for graph-based clustering and is not installed. "
        f'Install it with `pip install "vtea-core[{extra}]"` (or `pip install {package}`).'
    )


def shared_neighbor_graph(
    data: np.ndarray,
    *,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    min_weight: float = MIN_JACCARD,
) -> tuple[np.ndarray, np.ndarray]:
    """The SNN graph of `data` as (edges, weights).

    `edges` is an (n_edges, 2) array of node index pairs, i < j, each listed
    once; `weights` is the Jaccard overlap of the two nodes' neighbourhoods.

    A row of `data` is one object and a column one feature, so the nodes of
    the returned graph are in the same order as the rows of the measurement
    table - which is what lets the resulting community ids be joined back on
    as a feature.
    """
    from sklearn.neighbors import NearestNeighbors

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"expected a (n_objects, n_features) matrix, got shape {data.shape}")
    n_objects = data.shape[0]
    if n_objects < 2:
        raise ValueError("graph-based clustering needs at least two objects")

    # Capped at n_objects - 1 so a small run - a test, a crop with nine
    # objects in it - asks for no more neighbours than there are to ask for.
    # Queried with X=None, which is what makes sklearn leave each point out
    # of its own neighbourhood.
    k = int(max(1, min(n_neighbors, n_objects - 1)))
    model = NearestNeighbors(n_neighbors=k, metric=metric).fit(data)
    connectivity = model.kneighbors_graph(mode="connectivity").tocsr()

    # |N(i) ∩ N(j)| for every pair that shares at least one neighbour, then
    # Jaccard: the intersection over the union, the union taken from the two
    # rows' actual degrees rather than assumed to be 2k - ties in the
    # distances can leave a row holding one neighbour more than asked for,
    # and a weight above 1 would be silently wrong rather than loudly.
    shared = (connectivity @ connectivity.T).tocoo()
    degrees = np.diff(connectivity.indptr).astype(float)
    rows, columns, counts = shared.row, shared.col, shared.data.astype(float)
    upper = rows < columns
    rows, columns, counts = rows[upper], columns[upper], counts[upper]
    union = degrees[rows] + degrees[columns] - counts
    weights = np.divide(counts, union, out=np.zeros_like(counts), where=union > 0)

    keep = weights >= min_weight
    edges = np.column_stack([rows[keep], columns[keep]]).astype(np.int64)
    return edges, weights[keep]


def _relabel_by_size(membership: np.ndarray) -> np.ndarray:
    """Renumber communities so 0 is the largest, 1 the next, and so on.

    Community detection numbers its output in whatever order the algorithm
    happened to find things, which makes two runs of the same data hard to
    compare and puts no meaning at all in the id. Ordering by size gives the
    ids a reading - "cluster 0" is the population the tissue is mostly made
    of - and makes a colour map comparable between runs.
    """
    values, counts = np.unique(membership, return_counts=True)
    order = values[np.argsort(-counts, kind="stable")]
    ranks = {value: rank for rank, value in enumerate(order)}
    return np.array([ranks[value] for value in membership], dtype=np.int32)


def _igraph_from(edges: np.ndarray, weights: np.ndarray, n_nodes: int):
    import igraph

    graph = igraph.Graph(n=n_nodes, edges=[tuple(edge) for edge in edges], directed=False)
    graph.es["weight"] = [float(weight) for weight in weights]
    return graph


def louvain(
    data: np.ndarray,
    *,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    metric: str = "euclidean",
    random_state: int | None = None,
) -> np.ndarray:
    """Community assignments via Louvain modularity maximisation.

    `resolution` above 1 finds more, smaller communities; below 1, fewer and
    larger. `n_neighbors` sets how connected the graph is: too few and the
    graph fragments into communities that are an artefact of the sampling,
    too many and distinct populations are bridged into one.

    Uses python-igraph where it is installed and networkx otherwise - both
    implement the same algorithm, and neither is a hard dependency of
    vtea-core.
    """
    edges, weights = shared_neighbor_graph(data, n_neighbors=n_neighbors, metric=metric)
    n_nodes = len(np.asarray(data))

    try:
        graph = _igraph_from(edges, weights, n_nodes)
    except ImportError:
        return _louvain_networkx(edges, weights, n_nodes, resolution, random_state)

    # igraph draws from Python's own `random` module, so seeding it is how a
    # run is made reproducible - and the previous state is put back
    # afterwards, since a clustering step has no business reseeding the
    # random numbers the rest of the process is drawing from.
    with _seeded(random_state):
        try:
            membership = graph.community_multilevel(
                weights="weight", resolution=float(resolution)
            ).membership
        except TypeError:  # python-igraph < 0.10 has no resolution parameter
            membership = graph.community_multilevel(weights="weight").membership
    return _relabel_by_size(np.asarray(membership))


def _louvain_networkx(
    edges: np.ndarray,
    weights: np.ndarray,
    n_nodes: int,
    resolution: float,
    random_state: int | None,
) -> np.ndarray:
    try:
        import networkx as nx
    except ImportError as exc:  # neither backend present
        raise ImportError(_install_hint("python-igraph (or networkx)")) from exc

    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_weighted_edges_from(
        (int(a), int(b), float(weight)) for (a, b), weight in zip(edges, weights)
    )
    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=float(resolution), seed=random_state
    )
    membership = np.zeros(n_nodes, dtype=np.int32)
    for index, community in enumerate(communities):
        for node in community:
            membership[node] = index
    return _relabel_by_size(membership)


def leiden(
    data: np.ndarray,
    *,
    n_neighbors: int = 15,
    resolution: float = 1.0,
    metric: str = "euclidean",
    n_iterations: int = 2,
    random_state: int | None = None,
) -> np.ndarray:
    """Community assignments via the Leiden algorithm.

    Leiden is Louvain with a refinement pass that guarantees every community
    it returns is internally connected - the one thing Louvain can silently
    get wrong, and which on tissue data shows up as a "population" that is
    really two populations that never touch.

    `n_iterations` runs the whole optimisation repeatedly (-1 iterates until
    it stops improving), and `resolution` moves the granularity as it does
    for Louvain.
    """
    edges, weights = shared_neighbor_graph(data, n_neighbors=n_neighbors, metric=metric)
    n_nodes = len(np.asarray(data))

    try:
        graph = _igraph_from(edges, weights, n_nodes)
    except ImportError as exc:
        raise ImportError(_install_hint("python-igraph")) from exc

    try:
        import leidenalg
    except ImportError:
        # python-igraph carries its own Leiden implementation. Its
        # modularity objective is the same one leidenalg's
        # RBConfigurationVertexPartition optimises, so this is the same
        # clustering by a different route rather than a lesser fallback.
        with _seeded(random_state):
            partition = graph.community_leiden(
                objective_function="modularity",
                weights="weight",
                resolution=float(resolution),
                n_iterations=int(n_iterations),
            )
        return _relabel_by_size(np.asarray(partition.membership))

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        n_iterations=int(n_iterations),
        seed=random_state,
    )
    return _relabel_by_size(np.asarray(partition.membership))
