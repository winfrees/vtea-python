"""Which neighbourhood features find tissue structure: a benchmark.

Synthetic tissues with a known structure, each cell labelled with the zone
it lies in; windows are drawn around every cell, described by a feature set,
clustered, and scored by how well the clusters recover the zones (adjusted
Rand index of each cell's own window), how stable they are over clustering
seeds, and how long they take. The defaults offered in the workbench come
from this table rather than from a guess.

Tissues (`SCENARIOS`), 2D or 3D (a slab with anisotropic spacing):

| name | zones | what it tests |
| --- | --- | --- |
| `halves` | two halves with overlapping class mixes | plain composition differences |
| `bands` | alternating bands, three kinds, 40 um wide | thin elongated compartments (tubules, interstitium) |
| `aggregates` | round immune aggregates in a background | small compact structures |
| `infiltration` | front / transition / hinterland of a gradient | alteration rather than compartments |

Cells sit on a jittered lattice about 10 um apart; each zone has its own mix
of five classes, and each class its own marker profile plus noise, so the
`markers` sets can work without classes at all.

Run `run_benchmark()` (or `scripts/benchmark_neighborhood_features.py`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vtea_core.hierarchy import features as F
from vtea_core.hierarchy.hierarchy import Hierarchy

CLASSES = ("epithelial", "stromal", "endothelial", "immune", "other")
N_MARKERS = 6

# Class mix of each zone (rows sum to 1), per scenario.
_MIXES = {
    "halves": np.array([[0.55, 0.20, 0.10, 0.10, 0.05], [0.15, 0.40, 0.15, 0.25, 0.05]]),
    "bands": np.array(
        [[0.75, 0.10, 0.05, 0.05, 0.05], [0.15, 0.55, 0.10, 0.15, 0.05], [0.10, 0.20, 0.55, 0.10, 0.05]]
    ),
    "aggregates": np.array([[0.55, 0.25, 0.10, 0.05, 0.05], [0.10, 0.10, 0.05, 0.70, 0.05]]),
    "infiltration": np.array(
        [[0.15, 0.15, 0.05, 0.60, 0.05], [0.35, 0.20, 0.07, 0.33, 0.05], [0.60, 0.22, 0.08, 0.05, 0.05]]
    ),
}
SCENARIOS = tuple(_MIXES)


@dataclass
class Tissue:
    name: str
    table: pd.DataFrame  # object_id, centroid-*, class, marker_*
    truth: np.ndarray  # zone per cell
    spacing: tuple[float, ...]  # um per voxel, per axis


def make_tissue(
    scenario: str, *, ndim: int = 2, size: float = 400.0, step: float = 10.0, seed: int = 0
) -> Tissue:
    """A synthetic tissue: positions, zones, classes and markers."""
    if scenario not in _MIXES:
        raise ValueError(f"unknown scenario {scenario!r}, expected one of {SCENARIOS}")
    rng = np.random.default_rng(seed)
    if ndim == 2:
        axes = [np.arange(0, size, step)] * 2
    elif ndim == 3:
        axes = [np.arange(0, 60.0, step), np.arange(0, size, step), np.arange(0, size, step)]
    else:
        raise ValueError("ndim must be 2 or 3")
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, ndim)
    um = grid + rng.uniform(-0.3, 0.3, grid.shape) * step
    y, x = um[:, -2], um[:, -1]
    if scenario == "halves":
        truth = (x >= size / 2).astype(int)
    elif scenario == "bands":
        truth = (np.floor(x / 40.0).astype(int)) % 3
    elif scenario == "aggregates":
        centres = rng.uniform(60, size - 60, size=(max(3, int(size / 130)) ** 2, 2))
        d = np.min(np.linalg.norm(np.c_[y, x][:, None, :] - centres[None], axis=2), axis=1)
        truth = (d < 40.0).astype(int)
    else:
        truth = np.digitize(x, [size / 3, 2 * size / 3])
    mix = _MIXES[scenario]
    classes = np.array([rng.choice(len(CLASSES), p=mix[z]) for z in truth])
    profiles = rng.uniform(0, 1, size=(len(CLASSES), N_MARKERS)) * 100
    markers = profiles[classes] + rng.normal(0, 15, size=(len(classes), N_MARKERS))
    # Voxels: 0.5 um in plane, 2 um in z, so 3D distances need the spacing.
    spacing = (0.5, 0.5) if ndim == 2 else (2.0, 0.5, 0.5)
    voxels = um / np.asarray(spacing)
    table = pd.DataFrame({"object_id": np.arange(1, len(um) + 1)})
    for axis in range(ndim):
        table[f"centroid-{axis}"] = voxels[:, axis]
    table["class"] = np.array(CLASSES, dtype=object)[classes]
    for k in range(N_MARKERS):
        table[f"marker_{k}"] = markers[:, k]
    return Tissue(scenario, table, truth, spacing)


# A feature set: (hierarchy, windows level name, radius) -> (matrix, normalize)
FeatureSet = Callable[[Hierarchy, str, float], tuple[np.ndarray, str]]


def _fractions(level) -> pd.DataFrame:
    return level.table[[c for c in level.table.columns if str(c).endswith("_ClassFraction")]]


def _markers(h: Hierarchy, name: str) -> pd.DataFrame:
    level = h[name]
    return level.table[[c for c in level.table.columns if str(c).startswith("mean_marker_")]]


def _with(*extra: str) -> FeatureSet:
    def make(h, name, radius):
        blocks = [_fractions(h[name])]
        for family in extra:
            if family == "markers":
                blocks.append(_markers(h, name))
            else:
                out = h.describe(name, family, classes="class", add=False)
                keep = {
                    "diversity": ["shannon", "simpson", "dominant_fraction"],
                    "arrangement": ["radius_of_gyration", "seed_offset", "member_nn_distance"],
                    "asymmetry": ["asymmetry"],
                }[family]
                blocks.append(out[keep])
        return pd.concat(blocks, axis=1).to_numpy(float), "zscore"

    return make


FEATURE_SETS: dict[str, FeatureSet] = {
    "composition": lambda h, n, r: (_fractions(h[n]).to_numpy(float), "none"),
    "composition, clr": lambda h, n, r: (transform(h, n, "clr"), "none"),
    "composition, sqrt": lambda h, n, r: (transform(h, n, "sqrt"), "none"),
    "weighted (sigma r/2)": lambda h, n, r: (
        h.describe(n, "weighted", classes="class", sigma=r / 2, add=False).to_numpy(float),
        "none",
    ),
    "multiscale (r/2, r, 2r)": lambda h, n, r: (
        h.describe(n, "multiscale", classes="class", radii=[r / 2, r, 2 * r], add=False).to_numpy(float),
        "none",
    ),
    "composition + diversity": _with("diversity"),
    "composition + arrangement": _with("arrangement"),
    "composition + asymmetry": _with("asymmetry"),
    "markers (member means)": lambda h, n, r: (_markers(h, n).to_numpy(float), "zscore"),
    "composition + markers": _with("markers"),
}


def transform(h: Hierarchy, name: str, how: str) -> np.ndarray:
    return F.transform_composition(_fractions(h[name]), how).to_numpy(float)


def _cluster(matrix, method, k, seed, normalize):
    from vtea_core import clustering

    matrix = np.nan_to_num(matrix)
    if method == "kmeans":
        return clustering.kmeans(matrix, k, random_state=seed, normalize=normalize)
    if method == "gaussian_mixture":
        return clustering.gaussian_mixture(matrix, k, random_state=seed, normalize=normalize)
    if method == "leiden":
        return clustering.leiden(matrix, n_neighbors=15, random_state=seed, normalize=normalize, resolution=0.3)
    raise ValueError(method)


def run_benchmark(
    scenarios=SCENARIOS,
    feature_sets=None,
    methods=("kmeans", "gaussian_mixture", "leiden"),
    *,
    ndims=(2, 3),
    radius: float = 30.0,
    size: float = 400.0,
    tissue_seeds=(0, 1),
    cluster_seeds=(0, 1, 2),
) -> pd.DataFrame:
    """One row per (scenario, ndim, feature set, method): mean and SD of the
    ARI over tissue and clustering seeds, stability (mean ARI between the
    clusterings of the same tissue with different seeds), features, seconds
    (features + one clustering)."""
    from sklearn.metrics import adjusted_rand_score

    feature_sets = FEATURE_SETS if feature_sets is None else {k: FEATURE_SETS[k] for k in feature_sets}
    rows = []
    for scenario in scenarios:
        for ndim in ndims:
            results: dict[tuple[str, str], dict[str, list]] = {}
            for tissue_seed in tissue_seeds:
                tissue = make_tissue(scenario, ndim=ndim, size=size, seed=tissue_seed)
                from vtea_core.data import Spacing

                h = Hierarchy(tissue.table, spacing=Spacing(values=tissue.spacing, unit="um"))
                markers = ",".join(c for c in tissue.table.columns if c.startswith("marker_"))
                h.add_windows("Objects", "W", radius=radius, classes="class", features=markers)
                k = len(np.unique(tissue.truth))
                for set_name, make in feature_sets.items():
                    started = time.perf_counter()
                    matrix, normalize = make(h, "W", radius)
                    feature_seconds = time.perf_counter() - started
                    for method in methods:
                        entry = results.setdefault((set_name, method), {"ari": [], "stab": [], "sec": [], "nf": 0})
                        labels = []
                        for cluster_seed in cluster_seeds:
                            started = time.perf_counter()
                            found = _cluster(matrix, method, k, cluster_seed, normalize)
                            entry["sec"].append(feature_seconds + time.perf_counter() - started)
                            labels.append(found)
                            entry["ari"].append(adjusted_rand_score(tissue.truth, found))
                        entry["stab"] += [
                            adjusted_rand_score(labels[i], labels[j])
                            for i in range(len(labels))
                            for j in range(i + 1, len(labels))
                        ]
                        entry["nf"] = matrix.shape[1]
            for (set_name, method), entry in results.items():
                rows.append(
                    {
                        "scenario": scenario,
                        "ndim": ndim,
                        "features": set_name,
                        "method": method,
                        "ari": float(np.mean(entry["ari"])),
                        "ari_sd": float(np.std(entry["ari"])),
                        "stability": float(np.mean(entry["stab"])) if entry["stab"] else np.nan,
                        "n_features": entry["nf"],
                        "seconds": float(np.mean(entry["sec"])),
                    }
                )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean ARI per (feature set, method) over every scenario and ndim, with
    the worst scenario's ARI, sorted best first."""
    grouped = results.groupby(["features", "method"])
    out = pd.DataFrame(
        {
            "mean_ari": grouped["ari"].mean(),
            "worst_ari": grouped["ari"].min(),
            "stability": grouped["stability"].mean(),
            "seconds": grouped["seconds"].mean(),
        }
    )
    return out.sort_values("mean_ari", ascending=False)
