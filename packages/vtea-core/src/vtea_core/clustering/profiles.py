"""What each cluster is: its profile over the features it was clustered on.

A clustering hands back numbers - cluster 3, cluster 7 - and the work that
turns them into cell types is reading what distinguishes each one: which
markers it is bright for, which it lacks, how big it is. That is a table of
clusters by features, usually drawn as a heat map, and the clusters'
*top markers* are what a name is chosen from ("CD3, CD8 high: CD8 T cells").

`cluster_profiles` builds that table two ways:

- **mean**: each cluster's mean of each feature, in the feature's own units -
  what the cluster actually measured;
- **z**: the same mean as a z-score against all the objects profiled
  ((cluster mean - overall mean) / overall standard deviation), so features
  in unrelated units can share one colour scale and "bright for CD8" means
  the same thing for every marker.

`top_markers` reads the z table: per cluster, the features it is most above
(or below) the rest. Objects with a cluster id < 0 (not clustered, e.g.
outside the population a clustering was run on) are left out.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def cluster_profiles(
    data: pd.DataFrame, labels: Sequence[int] | np.ndarray, features: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """Per-cluster profiles of `features` over the rows of `data`.

    `labels` has one cluster id per row (< 0 = not clustered). Returns
    {"mean", "z", "counts"}: two cluster x feature tables (index: cluster id,
    sorted) and a table of each cluster's object count and fraction.
    """
    labels = np.asarray(labels)
    if len(labels) != len(data):
        raise ValueError(f"{len(labels)} labels for {len(data)} rows")
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise KeyError(f"features not in the table: {missing}")
    keep = labels >= 0
    values = data.loc[keep, list(features)].astype(float)
    ids = labels[keep]
    mean = values.groupby(ids).mean()
    overall = values.mean()
    spread = values.std(ddof=0).replace(0, np.nan)
    z = ((mean - overall) / spread).fillna(0.0)
    n = pd.Series(ids).value_counts().sort_index()
    counts = pd.DataFrame({"objects": n.values, "fraction": n.values / max(int(keep.sum()), 1)},
                          index=n.index)
    for table in (mean, z, counts):
        table.index.name = "cluster"
    return {"mean": mean, "z": z, "counts": counts}


def top_markers(z: pd.DataFrame, n: int = 3, low: bool = False, threshold: float = 0.5) -> dict[int, list[str]]:
    """Per cluster, up to `n` features whose z-score is highest (or, with `low`,
    lowest), keeping only those at least `threshold` from 0."""
    out = {}
    for cluster, row in z.iterrows():
        ordered = row.sort_values(ascending=low)
        picked = [f for f, v in ordered.items() if (v <= -threshold if low else v >= threshold)]
        out[cluster] = picked[:n]
    return out


def describe_clusters(z: pd.DataFrame, n: int = 3, threshold: float = 0.5) -> dict[int, str]:
    """A short description per cluster from its top markers, e.g. "CD8+ CD3+ / CD4-"."""
    high = top_markers(z, n, threshold=threshold)
    low = top_markers(z, n, low=True, threshold=threshold)
    out = {}
    for cluster in z.index:
        parts = [f"{f}+" for f in high[cluster]]
        lows = [f"{f}-" for f in low[cluster]]
        text = " ".join(parts)
        if lows:
            text = (text + " / " if text else "") + " ".join(lows)
        out[cluster] = text or "no distinctive feature"
    return out
