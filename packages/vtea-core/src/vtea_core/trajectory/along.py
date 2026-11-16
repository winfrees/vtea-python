"""What changes along a trajectory: features, classes and neighbourhoods by pseudotime.

Once each object has a pseudotime on each lineage (`slingshot`), these tables
describe the lineages, for plots and for tests:

- `feature_trends`: the mean (and standard error) of each feature in bins of
  pseudotime, per lineage - a marker's "pseudotime spectrum" (e.g. THY1 rising
  along one branch, PROM1 along another).
- `association`: does a feature change along a lineage? Spearman correlation
  with pseudotime and its p-value, Benjamini-Hochberg q-values over all
  (lineage, feature) tests, and the start-vs-end difference (mean of the last
  fifth minus the first fifth of the lineage, in standard deviations of the
  feature) - the questions of tradeSeq's `associationTest` and
  `startVsEndTest` (Van den Berge et al., Nat Commun 11:1201, 2020), answered
  without the GAM.
- `composition_along`: the fraction of each class (cell type, cluster,
  neighbourhood type) in bins of pseudotime - how the mix of cell types or
  of neighbourhoods changes along a lineage.

Objects count for a lineage when their pseudotime on it is finite; their
lineage weight weights the means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pseudotime_bins(pseudotime, n_bins: int = 20):
    """Bin edges (n_bins + 1) spanning the finite pseudotimes of one lineage."""
    t = np.asarray(pseudotime, float)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return np.linspace(0, 1, n_bins + 1)
    lo, hi = float(t.min()), float(t.max())
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_bins + 1)


def _bin(t, edges):
    return np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2)


def _lineage_names(n, names):
    return list(names) if names is not None else [f"t{j + 1}" for j in range(n)]


def feature_trends(
    pseudotime, values, *, weights=None, n_bins: int = 20, lineages=None
) -> pd.DataFrame:
    """Long table, one row per (lineage, bin, feature): `lineage`, `bin`,
    `pseudotime` (bin centre), `feature`, `mean`, `sem`, `objects`.

    `pseudotime`: (objects, lineages); `values`: DataFrame (objects x features);
    `weights`: (objects, lineages), default 1."""
    t_all = np.atleast_2d(np.asarray(pseudotime, float))
    if t_all.shape[0] != len(values):
        t_all = t_all.T
    w_all = (
        np.ones_like(t_all) if weights is None else np.asarray(weights, float).reshape(t_all.shape)
    )
    names = _lineage_names(t_all.shape[1], lineages)
    x = values.to_numpy(float)
    rows = []
    for j, name in enumerate(names):
        on = np.isfinite(t_all[:, j]) & (w_all[:, j] > 0)
        if not on.any():
            continue
        edges = pseudotime_bins(t_all[on, j], n_bins)
        which = _bin(t_all[on, j], edges)
        w = w_all[on, j]
        centres = (edges[:-1] + edges[1:]) / 2
        for k, feature in enumerate(values.columns):
            v = x[on, k]
            ok = np.isfinite(v)
            wb = np.bincount(which[ok], weights=w[ok], minlength=n_bins)
            s1 = np.bincount(which[ok], weights=w[ok] * v[ok], minlength=n_bins)
            s2 = np.bincount(which[ok], weights=w[ok] * v[ok] ** 2, minlength=n_bins)
            count = np.bincount(which[ok], minlength=n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = s1 / wb
                var = np.maximum(s2 / wb - mean**2, 0.0)
                sem = np.sqrt(var / np.maximum(count, 1))
            for b in range(n_bins):
                rows.append((name, b, centres[b], feature, mean[b], sem[b], int(count[b])))
    return pd.DataFrame(
        rows, columns=["lineage", "bin", "pseudotime", "feature", "mean", "sem", "objects"]
    )


def association(
    pseudotime, values, *, weights=None, lineages=None, end_fraction: float = 0.2
) -> pd.DataFrame:
    """One row per (lineage, feature): `lineage`, `feature`, `objects`, `rho`
    (Spearman correlation with pseudotime), `p`, `q` (Benjamini-Hochberg over
    every row), `start mean`, `end mean`, `start vs end (SD)` (end minus start
    over the feature's SD on the lineage). Start and end are the first and last
    `end_fraction` of the lineage's objects by pseudotime."""
    from scipy.stats import spearmanr

    t_all = np.atleast_2d(np.asarray(pseudotime, float))
    if t_all.shape[0] != len(values):
        t_all = t_all.T
    w_all = (
        np.ones_like(t_all) if weights is None else np.asarray(weights, float).reshape(t_all.shape)
    )
    names = _lineage_names(t_all.shape[1], lineages)
    x = values.to_numpy(float)
    rows = []
    for j, name in enumerate(names):
        on = np.isfinite(t_all[:, j]) & (w_all[:, j] > 0)
        for k, feature in enumerate(values.columns):
            ok = on & np.isfinite(x[:, k])
            t, v = t_all[ok, j], x[ok, k]
            n = int(ok.sum())
            rho = p = start = end = effect = np.nan
            if n >= 3 and np.ptp(t) > 0:
                start_cut, end_cut = np.quantile(t, [end_fraction, 1 - end_fraction])
                start, end = float(v[t <= start_cut].mean()), float(v[t >= end_cut].mean())
                sd = float(v.std())
                effect = (end - start) / sd if sd > 0 else 0.0
                if np.ptp(v) > 0:
                    rho, p = spearmanr(t, v)
                    rho, p = float(rho), float(p)
            rows.append((name, feature, n, rho, p, start, end, effect))
    out = pd.DataFrame(
        rows,
        columns=[
            "lineage",
            "feature",
            "objects",
            "rho",
            "p",
            "start mean",
            "end mean",
            "start vs end (SD)",
        ],
    )
    out.insert(5, "q", benjamini_hochberg(out["p"].to_numpy(float)))
    return out


def composition_along(
    pseudotime, classes, *, weights=None, n_bins: int = 10, lineages=None
) -> pd.DataFrame:
    """Long table, one row per (lineage, bin, class): `lineage`, `bin`,
    `pseudotime` (bin centre), `class`, `objects` and `fraction` (of the bin's
    objects with a class)."""
    t_all = np.atleast_2d(np.asarray(pseudotime, float))
    classes = np.asarray(classes, dtype=object)
    if t_all.shape[0] != len(classes):
        t_all = t_all.T
    w_all = (
        np.ones_like(t_all) if weights is None else np.asarray(weights, float).reshape(t_all.shape)
    )
    names = _lineage_names(t_all.shape[1], lineages)
    has = ~pd.isna(pd.Series(classes, dtype=object)).to_numpy()
    values = sorted(
        {c for c in classes[has]}, key=lambda v: (not isinstance(v, (int, np.integer)), str(v))
    )
    rows = []
    for j, name in enumerate(names):
        on = np.isfinite(t_all[:, j]) & (w_all[:, j] > 0) & has
        if not on.any():
            continue
        edges = pseudotime_bins(t_all[on, j], n_bins)
        which = _bin(t_all[on, j], edges)
        centres = (edges[:-1] + edges[1:]) / 2
        cls = classes[on]
        totals = np.bincount(which, minlength=n_bins)
        for value in values:
            counts = np.bincount(which[cls == value], minlength=n_bins)
            for b in range(n_bins):
                frac = counts[b] / totals[b] if totals[b] else np.nan
                rows.append((name, b, centres[b], value, int(counts[b]), frac))
    return pd.DataFrame(
        rows, columns=["lineage", "bin", "pseudotime", "class", "objects", "fraction"]
    )


def benjamini_hochberg(p) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaN p-values stay NaN and are not counted)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q
