"""Features along lineages: a port of tradeSeq (Van den Berge et al., Nat Commun 11:1201, 2020).

tradeSeq asks, for every feature (a gene there, a marker intensity or a
neighbourhood fraction here), *how* it changes along a trajectory, with a
generalised additive model (GAM) per feature and Wald tests on its fitted
curves.

**`fit_gam`** (tradeSeq's `fitGAM`):

- Each object is given to **one** lineage, drawn at random in proportion to its
  lineage weights (Slingshot's), as tradeSeq does. Objects on a shared trunk
  are split among the lineages.
- The model is `y ~ s(t1, by = l1, bs = "cr", id = 1, k = nknots) + ... +
  s(tL, by = lL, ...) [+ offset]`: one cubic regression spline of pseudotime
  per lineage (`splines.cr_basis`), all with one smoothing parameter
  (`id = 1`). The same `nknots` knots serve every lineage, at quantiles of the
  assigned pseudotimes, with each lineage's end moved onto a knot.
- `family="gaussian"` (identity link) for continuous measurements such as
  imaging intensities (asinh / z-scored), or `family="nb"` (log link, theta
  estimated) for counts, tradeSeq's default.
- The smoothing parameter is chosen by `method="GCV.Cp"` (mgcv's default: GCV
  for the Gaussian, UBRE for the negative binomial) or `"REML"`. For the NB
  family it is chosen by performance iteration on the working model (Gu
  1992), not mgcv's outer Newton iteration; theta is re-estimated by maximum
  likelihood between fits.
- The coefficients' Bayesian covariance `Vp` (mgcv's) is kept for the tests.

**Tests** (tradeSeq's, on the linear predictor; for the NB family differences
are log fold changes, for the Gaussian plain differences):

| function | tradeSeq | null hypothesis |
| --- | --- | --- |
| `association_test` | `associationTest` | the feature is constant along each lineage |
| `start_vs_end_test` | `startVsEndTest` | same value at a lineage's start and end |
| `diff_end_test` | `diffEndTest` | same value at the ends of the lineages |
| `pattern_test` | `patternTest` | same pattern along the lineages (on each lineage's own pseudotime scale) |
| `early_drivers_test` | `earlyDriversTest` | same pattern between two knots (around a branch point) |

Each returns one row per feature: `waldStat`, `df`, `pvalue`, `q`
(Benjamini-Hochberg over the features, which tradeSeq leaves to the user), and
per lineage / pair columns when asked. `smoothers` gives the fitted curves with
95% bands for plots (tradeSeq's `plotSmoothers` / `predictSmooth`).

`evaluate_k` compares numbers of knots by AIC (tradeSeq's `evaluateK`).

Not ported: conditions within lineages (`conditions=`), zero-inflation weights,
and fold-change thresholds (`l2fc`; the tests are of no change).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi2

from vtea_core.trajectory.along import benjamini_hochberg
from vtea_core.trajectory.splines import cr_basis, cr_penalty, quantile_knots


@dataclass
class GAMFit:
    """`fit_gam`'s result: one GAM per feature on a shared design."""

    features: list
    lineages: list
    family: str
    knots: np.ndarray
    coefficients: np.ndarray  # (features, L * k)
    vp: np.ndarray  # (features, L * k, L * k)
    lam: np.ndarray  # (features,)
    scale: np.ndarray  # (features,) residual variance (Gaussian), 1 (NB)
    theta: np.ndarray  # (features,) NB dispersion, NaN for the Gaussian
    edf: np.ndarray  # (features,)
    aic: np.ndarray  # (features,)
    converged: np.ndarray  # (features,)
    ranges: np.ndarray  # (L, 2) pseudotime range of each lineage's objects
    assignment: np.ndarray  # (objects,) lineage index, -1 = none
    time: np.ndarray  # (objects,) pseudotime on the assigned lineage

    @property
    def n_knots(self) -> int:
        return len(self.knots)

    def lpmatrix(self, t, lineage) -> np.ndarray:
        """Rows of the design (mgcv's `predict(type = "lpmatrix")`) at
        pseudotimes `t` on lineage `lineage` (index or name)."""
        j = (
            self.lineages.index(lineage)
            if not isinstance(lineage, (int, np.integer))
            else int(lineage)
        )
        t = np.atleast_1d(np.asarray(t, float))
        k = self.n_knots
        out = np.zeros((len(t), k * len(self.lineages)))
        out[:, j * k : (j + 1) * k] = cr_basis(t, self.knots)
        return out

    def predict(self, feature, t, lineage, *, response: bool = True) -> np.ndarray:
        i = self.features.index(feature)
        eta = self.lpmatrix(t, lineage) @ self.coefficients[i]
        return np.exp(eta) if (response and self.family == "nb") else eta


# ---- fitting --------------------------------------------------------------------------------


def assign_lineages(weights, seed=0) -> np.ndarray:
    """One lineage per object, drawn in proportion to its weights (tradeSeq);
    -1 for objects with no weight."""
    w = np.nan_to_num(np.asarray(weights, float), nan=0.0)
    w = np.clip(w, 0, None)
    total = w.sum(axis=1)
    out = np.full(len(w), -1, np.int64)
    ok = total > 0
    if ok.any():
        # A stream of its own: the same seed elsewhere (e.g. a simulation) must not
        # give the same numbers, or the draw would follow whatever those made.
        rng = np.random.default_rng([int(seed), 0x7EADE5E0])
        cum = np.cumsum(w[ok] / total[ok, None], axis=1)
        u = rng.random(int(ok.sum()))[:, None]
        out[ok] = np.minimum((u > cum).sum(axis=1), w.shape[1] - 1)
    return out


def gam_knots(time, assignment, n_lineages, nknots) -> np.ndarray:
    """tradeSeq's knots: at quantiles of the assigned pseudotimes, with the end of
    every lineage moved onto its nearest knot so each curve ends on one. An end
    within 1% of the range of a knot already counts as on it (two lineages that
    end together share their last knot)."""
    t = time[assignment >= 0]
    knots = quantile_knots(t, nknots)
    tol = 0.01 * (t.max() - t.min())
    claimed = {0}
    ends = sorted(
        float(time[assignment == j].max()) for j in range(n_lineages) if (assignment == j).any()
    )
    for end in reversed(ends):
        gaps = np.abs(knots - end)
        if gaps.min() <= tol:
            claimed.add(int(np.argmin(gaps)))
            continue
        free = [i for i in range(1, len(knots)) if i not in claimed]
        if not free:
            break
        near = min(free, key=lambda i: gaps[i])
        knots[near] = end
        claimed.add(near)
        knots = np.sort(knots)
    if len(np.unique(knots)) < len(knots):
        knots = np.linspace(t.min(), t.max(), nknots)
    return knots


def fit_gam(
    pseudotime,
    weights,
    values,
    *,
    nknots: int = 6,
    family: str = "gaussian",
    offset=None,
    method: str = "GCV.Cp",
    seed: int = 0,
    lineages=None,
    assignment=None,
) -> GAMFit:
    """GAMs of each column of `values` (objects x features, DataFrame or array)
    along the lineages. `pseudotime`, `weights`: (objects, lineages), e.g. a
    `Trajectory`'s. `offset`: (objects,) on the link scale (tradeSeq uses log
    library size for counts). `assignment`: give the objects' lineages instead
    of drawing them (e.g. to compare with R on the same draw)."""
    if family not in ("gaussian", "nb"):
        raise ValueError(f"family must be 'gaussian' or 'nb', not {family!r}")
    if method not in ("GCV.Cp", "REML"):
        raise ValueError(f"method must be 'GCV.Cp' or 'REML', not {method!r}")
    t_all = np.asarray(pseudotime, float)
    if t_all.ndim == 1:
        t_all = t_all[:, None]
    w_all = np.asarray(weights, float).reshape(t_all.shape)
    w_all = np.where(np.isfinite(t_all), w_all, 0.0)
    frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(np.asarray(values, float))
    if len(frame) != len(t_all):
        raise ValueError("values and pseudotime have different numbers of objects")
    n_lin = t_all.shape[1]
    names = list(lineages) if lineages is not None else [f"t{j + 1}" for j in range(n_lin)]
    if assignment is None:
        assignment = assign_lineages(w_all, seed)
    assignment = np.asarray(assignment, np.int64)
    time = np.full(len(t_all), np.nan)
    on = assignment >= 0
    time[on] = t_all[np.flatnonzero(on), assignment[on]]
    assignment = np.where(np.isfinite(time), assignment, -1)
    used = assignment >= 0
    if used.sum() < nknots + 2:
        raise ValueError("too few objects on the lineages for the GAM")
    knots = gam_knots(time, assignment, n_lin, nknots)
    k = len(knots)
    basis = cr_basis(time[used], knots)
    x = np.zeros((int(used.sum()), k * n_lin))
    lin_used = assignment[used]
    for j in range(n_lin):
        rows = lin_used == j
        x[rows, j * k : (j + 1) * k] = basis[rows]
    s_block = cr_penalty(knots)
    penalty = np.kron(np.eye(n_lin), s_block)
    # Lineages with no objects: their coefficients are fixed at 0 by a ridge.
    empty = [j for j in range(n_lin) if not (lin_used == j).any()]
    ridge = np.zeros(k * n_lin)
    for j in empty:
        ridge[j * k : (j + 1) * k] = 1.0
    off = np.zeros(int(used.sum())) if offset is None else np.asarray(offset, float)[used]
    ranges = np.array(
        [
            [time[assignment == j].min(), time[assignment == j].max()]
            if (assignment == j).any()
            else [np.nan, np.nan]
            for j in range(n_lin)
        ]
    )
    y_all = frame.to_numpy(float)[used]
    n_feat, p = y_all.shape[1], k * n_lin
    out = {
        "coefficients": np.zeros((n_feat, p)),
        "vp": np.zeros((n_feat, p, p)),
        "lam": np.zeros(n_feat),
        "scale": np.ones(n_feat),
        "theta": np.full(n_feat, np.nan),
        "edf": np.zeros(n_feat),
        "aic": np.zeros(n_feat),
        "converged": np.zeros(n_feat, bool),
    }
    for f in range(n_feat):
        ok = np.isfinite(y_all[:, f])
        xf, yf, of = x[ok], y_all[ok, f], off[ok]
        if family == "gaussian":
            res = _fit_gaussian(xf, yf - of, penalty, ridge, method)
        else:
            res = _fit_nb(xf, yf, of, penalty, ridge, method)
        for key, column in out.items():
            column[f] = res[key]
    return GAMFit(
        features=[str(c) for c in frame.columns],
        lineages=names,
        family=family,
        knots=knots,
        ranges=ranges,
        assignment=assignment,
        time=time,
        **out,
    )


def _penalty_scale(xtwx, penalty):
    return np.trace(xtwx) / max(np.trace(penalty), 1e-300)


def _solve(xtwx, xtwz, penalty, ridge, lam):
    r = xtwx + lam * penalty + np.diag(ridge)
    try:
        c = np.linalg.cholesky(r)
        beta = np.linalg.solve(c.T, np.linalg.solve(c, xtwz))
        logdet = 2 * np.log(np.diag(c)).sum()
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(r, xtwz, rcond=None)[0]
        logdet = np.linalg.slogdet(r)[1]
    return r, beta, logdet


def _choose_lambda(score):
    """Minimise score(log lambda): a coarse grid, then a bounded refinement."""
    grid = np.linspace(-12, 20, 33)
    values = [score(g) for g in grid]
    i = int(np.nanargmin(values))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    if hi <= lo:
        return grid[i]
    res = minimize_scalar(score, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
    return res.x if res.fun <= values[i] else grid[i]


def _working_criterion(xtwx, xtwz, ztwz, n, penalty, ridge, method, scale_known, rank_s, null):
    """score(log lambda) of the (working) linear model, and the fit at lambda."""
    base = _penalty_scale(xtwx, penalty)

    def fit(rho):
        lam = np.exp(rho) * base
        r, beta, logdet = _solve(xtwx, xtwz, penalty, ridge, lam)
        rss = max(ztwz - 2 * beta @ xtwz + beta @ xtwx @ beta, 0.0)
        pen = lam * beta @ penalty @ beta
        edf = float(np.trace(np.linalg.solve(r, xtwx)))
        return lam, r, beta, logdet, rss, pen, edf

    def score(rho):
        lam, _, _, logdet, rss, pen, edf = fit(rho)
        if method == "REML":
            if scale_known:
                return 0.5 * (rss + pen) - 0.5 * rank_s * np.log(lam) + 0.5 * logdet
            d = max(rss + pen, 1e-300)
            return 0.5 * (n - null) * np.log(d) - 0.5 * rank_s * np.log(lam) + 0.5 * logdet
        if scale_known:  # UBRE (Mallows' Cp) with scale 1
            return rss / n - 1.0 + 2.0 * edf / n
        return n * rss / max(n - edf, 1e-6) ** 2  # GCV

    return score, fit


def _fit_gaussian(x, y, penalty, ridge, method):
    n, p = x.shape
    xtwx, xtwz, ztwz = x.T @ x, x.T @ y, float(y @ y)
    rank_s = np.linalg.matrix_rank(penalty)
    null = p - rank_s - int((ridge > 0).sum())
    score, fit = _working_criterion(
        xtwx, xtwz, ztwz, n, penalty, ridge, method, False, rank_s, null
    )
    rho = _choose_lambda(score)
    lam, r, beta, _, rss, pen, edf = fit(rho)
    scale = (rss + pen) / max(n - null, 1) if method == "REML" else rss / max(n - edf, 1e-6)
    rinv = np.linalg.inv(r)
    loglik = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
    return {
        "coefficients": beta,
        "vp": scale * rinv,
        "lam": lam,
        "scale": scale,
        "theta": np.nan,
        "edf": edf,
        "aic": -2 * loglik + 2 * (edf + 1),
        "converged": True,
    }


def _nb_loglik(y, mu, theta):
    return float(
        np.sum(
            gammaln(y + theta)
            - gammaln(theta)
            - gammaln(y + 1)
            + theta * np.log(theta / (theta + mu))
            + y * np.log(np.maximum(mu, 1e-300) / (theta + mu))
        )
    )


def _theta_ml(y, mu):
    res = minimize_scalar(
        lambda lt: -_nb_loglik(y, mu, np.exp(lt)), bounds=(-8, 12), method="bounded"
    )
    return float(np.exp(res.x))


def _fit_nb(x, y, offset, penalty, ridge, method, max_outer=10, max_inner=50):
    if np.any(y < 0):
        raise ValueError("the negative binomial family needs counts (no negative values)")
    n, p = x.shape
    rank_s = np.linalg.matrix_rank(penalty)
    null = p - rank_s - int((ridge > 0).sum())
    mu = np.maximum(y, 0.1)
    mu = np.full(n, max(y.mean(), 0.1)) * 0.5 + mu * 0.5
    eta = np.log(mu) - offset
    var = np.var(y)
    theta = max(y.mean() ** 2 / max(var - y.mean(), 1e-6), 0.1) if var > y.mean() else 100.0
    converged = False
    beta = np.zeros(p)
    lam, r = 1.0, np.eye(p)
    for _ in range(max_outer):
        for _ in range(max_inner):
            mu = np.exp(np.clip(eta + offset, -30, 30))
            w = mu**2 / (mu + mu**2 / theta)
            z = eta + (y - mu) / mu
            xw = x * w[:, None]
            xtwx, xtwz, ztwz = x.T @ xw, xw.T @ z, float(z @ (w * z))
            score, fit = _working_criterion(
                xtwx, xtwz, ztwz, n, penalty, ridge, method, True, rank_s, null
            )
            lam, r, beta_new, _, _, _, edf = fit(_choose_lambda(score))
            eta_new = x @ beta_new
            change = np.max(np.abs(eta_new - eta)) / (np.max(np.abs(eta)) + 1e-8)
            beta, eta = beta_new, eta_new
            if change < 1e-6:
                break
        mu = np.exp(np.clip(eta + offset, -30, 30))
        new_theta = _theta_ml(y, mu)
        if abs(np.log(new_theta) - np.log(theta)) < 1e-4:
            theta = new_theta
            converged = True
            break
        theta = new_theta
    mu = np.exp(np.clip(eta + offset, -30, 30))
    w = mu**2 / (mu + mu**2 / theta)
    xtwx = x.T @ (x * w[:, None])
    r = xtwx + lam * penalty + np.diag(ridge)
    rinv = np.linalg.inv(r)
    edf = float(np.trace(rinv @ xtwx))
    return {
        "coefficients": beta,
        "vp": rinv,
        "lam": lam,
        "scale": 1.0,
        "theta": theta,
        "edf": edf,
        "aic": -2 * _nb_loglik(y, mu, theta) + 2 * (edf + 1),
        "converged": converged,
    }


def evaluate_k(
    pseudotime, weights, values, *, knots=range(3, 11), family="gaussian", seed=0, **kwargs
) -> pd.DataFrame:
    """tradeSeq's `evaluateK`: fit with each number of knots and compare AIC.
    One row per feature, one column per k (AIC minus the feature's AIC at the
    smallest k), plus `best k`. Too few knots smooth over short-lived changes
    (and can then bend a curve's ends, which start-vs-end and diffEnd read);
    choose the k past which the AIC stops falling for most features."""
    assignment = assign_lineages(
        np.where(np.isfinite(np.asarray(pseudotime, float)), np.asarray(weights, float), 0.0), seed
    )
    columns = {}
    for k in knots:
        fit = fit_gam(
            pseudotime,
            weights,
            values,
            nknots=int(k),
            family=family,
            assignment=assignment,
            **kwargs,
        )
        columns[int(k)] = fit.aic
        features = fit.features
    table = pd.DataFrame(columns, index=features)
    best = table.idxmin(axis=1)
    table = table.sub(table.iloc[:, 0], axis=0)
    table["best k"] = best
    return table


# ---- tests ------------------------------------------------------------------------------------


def wald(est, sigma, eigen_thresh=None):
    """(statistic, df, p) of the Wald test est = 0 with covariance sigma, through
    an eigen pseudo-inverse: eigenvalues below `eigen_thresh` times the largest
    are dropped (default sqrt(eps), as MASS::ginv)."""
    sigma = (np.asarray(sigma, float) + np.asarray(sigma, float).T) / 2
    vals, vecs = np.linalg.eigh(sigma)
    top = vals.max(initial=0.0)
    if top <= 0:
        return np.nan, 0, np.nan
    tol = (eigen_thresh if eigen_thresh is not None else np.sqrt(np.finfo(float).eps)) * top
    keep = vals > tol
    proj = vecs[:, keep].T @ np.asarray(est, float)
    stat = float(np.sum(proj**2 / vals[keep]))
    df = int(keep.sum())
    return stat, df, float(chi2.sf(stat, df))


def _test_table(fit, contrasts, eigen_thresh=None, effects=None):
    """contrasts: {name: (c, p) matrix} -> one row per feature; name "" is the
    global test."""
    rows = []
    for i, feature in enumerate(fit.features):
        beta, vp = fit.coefficients[i], fit.vp[i]
        row = {"feature": feature}
        for name, lmat in contrasts.items():
            est = lmat @ beta
            stat, df, p = wald(est, lmat @ vp @ lmat.T, eigen_thresh)
            suffix = f"_{name}" if name else ""
            row[f"waldStat{suffix}"], row[f"df{suffix}"], row[f"pvalue{suffix}"] = stat, df, p
        for name, lmat in (effects or {}).items():
            row[name] = (
                float(np.mean(lmat @ beta)) if lmat.shape[0] > 1 else float((lmat @ beta)[0])
            )
        rows.append(row)
    out = pd.DataFrame(rows).set_index("feature")
    if "pvalue" in out:
        out.insert(
            out.columns.get_loc("pvalue") + 1, "q", benjamini_hochberg(out["pvalue"].to_numpy())
        )
    return out


def _points(fit, j, n_points):
    lo, hi = fit.ranges[j]
    return np.linspace(lo, hi, n_points)


def association_test(
    fit: GAMFit, *, global_test=True, lineages=False, n_points=None, contrast="start"
):
    """Is the feature constant along each lineage? Contrasts between `n_points`
    (default 2 x nknots) equally spaced pseudotimes of each lineage: each against
    the start ("start"), the end ("end") or the next ("consecutive")."""
    n_points = n_points or 2 * fit.n_knots
    per = {}
    for j, name in enumerate(fit.lineages):
        if not np.isfinite(fit.ranges[j]).all():
            continue
        lp = fit.lpmatrix(_points(fit, j, n_points), j)
        if contrast == "start":
            per[name] = lp[1:] - lp[0]
        elif contrast == "end":
            per[name] = lp[:-1] - lp[-1]
        elif contrast == "consecutive":
            per[name] = lp[1:] - lp[:-1]
        else:
            raise ValueError(f"unknown contrast {contrast!r}")
    contrasts = {}
    if global_test:
        contrasts[""] = np.vstack(list(per.values()))
    if lineages:
        contrasts.update(per)
    out = _test_table(fit, contrasts)
    if per:  # tradeSeq's meanLogFC: the mean absolute contrast
        stacked = np.vstack(list(per.values()))
        out["meanEffect"] = [float(np.mean(np.abs(stacked @ b))) for b in fit.coefficients]
    return out


def start_vs_end_test(fit: GAMFit, *, global_test=True, lineages=False, pseudotime_values=None):
    """Same value at the start and end of each lineage? `pseudotime_values`:
    (start, end) to compare instead of each lineage's range."""
    per, effects = {}, {}
    for j, name in enumerate(fit.lineages):
        if not np.isfinite(fit.ranges[j]).all():
            continue
        a, b = pseudotime_values if pseudotime_values is not None else fit.ranges[j]
        lmat = fit.lpmatrix([b], j) - fit.lpmatrix([a], j)
        per[name] = lmat
        effects[f"effect_{name}"] = lmat
    contrasts = {"": np.vstack(list(per.values()))} if global_test else {}
    if lineages:
        contrasts.update(per)
    return _test_table(fit, contrasts, effects=effects)


def _pairs(fit):
    ok = [j for j in range(len(fit.lineages)) if np.isfinite(fit.ranges[j]).all()]
    return [(a, b) for i, a in enumerate(ok) for b in ok[i + 1 :]]


def diff_end_test(fit: GAMFit, *, global_test=True, pairwise=False):
    """Same value at the ends of the lineages?"""
    per, effects = {}, {}
    for a, b in _pairs(fit):
        lmat = fit.lpmatrix([fit.ranges[a][1]], a) - fit.lpmatrix([fit.ranges[b][1]], b)
        name = f"{fit.lineages[a]}vs{fit.lineages[b]}"
        per[name] = lmat
        effects[f"effect_{name}"] = lmat
    if not per:
        raise ValueError("diff_end_test needs at least two lineages")
    contrasts = {"": np.vstack(list(per.values()))} if global_test else {}
    if pairwise:
        contrasts.update(per)
    return _test_table(fit, contrasts, effects=effects)


def pattern_test(
    fit: GAMFit, *, global_test=True, pairwise=False, n_points=None, eigen_thresh=1e-2
):
    """Same pattern along the lineages? Compares the lineages at `n_points`
    (default 2 x nknots) equally spaced points of each one's own pseudotime
    range, so lineages of different lengths are compared start to start and
    end to end."""
    n_points = n_points or 2 * fit.n_knots
    per = {}
    for a, b in _pairs(fit):
        per[f"{fit.lineages[a]}vs{fit.lineages[b]}"] = fit.lpmatrix(
            _points(fit, a, n_points), a
        ) - fit.lpmatrix(_points(fit, b, n_points), b)
    if not per:
        raise ValueError("pattern_test needs at least two lineages")
    contrasts = {"": np.vstack(list(per.values()))} if global_test else {}
    if pairwise:
        contrasts.update(per)
    return _test_table(fit, contrasts, eigen_thresh=eigen_thresh)


def early_drivers_test(
    fit: GAMFit, knots=(0, 2), *, global_test=True, pairwise=False, n_points=None, eigen_thresh=1e-2
):
    """Same pattern between knots `knots` (indices into `fit.knots`, e.g. on
    either side of a branch point), on the same pseudotimes for every lineage."""
    n_points = n_points or 2 * fit.n_knots
    lo, hi = fit.knots[knots[0]], fit.knots[knots[1]]
    t = np.linspace(lo, hi, n_points)
    per = {}
    for a, b in _pairs(fit):
        per[f"{fit.lineages[a]}vs{fit.lineages[b]}"] = fit.lpmatrix(t, a) - fit.lpmatrix(t, b)
    if not per:
        raise ValueError("early_drivers_test needs at least two lineages")
    contrasts = {"": np.vstack(list(per.values()))} if global_test else {}
    if pairwise:
        contrasts.update(per)
    return _test_table(fit, contrasts, eigen_thresh=eigen_thresh)


def smoothers(fit: GAMFit, feature, n_points: int = 100) -> pd.DataFrame:
    """The fitted curve of `feature` on each lineage, with a 95% band:
    `lineage`, `pseudotime`, `fit`, `lower`, `upper` (response scale)."""
    i = fit.features.index(feature)
    rows = []
    for j, name in enumerate(fit.lineages):
        if not np.isfinite(fit.ranges[j]).all():
            continue
        t = _points(fit, j, n_points)
        lp = fit.lpmatrix(t, j)
        eta = lp @ fit.coefficients[i]
        se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", lp, fit.vp[i], lp), 0))
        lo, hi = eta - 1.96 * se, eta + 1.96 * se
        if fit.family == "nb":
            eta, lo, hi = np.exp(eta), np.exp(lo), np.exp(hi)
        rows.append(
            pd.DataFrame({"lineage": name, "pseudotime": t, "fit": eta, "lower": lo, "upper": hi})
        )
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["lineage", "pseudotime", "fit", "lower", "upper"])
    )
