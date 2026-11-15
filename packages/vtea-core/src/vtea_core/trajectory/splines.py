"""Penalized cubic regression splines: what Slingshot's smoother and tradeSeq's GAM share.

- `cr_basis` / `cr_penalty`: the cubic regression spline of mgcv (`bs = "cr"`;
  Wood, *Generalized Additive Models*, 2nd ed., 2017, section 5.3.1). A natural
  cubic spline parametrised by its values at the knots: the basis rows sum to 1,
  the penalty is the integrated squared second derivative, and outside the knots
  the spline continues as a straight line. tradeSeq's `fitGAM` uses this basis,
  one copy per lineage (`s(t_l, by = l_l, bs = "cr", id = 1, k = nknots)`).
- `smoothing_spline`: a weighted penalized regression spline whose smoothing is
  set by its **effective degrees of freedom**, as Slingshot calls
  `smooth.spline(lambda, x, w = w, df = 5)`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def _second_derivative_map(knots):
    """(F, S): F maps the values at the knots to the second derivatives there
    (zero at both ends: a natural spline); S = D' B^-1 D is the penalty."""
    x = np.asarray(knots, float)
    k = len(x)
    h = np.diff(x)
    if k < 3:
        return np.zeros((k, k)), np.zeros((k, k))
    d = np.zeros((k - 2, k))
    b = np.zeros((k - 2, k - 2))
    for i in range(k - 2):
        d[i, i] = 1.0 / h[i]
        d[i, i + 1] = -1.0 / h[i] - 1.0 / h[i + 1]
        d[i, i + 2] = 1.0 / h[i + 1]
        b[i, i] = (h[i] + h[i + 1]) / 3.0
        if i < k - 3:
            b[i, i + 1] = b[i + 1, i] = h[i + 1] / 6.0
    binv_d = np.linalg.solve(b, d)
    f = np.vstack([np.zeros(k), binv_d, np.zeros(k)])
    return f, d.T @ binv_d


def cr_penalty(knots) -> np.ndarray:
    """(k, k) penalty of the cubic regression spline: beta' S beta = integral of
    f''(t)^2. Rank k - 2 (constants and straight lines are not penalised)."""
    return _second_derivative_map(knots)[1]


def cr_basis(t, knots) -> np.ndarray:
    """(len(t), k) cubic regression spline basis at `t` for `knots` (sorted,
    distinct). Linear beyond the end knots, as in mgcv."""
    x = np.asarray(knots, float)
    t = np.asarray(t, float)
    k = len(x)
    f, _ = _second_derivative_map(x)
    out = np.zeros((len(t), k))
    lo, hi = x[0], x[-1]
    inside = (t >= lo) & (t <= hi)
    if inside.any():
        ti = t[inside]
        j = np.clip(np.searchsorted(x, ti, side="right") - 1, 0, k - 2)
        h = x[j + 1] - x[j]
        am = (x[j + 1] - ti) / h
        ap = (ti - x[j]) / h
        cm = ((x[j + 1] - ti) ** 3 / h - h * (x[j + 1] - ti)) / 6.0
        cp = ((ti - x[j]) ** 3 / h - h * (ti - x[j])) / 6.0
        rows = np.flatnonzero(inside)
        block = cm[:, None] * f[j] + cp[:, None] * f[j + 1]
        block[np.arange(len(ti)), j] += am
        block[np.arange(len(ti)), j + 1] += ap
        out[rows] = block
    for side, mask in (("lo", t < lo), ("hi", t > hi)):
        if not mask.any():
            continue
        # f(t) = f(end) + f'(end) (t - end): value and slope at the end knot.
        end = lo if side == "lo" else hi
        value = np.zeros(k)
        value[0 if side == "lo" else k - 1] = 1.0
        h0 = x[1] - x[0] if side == "lo" else x[-1] - x[-2]
        if side == "lo":  # f'(x0) = (b1 - b0)/h0 - h0/6 (2 d0 + d1)
            slope = -np.eye(k)[0] / h0 + np.eye(k)[1] / h0 - h0 / 6.0 * (2 * f[0] + f[1])
        else:  # f'(xk) = (bk - bk-1)/h + h/6 (dk-1 + 2 dk)
            slope = (
                np.eye(k)[k - 1] / h0 - np.eye(k)[k - 2] / h0 + h0 / 6.0 * (f[k - 2] + 2 * f[k - 1])
            )
        out[mask] = value[None, :] + (t[mask] - end)[:, None] * slope[None, :]
    return out


def quantile_knots(t, k: int) -> np.ndarray:
    """k knots at quantiles of `t`, evenly spaced instead if quantiles repeat."""
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    knots = np.quantile(t, np.linspace(0, 1, k))
    if len(np.unique(knots)) < k:
        knots = np.linspace(t.min(), t.max(), k)
    return knots


def edf(xtwx, penalty, lam) -> float:
    """Effective degrees of freedom tr((X'WX + lam S)^-1 X'WX)."""
    r = xtwx + lam * penalty
    return float(np.trace(np.linalg.solve(r, xtwx)))


def lambda_for_df(xtwx, penalty, df) -> float:
    """The smoothing parameter giving `df` effective degrees of freedom (between
    the penalty's null-space dimension and the number of coefficients)."""
    p = xtwx.shape[0]
    null = p - np.linalg.matrix_rank(penalty)
    df = float(np.clip(df, null + 1e-6, p - 1e-6))
    scale = np.trace(xtwx) / max(np.trace(penalty), 1e-300)

    def gap(log_lam):
        return edf(xtwx, penalty, np.exp(log_lam) * scale) - df

    lo, hi = -30.0, 30.0
    if gap(lo) < 0:
        return float(np.exp(lo) * scale)
    if gap(hi) > 0:
        return float(np.exp(hi) * scale)
    return float(np.exp(brentq(gap, lo, hi, xtol=1e-6)) * scale)


def smoothing_spline(t, y, weights=None, *, df: float = 5.0, n_knots: int = 50):
    """A function: the weighted penalized cubic spline of y (n, or n x d) on t
    with `df` effective degrees of freedom (one smoothing parameter for every
    column of y, from the weights and t only). Knots at quantiles of t, at most
    `n_knots` (R's smooth.spline also uses a reduced knot set for large n)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, float)
    k = int(min(n_knots, len(np.unique(t))))
    if k < 4:
        a = np.vstack([np.ones_like(t), t]).T
        coef = np.linalg.lstsq(a * np.sqrt(w)[:, None], (y.T * np.sqrt(w)).T, rcond=None)[0]
        return lambda s: np.vstack([np.ones_like(s), s]).T @ coef
    knots = quantile_knots(t, k)
    x = cr_basis(t, knots)
    s = cr_penalty(knots)
    xtwx = x.T @ (x * w[:, None])
    lam = lambda_for_df(xtwx, s, df)
    coef = np.linalg.solve(xtwx + lam * s, x.T @ (w[:, None] * (y if y.ndim > 1 else y[:, None])))
    if y.ndim == 1:
        coef = coef[:, 0]
    return lambda s_: cr_basis(np.asarray(s_, float), knots) @ coef
