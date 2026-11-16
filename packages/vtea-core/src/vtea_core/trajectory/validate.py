"""Validation of the trajectory engine: against a known truth, and against R.

Two kinds of evidence, both reproducible:

1. **Simulation** (`benchmark`): datasets from `simulate.simulate_trajectory`
   whose topology, pseudotime and feature behaviour are known. For Slingshot:
   is the topology found (`topology_correct`), how well does pseudotime follow
   the true time (`pseudotime_spearman`, worst lineage), and are objects past a
   branch point put on their own lineage (`branch_accuracy`)? For each tradeSeq
   test: the type I error on null features at alpha (should be <= alpha), the
   power on features the test should find (at BH q < alpha), and the false
   discovery rate among the discoveries (should be <= alpha).
2. **Parity with R** (`compare_with_r`): the same data through R slingshot /
   tradeSeq (`r_engine`) and through the ports. Lineages identical,
   pseudotime correlation, and the agreement of Wald statistics and p-values
   (on the same lineage assignment, so only the implementations differ).
   Needs R with the packages; `tests/test_trajectory_r_parity.py` runs it
   when they are installed, and the `trajectory-r-parity` CI workflow installs
   them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vtea_core.trajectory.simulate import simulate_trajectory
from vtea_core.trajectory.slingshot import Trajectory, slingshot
from vtea_core.trajectory.tradeseq import (
    assign_lineages,
    association_test,
    diff_end_test,
    fit_gam,
    pattern_test,
    start_vs_end_test,
)

TESTS = {
    "association": association_test,
    "start_vs_end": start_vs_end_test,
    "diff_end": diff_end_test,
    "pattern": pattern_test,
}


def _match(found, truth):
    """{found lineage index: true lineage index} by identical cluster paths."""
    index = {tuple(lin): j for j, lin in enumerate(truth)}
    return {i: index.get(tuple(lin)) for i, lin in enumerate(found)}


def validate_lineages(sim: dict, traj: Trajectory) -> dict:
    """Topology, pseudotime and branch assignment of `traj` against `sim`."""
    match = _match(traj.lineages, sim["lineages"])
    correct = sorted(map(tuple, traj.lineages)) == sorted(map(tuple, sim["lineages"]))
    rhos = []
    for j in range(traj.n_lineages):
        on = np.isfinite(traj.pseudotime[:, j])
        if on.sum() > 2:
            rhos.append(spearmanr(traj.pseudotime[on, j], sim["time"][on])[0])
    membership = sim["membership"]
    own = membership.sum(axis=1) == 1  # past every branch point: on one true lineage
    accuracy = np.nan
    if correct and own.any() and traj.n_lineages > 1:
        main = traj.main_lineage() - 1
        mapped = np.array([match.get(int(m)) if m >= 0 else -1 for m in main], dtype=object)
        truth = membership[own].argmax(axis=1)
        accuracy = float(np.mean([m == t for m, t in zip(mapped[own], truth)]))
    return {
        "topology_correct": bool(correct),
        "n_lineages": traj.n_lineages,
        "pseudotime_spearman": float(min(rhos)) if rhos else np.nan,
        "branch_accuracy": accuracy,
    }


def validate_tests(sim: dict, fit, alpha: float = 0.05) -> pd.DataFrame:
    """One row per tradeSeq test: `type_I_error` (null features with p < alpha),
    `power` (features the test should find, q < alpha), `fdr` (discoveries at q
    < alpha that the test should not have made), and their counts."""
    truth = sim["truth"]
    rows = []
    n_lin = len(fit.lineages)
    for name, test in TESTS.items():
        if name in ("diff_end", "pattern") and n_lin < 2:
            continue
        res = test(fit).reindex(truth.index)
        expected = truth[name].astype(object)
        null = truth["kind"] == "null"
        found = res["q"] < alpha
        positives = expected == True
        judged = expected.notna()
        n_found = int((found & judged).sum())
        rows.append(
            {
                "test": name,
                "type_I_error": float((res["pvalue"][null] < alpha).mean()),
                "power": float(found[positives].mean()) if positives.any() else np.nan,
                "fdr": float((found & judged & ~positives).sum() / n_found) if n_found else 0.0,
                "n_null": int(null.sum()),
                "n_positive": int(positives.sum()),
            }
        )
    return pd.DataFrame(rows)


def run_one(
    topology="bifurcation",
    seed=0,
    *,
    distance="slingshot",
    family="gaussian",
    nknots=6,
    alpha=0.05,
    use_true_time=False,
    **sim_kwargs,
) -> tuple[dict, pd.DataFrame]:
    """Simulate, find the trajectory, fit the GAMs, and score both. With
    `use_true_time`, the GAMs use the true pseudotime and lineages (tests the
    statistics alone, not Slingshot)."""
    sim = simulate_trajectory(topology, seed=seed, family=family, **sim_kwargs)
    traj = slingshot(sim["points"], sim["labels"], sim["start"], distance=distance)
    lineage_scores = validate_lineages(sim, traj)
    if use_true_time:
        pt = np.where(sim["membership"], sim["time"][:, None], np.nan)
        weights = sim["membership"].astype(float)
    else:
        pt, weights = traj.pseudotime, traj.weights
        # Lineages in the simulation's order, so the truth's lineage indices apply.
        match = _match(traj.lineages, sim["lineages"])
        if all(v is not None for v in match.values()):
            order = np.argsort([match[i] for i in range(traj.n_lineages)])
            pt, weights = pt[:, order], weights[:, order]
    fit = fit_gam(pt, weights, sim["values"], nknots=nknots, family=family, seed=seed)
    tests = validate_tests(sim, fit, alpha)
    return lineage_scores, tests


def benchmark(
    topologies=("linear", "bifurcation", "trifurcation", "tree"),
    seeds=range(5),
    *,
    distances=("slingshot", "euclidean"),
    families=("gaussian",),
    **kwargs,
) -> dict:
    """Every topology x seed x distance x family. Returns {"lineages": one row
    per run, "tests": one row per run and test, "summary_lineages",
    "summary_tests"} (means over seeds)."""
    lin_rows, test_rows = [], []
    for topology in topologies:
        for family in families:
            for distance in distances:
                for seed in seeds:
                    scores, tests = run_one(
                        topology, seed, distance=distance, family=family, **kwargs
                    )
                    key = {
                        "topology": topology,
                        "family": family,
                        "distance": distance,
                        "seed": seed,
                    }
                    lin_rows.append({**key, **scores})
                    test_rows.append(tests.assign(**key))
    lineages = pd.DataFrame(lin_rows)
    tests = pd.concat(test_rows, ignore_index=True)
    return {
        "lineages": lineages,
        "tests": tests,
        "summary_lineages": lineages.groupby(["topology", "distance", "family"], sort=False)[
            ["topology_correct", "pseudotime_spearman", "branch_accuracy"]
        ].mean(),
        "summary_tests": tests.groupby(["family", "test"], sort=False)[
            ["type_I_error", "power", "fdr"]
        ].mean(),
    }


def compare_with_r(sim: dict, *, distance="slingshot", nknots=6, family="gaussian", seed=0) -> dict:
    """Parity of the ports with R slingshot and tradeSeq on one dataset: lineage
    sets, pseudotime correlation per lineage, and for each tradeSeq test the
    Spearman correlation of the Wald statistics, the largest difference of
    -log10 p, and the agreement of calls at p < 0.05. The GAMs get the same
    one-hot lineage assignment in both, drawn here from R's pseudotime and
    weights."""
    from vtea_core.trajectory import r_engine

    py = slingshot(sim["points"], sim["labels"], sim["start"], distance=distance)
    r = r_engine.slingshot_r(
        sim["points"], sim["labels"], sim["start"], distance=distance, seed=seed
    )
    out = {"lineages_equal": sorted(map(tuple, py.lineages)) == sorted(map(tuple, r.lineages))}
    if out["lineages_equal"]:
        order = [r.lineages.index(lin) for lin in py.lineages]
        rho = []
        for j, k in enumerate(order):
            on = np.isfinite(py.pseudotime[:, j]) & np.isfinite(r.pseudotime[:, k])
            rho.append(spearmanr(py.pseudotime[on, j], r.pseudotime[on, k])[0])
        out["pseudotime_spearman"] = float(min(rho))
        out["weights_agreement"] = float(np.mean((py.weights > 0) == (r.weights[:, order] > 0)))
    # tradeSeq on R's trajectory, one lineage per object in both.
    assignment = assign_lineages(r.weights, seed)
    onehot = np.zeros_like(r.weights)
    on = assignment >= 0
    onehot[np.flatnonzero(on), assignment[on]] = 1.0
    pt = np.where(onehot > 0, np.nan_to_num(r.pseudotime), np.nan)
    fit = fit_gam(pt, onehot, sim["values"], nknots=nknots, family=family, assignment=assignment)
    rt = r_engine.tradeseq_r(pt, onehot, sim["values"], nknots=nknots, family=family, seed=seed)
    for name, test in TESTS.items():
        if name not in rt:
            continue
        a = test(fit)
        b = rt[name].reindex(a.index)
        ok = a["pvalue"].notna() & b["pvalue"].notna()
        la = -np.log10(np.clip(a["pvalue"][ok], 1e-300, 1))
        lb = -np.log10(np.clip(b["pvalue"][ok], 1e-300, 1))
        out[f"{name}_wald_spearman"] = float(spearmanr(a["waldStat"][ok], b["waldStat"][ok])[0])
        out[f"{name}_max_log10p_diff"] = float(np.max(np.abs(la - lb))) if ok.any() else np.nan
        out[f"{name}_call_agreement"] = float(
            np.mean((a["pvalue"][ok] < 0.05) == (b["pvalue"][ok] < 0.05))
        )
    return out
