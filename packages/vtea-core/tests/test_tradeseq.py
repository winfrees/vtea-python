"""tradeSeq port: splines, GAM fits, Wald tests, and their calibration on simulations."""

import numpy as np
import pandas as pd
import pytest

from vtea_core.trajectory.simulate import simulate_trajectory
from vtea_core.trajectory.splines import cr_basis, cr_penalty, lambda_for_df, smoothing_spline
from vtea_core.trajectory.tradeseq import (
    assign_lineages,
    association_test,
    diff_end_test,
    early_drivers_test,
    fit_gam,
    gam_knots,
    pattern_test,
    smoothers,
    start_vs_end_test,
    wald,
)
from vtea_core.trajectory.validate import benchmark, run_one, validate_lineages

# ---- splines --------------------------------------------------------------------------------


def test_cr_basis_is_a_natural_cubic_spline():
    knots = np.array([0, 1, 2.5, 3, 5.0])
    t = np.linspace(-1, 6, 7001)
    x = cr_basis(t, knots)
    assert np.allclose(x.sum(axis=1), 1)  # constants are reproduced
    assert np.allclose(x @ (2 * knots + 1), 2 * t + 1)  # so are straight lines (also outside)
    b = np.random.default_rng(0).normal(size=5)
    f = x @ b
    d2 = np.gradient(np.gradient(f, t), t)
    inside = (t >= 0) & (t <= 5)
    # The penalty is the integrated squared second derivative.
    assert b @ cr_penalty(knots) @ b == pytest.approx(
        np.trapezoid(d2[inside] ** 2, t[inside]), rel=1e-3
    )
    assert np.allclose(
        x[np.searchsorted(t, 2.5)], np.eye(5)[2], atol=1e-3
    )  # value at a knot = its coefficient


def test_lambda_for_df_and_smoothing_spline():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 10, 400))
    knots = np.quantile(t, np.linspace(0, 1, 20))
    x = cr_basis(t, knots)
    xtx, s = x.T @ x, cr_penalty(knots)
    lam = lambda_for_df(xtx, s, 5.0)
    assert np.trace(np.linalg.solve(xtx + lam * s, xtx)) == pytest.approx(5.0, abs=1e-3)
    y = np.sin(t / 2) + rng.normal(0, 0.2, len(t))
    smooth = smoothing_spline(t, y, df=5)
    assert np.sqrt(np.mean((smooth(t) - np.sin(t / 2)) ** 2)) < 0.08


# ---- fitting --------------------------------------------------------------------------------


def two_lineages(n=1500, seed=0):
    """A trunk (t < 3) then two branches to t = 6; true time and membership."""
    rng = np.random.default_rng(seed)
    seg = rng.integers(0, 3, n)
    time = np.where(seg == 0, 3, 3) * rng.uniform(0, 1, n) + np.where(seg == 0, 0, 3)
    member = np.column_stack([(seg == 0) | (seg == 1), (seg == 0) | (seg == 2)])
    pt = np.where(member, time[:, None], np.nan)
    return rng, seg, time, pt, member.astype(float)


def test_assignment_draws_in_proportion_to_weights():
    w = np.tile([[3.0, 1.0]], (20000, 1))
    a = assign_lineages(w, seed=1)
    assert (a == 0).mean() == pytest.approx(0.75, abs=0.02)
    assert (assign_lineages(np.zeros((3, 2))) == -1).all()
    # Its own random stream: not the same numbers as a generator with the same seed.
    u = np.random.default_rng(1).random(20000)
    assert abs(np.corrcoef(u, a)[0, 1]) < 0.05


def test_knots_end_every_lineage():
    _, seg, time, pt, w = two_lineages()
    a = assign_lineages(w, 0)
    t = np.where(a >= 0, pt[np.arange(len(pt)), np.maximum(a, 0)], np.nan)
    knots = gam_knots(t, a, 2, 6)
    assert len(knots) == 6 and np.all(np.diff(knots) > 0)
    for j in range(2):  # on a knot, or within 1% of the range of one
        assert np.abs(knots - t[a == j].max()).min() <= 0.01 * (t.max() - t.min()) + 1e-12


def test_gaussian_fit_recovers_the_curves():
    rng, seg, time, pt, w = two_lineages()
    truth = np.where(seg == 1, np.sin(time), np.where(seg == 2, -0.5 * (time - 3), 0.0))
    y = truth + rng.normal(0, 0.3, len(time))
    for method in ("GCV.Cp", "REML"):
        fit = fit_gam(pt, w, pd.DataFrame({"y": y}), method=method)
        assert fit.family == "gaussian" and fit.scale[0] == pytest.approx(0.09, rel=0.2)
        assert fit.predict("y", [5.5], "t1")[0] == pytest.approx(np.sin(5.5), abs=0.15)
        assert fit.predict("y", [5.5], "t2")[0] == pytest.approx(-1.25, abs=0.15)
    curve = smoothers(fit, "y", n_points=20)
    assert set(curve["lineage"]) == {"t1", "t2"} and (curve["lower"] < curve["upper"]).all()


def test_negative_binomial_fit_and_theta():
    rng, seg, time, pt, w = two_lineages(n=3000)
    mu = 20 * np.exp(np.where(seg == 1, 0.3 * (time - 3), 0.0))
    y = rng.negative_binomial(4, 4 / (4 + mu)).astype(float)
    fit = fit_gam(pt, w, pd.DataFrame({"y": y}), family="nb")
    assert fit.converged[0]
    assert fit.theta[0] == pytest.approx(4, rel=0.3)
    assert fit.predict("y", [6.0], "t1")[0] == pytest.approx(20 * np.exp(0.9), rel=0.15)
    assert fit.predict("y", [6.0], "t2")[0] == pytest.approx(20, rel=0.15)
    with pytest.raises(ValueError):
        fit_gam(pt, w, pd.DataFrame({"y": -y}), family="nb")


def test_wald_pseudo_inverse():
    sigma = np.diag([1.0, 4.0, 1e-12])
    stat, df, p = wald(np.array([1.0, 2.0, 5.0]), sigma)
    assert df == 2 and stat == pytest.approx(2.0) and 0 < p < 1


def test_each_test_finds_its_own_kind_of_change():
    rng, seg, time, pt, w = two_lineages(n=3000)
    noise = lambda: rng.normal(0, 0.5, len(time))
    frame = pd.DataFrame(
        {
            "null": noise(),
            "rise_t1": np.where(seg == 1, time - 3, 0.0) + noise(),  # everything
            "shared": time / 3 + noise(),  # association, start vs end; not diff end / pattern
            "bump_t1": np.where(seg == 1, 2 * np.exp(-0.5 * ((time - 4.5) / 0.4) ** 2), 0)
            + noise(),
        }
    )
    fit = fit_gam(pt, w, frame, seed=3)
    p = {
        name: test(fit)["pvalue"]
        for name, test in (
            ("assoc", association_test),
            ("sve", start_vs_end_test),
            ("end", diff_end_test),
            ("pattern", pattern_test),
        )
    }
    small, large = 1e-4, 0.01
    assert (
        p["assoc"]["rise_t1"] < small
        and p["assoc"]["shared"] < small
        and p["assoc"]["bump_t1"] < small
    )
    assert (
        p["sve"]["rise_t1"] < small and p["sve"]["shared"] < small and p["sve"]["bump_t1"] > large
    )
    # (Not asked: diffEnd on the bump. A cubic spline with 6 knots rings after a narrow
    # bump, so the ends can differ a little; tradeSeq's GAM does the same.)
    assert p["end"]["rise_t1"] < small and p["end"]["shared"] > large
    assert (
        p["pattern"]["rise_t1"] < small
        and p["pattern"]["bump_t1"] < small
        and p["pattern"]["shared"] > large
    )
    assert all(p[k]["null"] > large for k in p)
    per = association_test(fit, lineages=True)
    assert per.loc["rise_t1", "pvalue_t1"] < small and per.loc["rise_t1", "pvalue_t2"] > large
    sve = start_vs_end_test(fit, lineages=True)
    assert sve.loc["rise_t1", "effect_t1"] == pytest.approx(3, abs=0.3)
    pair = diff_end_test(fit, pairwise=True)
    assert "pvalue_t1vst2" in pair and pair.loc["rise_t1", "effect_t1vst2"] == pytest.approx(
        3, abs=0.3
    )
    early = early_drivers_test(fit, knots=(0, 2))
    assert early.loc["null", "pvalue"] > large
    assert {"waldStat", "df", "pvalue", "q"} <= set(association_test(fit).columns)


def test_null_features_are_not_called_too_often():
    """Calibration: on 300 null features, every test's type I error at 0.05 is at
    most about nominal (within binomial noise)."""
    rng, seg, time, pt, w = two_lineages(n=1500, seed=5)
    frame = pd.DataFrame(rng.normal(0, 1, (len(time), 300)))
    fit = fit_gam(pt, w, frame, seed=5)
    bound = 0.05 + 3 * np.sqrt(0.05 * 0.95 / 300)
    for test in (association_test, start_vs_end_test, diff_end_test, pattern_test):
        assert (test(fit)["pvalue"] < 0.05).mean() <= bound, test.__name__


# ---- validation on simulations -------------------------------------------------------------


@pytest.mark.parametrize("topology", ["linear", "bifurcation", "trifurcation", "tree"])
def test_slingshot_recovers_simulated_topologies(topology):
    sim = simulate_trajectory(topology, seed=1)
    from vtea_core.trajectory import slingshot

    scores = validate_lineages(sim, slingshot(sim["points"], sim["labels"], sim["start"]))
    assert scores["topology_correct"]
    assert scores["pseudotime_spearman"] > 0.93
    if topology != "linear":
        assert scores["branch_accuracy"] > 0.8


@pytest.mark.parametrize("family", ["gaussian", "nb"])
def test_end_to_end_power_and_error_rates(family):
    _, tests = run_one("bifurcation", seed=2, family=family, n_features=80)
    tests = tests.set_index("test")
    assert (tests["type_I_error"] <= 0.15).all()
    assert (tests["power"] >= 0.7).all()
    assert (tests["fdr"] <= 0.2).all()


def test_benchmark_tables():
    b = benchmark(["linear"], seeds=[0], distances=["slingshot"], n_features=20)
    assert set(b) == {"lineages", "tests", "summary_lineages", "summary_tests"}
    assert b["lineages"]["topology_correct"].all()


# ---- the R engine's plumbing (without R) ------------------------------------------------------


def test_r_engine_writes_scripts_and_reads_results(monkeypatch):
    """The R scripts get every setting filled in, and what R writes is read back
    (a stand-in for Rscript writes the files R would)."""
    from vtea_core.trajectory import r_engine

    sim = simulate_trajectory("bifurcation", seed=0, n_features=4)
    scripts = []

    def fake_run(script, folder, timeout):
        scripts.append(script)
        n = len(sim["labels"])
        if "slingshot(" in script:
            pt = np.where(sim["membership"], sim["time"][:, None], np.nan)
            pd.DataFrame(pt).to_csv(folder / "pseudotime.csv", index=False)
            pd.DataFrame(sim["membership"].astype(float)).to_csv(folder / "weights.csv", index=False)
            (folder / "lineages.txt").write_text("\n".join("\t".join(lin) for lin in sim["lineages"]))
            for i in range(2):
                pd.DataFrame(np.zeros((150, 3))).to_csv(folder / f"curve{i + 1}.csv", index=False)
        else:
            table = pd.DataFrame({"waldStat": 1.0, "df": 2, "pvalue": 0.5},
                                 index=sim["values"].columns)
            for name in ("association", "start_vs_end", "diff_end", "pattern"):
                table.to_csv(folder / f"{name}.csv")
        assert len(pd.read_csv(folder / ("points.csv" if "slingshot(" in script else "values.csv"))) == n

    monkeypatch.setattr(r_engine, "_run", fake_run)
    traj = r_engine.slingshot_r(sim["points"], sim["labels"], sim["start"], ends=["a.1"])
    assert traj.lineages == sim["lineages"] and traj.pseudotime.shape == (len(sim["labels"]), 2)
    for placeholder in ("SEED", "START", "ENDS", "DIST", "SHRINK", "EXTEND", "REWEIGHT", "REASSIGN",
                        "THRESH", "MAXIT", "STRETCH", "APPROX"):
        assert placeholder not in scripts[0], placeholder
    assert 'start.clus = "s0.0"' in scripts[0] and 'end.clus = c("a.1")' in scripts[0]
    out = r_engine.tradeseq_r(traj.pseudotime, traj.weights, sim["values"], nknots=5, family="gaussian")
    assert set(out) == {"association", "start_vs_end", "diff_end", "pattern"}
    assert "nknots = 5" in scripts[1] and 'FAMILY' not in scripts[1] and '"gaussian"' in scripts[1]


def test_r_status_without_r(monkeypatch):
    from vtea_core.trajectory import r_engine

    monkeypatch.setattr(r_engine, "rscript", lambda: None)
    assert r_engine.r_status()["rscript"] is None and not r_engine.r_available()


def test_evaluate_k_prefers_more_knots_for_a_wiggly_feature():
    from vtea_core.trajectory.tradeseq import evaluate_k

    rng, seg, time, pt, w = two_lineages(n=2000)
    frame = pd.DataFrame({"wiggly": np.sin(3 * time) + rng.normal(0, 0.3, len(time)),
                          "straight": time + rng.normal(0, 0.3, len(time))})
    table = evaluate_k(pt, w, frame, knots=[3, 5, 8, 12])
    assert table.loc["wiggly", "best k"] >= 8
    assert table.loc["wiggly", 12] < table.loc["wiggly", 3]
