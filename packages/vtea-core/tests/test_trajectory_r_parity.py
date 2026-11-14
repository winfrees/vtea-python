"""Parity of the Slingshot and tradeSeq ports with the R packages.

Runs only where R with Bioconductor slingshot and tradeSeq is installed (the
`trajectory-r-parity` workflow installs them); skipped otherwise. The
thresholds are what "the same method" should give on simulated data with a
clear structure: identical lineages, near-identical pseudotime ordering, and
Wald statistics that rank features the same way, with the same calls at
p < 0.05 for nearly all features.
"""

import pytest

from vtea_core.trajectory.r_engine import r_available

pytestmark = pytest.mark.skipif(
    not r_available("slingshot", "tradeSeq"),
    reason="R with slingshot and tradeSeq is not installed",
)


@pytest.mark.parametrize("topology", ["linear", "bifurcation", "trifurcation"])
def test_slingshot_and_tradeseq_agree_with_r(topology):
    from vtea_core.trajectory.simulate import simulate_trajectory
    from vtea_core.trajectory.validate import compare_with_r

    sim = simulate_trajectory(topology, seed=3, n_features=60)
    out = compare_with_r(sim, family="gaussian")
    assert out["lineages_equal"], out
    assert out["pseudotime_spearman"] > 0.98, out
    assert out["weights_agreement"] > 0.9, out
    for test in ("association", "start_vs_end", "diff_end", "pattern"):
        if f"{test}_wald_spearman" in out:
            assert out[f"{test}_wald_spearman"] > 0.9, (test, out)
            assert out[f"{test}_call_agreement"] > 0.9, (test, out)


def test_negative_binomial_agrees_with_r():
    from vtea_core.trajectory.simulate import simulate_trajectory
    from vtea_core.trajectory.validate import compare_with_r

    sim = simulate_trajectory("bifurcation", seed=4, n_features=40, family="nb")
    out = compare_with_r(sim, family="nb")
    assert out["lineages_equal"], out
    for test in ("association", "start_vs_end", "diff_end", "pattern"):
        assert out[f"{test}_wald_spearman"] > 0.85, (test, out)
        assert out[f"{test}_call_agreement"] > 0.85, (test, out)
