"""Validate vtea_core.trajectory (Slingshot and tradeSeq ports) and write a report.

    python scripts/validate_trajectory.py --seeds 10 --out docs/trajectory-validation
    python scripts/validate_trajectory.py --seeds 10 --r --out report   # with R parity

1. Simulation benchmark: four topologies (linear, bifurcation, trifurcation, a
   tree with two branch points) x seeds x both cluster distances x Gaussian /
   negative-binomial features. Lineage recovery and the type I error, power
   and FDR of each tradeSeq test, from estimated and from true pseudotime.
2. A known failure mode: no cluster covers the branch point.
3. With --r: parity with R slingshot / tradeSeq (needs R and the packages).

Writes CSV tables and REPORT.md to --out.
"""

from __future__ import annotations

import argparse
import datetime
import platform
import time
from pathlib import Path

import pandas as pd
from vtea_core.trajectory import validate
from vtea_core.trajectory.simulate import simulate_trajectory


def _md(frame: pd.DataFrame, digits=3) -> str:
    frame = frame.round(digits).reset_index()
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in frame.to_numpy()
    ]
    return "\n".join([head, rule, *body])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--r", action="store_true", help="also compare with R slingshot / tradeSeq"
    )
    parser.add_argument("--out", default="trajectory-validation")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seeds)
    started = time.perf_counter()

    est = validate.benchmark(seeds=seeds, families=("gaussian", "nb"))
    true = validate.benchmark(
        seeds=seeds,
        families=("gaussian", "nb"),
        distances=("slingshot",),
        use_true_time=True,
    )
    no_branch = validate.benchmark(
        seeds=seeds, families=("gaussian",), branch_overlap=0.0, n_features=10
    )
    est["lineages"].to_csv(out / "lineages.csv", index=False)
    est["tests"].to_csv(out / "tests_estimated_pseudotime.csv", index=False)
    true["tests"].to_csv(out / "tests_true_pseudotime.csv", index=False)
    no_branch["lineages"].to_csv(
        out / "lineages_no_branch_point_cluster.csv", index=False
    )

    parity = None
    if args.r:
        from vtea_core.trajectory.r_engine import r_available, r_status

        if r_available("slingshot", "tradeSeq"):
            rows = []
            for family in ("gaussian", "nb"):
                for topology in ("linear", "bifurcation", "trifurcation", "tree"):
                    for seed in range(min(args.seeds, 3)):
                        sim = simulate_trajectory(
                            topology, seed=seed, family=family, n_features=40
                        )
                        try:
                            res = validate.compare_with_r(sim, family=family, seed=seed)
                        except Exception as e:  # noqa: BLE001 (reported, not fatal)
                            res = {"error": str(e)[:300]}
                        rows.append(
                            {
                                "family": family,
                                "topology": topology,
                                "seed": seed,
                                **res,
                            }
                        )
            parity = pd.DataFrame(rows)
            parity.to_csv(out / "r_parity.csv", index=False)
            status = r_status()
        else:
            status = {"rscript": None}
    lin = est["summary_lineages"].unstack("distance")
    lin.columns = [f"{a} ({b})" for a, b in lin.columns]
    report = [
        "# Trajectory validation report",
        "",
        f"Generated {datetime.date.today()} by `scripts/validate_trajectory.py --seeds {args.seeds}"
        + (" --r" if args.r else "")
        + f"` (Python {platform.python_version()}, "
        f"{time.perf_counter() - started:.0f} s). Simulated data from "
        "`vtea_core.trajectory.simulate.simulate_trajectory` (300 objects per segment, 2 clusters per "
        "segment, a cluster at each branch point, 60 features: half null, the rest rising on one "
        "lineage, changing at one end, a transient bump, or a rise shared by every lineage).",
        "",
        "## 1. Lineages (Slingshot port)",
        "",
        "Mean over seeds. `topology_correct`: the lineages found are exactly the true ones. "
        "`pseudotime_spearman`: worst lineage's rank correlation with the true time. "
        "`branch_accuracy`: objects past a branch point whose main lineage is their own.",
        "",
        _md(lin),
        "",
        "## 2. Tests (tradeSeq port), on Slingshot's pseudotime",
        "",
        "Mean over topologies and seeds; Slingshot distance. `type_I_error`: null features with "
        "p < 0.05 (should be <= 0.05). `power`: features the test should find, at BH q < 0.05. "
        "`fdr`: share of discoveries (q < 0.05) the test should not have made (should be <= 0.05).",
        "",
        _md(
            est["tests"][est["tests"]["distance"] == "slingshot"]
            .groupby(["family", "test"], sort=False)[["type_I_error", "power", "fdr"]]
            .mean()
        ),
        "",
        "## 3. Tests on the true pseudotime",
        "",
        "The same, with the true pseudotime and lineages: the statistics alone, without "
        "trajectory-inference error.",
        "",
        _md(true["summary_tests"]),
        "",
        "## 4. Known failure mode: no cluster at the branch point",
        "",
        "When the first clusters of the branches meet at the branch point (no cluster covers it), "
        "the cluster tree joins them to each other instead of to the trunk, with either distance. "
        "Re-cluster (or merge) so one cluster covers each branch point, or check the tree on the map.",
        "",
        _md(no_branch["summary_lineages"][["topology_correct"]]),
        "",
        "## 5. Parity with R (slingshot, tradeSeq)",
        "",
    ]
    if parity is not None and len(parity):
        report += [
            f"R: {status.get('R')}, slingshot {status.get('slingshot')}, tradeSeq "
            f"{status.get('tradeSeq')}.",
            "",
            _md(parity.set_index(["family", "topology", "seed"])),
            "",
        ]
    elif args.r:
        report += ["R with slingshot and tradeSeq was not found: not run.", ""]
    else:
        report += [
            "Not run here (needs R with Bioconductor slingshot and tradeSeq). Run the "
            "`trajectory-r-parity` GitHub workflow, or this script with `--r`.",
            "",
        ]
    (out / "REPORT.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
