# Trajectory validation report

Generated 2026-11-14 by `scripts/validate_trajectory.py --seeds 10` (Python 3.13.18, 256 s). Simulated data from `vtea_core.trajectory.simulate.simulate_trajectory` (300 objects per segment, 2 clusters per segment, a cluster at each branch point, 60 features: half null, the rest rising on one lineage, changing at one end, a transient bump, or a rise shared by every lineage).

## 1. Lineages (Slingshot port)

Mean over seeds. `topology_correct`: the lineages found are exactly the true ones. `pseudotime_spearman`: worst lineage's rank correlation with the true time. `branch_accuracy`: objects past a branch point whose main lineage is their own.

| topology | family | topology_correct (slingshot) | topology_correct (euclidean) | pseudotime_spearman (slingshot) | pseudotime_spearman (euclidean) | branch_accuracy (slingshot) | branch_accuracy (euclidean) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear | gaussian | 1.0 | 1.0 | 0.981 | 0.981 |  |  |
| linear | nb | 1.0 | 1.0 | 0.981 | 0.981 |  |  |
| bifurcation | gaussian | 1.0 | 1.0 | 0.975 | 0.975 | 0.931 | 0.931 |
| bifurcation | nb | 1.0 | 1.0 | 0.975 | 0.975 | 0.931 | 0.931 |
| trifurcation | gaussian | 0.6 | 0.0 | 0.964 | 0.95 | 0.877 |  |
| trifurcation | nb | 0.6 | 0.0 | 0.964 | 0.95 | 0.877 |  |
| tree | gaussian | 1.0 | 1.0 | 0.976 | 0.976 | 0.947 | 0.946 |
| tree | nb | 1.0 | 1.0 | 0.976 | 0.976 | 0.947 | 0.946 |

## 2. Tests (tradeSeq port), on Slingshot's pseudotime

Mean over topologies and seeds; Slingshot distance. `type_I_error`: null features with p < 0.05 (should be <= 0.05). `power`: features the test should find, at BH q < 0.05. `fdr`: share of discoveries (q < 0.05) the test should not have made (should be <= 0.05).

| family | test | type_I_error | power | fdr |
| --- | --- | --- | --- | --- |
| gaussian | association | 0.014 | 0.885 | 0.005 |
| gaussian | start_vs_end | 0.053 | 0.797 | 0.036 |
| nb | association | 0.019 | 1.0 | 0.006 |
| nb | start_vs_end | 0.051 | 0.91 | 0.082 |
| gaussian | diff_end | 0.052 | 0.783 | 0.071 |
| gaussian | pattern | 0.049 | 0.909 | 0.041 |
| nb | diff_end | 0.043 | 0.858 | 0.089 |
| nb | pattern | 0.047 | 1.0 | 0.036 |

## 3. Tests on the true pseudotime

The same, with the true pseudotime and lineages: the statistics alone, without trajectory-inference error.

| family | test | type_I_error | power | fdr |
| --- | --- | --- | --- | --- |
| gaussian | association | 0.017 | 0.958 | 0.008 |
| gaussian | start_vs_end | 0.05 | 0.898 | 0.035 |
| nb | association | 0.015 | 1.0 | 0.01 |
| nb | start_vs_end | 0.051 | 0.997 | 0.088 |
| gaussian | diff_end | 0.06 | 0.906 | 0.058 |
| gaussian | pattern | 0.049 | 0.971 | 0.035 |
| nb | diff_end | 0.044 | 0.998 | 0.085 |
| nb | pattern | 0.04 | 1.0 | 0.032 |

## 4. Known failure mode: no cluster at the branch point

When the first clusters of the branches meet at the branch point (no cluster covers it), the cluster tree joins them to each other instead of to the trunk, with either distance. Re-cluster (or merge) so one cluster covers each branch point, or check the tree on the map.

| topology | distance | family | topology_correct |
| --- | --- | --- | --- |
| linear | slingshot | gaussian | 1.0 |
| linear | euclidean | gaussian | 1.0 |
| bifurcation | slingshot | gaussian | 0.0 |
| bifurcation | euclidean | gaussian | 0.0 |
| trifurcation | slingshot | gaussian | 0.0 |
| trifurcation | euclidean | gaussian | 0.0 |
| tree | slingshot | gaussian | 0.0 |
| tree | euclidean | gaussian | 0.0 |

## 5. Parity with R (slingshot, tradeSeq)

Not run here (needs R with Bioconductor slingshot and tradeSeq). Run the `trajectory-r-parity` GitHub workflow, or this script with `--r`.
