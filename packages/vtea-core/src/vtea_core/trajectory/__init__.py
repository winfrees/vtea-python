"""Trajectories: lineages and pseudotime through cell types and neighbourhood types.

Clustering, classification and neighbourhood analysis say *what* each object
is. A trajectory asks how those types connect: which states lie between a
healthy and an injured cell type, where lineages branch, and which markers
(or neighbourhood compositions) change along each branch. The approach is the
one used for proximal tubule repair states in human kidney (Asghari et al.,
Sci Adv 11:eadv8918, 2025, figures 5 and 6): types projected into a reduced
space, lineages found from a chosen start type with Slingshot, marker
"pseudotime spectra" along each lineage, and tradeSeq tests of what changes.

- `slingshot`: the cluster tree (MST of cluster distances), one lineage per
  leaf from the start cluster, principal curves and pseudotime per lineage
  (`Trajectory`). See `slingshot.py`.
- `fit_gam` and its tests (`association_test`, `start_vs_end_test`,
  `diff_end_test`, `pattern_test`, `early_drivers_test`) and `smoothers`: a
  port of tradeSeq - a GAM per feature along the lineages and Wald tests on
  its fitted curves. See `tradeseq.py`.
- `feature_trends`, `association`, `composition_along`: quick descriptive
  summaries (binned means, Spearman with pseudotime, class or
  neighbourhood-type fractions by pseudotime). See `along.py`.
- `r_engine`: the R packages themselves (Bioconductor slingshot and
  tradeSeq through Rscript), when installed.
- `simulate` and `validate`: simulated trajectories with a known truth, the
  validation benchmark, and parity checks with R. See docs/TRAJECTORIES.md.

Run on cell types, the space is the objects' features (or a map of them); run
on neighbourhood types, it is the neighbourhoods' compositions, so lineages
follow how a tissue niche changes into another.

One dataset at a time: atlas / cohort trajectories over many datasets (a fixed
pipeline applied to each, then integration) are planned on top of these
functions - see docs/TRAJECTORIES.md.
"""

from vtea_core.trajectory.along import (
    association,
    benjamini_hochberg,
    composition_along,
    feature_trends,
    pseudotime_bins,
)
from vtea_core.trajectory.slingshot import (
    Trajectory,
    cluster_distances,
    lineage_tree,
    slingshot,
)
from vtea_core.trajectory.tradeseq import (
    GAMFit,
    assign_lineages,
    association_test,
    diff_end_test,
    early_drivers_test,
    evaluate_k,
    fit_gam,
    pattern_test,
    smoothers,
    start_vs_end_test,
)

__all__ = [
    "GAMFit",
    "Trajectory",
    "assign_lineages",
    "association",
    "association_test",
    "benjamini_hochberg",
    "cluster_distances",
    "composition_along",
    "diff_end_test",
    "early_drivers_test",
    "evaluate_k",
    "feature_trends",
    "fit_gam",
    "lineage_tree",
    "pattern_test",
    "pseudotime_bins",
    "slingshot",
    "smoothers",
    "start_vs_end_test",
]
