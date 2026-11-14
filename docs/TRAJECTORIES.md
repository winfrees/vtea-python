# Trajectories: lineages through cell types and neighbourhood types

Clustering, classification and neighbourhood analysis say *what* each object
is. A trajectory asks how those types connect. Which states lie between a
healthy cell type and an injured one? Where do lineages branch? Which markers,
or which neighbourhood compositions, change along each branch?

`vtea_core.trajectory` answers these for **one dataset**. It follows the
approach used for proximal tubule (PT) repair states in the human kidney:
Asghari, Sabo, …, Winfree, El-Achkar, Dagher, *Sci Adv* 11:eadv8918 (2025),
[PMC12356270](https://pmc.ncbi.nlm.nih.gov/articles/PMC12356270/), figures 5
and 6. In that paper:

1. PT cells were subset and reclustered.
2. They were projected into UMAP.
3. Trajectories were found with Slingshot (R), starting from a cluster chosen
   by its markers: the high-LRP2 healthy PT cluster, or the
   THY1+PROM1+VCAM1+ cluster.
4. The pseudotime "spectra" of THY1, PROM1 and VCAM1 were read along each
   lineage.

Figure 8 of the same paper types neighbourhoods (VTEA "spatial by cell",
55 µm) and relates them to cell types and disease. Here, a trajectory can be
run on those neighbourhood types too.

## What it does

```
 objects x features (or a map)      clusters (types)       start type
          │                               │                    │
          └──────────────┬────────────────┘                    │
                         ▼                                     │
            cluster means + covariances                        │
                         ▼                                     │
     cluster distances (Slingshot: (S_i+S_j)^-1 scaled) ──▶ MST, rooted
                         ▼                                     at start
         lineages: start ──▶ each leaf (forced ends = leaves)
                         ▼
     principal curve per lineage (iterate: project, smooth, shrink
     shared trunk towards the mean curve)
                         ▼
   per object, per lineage: pseudotime (arc length), weight 0..1
                         ▼
   feature_trends · association · composition_along
```

| function | returns |
| --- | --- |
| `slingshot(points, labels, start, ends=None, distance="slingshot", ...)` | a `Trajectory`: clusters, centres, tree edges, lineages, curves, `pseudotime` and `weights` (objects × lineages), `average_pseudotime()`, `main_lineage()`, `to_dict()`. Port of R slingshot, with its arguments and defaults |
| `lineage_tree(points, labels, start, ends=None)` | the cluster tree and the lineages only (`getLineages`) |
| `fit_gam(pseudotime, weights, values, nknots=6, family="gaussian")` | a `GAMFit`: one GAM per feature along the lineages. Port of tradeSeq `fitGAM` |
| `association_test`, `start_vs_end_test`, `diff_end_test`, `pattern_test`, `early_drivers_test` | tradeSeq's Wald tests, one row per feature: `waldStat`, `df`, `pvalue`, BH `q`, per lineage / pair columns |
| `smoothers(fit, feature)` | the fitted curves with 95% bands (`plotSmoothers`) |
| `evaluate_k(pseudotime, weights, values, knots=range(3, 11))` | AIC by number of knots (`evaluateK`) |
| `feature_trends`, `association`, `composition_along` | quick descriptive summaries: binned means (the pseudotime spectrum), Spearman ρ with pseudotime, class or neighbourhood-type fractions by pseudotime |
| `r_engine.slingshot_r`, `r_engine.tradeseq_r` | the R packages themselves, through Rscript, when R with slingshot and tradeSeq is installed |
| `simulate.simulate_trajectory`, `validate.benchmark`, `validate.compare_with_r` | simulated data with a known truth, the validation benchmark, and parity with R |

**Cell types or neighbourhood types.**

- *Cell types*: the points are the objects' features (typically PCA of
  asinh / z-scored intensities) or a map such as UMAP. The labels are
  phenotypes, clusters or predicted classes.
- *Neighbourhood types*: the points are each object's neighbourhood
  composition (`Class_<c>_ClassFraction`, or IBIF napari's
  `<name> fraction <class>`). The labels are the neighbourhood types. Lineages
  then follow how one niche turns into another, e.g. a THY1+ PT-rich,
  immune-poor niche into a PROM1+ PT, immune-rich one.

`composition_along` connects the two. Along a cell-type lineage it shows which
neighbourhood types the cells are in. Along a neighbourhood lineage it shows
which cell types the neighbourhoods are made of.

### Slingshot: what is ported

The port follows R slingshot (Street et al., *BMC Genomics* 19:477, 2018;
`getLineages` + `getCurves`) with its defaults:

- **Cluster tree.** Covariance-scaled cluster distances
  (`dist.method = "slingshot"`) or Euclidean. Minimum spanning tree rooted at
  the start cluster; forced end clusters (`end.clus`).
- **Initial curves.** Polylines through the cluster centres, extended past the
  end centres (`extend = "y"`).
- **Principal curves.** A weighted cubic smoothing spline with df = 5 for each
  coordinate against arc length, evaluated at 150 points (`approx_points`).
  Projection with `stretch = 2`.
- **Shrinkage.** At each branch point, deepest first, curves are shrunk
  towards their average by the cosine-kernel survival function over the
  shared objects' arc lengths (`shrink = 1`).
- **Weights.** Reweighting (1 − Z², Z the distance quantile) and reassignment
  (Z < 0.5 added; Z > 0.9 with weight < 0.1 dropped).
- **Convergence.** Relative change of the total distance below 0.001, at most
  15 iterations.

Differences from R:

- The smoothing spline uses up to 50 knots at quantiles, a reduced knot set
  as R's `smooth.spline` also uses for large n. Its smoothing is set to df = 5
  exactly.
- Pseudotime is shifted so each lineage's earliest object is at 0.
- Not ported: the "omega" cluster that splits the graph into separate trees,
  and the loess smoother.
- Speed: projection uses a k-d tree on a densely resampled curve. 200,000
  objects in 10 dimensions take about 10 s.

### tradeSeq: what is ported

The port follows tradeSeq (Van den Berge et al., *Nat Commun* 11:1201,
2020).

- **`fitGAM`.**
  - Each object is assigned to one lineage, drawn in proportion to its
    weights.
  - One cubic regression spline per lineage (mgcv `bs = "cr"`), all sharing
    one smoothing parameter (`id = 1`).
  - `nknots` knots at quantiles of the assigned pseudotimes, with the lineage
    ends moved onto knots.
  - Families:
    - `family="gaussian"` (identity link) for continuous imaging measurements
      (asinh / z-scored intensities, neighbourhood fractions);
    - `family="nb"` (log link, theta by maximum likelihood) for counts.
      This is tradeSeq's default; an `offset` gives log library size.
  - The smoothing parameter is chosen by GCV / UBRE (mgcv's default
    `GCV.Cp`) or `REML`. For the NB family it is chosen by performance
    iteration on the working model rather than mgcv's outer Newton
    optimisation. Inference uses mgcv's Bayesian covariance `Vp`.
- **Tests.** Wald tests on contrasts of the linear predictor, with an eigen
  pseudo-inverse (eigenvalue threshold 1e-2 for the pattern tests, as
  tradeSeq):
  - `associationTest`: 2 × nknots points per lineage against the start;
  - `startVsEndTest`;
  - `diffEndTest`;
  - `patternTest`: each lineage on its own pseudotime scale;
  - `earlyDriversTest`: between two knots.

  BH q-values are added.
- **Not ported:** `conditions`, zero-inflation weights, and fold-change
  thresholds (`l2fc`).

For imaging data the Gaussian family is the right one. Intensities are not
counts, and the negative binomial would treat them as such.

## Validation

Two kinds of evidence. Both can be re-run with `scripts/validate_trajectory.py`.

**1. Simulations with a known truth.** The current report is
[trajectory-validation/REPORT.md](trajectory-validation/REPORT.md): 10 seeds,
Gaussian and NB features, and the per-run CSV tables. The data:

- four topologies: linear, bifurcation, trifurcation, and a tree with two
  branch points;
- 1,500–1,800 objects, clusters as a clustering would cut them;
- 60 features, half null, the rest of known kinds: a rise on one lineage, a
  change at one end, a transient bump, a rise shared by every lineage.

Results:

- **Lineages:**
  - The topology was recovered in every linear, bifurcation and tree run.
  - Trifurcations: 6/10 with Slingshot's distance, 0/10 with Euclidean.
    Euclidean was never better, so Slingshot's distance is the default.
  - Pseudotime vs. the true time: Spearman 0.95–0.98 on the worst lineage.
  - 88–95% of objects past a branch point were put on their own lineage.
- **Type I error** at 0.05 on null features:
  - 0.01–0.02 for the association test (conservative);
  - 0.04–0.06 for start-vs-end, diffEnd and pattern.
- **Power** at q < 0.05:
  - on Slingshot's pseudotime: 0.80–1.00;
  - on the true pseudotime: 0.90–1.00.
- **False discovery rate:**
  - at or below 0.05 for association and pattern;
  - 0.06–0.09 for start-vs-end and diffEnd.

  Almost all of that excess comes from narrow transient bumps. A 6-knot
  spline cannot follow them and bends the curve's ends. The GAM is the same
  as tradeSeq's, so this is a property of the method. With 8 knots the share
  of such features called falls from 56% to 17% (NB start-vs-end, linear and
  tree topologies). Use `evaluate_k` to choose the number of knots. On those
  same topologies the NB start-vs-end type I error was 0.07–0.08, slightly
  liberal.
- **Known failure mode** (section 4 of the report): if no cluster covers a
  branch point, the cluster tree joins the first clusters of the branches to
  each other, with either distance. This is inherent to cluster-tree methods,
  R slingshot included. Re-cluster or merge so one cluster spans each branch
  point, and check the tree on the map.

**2. Parity with R** (`validate.compare_with_r`,
`tests/test_trajectory_r_parity.py`). The same simulated data are run through
R slingshot / tradeSeq and through the ports, comparing:

- that the lineages are identical;
- the pseudotime rank correlation (> 0.98 required);
- the lineage weights;
- for each test, the rank correlation of the Wald statistics and the
  agreement of calls at p < 0.05 (> 0.9 required).

The GAMs get the same lineage assignment in both, so only the
implementations differ. **Status: not yet run.** It needs R with Bioconductor
slingshot and tradeSeq. They could not be installed where the ports were
written: CRAN and Bioconductor were not reachable. The
`trajectory-r-parity` GitHub workflow installs them, runs the parity tests
and uploads the full validation report. Run it, from the Actions tab, before
relying on parity.

## Choosing the start

Pseudotime has no direction of its own: the start cluster gives it one. Choose
it from biology:

- a healthy reference type (high LRP2 PT), with lineages towards injured or
  repair states;
- or a candidate progenitor or transitional state (THY1+PROM1+VCAM1+), with
  lineages towards its possible fates.

Running both, as the paper did, is a robustness check. So is running on each
condition separately and then together (reference only, disease only,
integrated).

## Not in scope yet: atlas and cohort trajectories

The aim is a pipeline built and checked on one dataset in IBIF napari or VTEA
that is then applied **unchanged** to many datasets. Its results are then
analysed together: an atlas, or a cohort meta-analysis across reference and
disease tissue. That needs more than running `slingshot` on a concatenated
table. The plan:

1. **A frozen pipeline.** Everything that turns an image into types and
   pseudotime is saved as one versioned, hashable specification:
   - segmentation;
   - measurements and feature scaling (asinh cofactor, z-score);
   - the phenotype rules or the classification model (by hash);
   - the neighbourhood definition (radius / k) and neighbourhood types;
   - the trajectory settings (features, space, start and end types, distance).

   Every dataset is run with it, and the specification and vtea-core version
   are recorded with each result. The pieces exist: protocols
   (`workflow.io`), model cards (`classification`), workspace saving in IBIF
   napari. What is missing is the bundle that ties them together.
2. **A shared reference space.** Trajectories from different datasets are
   only comparable in a common space with common types:
   - Types come from rules or a model applied to each dataset, not from
     clustering each one on its own. Phase 5 classification is this step.
   - The trajectory space is a *reference* PCA / UMAP fitted on reference
     datasets, with every other dataset projected into it (`umap` can
     transform new objects). It is not a new embedding per dataset.
   - Per-dataset normalisation (z-score within dataset, or quantile
     normalisation to the reference) comes first. Batch correction (e.g.
     Harmony on the PCA, or a reference-mapping model) is optional, and its
     effect is reported.
3. **Trajectory on the reference, pseudotime for everyone.** The lineages
   and curves are fitted once on the reference (or on a balanced subsample of
   every dataset). Each dataset's objects are then *projected* onto those
   curves to get pseudotime, as `_project` does. This keeps "pseudotime 0.7
   on lineage t2" meaning the same in every sample.
4. **Per-sample summaries, then statistics across samples.** The unit of
   replication is the sample or donor, not the cell. For each sample and
   lineage, compute:
   - the fraction of its objects on each lineage;
   - the pseudotime distribution (median, quantiles, density in bins);
   - feature trends;
   - the neighbourhood-type composition along pseudotime.

   Then compare conditions with sample-level models: mixed models with donor
   as a random effect; condition × pseudotime interaction for "does THY1 rise
   later in disease"; odds ratios for lineage occupancy, as the paper's
   figure 8 does for neighbourhoods; and differential progression tests
   (condiments-style: Roux de Bézieux et al., *Nat Commun* 15:833, 2024).
5. **Robustness reported by default.**
   - Trajectories per condition and integrated (the paper's fig. S12).
   - Bootstrap over samples for lineage stability.
   - Alternative start clusters.
   - Leave-one-dataset-out.
6. **Scale and provenance.** Object tables are in Parquet / DuckDB
   (`measurements.store`), one per dataset, with a cohort manifest: dataset,
   donor, condition, tissue, acquisition, and the pipeline hash. Fitting on a
   subsample and projecting everyone else keeps millions of objects
   tractable.

Each step reuses the single-dataset functions here. The new pieces are the
pipeline bundle, the reference space with projection, curve projection as a
public function, and the sample-level statistics. IBIF napari keeps its
Trajectory tab for one dataset. The cohort analysis is a separate, headless
workflow (vtea-core plus a cohort manifest) that can run on a cluster.
