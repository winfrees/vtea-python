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
| `slingshot(points, labels, start, ends=None, distance="slingshot")` | a `Trajectory`: clusters, centres, MST edges, lineages, curves, `pseudotime` and `weights` (objects × lineages), `average_pseudotime()`, `main_lineage()`, `to_dict()` |
| `lineage_tree(points, labels, start, ends=None)` | the cluster tree and the lineages only |
| `cluster_distances(points, labels, clusters, distance=…)` | the cluster distance matrix |
| `feature_trends(pseudotime, values, weights=…, n_bins=20)` | mean and SEM of each feature in bins of pseudotime per lineage (the pseudotime spectrum) |
| `association(pseudotime, values, weights=…)` | per lineage and feature: Spearman ρ with pseudotime, p, BH q, start / end means and start-vs-end in SDs |
| `composition_along(pseudotime, classes, weights=…, n_bins=10)` | fraction of each class (cell type, cluster, neighbourhood type) in bins of pseudotime per lineage |

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

### Differences from Slingshot (R)

- The distance, MST, forced ends, lineages from a start cluster, iterative
  principal curves, shrinkage of shared lineages and lineage weights follow
  Street et al., *BMC Genomics* 19:477 (2018).
- The smoother is a Gaussian kernel on binned arc length rather than
  `smooth.spline`, so it is O(n) in the objects.
- Projection onto curves uses a k-d tree of a densely resampled curve. On 4
  CPU cores, 200,000 objects in 10 dimensions take about 4 s.
- The artificial "omega" cluster, which lets the graph fall apart into
  separate trees, is not ported. Every cluster is on one tree from the start.
- `association` asks tradeSeq's questions (Van den Berge et al., *Nat Commun*
  11:1201, 2020): does a feature change along a lineage, and differ between
  its start and end? It answers them with rank correlation and the means of
  the first and last fifth of the lineage, not a negative-binomial GAM.
  Imaging intensities are not counts, and with tens of thousands of objects
  almost every association is "significant", so read ρ and the start-vs-end
  effect size, not only q.

### Choosing the start

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
