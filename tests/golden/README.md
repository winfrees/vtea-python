# Golden-dataset parity harness

Phase 0 deliverable (see `docs/PORT_PLAN.md`): a fixture set of Java-VTEA
outputs that every later phase's Python port is diffed against for
numerical/behavioral parity.

## What's here

- `compare.py` — comparison utilities (`segmentation_iou`,
  `feature_table_diff`, `cluster_assignment_ari`). Unit-tested in
  `test_compare.py`, no fixtures required.
- `fixtures.py` — loaders for the fixture files described below.
- `test_parity.py` — the actual parity tests. Skipped as a whole until
  `fixtures/` is populated; individual tests are further skipped until the
  corresponding `vtea-core` functionality lands (each names the phase it's
  blocked on).
- `fixtures/` — not checked in (git-ignored, see `.gitignore`). Populate it
  per the instructions below.

## Generating fixtures

Fixtures are produced by `GoldenFixtureGenerator.java` in the source Java
repo, run via a GitHub Actions workflow (`generate-golden-fixtures.yml`)
rather than locally — the Maven build needs `maven.scijava.org`, which is
blocked in network-restricted sandboxes.

1. Go to the [workflow's Actions page](https://github.com/winfrees/volumetric-tissue-exploration-analysis/actions/workflows/generate-golden-fixtures.yml)
   in the Java repo and run it (`workflow_dispatch`) against the branch you
   want fixtures from.
2. Download the `golden-fixtures` artifact from the completed run.
3. Unzip its contents into `tests/golden/fixtures/` in this repo.

The generator produces two kinds of fixture:

**Image-derived** (one set per sample TIFF — `AQtest_human_crop` and
`C1-IU_VTEA_ExampleData_001`), from the real `SingleThreshold3D`
segmentation and measurement plugins:

```
fixtures/
  <dataset>_segmentation_singlethreshold.tif   label mask (uint16)
  <dataset>_measurements.csv                   object_id,count,mean,sum,stddev,min,max
  <dataset>_metadata.txt                       dimensions, threshold used, object count
```

Note: `SingleThreshold3D` produces exactly one object (all above-threshold
voxels) — it validates image I/O, thresholding, and measurement extraction
end-to-end, but isn't a multi-object fixture. Multi-object segmentation
fixtures should be added once those methods are ported in Phase 2.

**Synthetic** (deterministic, seed=42, independent of image segmentation —
isolates clustering/DR algorithmic parity), from the real `KMeans` and
`PCAReduction` implementations:

```
fixtures/
  synthetic_clustering_input.csv      point_id,x,y (300 points, 3 clusters)
  synthetic_clustering_kmeans_k3.csv  point_id,cluster
  synthetic_clustering_pca.csv        point_id,pc1,pc2
```

## Comparison tolerances

Not exact bit-for-bit parity — algorithmic equivalence:

- **Segmentation**: IoU on foreground/background (not label-id equality —
  label IDs aren't expected to match across independent runs).
- **Measurements**: relative tolerance (default `rtol=1e-3`) per numeric
  column, row-aligned on `object_id`.
- **Clustering**: Adjusted Rand Index (ARI) between assignments, not
  cluster-id equality (ids are arbitrary/permutable).
- **PCA**: compare explained variance, not raw components directly — sign
  and axis order aren't guaranteed to match across implementations.

## Running

```bash
pip install -e "packages/vtea-core[dev]"   # provides numpy/pandas/sklearn/tifffile
pytest tests/golden
```
