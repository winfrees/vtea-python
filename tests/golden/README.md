# Golden-dataset parity harness

Phase 0 deliverable (see `docs/PORT_PLAN.md`): a fixture set of Java-VTEA
outputs that every later phase's Python port is diffed against for
numerical/behavioral parity.

## What's here

- `compare.py` — comparison utilities (`segmentation_iou`,
  `feature_table_diff`, `cluster_assignment_ari`). Unit-tested in
  `test_compare.py`, no fixtures required.
- `fixtures.py` — loaders for the fixture files described below.
- `test_parity.py` — the actual parity tests, with real assertions against
  `vtea-core`. Skipped as a whole until `fixtures/` is populated; the
  image-derived tests also skip until the sample TIFFs are in `data/`.
- `fixtures/`, `data/` — not checked in (git-ignored, see `.gitignore`).
  Populate them per the instructions below.

**Status (2026-09-25): no fixtures have ever been generated.** The Java
workflow below has not been run, and `GoldenFixtureGenerator.java` has never
been compiled. Until it is, nothing here has compared a Python number with a
Java one. This is milestone M1 in `docs/PORT_PLAN.md`'s "Path forward".

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
4. Copy `AQtest_human_crop.tif` and `C1-IU_VTEA_ExampleData_001.tif` from the
   Java repo's root into `tests/golden/data/` - the image-derived tests
   re-run the segmentation on them.

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
end-to-end, but isn't a multi-object fixture.

**LayerCake3D** (the Java default segmentation, ported as
`vtea_core.segmentation.layercake_3d`) has a parity test waiting for its
fixture, which the generator does not write yet (milestone M2). The test
expects, per dataset:

```
fixtures/
  <dataset>_segmentation_layercake3d.tif   label image (ZYX) from LayerCake3DSingleThreshold
  <dataset>_metadata.txt                   plus layercake_threshold, layercake_offset,
                                           layercake_min, layercake_max, layercake_watershed
```

It compares foreground IoU and objects matched by centroid rather than
label ids - see the test's docstring for why a few per cent may differ.

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
- **PCA**: per-component absolute correlation (> 0.999) — sign, centring
  and scale aren't guaranteed to match across implementations; the axes are.

Known differences to expect: Java converts the thresholded stack to 8-bit
before collecting voxels (`SingleThreshold.process`), and truncates the
threshold to an int; the test does the latter, and the IoU tolerance absorbs
the former only if the conversion is lossless. Java's standard deviation
divides by n - 1; `vtea-core` matches it.

## Running

```bash
pip install -e "packages/vtea-core[dev]"   # provides numpy/pandas/sklearn/tifffile
pytest tests/golden
```
