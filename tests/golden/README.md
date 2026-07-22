# Golden-dataset parity harness

Phase 0 deliverable (see `docs/PORT_PLAN.md`): a fixture set of Java-VTEA
outputs — segmentation masks, feature tables, cluster/DR assignments — run
against representative datasets, that every later phase's Python port is
diffed against for numerical/behavioral parity.

**Not yet populated.** Producing the fixtures requires running the Java
application's pipeline (Maven build + a display or headless-ImageJ setup)
against representative datasets and capturing its outputs; that hasn't been
done in this environment yet. Candidate seed datasets already exist in the
Java repo root: `AQtest_human_crop.tif`, `C1-IU_VTEA_ExampleData_001.tif`.

Planned structure once populated:

```
tests/golden/
  data/                     input images (git-ignored; fetch via a script or Git LFS)
  fixtures/<dataset>/
    segmentation_<method>.tif   label masks from the Java pipeline
    measurements.parquet        per-object feature table from the Java pipeline
    clustering_<method>.parquet cluster assignments
    reduction_<method>.parquet  DR embeddings
  test_parity.py             diffs vtea-core output against the fixtures above
```

Comparison tolerances (segmentation IoU threshold, feature-table numeric
tolerance, cluster-assignment ARI threshold) should be defined per-dataset
once the harness is built, since exact bit-for-bit parity isn't the goal —
algorithmic equivalence is.
