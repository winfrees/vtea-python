# Phase 0 Review

Audit of Phase 0 ("Foundations & parity harness") against the scope defined
in `PORT_PLAN.md`, done before starting Phase 1.

## Scope checklist

| Item | Status |
|---|---|
| `vtea-core` / `vtea-napari` package skeletons | Done |
| CI (lint + test, both packages) | Done, see caveat below |
| Dependency choices locked in | Done |
| Golden-dataset regression harness | Partially done — see below |

## What's verified vs. what isn't

**Verified** (installed, tested, or run for real in this pass):

- `vtea-core` and `vtea-napari` both `pip install -e` cleanly and pass their
  test suites, in both a local venv and GitHub Actions.
- The npe2 plugin manifest (`vtea-napari/src/vtea_napari/napari.yaml`) is
  valid per `npe2 validate`.
- `tests/golden/compare.py` (`segmentation_iou`, `feature_table_diff`,
  `cluster_assignment_ari`) is unit-tested — 10 tests, all passing.
- `tests/golden/test_parity.py` skips cleanly (whole-module skip when
  fixtures are absent, per-test skip naming the blocking phase otherwise) —
  confirmed it doesn't silently pass or error.
- CI actually runs on GitHub, not just locally — this caught a real bug
  (below) that local testing missed.
- License: swapped GitHub's default GPLv3 auto-init for GPL-2.0, matching
  the source Java repo.

**Not verified** (written, not run):

- `GoldenFixtureGenerator.java` (in the Java repo) — cannot be
  compiled or run in this environment. The Maven build resolves
  dependencies through `maven.scijava.org`, which this sandbox's network
  policy blocks (403 on every SciJava-hosted artifact: ImageJ, ImgLib2, N5,
  MorphoLibJ, Bio-Formats, Renjin, tsne, CUDA platform). I wrote it against
  API signatures verified by reading the existing JUnit test suite
  (`SingleThresholdTest`, `KMeansTest`, `MeanTest`, `TestDataBuilder`,
  `PCAReduction.process`) rather than by compiling it, so treat it as
  best-effort until CI actually builds it.
- The `generate-golden-fixtures.yml` workflow in the Java repo — written,
  pushed, but not yet triggered (this session's GitHub App integration has
  Contents write but not Actions write on that repo — 403 on
  `workflow_dispatch`). **Owner action needed:** trigger it manually from
  the Actions tab, or grant the integration Actions write access.
- The actual fixture files (`tests/golden/fixtures/`) — don't exist yet,
  because the step above hasn't run. `test_parity.py` is fully wired to
  consume them the moment they're generated and copied in; no code changes
  needed on this side.

## A bug CI caught that local testing didn't

The first two pushes to `vtea-python` (`bce5226`, `54edb11`) both failed CI:
`ruff: command not found` in the `test-napari` job's lint step.
`vtea-napari`'s `[dev]` extra listed `pytest`/`pytest-qt`/`napari[pyqt5]`
but not `ruff` — invisible locally because my dev venv had `vtea-core[dev]`
(which does list `ruff`) installed alongside it, so `ruff` was on `PATH`
either way. CI installs each job's dependencies in isolation, so the gap
showed up immediately there.

Fixed in `1facc40` (added `ruff` to `vtea-napari`'s `dev` extra) and
confirmed by reproducing CI's exact install in a throwaway venv before
pushing, not just re-running the existing local venv.

**Lesson for later phases:** local "it passed" isn't sufficient signal by
itself — verify against CI, or at minimum a clean install, before treating
a package's test result as trustworthy. This applies more as more packages
and optional-dependency combinations get added.

## Readiness for Phase 1

Green light, with two caveats to close out in parallel rather than as a
blocker:

1. The Java-side fixture generator needs its first real compile/run
   (via the workflow, once triggered) before it can be trusted. If it
   doesn't compile cleanly, that's a quick fix against real compiler
   errors rather than a redesign — the logic was written directly against
   verified API signatures, not guessed.
2. Phase 1 (`VolumeDataset` + I/O) doesn't actually depend on the golden
   fixtures existing yet — parity validation is what the fixtures are for,
   and that's Phase 2+ territory (segmentation, measurements, clustering,
   reduction). Phase 1 can proceed now; the fixture generation should just
   land before Phase 2's segmentation/measurement work needs something to
   validate against.
