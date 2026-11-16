"""Run the reference R packages: Bioconductor `slingshot` and `tradeSeq`.

The Python ports (`slingshot.slingshot`, `tradeseq.fit_gam` and the tests)
follow the R packages; this module runs the R packages themselves, for users
who want the reference implementation and for checking the ports against it
(`validate.compare_with_r`). It needs R with slingshot and tradeSeq installed
(`BiocManager::install(c("slingshot", "tradeSeq"))`); `r_status()` says what
is there. Data go to R as CSV files in a temporary folder and come back the
same way: no rpy2.

`IBIF_RSCRIPT` / `VTEA_RSCRIPT` point to an Rscript that is not on the PATH
(e.g. a conda environment's).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from vtea_core.trajectory.slingshot import Trajectory, cluster_distances


def rscript() -> str | None:
    """The Rscript to use, or None."""
    for var in ("VTEA_RSCRIPT", "IBIF_RSCRIPT"):
        path = os.environ.get(var)
        if path and Path(path).exists():
            return path
    return shutil.which("Rscript")


@lru_cache(maxsize=4)
def _status(path):
    if path is None:
        return {"rscript": None, "slingshot": None, "tradeSeq": None}
    code = (
        'v <- function(p) if (requireNamespace(p, quietly = TRUE)) as.character(packageVersion(p)) else "";'
        'cat(R.version$major, ".", R.version$minor, "\\n", v("slingshot"), "\\n", v("tradeSeq"), "\\n", sep = "")'
    )
    try:
        out = subprocess.run(
            [path, "-e", code], capture_output=True, text=True, timeout=120, check=False
        )
        lines = out.stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return {"rscript": None, "slingshot": None, "tradeSeq": None}
    lines += [""] * (3 - len(lines))
    return {
        "rscript": path,
        "R": lines[0] or None,
        "slingshot": lines[1] or None,
        "tradeSeq": lines[2] or None,
    }


def r_status() -> dict:
    """{"rscript", "R", "slingshot", "tradeSeq"}: paths and versions, None when
    missing."""
    return dict(_status(rscript()))


def r_available(*packages: str) -> bool:
    status = r_status()
    return status["rscript"] is not None and all(
        status.get(p) for p in packages or ("slingshot", "tradeSeq")
    )


def _run(script: str, folder: Path, timeout: int):
    path = rscript()
    if path is None:
        raise RuntimeError("Rscript was not found (install R, or set VTEA_RSCRIPT)")
    (folder / "run.R").write_text(script)
    out = subprocess.run(
        [path, str(folder / "run.R")],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=folder,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(f"R failed:\n{out.stderr[-4000:]}")
    return out


def _r_str(value):
    return json.dumps(str(value))


SLINGSHOT_R = r"""
suppressPackageStartupMessages(library(slingshot))
X <- as.matrix(read.csv("points.csv", check.names = FALSE))
cl <- as.character(read.csv("labels.csv", colClasses = "character")$label)
keep <- !is.na(cl) & cl != ""
set.seed(SEED)
sds <- slingshot(X[keep, , drop = FALSE], clusterLabels = cl[keep], start.clus = START, end.clus = ENDS,
                 dist.method = DIST, shrink = SHRINK, extend = EXTEND, reweight = REWEIGHT,
                 reassign = REASSIGN, thresh = THRESH, maxit = MAXIT, stretch = STRETCH,
                 approx_points = APPROX)
pt <- slingPseudotime(sds, na = TRUE)
w <- slingCurveWeights(sds)
out_pt <- matrix(NA_real_, nrow(X), ncol(pt)); out_pt[keep, ] <- pt
out_w <- matrix(0, nrow(X), ncol(w)); out_w[keep, ] <- w
write.csv(out_pt, "pseudotime.csv", row.names = FALSE)
write.csv(out_w, "weights.csv", row.names = FALSE)
lin <- slingLineages(sds)
writeLines(vapply(lin, function(l) paste(l, collapse = "\t"), ""), "lineages.txt")
crv <- slingCurves(sds)
for (i in seq_along(crv)) write.csv(crv[[i]]$s[crv[[i]]$ord, , drop = FALSE], sprintf("curve%d.csv", i),
                                    row.names = FALSE)
"""


def slingshot_r(
    points,
    labels,
    start,
    *,
    ends=None,
    distance="slingshot",
    shrink=1.0,
    extend="y",
    reweight=True,
    reassign=True,
    thresh=0.001,
    maxit=15,
    stretch=2.0,
    approx_points=150,
    seed=0,
    timeout=3600,
) -> Trajectory:
    """R slingshot on the same inputs as `slingshot.slingshot`, returned as a
    `Trajectory` (lineages and curves from R; cluster centres and distances
    computed here)."""
    points = np.asarray(points, float)
    labels = np.asarray(labels, dtype=object)
    names = ["" if pd.isna(v) else str(v) for v in labels]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        pd.DataFrame(points, columns=[f"d{i + 1}" for i in range(points.shape[1])]).to_csv(
            folder / "points.csv", index=False
        )
        pd.DataFrame({"label": names}).to_csv(folder / "labels.csv", index=False)
        ends_r = "NULL" if not ends else "c(" + ", ".join(_r_str(e) for e in ends) + ")"
        script = SLINGSHOT_R
        for key, value in {
            "SEED": str(int(seed)),
            "START": _r_str(start),
            "ENDS": ends_r,
            "DIST": _r_str("slingshot" if distance == "slingshot" else "euclidean"),
            "SHRINK": repr(float(shrink)),
            "EXTEND": _r_str(extend),
            "REWEIGHT": "TRUE" if reweight else "FALSE",
            "REASSIGN": "TRUE" if reassign else "FALSE",
            "THRESH": repr(float(thresh)),
            "MAXIT": str(int(maxit)),
            "STRETCH": repr(float(stretch)),
            "APPROX": str(int(approx_points)) if approx_points else "FALSE",
        }.items():
            script = script.replace(key, value)
        _run(script, folder, timeout)
        pt = pd.read_csv(folder / "pseudotime.csv").to_numpy(float)
        w = pd.read_csv(folder / "weights.csv").to_numpy(float)
        lineages = [
            line.split("\t") for line in (folder / "lineages.txt").read_text().splitlines() if line
        ]
        curves = [
            pd.read_csv(folder / f"curve{i + 1}.csv").to_numpy(float) for i in range(len(lineages))
        ]
    clusters = sorted({n for n in names if n})
    ok = np.array([bool(n) for n in names])
    lab = np.array(names, dtype=object)
    centers = np.vstack([points[lab == c].mean(axis=0) for c in clusters])
    edges = sorted({(lin[i], lin[i + 1]) for lin in lineages for i in range(len(lin) - 1)})
    return Trajectory(
        clusters=clusters,
        centers=centers,
        edges=edges,
        start=start,
        ends=list(ends or []),
        lineages=lineages,
        curves=curves,
        pseudotime=pt,
        weights=np.where(np.isfinite(pt), w, 0.0),
        distances=cluster_distances(points[ok], lab[ok], clusters, distance=distance),
    )


TRADESEQ_R = r"""
suppressPackageStartupMessages({library(tradeSeq); library(mgcv)})
Y <- t(as.matrix(read.csv("values.csv", check.names = FALSE)))
pt <- as.matrix(read.csv("pseudotime.csv")); w <- as.matrix(read.csv("weights.csv"))
pt[is.na(pt)] <- 0
off <- if (file.exists("offset.csv")) read.csv("offset.csv")$offset else NULL
set.seed(SEED)
args <- list(counts = Y, pseudotime = pt, cellWeights = w, nknots = NKNOTS, sce = FALSE, verbose = FALSE)
if (!is.null(off)) args$offset <- off
if (FAMILY != "nb") args$family <- FAMILY
models <- do.call(fitGAM, args)
saveRDS(models, "models.rds")
write.csv(associationTest(models, global = TRUE, lineages = TRUE), "association.csv")
write.csv(startVsEndTest(models, global = TRUE, lineages = TRUE), "start_vs_end.csv")
if (ncol(pt) > 1) {
  write.csv(diffEndTest(models, global = TRUE, pairwise = TRUE), "diff_end.csv")
  write.csv(patternTest(models, global = TRUE, pairwise = TRUE), "pattern.csv")
}
"""


def tradeseq_r(
    pseudotime, weights, values, *, nknots=6, family="gaussian", offset=None, seed=0, timeout=7200
) -> dict:
    """R tradeSeq `fitGAM` and its tests on the same inputs as `fit_gam`.
    Returns {"association", "start_vs_end", "diff_end", "pattern"} tables (as R
    writes them, one row per feature). Give one-hot `weights` (each object on
    one lineage) to compare with Python on the same lineage assignment; tradeSeq
    otherwise draws the assignment with R's random numbers."""
    frame = values if isinstance(values, pd.DataFrame) else pd.DataFrame(np.asarray(values, float))
    pt = np.asarray(pseudotime, float)
    w = np.asarray(weights, float).reshape(pt.shape)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        frame.to_csv(folder / "values.csv", index=False)
        pd.DataFrame(pt, columns=[f"Lineage{j + 1}" for j in range(pt.shape[1])]).to_csv(
            folder / "pseudotime.csv", index=False
        )
        pd.DataFrame(w, columns=[f"Lineage{j + 1}" for j in range(pt.shape[1])]).to_csv(
            folder / "weights.csv", index=False
        )
        if offset is not None:
            pd.DataFrame({"offset": np.asarray(offset, float)}).to_csv(
                folder / "offset.csv", index=False
            )
        script = (
            TRADESEQ_R.replace("SEED", str(int(seed)))
            .replace("NKNOTS", str(int(nknots)))
            .replace("FAMILY", _r_str(family))
        )
        _run(script, folder, timeout)
        out = {}
        for name in ("association", "start_vs_end", "diff_end", "pattern"):
            path = folder / f"{name}.csv"
            if path.exists():
                out[name] = pd.read_csv(path, index_col=0)
    return out
