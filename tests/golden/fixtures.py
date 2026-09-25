"""Loaders for golden-dataset fixtures produced by GoldenFixtureGenerator.java.

See README.md for how fixtures are generated and where to place them.
Every loader raises a clear, actionable FileNotFoundError if the fixture
directory hasn't been populated yet, rather than failing obscurely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).parent / "fixtures"
# The sample TIFFs the Java generator segmented (they live at the Java repo's
# root, ~46 MB together, so they are not committed here either).
DATA_DIR = Path(__file__).parent / "data"

_MISSING_MSG = (
    "Golden fixtures not found at {path}. Generate them by running the "
    "'Generate golden-dataset fixtures' GitHub Actions workflow in the "
    "volumetric-tissue-exploration-analysis repo, then copy the "
    "downloaded artifact's contents into tests/golden/fixtures/. "
    "See tests/golden/README.md."
)


def fixtures_available() -> bool:
    return FIXTURES_DIR.exists() and any(FIXTURES_DIR.iterdir())


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(_MISSING_MSG.format(path=path))
    return path


def load_measurements(dataset_basename: str) -> pd.DataFrame:
    """Per-object measurement table from the Java SingleThreshold+measurements run."""
    path = _require(FIXTURES_DIR / f"{dataset_basename}_measurements.csv")
    return pd.read_csv(path)


def load_label_mask(dataset_basename: str) -> np.ndarray:
    """Label mask (ZYX, uint16) from the Java SingleThreshold segmentation run."""
    import tifffile

    path = _require(FIXTURES_DIR / f"{dataset_basename}_segmentation_singlethreshold.tif")
    return tifffile.imread(path)


def load_metadata(dataset_basename: str) -> dict[str, str]:
    """key=value pairs from the generator's <dataset>_metadata.txt."""
    path = _require(FIXTURES_DIR / f"{dataset_basename}_metadata.txt")
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    return {key.strip(): value.strip() for key, value in pairs}


def load_source_channel(dataset_basename: str, channel: int = 0) -> np.ndarray:
    """One channel (ZYX) of the sample TIFF the Java fixtures were made from."""
    from vtea_core.io import read_tiff

    path = DATA_DIR / f"{dataset_basename}.tif"
    if not path.exists():
        raise FileNotFoundError(
            f"Sample image not found at {path}. Copy {path.name} from the root of the "
            "volumetric-tissue-exploration-analysis repo into tests/golden/data/."
        )
    return np.asarray(read_tiff(path).channel(channel))


def load_synthetic_clustering_input() -> pd.DataFrame:
    return pd.read_csv(_require(FIXTURES_DIR / "synthetic_clustering_input.csv"))


def load_synthetic_kmeans_k3() -> pd.DataFrame:
    return pd.read_csv(_require(FIXTURES_DIR / "synthetic_clustering_kmeans_k3.csv"))


def load_synthetic_pca() -> pd.DataFrame:
    return pd.read_csv(_require(FIXTURES_DIR / "synthetic_clustering_pca.csv"))
