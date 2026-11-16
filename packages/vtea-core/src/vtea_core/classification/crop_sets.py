"""Training sets of object crops exported by VTEA, as NephNet read them.

The Java VTEA exports labelled nuclei for NephNet as one CSV file per class.
Each row is one object: its class (column 0), its object id (column 1), then
the voxels of its crop, flattened Z-Y-X (7 x 32 x 32 = 7,168 values for
NephNet3D), written as signed bytes (-128..127 for 128..255). NephNet's
`utils/toHDF5.py` (ntalukde/NephNet) turned these into an HDF5 file; this
reads the CSVs directly, so existing VTEA training sets can be used without
re-exporting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NEPHNET_CROP = (7, 32, 32)


def read_vtea_crops(path, shape=NEPHNET_CROP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(crops (N, Z, Y, X) uint8, classes (N,), object ids (N,)) from one VTEA crop
    CSV, or from every CSV in a folder (in name order).

    Signed bytes are read back as unsigned (v + 256 for v < 0). A row whose
    voxel count is not prod(`shape`) raises ValueError, naming the file."""
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"no CSV files in {path}")
    n_vox = int(np.prod(shape))
    crops, classes, ids = [], [], []
    for file in files:
        table = pd.read_csv(file, header=None)
        if table.shape[1] - 2 != n_vox:
            raise ValueError(
                f"{file.name}: rows have {table.shape[1] - 2} voxels, expected {n_vox} "
                f"for crops of {shape}"
            )
        voxels = table.iloc[:, 2:].to_numpy(dtype=np.int64)
        voxels = np.where(voxels < 0, voxels + 256, voxels)
        if voxels.min(initial=0) < 0 or voxels.max(initial=0) > 255:
            raise ValueError(f"{file.name}: voxel values outside the byte range")
        crops.append(voxels.astype(np.uint8).reshape(len(table), *shape))
        classes.append(table.iloc[:, 0].to_numpy())
        ids.append(table.iloc[:, 1].to_numpy(dtype=np.int64))
    return np.concatenate(crops), np.concatenate(classes), np.concatenate(ids)


def write_vtea_crops(path, crops, classes, ids) -> Path:
    """Write crops in VTEA's export format (one CSV per class, signed bytes), the
    inverse of `read_vtea_crops`; for tests and for sharing training sets."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    crops = np.asarray(crops)
    classes, ids = np.asarray(classes), np.asarray(ids, dtype=np.int64)
    for value in pd.unique(classes):
        rows = classes == value
        voxels = crops[rows].reshape(rows.sum(), -1).astype(np.int64)
        voxels = np.where(voxels > 127, voxels - 256, voxels)
        table = pd.DataFrame(np.column_stack([classes[rows].astype(object), ids[rows], voxels]))
        table.to_csv(path / f"class_{value}.csv", header=False, index=False)
    return path
