"""Deep-learning-based segmentation: Cellpose.

Ports vtea.objects.Segmentation.CellposeSegmentation (and its Chunked
variant) from the Java codebase. There it ran over a Py4J subprocess bridge
to python/cellpose_server.py; here it's an ordinary in-process `cellpose`
import, consistent with every other function in this module - see
PORT_PLAN.md's "Why deep learning isn't a separate module".

This module has no import-time dependency on `torch`/`cellpose` (the
`cellpose` package is only imported inside cellpose_segmentation(), and only
when the caller doesn't supply their own model), so the rest of the test
suite doesn't need the `deeplearning` extra installed.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class _CellposeModelLike(Protocol):
    def eval(self, x: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, Any, Any]: ...


def cellpose_segmentation(
    volume: np.ndarray,
    *,
    model: _CellposeModelLike | None = None,
    pretrained_model: str = "cpsam_v2",
    gpu: bool = False,
    do_3D: bool = False,
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
) -> np.ndarray:
    """Segments a single-channel intensity volume with Cellpose.

    volume: (Z, Y, X) for a 3D stack, or (Y, X) for a single 2D image.

    model: an existing cellpose.models.CellposeModel-like object (must
        expose `.eval(x, **kwargs) -> (masks, flows, styles)`, matching
        cellpose's own return convention); pass one in to reuse a loaded
        model across calls, or to inject a stub for testing. If None, a
        real `cellpose.models.CellposeModel` is constructed from
        `pretrained_model`/`gpu` (this is the only place `cellpose` gets
        imported).

    Cellpose 4.x's models require 3-channel input ("Images must have 3
    channels" per CellposeModel.eval's docstring - its default model is a
    SAM-based foundation model, not the single/dual-channel U-Nets older
    Cellpose versions used). `volume` is replicated across a new trailing
    channel axis before being passed to the model, since single-channel
    fluorescence data has no real second/third channel to provide.

    Returns an integer label array the same shape as `volume` (0 =
    background), matching this module's other segmentation functions.
    """
    if model is None:
        from cellpose.models import CellposeModel

        model = CellposeModel(pretrained_model=pretrained_model, gpu=gpu)

    stacked = np.stack([volume, volume, volume], axis=-1)
    eval_kwargs: dict[str, Any] = {
        "channel_axis": -1,
        "do_3D": do_3D,
        "diameter": diameter,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
    }
    if do_3D:
        eval_kwargs["z_axis"] = 0

    masks, _flows, _styles = model.eval(stacked, **eval_kwargs)
    return np.asarray(masks)
