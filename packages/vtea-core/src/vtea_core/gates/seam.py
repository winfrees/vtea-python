"""A ready-made gate over the objects a tile boundary went through.

Phase L3 joins three columns onto the measurement table - `n_fragments`,
`seam_rule` and `seam_confidence` - so that a seam-crossing object is
selectable with the gating machinery that already exists. That claim is
true, and it is also useless to anybody who does not know the columns are
there. This is the shortcut: one call, a real `Gate`, and everything
downstream - membership, the gallery, the highlight on the image, saving to
JSON - treats it as the ordinary gate it is.

Deliberately a gate and not a special selection mode. A reviewer who wants
to narrow it further should be able to intersect it with a size gate or a
brightness gate, and a gate is the thing that composes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vtea_core.gates.gate import Gate
from vtea_core.gates.polygon import rectangle_vertices

# The columns LabelLedger.to_frame contributes, and what a seam gate is
# drawn over.
CONFIDENCE_COLUMN = "seam_confidence"
FRAGMENTS_COLUMN = "n_fragments"

# Below this, an object is worth a person's attention. An object no seam
# went near scores 1.0, so anything under 1 is already a seam object; the
# default is lower than that because the interesting ones are the weakly
# joined rather than merely the joined.
DEFAULT_THRESHOLD = 0.8

SEAM_GATE_NAME = "Seam objects"
# Amber, to read as a review gate rather than as one of the analysis gates
# a user drew themselves.
SEAM_GATE_COLOR = "#f0a500"


def has_seam_columns(frame: pd.DataFrame) -> bool:
    """Whether this table came from a blocked run at all.

    An in-memory run has no seams and no such columns, and offering a seam
    gate there would be offering to select nothing.
    """
    return CONFIDENCE_COLUMN in frame.columns and FRAGMENTS_COLUMN in frame.columns


def seam_gate(
    frame: pd.DataFrame,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    name: str = SEAM_GATE_NAME,
    parent_id: str | None = None,
) -> Gate:
    """A gate selecting the objects a seam ran through and left uncertain.

    Drawn over confidence against fragment count, which is the pair that
    makes the population legible: a cluster at (1.0, 1) is everything that
    was never cut, and what a reviewer wants is everything else.

    The upper bound on fragments comes from the data rather than from a
    constant, so the rectangle actually contains the rows it is meant to -
    a fixed ceiling would silently miss the vessel that ended up in nine
    tiles, which is exactly the object worth looking at.
    """
    if not has_seam_columns(frame):
        raise ValueError(
            f"this table has no seam columns, so nothing was reconciled across tiles - "
            f"expected {CONFIDENCE_COLUMN!r} and {FRAGMENTS_COLUMN!r}, found "
            f"{list(frame.columns)}"
        )
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    fragments = pd.to_numeric(frame[FRAGMENTS_COLUMN], errors="coerce")
    ceiling = float(np.nanmax(fragments)) if len(fragments) else 1.0
    return Gate(
        name=name,
        x_axis=CONFIDENCE_COLUMN,
        y_axis=FRAGMENTS_COLUMN,
        # From just below zero, so an object flagged with confidence 0.0 -
        # one no tile contained - is inside the gate rather than on its edge.
        vertices=rectangle_vertices(-0.01, 0.0, float(threshold), max(ceiling, 1.0) + 0.5),
        parent_id=parent_id,
        color=SEAM_GATE_COLOR,
    )


def seam_table(frame: pd.DataFrame, *, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """The seam-crossing objects, least confident first.

    The same rows the gate selects, as a table to read rather than a shape
    to draw - what a review pane shows beside the plot.
    """
    if not has_seam_columns(frame):
        return pd.DataFrame()
    columns = [
        column
        for column in ("object_id", FRAGMENTS_COLUMN, "seam_rule", CONFIDENCE_COLUMN)
        if column in frame.columns
    ]
    confidence = pd.to_numeric(frame[CONFIDENCE_COLUMN], errors="coerce")
    selected = frame.loc[confidence <= threshold, columns]
    return selected.sort_values(CONFIDENCE_COLUMN).reset_index(drop=True)
