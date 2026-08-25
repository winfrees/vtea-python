"""A single pipeline step rendered as a card: thumbnail, position, name,
parameter summary, Edit/Delete buttons.

Renders the same information vtea.protocol.blockstepgui's
AbstractMicroBlockStepGUI-derived cards showed (position number, headline,
comment, Edit/Delete) as a plain qtpy widget instead of hand-built Swing,
plus a thumbnail preview of that step's last-run output (see
protocol_builder.py's run_pipeline(), which threads a Pipeline.run()
result's per-step arrays back onto each card via the shared
thumbnail.array_to_pixmap() helper the gallery view also uses).
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from vtea_core.workflow import Step

from vtea_napari.widgets.thumbnail import array_to_pixmap, max_projection

_THUMBNAIL_SIZE = 48


def summarize_params(step: Step) -> str:
    if not step.params:
        return "(default parameters)"
    return ", ".join(f"{key}={value}" for key, value in step.params.items())


def summarize_channel(step: Step) -> str:
    """Which channel this step runs on, shown on the card so a per-step
    channel choice is visible without opening the Edit dialog."""
    return "all channels" if step.channel is None else f"channel {step.channel}"


class StepCardWidget(QFrame):
    """Emits edit_requested/delete_requested; the parent widget owns the Pipeline."""

    edit_requested = Signal()
    delete_requested = Signal()

    def __init__(self, position: int, step: Step, parent=None, thumbnail: np.ndarray | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        outer = QHBoxLayout(self)

        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(_THUMBNAIL_SIZE, _THUMBNAIL_SIZE)
        self.set_thumbnail(thumbnail)
        outer.addWidget(self.thumbnail_label)

        position_label = QLabel(f"{position}.")
        outer.addWidget(position_label)

        text_column = QVBoxLayout()
        headline = QLabel(f"{step.category}.{step.function_name}")
        headline.setStyleSheet("font-weight: bold;")
        text_column.addWidget(headline)
        comment_text = step.comment if step.comment else summarize_params(step)
        text_column.addWidget(QLabel(comment_text))
        self.channel_label = QLabel(summarize_channel(step))
        self.channel_label.setStyleSheet("color: gray;")
        text_column.addWidget(self.channel_label)
        outer.addLayout(text_column)

        outer.addStretch()

        edit_button = QPushButton("Edit")
        edit_button.clicked.connect(self.edit_requested.emit)
        outer.addWidget(edit_button)

        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.delete_requested.emit)
        outer.addWidget(delete_button)

    def set_thumbnail(self, array: np.ndarray | None) -> None:
        """Shows a preview of a 2D+ numeric or boolean array (e.g. this
        step's last-run output); clears the preview for anything else
        (None, a scalar, a DataFrame, a fitted model, ...)."""
        previewable = isinstance(array, np.ndarray) and array.ndim >= 2 and (
            array.dtype == bool or np.issubdtype(array.dtype, np.number)
        )
        if not previewable:
            self.thumbnail_label.clear()
            return
        pixmap = array_to_pixmap(max_projection(array), size=_THUMBNAIL_SIZE)
        self.thumbnail_label.setPixmap(pixmap)
