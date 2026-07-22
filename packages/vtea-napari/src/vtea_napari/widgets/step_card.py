"""A single pipeline step rendered as a card: position, name, parameter
summary, Edit/Delete buttons.

Renders the same information vtea.protocol.blockstepgui's
AbstractMicroBlockStepGUI-derived cards showed (position number, headline,
comment, Edit/Delete) as a plain qtpy widget instead of hand-built Swing.
Thumbnail previews aren't included in this first pass - see PORT_PLAN.md.
"""

from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from vtea_core.workflow import Step


def summarize_params(step: Step) -> str:
    if not step.params:
        return "(default parameters)"
    return ", ".join(f"{key}={value}" for key, value in step.params.items())


class StepCardWidget(QFrame):
    """Emits edit_requested/delete_requested; the parent widget owns the Pipeline."""

    edit_requested = Signal()
    delete_requested = Signal()

    def __init__(self, position: int, step: Step, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        outer = QHBoxLayout(self)

        position_label = QLabel(f"{position}.")
        outer.addWidget(position_label)

        text_column = QVBoxLayout()
        headline = QLabel(f"{step.category}.{step.function_name}")
        headline.setStyleSheet("font-weight: bold;")
        text_column.addWidget(headline)
        comment_text = step.comment if step.comment else summarize_params(step)
        text_column.addWidget(QLabel(comment_text))
        outer.addLayout(text_column)

        outer.addStretch()

        edit_button = QPushButton("Edit")
        edit_button.clicked.connect(self.edit_requested.emit)
        outer.addWidget(edit_button)

        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self.delete_requested.emit)
        outer.addWidget(delete_button)
