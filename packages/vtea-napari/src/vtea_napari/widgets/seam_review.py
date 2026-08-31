"""The objects a tile boundary went through, and what to do about them.

Phase L3 reconciles objects across seams and records how, per object: which
rule decided it, how many fragments it was assembled from, and how strong
the evidence was. The few percent of objects a seam ran through are the ones
worth looking at by eye, and this is where a person looks at them.

**Reject, do not edit.** A reviewer can exclude an object from the analysis;
they cannot redraw it. Correcting a boundary means rewriting voxels in a
label array that may be 33 GB, which is the random-access editing problem
docs/LARGE_IMAGES.md deliberately puts out of scope - chunk write
amplification and undo are a project of their own. Excluding an object is a
table operation, it is recorded beside the objects the seam policy itself
dropped, and it is honest about what it did. The interface says so rather
than leaving somebody hunting for an edit tool that is intentionally absent.

Beside this, the ordinary gating machinery already works: `seam_gate` puts
a real gate on the plot over the same rows, so a reviewer can intersect it
with a size or brightness gate and open the gallery on what survives.
"""

from __future__ import annotations

import pandas as pd
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from vtea_core.gates import DEFAULT_THRESHOLD, has_seam_columns, seam_table

_COLUMNS = ("Object", "Fragments", "Rule", "Confidence")
_FIELDS = ("object_id", "n_fragments", "seam_rule", "seam_confidence")

REJECTED_REASON = "rejected in review"

NOTHING_TO_REVIEW = (
    "No seam information. This table came from an in-memory run, which has "
    "no tile boundaries to reconcile."
)


class SeamReviewWidget(QWidget):
    """Seam-crossing objects, least confident first, with a way to exclude
    one."""

    object_selected = Signal(int)
    objects_rejected = Signal(object)
    gate_requested = Signal(float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._frame: pd.DataFrame | None = None
        self._ledger = None

        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Confidence below:"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setDecimals(2)
        self.threshold.setRange(0.01, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(DEFAULT_THRESHOLD)
        self.threshold.valueChanged.connect(lambda _value: self.refresh())
        controls.addWidget(self.threshold)
        controls.addStretch()
        layout.addLayout(controls)

        self.summary_label = QLabel(NOTHING_TO_REVIEW)
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.gate_button = QPushButton("Gate these objects")
        self.gate_button.setToolTip(
            "Put a gate on the plot over the objects listed here, so they can be "
            "intersected with the gates already drawn and opened in the gallery."
        )
        self.gate_button.clicked.connect(
            lambda: self.gate_requested.emit(self.threshold.value())
        )
        buttons.addWidget(self.gate_button)
        self.reject_button = QPushButton("Reject object")
        self.reject_button.setToolTip(
            "Exclude this object from the analysis. It stays in the image - a seam "
            "cannot be redrawn here - but it is recorded as excluded and drops out of "
            "the table and of anything filtered from it."
        )
        self.reject_button.clicked.connect(self.reject_selected)
        buttons.addWidget(self.reject_button)
        buttons.addStretch()
        layout.addLayout(buttons)

        self._set_enabled(False)

    # -- data -------------------------------------------------------------

    def set_source(self, frame: pd.DataFrame | None, ledger=None) -> None:
        """The measurement table to review, and the ledger behind it."""
        self._frame = frame
        self._ledger = ledger
        self.refresh()

    @property
    def ledger(self):
        return self._ledger

    def rows(self) -> pd.DataFrame:
        """The seam objects currently listed."""
        if self._frame is None or not has_seam_columns(self._frame):
            return pd.DataFrame()
        listed = seam_table(self._frame, threshold=self.threshold.value())
        if self._ledger is not None and getattr(self._ledger, "dropped", None):
            listed = listed[~listed["object_id"].isin(set(self._ledger.dropped))]
        return listed.reset_index(drop=True)

    def refresh(self) -> None:
        listed = self.rows()
        self.table.setRowCount(len(listed))
        for row, (_index, record) in enumerate(listed.iterrows()):
            for column, name in enumerate(_FIELDS):
                value = record.get(name, "")
                text = f"{value:.2f}" if name == "seam_confidence" else str(value)
                self.table.setItem(row, column, QTableWidgetItem(text))
        self._set_enabled(bool(len(listed)))
        self.summary_label.setText(self._summary(listed))

    def _summary(self, listed: pd.DataFrame) -> str:
        if self._frame is None or not has_seam_columns(self._frame):
            return NOTHING_TO_REVIEW
        total = len(self._frame)
        parts = [
            (
                f"{len(listed):,} of {total:,} objects below "
                f"{self.threshold.value():.2f} confidence"
            )
        ]
        if self._ledger is not None:
            parts.append(f"{self._ledger.seam_exposed_fraction:.1%} touched a seam")
            if self._ledger.dropped:
                parts.append(f"{len(self._ledger.dropped):,} already excluded")
        return "; ".join(parts)

    # -- review -----------------------------------------------------------

    def selected_object(self) -> int | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return int(item.text()) if item is not None else None

    def reject_selected(self) -> int | None:
        """Exclude the selected object from the analysis.

        Recorded in the ledger beside the objects the seam policy dropped,
        with a reason that says a person did it - the same distinction
        `Association.MANUAL` draws, and for the same reason: a correction
        indistinguishable from an inference is worse than no correction.
        """
        object_id = self.selected_object()
        if object_id is None or self._ledger is None:
            return None
        self._ledger.dropped[int(object_id)] = REJECTED_REASON
        self.refresh()
        self.objects_rejected.emit(object_id)
        return object_id

    def _on_row_selected(self) -> None:
        object_id = self.selected_object()
        if object_id is not None:
            self.object_selected.emit(object_id)

    def _set_enabled(self, enabled: bool) -> None:
        # Gating needs a table with the columns on it; rejecting needs
        # somewhere to record the rejection, which is the ledger.
        self.gate_button.setEnabled(
            self._frame is not None and has_seam_columns(self._frame)
        )
        self.reject_button.setEnabled(enabled and self._ledger is not None)
