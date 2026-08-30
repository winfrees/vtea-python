"""Reviewing the association: what it was unsure about, and fixing it.

Every automated assignment is wrong somewhere. The point of keeping the
posterior on each link (see `vtea_core.objects.association`) was never the
number itself - it was to be able to put the few percent of cells worth a
person's attention in front of them, worst first, and let them settle it.

So this pane does three things and no more: say how much of the run worked,
list the links the method was least sure about, and let one be reassigned to
any of the parents that were actually considered - or to nothing. A
correction is recorded as `manual` on the link and remembered on the session,
so re-running the association step with different parameters corrects the
automated answers without quietly undoing the settled ones.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from vtea_core.objects import AssociationSet, ObjectRef, load_associations, save_associations

NO_PARENT = "(no parent)"

_COLUMNS = ("Child", "Parent", "p", "Margin", "Runner-up")

# Below this margin a link is worth looking at. Not a hard threshold on
# anything - it only decides what the list shows first.
DEFAULT_THRESHOLD = 0.9


class AssociationReviewWidget(QWidget):
    """The contested links, worst first, and a way to settle them."""

    link_selected = Signal(object, object)  # child ObjectRef, parent ObjectRef | None
    associations_changed = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._sources: dict[str, AssociationSet] = {}
        self._rows: list = []

        layout = QVBoxLayout(self)

        chooser = QHBoxLayout()
        chooser.addWidget(QLabel("Association:"))
        self.source_combo = QComboBox()
        self.source_combo.currentTextChanged.connect(lambda _name: self.refresh())
        chooser.addWidget(self.source_combo, 1)
        layout.addLayout(chooser)

        self.summary_label = QLabel("No associations yet.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        layout.addWidget(self.table, 1)

        # The correction itself: the parents this child was actually a
        # candidate for, so a reassignment is a choice between the answers
        # the evidence offered rather than free text.
        fix = QHBoxLayout()
        fix.addWidget(QLabel("Reassign to:"))
        self.parent_combo = QComboBox()
        fix.addWidget(self.parent_combo, 1)
        self.apply_button = QPushButton("Apply")
        self.apply_button.setToolTip("Record this parent as a manual decision")
        self.apply_button.clicked.connect(self.apply_reassignment)
        fix.addWidget(self.apply_button)
        layout.addLayout(fix)

        buttons = QHBoxLayout()
        save_button = QPushButton("Save…")
        save_button.setToolTip("Write the associations, corrections included, to JSON")
        save_button.clicked.connect(self.save_associations)
        buttons.addWidget(save_button)
        open_button = QPushButton("Open…")
        open_button.clicked.connect(self.load_associations)
        buttons.addWidget(open_button)
        buttons.addStretch()
        layout.addLayout(buttons)

        self._set_enabled(False)

    # -- data -------------------------------------------------------------

    def set_sources(self, sources: dict[str, AssociationSet]) -> None:
        """The association results available to review, by step name."""
        self._sources = dict(sources)
        current = self.source_combo.currentText()
        names = list(self._sources)
        if names != [self.source_combo.itemText(i) for i in range(self.source_combo.count())]:
            self.source_combo.blockSignals(True)
            self.source_combo.clear()
            self.source_combo.addItems(names)
            if current in names:
                self.source_combo.setCurrentText(current)
            self.source_combo.blockSignals(False)
        self.refresh()

    @property
    def associations(self) -> AssociationSet | None:
        return self._sources.get(self.source_combo.currentText())

    def refresh(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        associations = self.associations
        self._rows = [] if associations is None else associations.uncertain(threshold)

        if associations is None:
            self.summary_label.setText("No associations yet - run an association step.")
        elif not self._rows:
            # Worth saying explicitly: an empty list here means "nothing to
            # review", which reads the same as "nothing loaded" if it is not.
            self.summary_label.setText(f"{associations.summary()}. Nothing below the threshold.")
        else:
            self.summary_label.setText(f"{associations.summary()}. {len(self._rows)} to review:")

        self.table.setRowCount(len(self._rows))
        for row, link in enumerate(self._rows):
            runner_up = (
                f"{link.alternatives[0][0]} ({link.alternatives[0][1]:.2f})"
                if link.alternatives
                else ""
            )
            for column, text in enumerate(
                (
                    str(link.child),
                    str(link.parent),
                    f"{link.probability:.2f}",
                    f"{link.margin:.2f}",
                    runner_up,
                )
            ):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, row)
                self.table.setItem(row, column, item)
        self._set_enabled(bool(self._rows))
        self._fill_parent_choices()

    def selected_link(self):
        row = self.table.currentRow()
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def _set_enabled(self, enabled: bool) -> None:
        self.apply_button.setEnabled(enabled)
        self.parent_combo.setEnabled(enabled)

    def _fill_parent_choices(self) -> None:
        link = self.selected_link()
        self.parent_combo.clear()
        if link is None:
            return
        self.parent_combo.addItem(str(link.parent), link.parent)
        for candidate, probability in link.alternatives:
            self.parent_combo.addItem(f"{candidate} ({probability:.2f})", candidate)
        # Breaking the link is a real answer, not a failure to choose one:
        # a cytoplasm with no nucleus in the section is an ordinary result.
        self.parent_combo.addItem(NO_PARENT, None)

    def _on_row_selected(self) -> None:
        self._fill_parent_choices()
        link = self.selected_link()
        if link is not None:
            self.link_selected.emit(link.child, link.parent)

    # -- editing ----------------------------------------------------------

    def apply_reassignment(self) -> ObjectRef | None:
        """Record the chosen parent as a manual decision."""
        link = self.selected_link()
        associations = self.associations
        if link is None or associations is None:
            return None
        parent = self.parent_combo.currentData()
        if parent is None:
            associations.unassign(link.child)
        else:
            associations.set_parent(link.child, parent)
        self.refresh()
        self.associations_changed.emit()
        return parent

    # -- persistence ------------------------------------------------------

    def save_associations(self) -> None:
        associations = self.associations
        if associations is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save associations", "associations.json", "JSON (*.json)"
        )
        if path:
            save_associations(associations, path)
            self.summary_label.setText(f"Saved to {path}")

    def load_associations(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Open associations", "", "JSON (*.json)"
        )
        if not path:
            return
        loaded = load_associations(path)
        name = self.source_combo.currentText() or "associations"
        sources = dict(self._sources)
        sources[name] = loaded
        self.set_sources(sources)
        self.associations_changed.emit()
