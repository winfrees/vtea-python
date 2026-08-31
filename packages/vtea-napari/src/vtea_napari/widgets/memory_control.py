"""The memory budget, and the tile plan it implies.

A button showing how much memory this run may use and what that divides the
data into - "512x512x512 tiles, 216 of them, bounded by watershed_split_1" -
with a dialog to change it.

Why a control rather than a setting in a file. A user running a 40 GB
acquisition on a laptop and one running it on a 512 GB workstation are doing
the same analysis at very different tile sizes, and the tile size is not a
detail: it decides how long the run takes, how much of the data is read more
than once, and - through the seam policy - which objects get reconciled at
all. Somebody who cannot see that number cannot tell a slow run from a stuck
one, and somebody who cannot change it cannot trade time for memory on the
machine they actually have.

The detected value is the default and is usually right. What the dialog is
for is the case detection cannot see: a job sharing a node with three other
jobs, or a card with a viewer already holding two gigabytes of textures.
"""

from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from vtea_core.blocked import (
    DEFAULT_POLICY,
    MemoryBudget,
    detect_memory_budget,
    format_bytes,
    parse_size,
)

GIB = 1024**3


class MemoryDialog(QDialog):
    """Set the budget by hand, or put it back to what was detected."""

    def __init__(self, budget: MemoryBudget, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Memory budget")
        self._detected = detect_memory_budget()

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"Detected: {self._detected.describe()}.\n"
                "Lower this when something else is using the machine; raise it to\n"
                "process larger tiles, which is usually faster."
            )
        )

        row = QHBoxLayout()
        row.addWidget(QLabel("Budget (GiB):"))
        self.amount = QDoubleSpinBox()
        self.amount.setDecimals(1)
        self.amount.setRange(0.1, 4096.0)
        self.amount.setValue(budget.total_bytes / GIB)
        row.addWidget(self.amount)
        layout.addLayout(row)

        fraction_row = QHBoxLayout()
        fraction_row.addWidget(QLabel("Usable fraction:"))
        self.fraction = QDoubleSpinBox()
        self.fraction.setDecimals(2)
        self.fraction.setRange(0.05, 1.0)
        self.fraction.setSingleStep(0.05)
        self.fraction.setValue(budget.fraction)
        fraction_row.addWidget(self.fraction)
        layout.addLayout(fraction_row)
        layout.addWidget(
            QLabel(
                "The rest is the interpreter, napari's own copies of what is on\n"
                "screen, and the fact that peak usage exceeds the sum of what you\n"
                "meant to allocate."
            )
        )

        self.use_detected = QCheckBox("Use the detected value")
        self.use_detected.setChecked(not budget.is_measured or budget.source != "user")
        self.use_detected.toggled.connect(self._on_detected_toggled)
        layout.addWidget(self.use_detected)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._on_detected_toggled(self.use_detected.isChecked())

    def _on_detected_toggled(self, checked: bool) -> None:
        self.amount.setEnabled(not checked)
        if checked:
            self.amount.setValue(self._detected.total_bytes / GIB)

    def budget(self) -> MemoryBudget:
        if self.use_detected.isChecked():
            return detect_memory_budget(fraction=self.fraction.value())
        return MemoryBudget(
            total_bytes=int(self.amount.value() * GIB),
            fraction=self.fraction.value(),
            source="user",
        )


class MemoryControl(QWidget):
    """A button showing the budget and what it divides the data into."""

    budget_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._budget = detect_memory_budget()
        self._plan = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton()
        self.button.setToolTip("How much memory this run may use, and the resulting tiles")
        self.button.clicked.connect(self.edit)
        layout.addWidget(self.button)
        self._refresh()

    def budget(self) -> MemoryBudget:
        return self._budget

    def set_budget(self, budget: MemoryBudget) -> None:
        self._budget = budget
        self._refresh()
        self.budget_changed.emit(budget)

    def set_plan(self, plan) -> None:
        """Show what the budget actually divided the data into.

        Set after a plan is computed rather than guessed from the budget:
        the tile size depends on the protocol's heaviest step, and saying so
        is the difference between a number and an explanation.
        """
        self._plan = plan
        self._refresh()

    def edit(self) -> None:
        dialog = MemoryDialog(self._budget, self)
        if dialog.exec_():
            self.set_budget(dialog.budget())

    def describe(self) -> str:
        if self._plan is None:
            return f"Memory: {format_bytes(self._budget.usable_bytes)} usable"
        if self._plan.is_single_tile:
            return f"Memory: {format_bytes(self._budget.usable_bytes)} - fits in one piece"
        tile = "x".join(str(size) for size in self._plan.tile)
        return f"Memory: {self._plan.n_tiles:,} tiles of {tile}"

    def _refresh(self) -> None:
        self.button.setText(self.describe())
        tooltip = self._budget.describe()
        if self._plan is not None:
            tooltip = self._plan.describe()
        self.button.setToolTip(tooltip)


__all__ = ["DEFAULT_POLICY", "MemoryControl", "MemoryDialog", "parse_size"]
