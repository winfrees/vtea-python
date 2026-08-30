"""The voxel-size control, and the dialog behind it.

A button showing what the voxel size currently is, or that nobody has said.
It reads the value off the napari layer where the file recorded one, and
asks otherwise - because the alternative, quietly measuring in voxels, gets
anisotropy wrong in a way that looks entirely plausible on screen.

The reason this is a button rather than a passive label: `layer.scale` is
`(1, 1, 1)` both when a file says "one unit per voxel" and when it says
nothing at all, so "unknown" is a state the user has to be able to resolve.
"""

from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from vtea_core.data import DEFAULT_UNIT, FROM_USER, Spacing, spacing_from_scale

# Axis names for the usual orderings, so the dialog says "Z" rather than
# "axis 0" where it can.
_AXIS_NAMES = {2: ("Y", "X"), 3: ("Z", "Y", "X"), 4: ("T", "Z", "Y", "X")}

UNKNOWN_TEXT = "Voxel size: not set"


def axis_names(ndim: int) -> tuple[str, ...]:
    return _AXIS_NAMES.get(ndim) or tuple(f"axis {index}" for index in range(ndim))


class SpacingDialog(QDialog):
    """Ask for the physical size of one voxel along each axis."""

    def __init__(self, spacing: Spacing, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Voxel size")
        layout = QVBoxLayout(self)

        explanation = QLabel(
            "The physical size of one voxel. Distances, dilation thicknesses "
            "and volumes are wrong without it - and wrong in a way that looks "
            "plausible, since z-steps are usually several times the lateral "
            "pixel size."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.spins: list[QDoubleSpinBox] = []
        names = axis_names(len(spacing.values))
        for name, value in zip(names, spacing.values, strict=False):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name}:"))
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(0.0001, 10000.0)
            spin.setValue(float(value))
            row.addWidget(spin, 1)
            layout.addLayout(row)
            self.spins.append(spin)

        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Unit:"))
        self.unit_edit = QLineEdit(spacing.unit or DEFAULT_UNIT)
        unit_row.addWidget(self.unit_edit, 1)
        layout.addLayout(unit_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def spacing(self) -> Spacing:
        """What was entered. Marked as coming from the user, so it is never
        mistaken for something the file supplied."""
        return Spacing(
            values=tuple(spin.value() for spin in self.spins),
            unit=self.unit_edit.text().strip() or DEFAULT_UNIT,
            source=FROM_USER,
        )


class SpacingControl(QPushButton):
    """Shows the voxel size and opens the dialog to change it."""

    spacing_changed = Signal(object)  # vtea_core.data.Spacing

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._spacing: Spacing | None = None
        self.clicked.connect(self.edit)
        self.refresh()

    def spacing(self) -> Spacing | None:
        return self._spacing

    def set_spacing(self, spacing: Spacing | None, announce: bool = True) -> None:
        self._spacing = spacing
        self.refresh()
        if announce:
            self.spacing_changed.emit(spacing)

    def read_from_layer(self, layer) -> None:
        """Take the voxel size from a napari layer's scale.

        Only when nothing is set yet, or what is set also came from the
        image: a value the user typed outlives switching between layers,
        which is the case where re-reading would be most annoying.
        """
        if layer is None or not hasattr(layer, "scale"):
            return
        if self._spacing is not None and self._spacing.source == FROM_USER:
            return
        self.set_spacing(spacing_from_scale(layer.scale))

    def refresh(self) -> None:
        if self._spacing is None or not self._spacing.is_known:
            self.setText(UNKNOWN_TEXT)
            self.setToolTip(
                "The image didn't record a voxel size. Click to set it - "
                "distances and volumes need it."
            )
            # Not an error, but not something to leave unnoticed either.
            self.setStyleSheet("QPushButton { color: #b06000; }")
            return
        self.setText(f"Voxel: {self._spacing.describe()}")
        origin = "from the image" if self._spacing.source == "metadata" else "set by hand"
        self.setToolTip(f"Physical size of one voxel ({origin}). Click to change.")
        self.setStyleSheet("")

    def edit(self) -> bool:
        """Open the dialog. Returns whether a spacing was accepted."""
        current = self._spacing or Spacing.unknown()
        dialog = SpacingDialog(current, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        self.set_spacing(dialog.spacing())
        return True

    def ensure_known(self) -> bool:
        """Ask for a voxel size if there isn't one, and report whether there
        is one now. Called before running anything whose result depends on
        it, so the prompt arrives at the moment it matters rather than as a
        warning nobody reads."""
        if self._spacing is not None and self._spacing.is_known:
            return True
        self.edit()
        return self._spacing is not None and self._spacing.is_known
