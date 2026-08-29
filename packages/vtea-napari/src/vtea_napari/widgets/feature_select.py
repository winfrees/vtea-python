"""Pick which measured features a clustering or reduction step works on.

A protocol that measures seven properties across four channels already has
28 features, and the derived ones add more. Scrolling a flat checklist that
long to find the six you want is the wrong shape of control, so this pairs
the checklist with a filter box and All/None/Invert over *what the filter is
currently showing* - which is what makes "select every mean_ch2 feature" one
gesture rather than twelve.

Each row's tooltip carries that feature's provenance from the
vtea_core.measurements.FeatureCatalog: what was measured, on which channel
and segmentation, by which step. The same record is what gets saved, so what
you see when choosing is what a reader sees later.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Tall enough to show a useful run of features without the dialog needing to
# be resized; the list scrolls past that.
MINIMUM_LIST_HEIGHT = 220


def describe_feature(descriptor) -> str:
    """A one-line provenance summary for a feature's tooltip."""
    if descriptor is None:
        return "No recorded provenance for this feature."
    parts = [descriptor.measurement or descriptor.name]
    if descriptor.channel is not None:
        parts.append(f"channel {descriptor.channel}")
    if descriptor.segmentation:
        parts.append(f"on {descriptor.segmentation}")
    if descriptor.produced_by:
        parts.append(f"by {descriptor.produced_by}")
    if descriptor.source_features:
        parts.append(f"from {len(descriptor.source_features)} feature(s)")
    if descriptor.units:
        parts.append(f"[{descriptor.units}]")
    return " · ".join(parts)


class FeatureSelectWidget(QWidget):
    """A filterable checklist of feature names.

    `selected()` returns the checked names in the order they were offered;
    an empty return means "every feature", which is also what an all-checked
    list means - see `selected_or_all`.
    """

    selection_changed = Signal()

    def __init__(
        self,
        features=(),
        selected=(),
        catalog=None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._features: list[str] = list(features)
        self._catalog = catalog

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("e.g. mean, _ch2, pca")
        self.filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_edit, 1)
        root.addLayout(filter_row)

        self.list = QListWidget()
        self.list.setMinimumHeight(MINIMUM_LIST_HEIGHT)
        self.list.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.list, 1)

        button_row = QHBoxLayout()
        for text, handler, tip in (
            ("All", self.select_all, "Check every feature the filter is showing"),
            ("None", self.select_none, "Uncheck every feature the filter is showing"),
            ("Invert", self.invert, "Flip every feature the filter is showing"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(handler)
            button_row.addWidget(button)
        button_row.addStretch()
        self.count_label = QLabel("")
        button_row.addWidget(self.count_label)
        root.addLayout(button_row)

        self.set_features(self._features, selected, catalog)

    # -- contents ---------------------------------------------------------

    def set_features(self, features, selected=(), catalog=None) -> None:
        """Rebuild the list. An empty `selected` checks everything: a step
        with no recorded selection uses every feature, and showing that as
        an empty list would misrepresent what it will do."""
        self._features = list(features)
        if catalog is not None:
            self._catalog = catalog
        wanted = set(selected) if selected else set(self._features)

        self.list.blockSignals(True)
        self.list.clear()
        for name in self._features:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if name in wanted else Qt.CheckState.Unchecked
            )
            descriptor = self._catalog.get(name) if self._catalog is not None else None
            item.setToolTip(describe_feature(descriptor))
            self.list.addItem(item)
        self.list.blockSignals(False)
        self._apply_filter(self.filter_edit.text())
        self._refresh_count()

    def selected(self) -> list[str]:
        return [
            self.list.item(row).text()
            for row in range(self.list.count())
            if self.list.item(row).checkState() == Qt.CheckState.Checked
        ]

    def selected_or_all(self) -> list[str]:
        """The selection to store on the step: empty when everything is
        checked, so a protocol doesn't pin a list that should grow when a
        later measurement step adds features."""
        chosen = self.selected()
        return [] if len(chosen) == len(self._features) else chosen

    # -- filtering and bulk actions ---------------------------------------

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for row in range(self.list.count()):
            item = self.list.item(row)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _visible_items(self):
        return [
            self.list.item(row)
            for row in range(self.list.count())
            if not self.list.item(row).isHidden()
        ]

    def _set_visible(self, state: Qt.CheckState) -> None:
        self.list.blockSignals(True)
        for item in self._visible_items():
            item.setCheckState(state)
        self.list.blockSignals(False)
        self._changed()

    def select_all(self) -> None:
        self._set_visible(Qt.CheckState.Checked)

    def select_none(self) -> None:
        self._set_visible(Qt.CheckState.Unchecked)

    def invert(self) -> None:
        self.list.blockSignals(True)
        for item in self._visible_items():
            item.setCheckState(
                Qt.CheckState.Unchecked
                if item.checkState() == Qt.CheckState.Checked
                else Qt.CheckState.Checked
            )
        self.list.blockSignals(False)
        self._changed()

    def _on_item_changed(self, _item) -> None:
        self._changed()

    def _changed(self) -> None:
        self._refresh_count()
        self.selection_changed.emit()

    def _refresh_count(self) -> None:
        total = len(self._features)
        chosen = len(self.selected())
        if not total:
            self.count_label.setText("No measured features yet.")
        else:
            self.count_label.setText(f"{chosen} of {total} selected")
