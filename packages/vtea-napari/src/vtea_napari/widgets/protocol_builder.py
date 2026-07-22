"""The protocol builder dock widget: Option A, a fully-functional GUI clone
of vtea.protocol's step-stack pipeline builder (see PORT_PLAN.md).

Owns a vtea_core.workflow.Pipeline and renders its steps as an ordered
stack of cards (StepCardWidget), matching ProtocolManagerMulti/
blockstepgui's actual layout (a plain top-to-bottom stack built by adding
steps from a category menu - not a node-graph editor, see PORT_PLAN.md's
"Protocol builder: Option A" section for why). Steps are added via category
+ function pickers, edited via a ParameterForm dialog, and removed with a
button on each card - the same operations the Java UI exposed, execution
handled by the shared Pipeline engine either here or from a script/notebook.
"""

from __future__ import annotations

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from vtea_core.workflow import STEP_REGISTRY, Pipeline, Step

from vtea_napari.widgets.param_form import ParameterForm
from vtea_napari.widgets.step_card import StepCardWidget


class EditStepDialog(QDialog):
    """A modal dialog wrapping a ParameterForm, pre-filled with the step's current params."""

    def __init__(self, step: Step, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit {step.category}.{step.function_name}")
        self.step = step

        layout = QVBoxLayout(self)
        self.form = ParameterForm(step.category, step.function_name)
        self.form.set_values(step.params)
        layout.addWidget(self.form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def updated_params(self) -> dict:
        return self.form.get_values()


class ProtocolBuilderWidget(QWidget):
    """The napari dock widget: a Pipeline plus the UI to build it."""

    def __init__(self, pipeline: Pipeline | None = None, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline if pipeline is not None else Pipeline()

        root = QVBoxLayout(self)

        add_row = QHBoxLayout()
        self.category_combo = QComboBox()
        self.category_combo.addItems(sorted(STEP_REGISTRY))
        self.category_combo.currentTextChanged.connect(self._refresh_function_choices)
        self.function_combo = QComboBox()
        add_button = QPushButton("Add Step")
        add_button.clicked.connect(self._add_step_from_selection)
        add_row.addWidget(QLabel("Category:"))
        add_row.addWidget(self.category_combo)
        add_row.addWidget(QLabel("Step:"))
        add_row.addWidget(self.function_combo)
        add_row.addWidget(add_button)
        root.addLayout(add_row)

        self._steps_container = QWidget()
        self._steps_layout = QVBoxLayout(self._steps_container)
        self._steps_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._steps_container)
        root.addWidget(scroll)

        self._refresh_function_choices(self.category_combo.currentText())
        self.refresh_steps()

    def _refresh_function_choices(self, category: str) -> None:
        self.function_combo.clear()
        if category:
            self.function_combo.addItems(sorted(STEP_REGISTRY[category]))

    def _add_step_from_selection(self) -> None:
        category = self.category_combo.currentText()
        function_name = self.function_combo.currentText()
        if not category or not function_name:
            return
        self.pipeline.add_step(Step(category=category, function_name=function_name))
        self.refresh_steps()

    def refresh_steps(self) -> None:
        while self._steps_layout.count() > 1:
            item = self._steps_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for position, step in enumerate(self.pipeline, start=1):
            card = StepCardWidget(position, step)
            card.edit_requested.connect(lambda s=step: self._edit_step(s))
            card.delete_requested.connect(lambda s=step: self._delete_step(s))
            self._steps_layout.insertWidget(self._steps_layout.count() - 1, card)

    def _edit_step(self, step: Step) -> None:
        dialog = EditStepDialog(step, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            step.params = dialog.updated_params()
            self.refresh_steps()

    def _delete_step(self, step: Step) -> None:
        index = self.pipeline.steps.index(step)
        self.pipeline.remove_step(index)
        self.refresh_steps()
