"""Editable-parameter introspection and a Qt form widget for step parameters.

vtea_core's functions use `from __future__ import annotations`, so their
type hints are plain strings at runtime (`inspect.signature` reflects this
without evaluating them - confirmed empirically). Auto-generating widgets
from a function's *full* signature (as magicgui does by default) also
doesn't distinguish data arguments (arrays/dataframes, resolved from the
Pipeline's run() context via Step.input_keys) from literal configuration
values (thresholds, cluster counts, ...) - only the latter belong in an
Edit-step form. Rather than fight either issue, DATA_PARAMETERS explicitly
lists each registered step's data argument names, and ParameterForm builds
plain qtpy widgets for everything else, inferring widget type from the
parameter's default value (when present) or a simple substring match
against its (string) annotation otherwise.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)
from vtea_core.workflow import DATA_PARAMETERS as _DATA_PARAMETERS
from vtea_core.workflow import get_step_function

# Which parameters are data (arrays, dataframes, models, ...) resolved from
# the pipeline's run() context via Step.input_keys, rather than editable
# literal form fields. This lives in vtea_core.workflow.wiring - it
# describes the functions, not this widget, and the Pipeline engine needs
# the same information to wire steps up. Re-exported here for callers that
# import it from this module.
DATA_PARAMETERS = _DATA_PARAMETERS


@dataclass
class ParamSpec:
    name: str
    kind: str  # "str", "int", "float", "bool" - the underlying type, even when optional
    default: Any
    required: bool
    optional: bool  # True if the parameter's default is literally None (e.g. `float | None = None`)


def _infer_kind(annotation: Any, default: Any) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "str"
    # default is None (or absent) - fall back to the annotation text, since
    # `from __future__ import annotations` leaves it a plain string like
    # "float | None" here, not an evaluated type.
    text = str(annotation)
    if "bool" in text:
        return "bool"
    if "int" in text:
        return "int"
    if "float" in text:
        return "float"
    return "str"


def editable_parameters(category: str, function_name: str) -> list[ParamSpec]:
    """The configuration (non-data) parameters of a registered step function."""
    func = get_step_function(category, function_name)
    data_names = DATA_PARAMETERS.get((category, function_name), set())
    sig = inspect.signature(func)

    specs = []
    for name, param in sig.parameters.items():
        if name in data_names:
            continue
        has_default = param.default is not inspect.Parameter.empty
        default = param.default if has_default else None
        specs.append(
            ParamSpec(
                name=name,
                kind=_infer_kind(param.annotation, default),
                default=default,
                required=not has_default,
                optional=has_default and default is None,
            )
        )
    return specs


class ParameterForm(QWidget):
    """A form of one widget per editable parameter of a registered step function."""

    def __init__(self, category: str, function_name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.specs = editable_parameters(category, function_name)
        self._field_widgets: dict[str, QWidget] = {}

        layout = QFormLayout(self)
        for spec in self.specs:
            field = self._make_field(spec)
            self._field_widgets[spec.name] = field
            layout.addRow(spec.name, field)

    def _make_field(self, spec: ParamSpec) -> QWidget:
        # Optional numeric/bool params (default is literally None) always get
        # a text field, empty by default - a QSpinBox/QCheckBox has no way to
        # represent "unset" distinctly from 0/False.
        if spec.optional:
            widget = QLineEdit()
            widget.setPlaceholderText("(not set)")
            return widget
        if spec.kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(spec.default))
            return widget
        if spec.kind == "int":
            widget = QSpinBox()
            widget.setRange(-(2**31), 2**31 - 1)
            widget.setValue(int(spec.default) if spec.default is not None else 0)
            return widget
        if spec.kind == "float":
            widget = QDoubleSpinBox()
            widget.setRange(-1e12, 1e12)
            widget.setDecimals(6)
            widget.setValue(float(spec.default) if spec.default is not None else 0.0)
            return widget
        if spec.name == "method" and spec.default in ("fixed", "otsu", "percentile"):
            widget = QComboBox()
            widget.addItems(["fixed", "otsu", "percentile"])
            widget.setCurrentText(str(spec.default))
            return widget
        widget = QLineEdit()
        if spec.default is not None:
            widget.setText(str(spec.default))
        return widget

    def get_values(self) -> dict[str, Any]:
        """Current form values, parsed back to Python types; empty optional
        text fields become None."""
        values: dict[str, Any] = {}
        for spec in self.specs:
            widget = self._field_widgets[spec.name]
            if spec.optional:
                text = widget.text()
                if not text:
                    values[spec.name] = None
                elif spec.kind == "int":
                    values[spec.name] = int(text)
                elif spec.kind == "float":
                    values[spec.name] = float(text)
                else:
                    values[spec.name] = text
            elif isinstance(widget, QCheckBox):
                values[spec.name] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[spec.name] = widget.value()
            elif isinstance(widget, QComboBox):
                values[spec.name] = widget.currentText()
            else:
                text = widget.text()
                values[spec.name] = text if text else None
        return values

    def set_values(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            widget = self._field_widgets.get(name)
            if widget is None:
                continue
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            else:
                widget.setText("" if value is None else str(value))
