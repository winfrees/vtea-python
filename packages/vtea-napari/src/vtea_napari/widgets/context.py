"""The Context pane: moving between length scales.

One slider runs through the levels of the analysis - the pieces of cells,
the cells, neighbourhoods of them, neighbourhoods of those - and every pane
follows it: the Object Explorer switches to that level's table and gates,
and the viewer shows that level. Above the cells, a level is drawn as a
hull outline per entity with a patterned fill, in colours chosen per level;
by default only the entities a gate selects are drawn, since neighbourhoods
overlap by design and a screen of overlapping hulls shows nothing.

Moving the slider can carry a selection with it: the entities gated on the
level being left are followed to the level arrived at - the neighbourhoods a
gated set of cells is in, the cells of a gated set of neighbourhoods, the
lysosomes of those cells.

A new level is defined from whichever level is showing (neighbourhoods of
it), with a radius slider that previews one neighbourhood at the chosen
size before anything is built. Definitions are saved with the protocol and
rebuilt on every run; the levels themselves are results. See
docs/CONTEXTS.md and vtea_core.context.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from vtea_core.context import (
    FILL_PATTERNS,
    NEIGHBORHOOD,
    ContextSpec,
    LevelDisplay,
    LevelSpec,
)
from vtea_core.context.geometry import level_outlines, pattern_for, typical_spacing
from vtea_core.neighborhoods import GRID, NEAREST, RADIUS, build_neighborhoods

from vtea_napari.session import AnalysisSession, session_for
from vtea_napari.widgets.explorer import (
    align_to,
    find_source_layer,
    highlight_array,
    placement_over,
)

logger = logging.getLogger(__name__)

LAYER_PREFIX = "Context"
SELECTION_LAYER = f"{LAYER_PREFIX}: selection"
PREVIEW_LAYER = f"{LAYER_PREFIX}: preview"

METHOD_LABELS = {
    RADIUS: "Around each entity, within a radius",
    NEAREST: "Each entity and its k nearest",
    GRID: "Grid points, within a radius",
}
CLASSIFY_METHODS = ("kmeans", "gaussian_mixture", "hierarchical", "leiden", "louvain")
NORMALIZE_CHOICES = ("none", "zscore", "robust", "minmax")
TIER_LABELS = {"subcellular": "subcellular", "cellular": "cellular", "neighborhood": "neighbourhood"}
# A column with this few distinct whole numbers is offered as a class.
MAX_CATEGORIES = 20


def _is_categorical(values) -> bool:
    import pandas as pd

    if pd.api.types.is_bool_dtype(values):
        return True
    if not pd.api.types.is_numeric_dtype(values):
        return False
    present = values.dropna()
    if present.empty or present.nunique() > MAX_CATEGORIES:
        return False
    return bool(np.all(np.mod(present.to_numpy(dtype=float), 1) == 0))


class _Ndim:
    """Stands in for data of a given dimensionality: `placement_over` only
    asks how many axes there are."""

    def __init__(self, ndim: int):
        self.ndim = ndim


class ContextWidget(QWidget):
    """A napari dock widget. `napari_viewer` is injected by napari; pass None
    to drive it without one (nothing is drawn). `session` is shared with the
    protocol builder and the Object Explorer."""

    def __init__(self, napari_viewer=None, parent=None, session: AnalysisSession | None = None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.session = session if session is not None else session_for(napari_viewer)
        self._layers: dict[str, list] = {}
        self._carried: tuple[str, np.ndarray] | None = None
        self._syncing = False

        root = QVBoxLayout(self)

        # -- the levels ----------------------------------------------------
        levels_box = QGroupBox("Level")
        levels_layout = QVBoxLayout(levels_box)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setPageStep(1)
        self.slider.valueChanged.connect(self._on_slider)
        levels_layout.addWidget(self.slider)
        self.stack_label = QLabel()
        self.stack_label.setWordWrap(True)
        levels_layout.addWidget(self.stack_label)
        self.level_label = QLabel()
        self.level_label.setWordWrap(True)
        levels_layout.addWidget(self.level_label)
        self.carry_check = QCheckBox("Carry the gated selection to the next level")
        self.carry_check.setChecked(True)
        self.carry_check.setToolTip(
            "Moving between levels follows what is gated on the level being left: the "
            "neighbourhoods gated cells are in, the cells of gated neighbourhoods, their pieces"
        )
        levels_layout.addWidget(self.carry_check)
        self.errors_label = QLabel()
        self.errors_label.setWordWrap(True)
        levels_layout.addWidget(self.errors_label)
        root.addWidget(levels_box)

        # -- defining the next level ---------------------------------------
        self.define_box = QGroupBox("Define neighbourhoods of this level")
        form = QFormLayout(self.define_box)
        self.name_edit = QLineEdit()
        form.addRow("Name:", self.name_edit)
        self.method_combo = QComboBox()
        for method, label in METHOD_LABELS.items():
            self.method_combo.addItem(label, method)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        form.addRow("Method:", self.method_combo)
        radius_row = QHBoxLayout()
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 1e6)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setValue(50.0)
        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setRange(1, 500)
        self.radius_slider.setValue(50)
        self.radius_slider.setToolTip("Drag to preview one neighbourhood at this radius")
        self.radius_slider.valueChanged.connect(lambda value: self.radius_spin.setValue(float(value)))
        self.radius_spin.valueChanged.connect(self._on_radius_changed)
        radius_row.addWidget(self.radius_spin)
        radius_row.addWidget(self.radius_slider)
        form.addRow("Radius:", radius_row)
        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        form.addRow(self.preview_label)
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 10000)
        self.k_spin.setValue(10)
        form.addRow("Neighbours (k):", self.k_spin)
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.0, 1e6)
        self.interval_spin.setSpecialValueText("= radius")
        form.addRow("Grid interval:", self.interval_spin)
        self.class_combo = QComboBox()
        self.class_combo.setToolTip(
            "Each entity's class: a neighbourhood is measured by how much of it is each class. "
            "Columns handed down from this level or above are not offered - a level cannot "
            "be defined on its own type."
        )
        form.addRow("Class column:", self.class_combo)
        self.features_edit = QLineEdit()
        self.features_edit.setPlaceholderText("optional: mean_ch1, volume")
        form.addRow("Member features:", self.features_edit)
        self.classify_check = QCheckBox("Cluster into types")
        self.classify_check.setChecked(True)
        form.addRow(self.classify_check)
        types_row = QHBoxLayout()
        self.classify_combo = QComboBox()
        self.classify_combo.addItems(CLASSIFY_METHODS)
        self.types_spin = QSpinBox()
        self.types_spin.setRange(1, 100)
        self.types_spin.setValue(5)
        self.normalize_combo = QComboBox()
        self.normalize_combo.addItems(NORMALIZE_CHOICES)
        types_row.addWidget(self.classify_combo)
        types_row.addWidget(self.types_spin)
        types_row.addWidget(self.normalize_combo)
        form.addRow("Types:", types_row)
        self.define_button = QPushButton("Add level")
        self.define_button.clicked.connect(self.define_level)
        form.addRow(self.define_button)
        root.addWidget(self.define_box)

        # -- how the level is drawn ----------------------------------------
        self.display_box = QGroupBox("Display")
        display = QFormLayout(self.display_box)
        self.outline_button = QPushButton()
        self.outline_button.clicked.connect(lambda: self._pick_color("outline_color"))
        display.addRow("Outline:", self.outline_button)
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(FILL_PATTERNS)
        self.pattern_combo.currentTextChanged.connect(
            lambda text: self.set_display(fill_pattern=text)
        )
        display.addRow("Fill pattern:", self.pattern_combo)
        self.fill_button = QPushButton()
        self.fill_button.clicked.connect(lambda: self._pick_color("fill_color"))
        display.addRow("Fill colour:", self.fill_button)
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_spin.valueChanged.connect(lambda value: self.set_display(opacity=value))
        display.addRow("Opacity:", self.opacity_spin)
        self.draw_combo = QComboBox()
        self.draw_combo.addItem("Gated in the Object Explorer", "gated")
        self.draw_combo.addItem("All (they overlap)", "all")
        self.draw_combo.currentIndexChanged.connect(
            lambda _index: self.set_display(draw=self.draw_combo.currentData())
        )
        display.addRow("Draw:", self.draw_combo)
        self.remove_button = QPushButton("Remove this level")
        self.remove_button.clicked.connect(self.remove_level)
        display.addRow(self.remove_button)
        root.addWidget(self.display_box)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        root.addStretch()

        self.session.context_changed.connect(self.refresh)
        self.session.data_changed.connect(self._follow_active_table)
        self.session.gates_changed.connect(self.redraw)
        self._on_method_changed()
        self.refresh()

    # -- reading the session ----------------------------------------------

    def levels(self) -> list:
        return self.session.context_levels()

    def current(self):
        graph = self.session.context_graph
        name = self.session.active_level
        return None if graph is None or name not in graph.levels else graph.level(name)

    def refresh(self) -> None:
        """Re-read the levels: the slider, the labels, the forms."""
        levels = self.levels()
        self._syncing = True
        self.slider.setRange(0, max(len(levels) - 1, 0))
        names = [level.name for level in levels]
        if self.session.active_level in names:
            self.slider.setValue(names.index(self.session.active_level))
        self._syncing = False
        self.slider.setEnabled(len(levels) > 1)
        self.stack_label.setText(" ▸ ".join(names) if names else "No levels yet: run a protocol.")
        errors = self.session.context_errors()
        self.errors_label.setText(
            "\n".join(f"Not built - {name}: {reason}" for name, reason in errors.items())
        )
        self._describe_current()
        self._refresh_define_form()
        self._refresh_display_form()
        self.redraw()

    def _follow_active_table(self) -> None:
        """The explorer's table menu is another way to change level; follow it."""
        table = self.session.active_table
        graph = self.session.context_graph
        if graph is not None and table in graph.levels and table != self.session.active_level:
            self.session.set_active_level(table)

    def _describe_current(self) -> None:
        level = self.current()
        if level is None:
            self.level_label.setText("")
            return
        text = f"{level.name}: {TIER_LABELS[level.tier]}, {len(level)} entities"
        parents = self.session.context_graph.parents_of(level.name)
        if parents:
            text += f"; part of {', '.join(parents)}"
        self.level_label.setText(text)

    def _refresh_define_form(self) -> None:
        level = self.current()
        self.define_box.setEnabled(level is not None)
        if level is None:
            return
        self.define_box.setTitle(f"Define neighbourhoods of {level.name}")
        if not self.name_edit.text() or self.name_edit.text() in self.session.context_graph.levels:
            self.name_edit.setText(self._next_name())
        new_rank = max(level.rank, 1) + 1
        current = self.class_combo.currentData()
        self.class_combo.clear()
        self.class_combo.addItem("(none - density only)", "")
        for column in level.table.columns:
            if column == level.id_column or level.reflected.get(column, -1) >= new_rank:
                continue
            if _is_categorical(level.table[column]):
                self.class_combo.addItem(str(column), str(column))
        self.class_combo.setCurrentIndex(max(self.class_combo.findData(current), 0))

    def _refresh_display_form(self) -> None:
        level = self.current()
        is_neighborhood = level is not None and level.tier == NEIGHBORHOOD
        self.display_box.setEnabled(is_neighborhood)
        if not is_neighborhood:
            return
        display = self.session.level_display(level.name)
        for widget in (self.pattern_combo, self.opacity_spin, self.draw_combo):
            widget.blockSignals(True)
        self.pattern_combo.setCurrentText(display.fill_pattern)
        self.opacity_spin.setValue(display.opacity)
        self.draw_combo.setCurrentIndex(max(self.draw_combo.findData(display.draw), 0))
        for widget in (self.pattern_combo, self.opacity_spin, self.draw_combo):
            widget.blockSignals(False)
        self.outline_button.setStyleSheet(f"background-color: {display.outline_color};")
        self.fill_button.setStyleSheet(f"background-color: {display.fill_color};")
        self.remove_button.setEnabled(self.session.context_spec.get(level.name) is not None)

    def _next_name(self) -> str:
        taken = set(self.session.context_graph.levels) if self.session.context_graph else set()
        taken |= {level.name for level in self.session.context_spec}
        index = 1
        while f"nbhd_{index}" in taken:
            index += 1
        return f"nbhd_{index}"

    def _on_method_changed(self, *_args) -> None:
        method = self.method_combo.currentData()
        self.k_spin.setEnabled(method == NEAREST)
        self.radius_spin.setEnabled(method in (RADIUS, GRID))
        self.radius_slider.setEnabled(method in (RADIUS, GRID))
        self.interval_spin.setEnabled(method == GRID)

    # -- moving between levels --------------------------------------------

    def _on_slider(self, value: int) -> None:
        if self._syncing:
            return
        levels = self.levels()
        if not 0 <= value < len(levels):
            return
        self.go_to(levels[value].name)

    def go_to(self, name: str) -> None:
        """Show level `name`, carrying the gated selection from the level
        being left when that is switched on."""
        previous = self.session.active_level
        carried = None
        if self.carry_check.isChecked() and previous and previous != name:
            ids = self.session.gated_ids(previous)
            if len(ids):
                carried = (name, self.session.related_ids(previous, ids, name))
        self._carried = carried
        self.session.set_active_level(name)
        if carried is not None:
            self.status_label.setText(
                f"{len(carried[1])} {name} entities related to the gated {previous} selection."
            )
        self.redraw()

    # -- defining levels --------------------------------------------------

    def level_spec(self) -> LevelSpec | None:
        """What the define form says, as a level definition."""
        level = self.current()
        if level is None:
            return None
        method = self.method_combo.currentData()
        build = {"method": method}
        if method in (RADIUS, GRID):
            build["radius"] = float(self.radius_spin.value())
        if method == NEAREST:
            build["k"] = int(self.k_spin.value())
        if method == GRID and self.interval_spin.value() > 0:
            build["interval"] = float(self.interval_spin.value())
        measure = {}
        if self.class_combo.currentData():
            measure["class_column"] = self.class_combo.currentData()
        if self.features_edit.text().strip():
            measure["features"] = self.features_edit.text().strip()
        classify = {}
        if self.classify_check.isChecked() and measure:
            classify = {
                "method": self.classify_combo.currentText(),
                "n_clusters": int(self.types_spin.value()),
                "normalize": self.normalize_combo.currentText(),
            }
            if "class_column" not in measure:
                classify["features"] = ",".join(
                    f"mean_{name.strip()}" for name in measure["features"].split(",") if name.strip()
                )
        return LevelSpec(
            name=self.name_edit.text().strip() or self._next_name(),
            tier=NEIGHBORHOOD,
            source=level.name,
            build=build,
            measure=measure,
            classify=classify,
        )

    def _spec_with_base(self) -> ContextSpec:
        """The saved definitions, with the levels the run defined by itself
        put in front, so a new level can be built on one of those."""
        graph = self.session.context_graph
        spec = self.session.context_spec
        if graph is None or graph.spec is None:
            return spec
        defined = {level.name for level in spec}
        return ContextSpec([level for level in graph.spec if level.name not in defined] + list(spec))

    def define_level(self) -> str | None:
        """Add the level the form describes, rebuild, and go to it."""
        new = self.level_spec()
        if new is None:
            return None
        spec = self._spec_with_base()
        try:
            spec.add(new)
        except ValueError as error:
            self.status_label.setText(f"Could not add {new.name}: {error}")
            return None
        self.session.set_context_spec(spec)
        errors = self.session.context_errors()
        if new.name in errors:
            self.status_label.setText(f"{new.name} was saved but not built: {errors[new.name]}")
            return None
        self._remove_layer_group(PREVIEW_LAYER)
        self._carried = None
        self.session.set_active_level(new.name)
        self.name_edit.setText(self._next_name())
        level = self.session.context_graph.level(new.name)
        self.status_label.setText(
            f"{new.name}: {len(level)} neighbourhoods of {new.source}. Gate them in the Object "
            f"Explorer to draw them; their type is handed down to every level below."
        )
        return new.name

    def remove_level(self) -> None:
        """Remove the current level's definition, and every level built on it."""
        level = self.current()
        if level is None or self.session.context_spec.get(level.name) is None:
            return
        doomed = {level.name}
        changed = True
        spec = list(self.session.context_spec)
        while changed:
            changed = False
            for other in spec:
                if other.tier == NEIGHBORHOOD and other.source in doomed and other.name not in doomed:
                    doomed.add(other.name)
                    changed = True
        for name in doomed:
            self._remove_layer_group(name)
        source = self.session.context_spec.get(level.name).source
        self.session.set_context_spec(ContextSpec([s for s in spec if s.name not in doomed]))
        self.session.set_active_level(source)
        self.status_label.setText(f"Removed {', '.join(sorted(doomed))}.")

    # -- display ----------------------------------------------------------

    def set_display(self, **changes) -> None:
        """Change how the current level is drawn; saved with its definition."""
        level = self.current()
        if level is None or level.tier != NEIGHBORHOOD:
            return
        spec = self._spec_with_base()
        definition = spec.get(level.name)
        if definition is None:
            return
        try:
            definition.display = replace(definition.display, **changes)
            LevelDisplay(**definition.display.to_dict())  # validate
        except (TypeError, ValueError) as error:
            self.status_label.setText(str(error))
            return
        self.session.context_spec = spec
        self._refresh_display_form()
        self.redraw()

    def _pick_color(self, field: str) -> None:
        from qtpy.QtGui import QColor

        current = getattr(self.session.level_display(self.session.active_level), field)
        color = QColorDialog.getColor(QColor(current), self, "Colour")
        if color.isValid():
            self.set_display(**{field: color.name()})

    # -- drawing ----------------------------------------------------------

    def _source_layer(self):
        return find_source_layer(self.viewer, self.session)

    def _remove_layer_group(self, key: str) -> None:
        for layer in self._layers.pop(key, []):
            if self.viewer is not None and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)

    def _to_world(self, coordinates: np.ndarray) -> np.ndarray:
        """Put back the channel axis a (z, c, y, x) source image has, so the
        shapes land on the same axes as the data."""
        source = self._source_layer()
        channel_axis = self.session.channel_axis
        source_ndim = getattr(getattr(source, "data", None), "ndim", None)
        if channel_axis is not None and source_ndim == coordinates.shape[1] + 1:
            coordinates = np.insert(coordinates, channel_axis, 0.0, axis=1)
        return coordinates

    def drawn_ids(self, name: str):
        """Which entities of a level are drawn: those carried to it from the
        level just left, else those its gates select - or all of them, when
        its display asks for that. None means all."""
        if self._carried is not None and self._carried[0] == name:
            return self._carried[1]
        if self.session.level_display(name).draw == "all":
            return None
        return self.session.gated_ids(name)

    def redraw(self) -> None:
        """Outlines for the level being looked at (when it is above the
        cells), and the carried selection on a labelled level."""
        if self.viewer is None:
            return
        graph = self.session.context_graph
        for key in [key for key in self._layers if key not in (PREVIEW_LAYER,)]:
            self._remove_layer_group(key)
        if graph is None:
            return
        active = self.current()
        if active is None:
            return
        if active.tier == NEIGHBORHOOD:
            ids = self.drawn_ids(active.name)
            if ids is None or len(ids):
                self._draw_outlines(active.name, ids)
        elif self._carried is not None and self._carried[0] == active.name:
            self._draw_selection(active, self._carried[1])

    def _draw_selection(self, level, ids) -> None:
        labels = self.session.labels(level.name)
        if labels is None or not len(ids):
            return
        data = align_to(highlight_array(labels, ids), self._source_layer(), self.session.channel_axis)
        layer = self.viewer.add_labels(
            data, name=SELECTION_LAYER, **placement_over(self._source_layer(), data)
        )
        self._layers[SELECTION_LAYER] = [layer]

    def _draw_outlines(
        self, name: str, ids, *, graph=None, key: str | None = None, display=None
    ) -> None:
        graph = graph or self.session.context_graph
        display = display or self.session.level_display(name)
        outlines = level_outlines(graph, name, ids)
        if not outlines:
            return
        shapes = [
            self._to_world(
                polygon if z is None else np.column_stack([np.full(len(polygon), z), polygon])
            )
            for _entity, z, polygon in outlines
        ]
        placement = placement_over(self._source_layer(), _Ndim(shapes[0].shape[1]))
        extent = typical_spacing(np.vstack([polygon for _e, _z, polygon in outlines]))
        width = max(0.5, extent / 6)
        face = display.fill_color if display.fill_pattern == "solid" else "transparent"
        layers = [
            self.viewer.add_shapes(
                shapes,
                shape_type="polygon",
                edge_color=display.outline_color,
                face_color=face,
                edge_width=width,
                opacity=display.opacity,
                name=f"{LAYER_PREFIX}: {name}",
                **placement,
            )
        ]
        fill = self._fill_layer(name, outlines, display, width, placement)
        if fill is not None:
            layers.append(fill)
        self._layers[key or name] = layers

    def _fill_layer(self, name, outlines, display, width, placement):
        segments, dots = [], []
        for _entity, z, polygon in outlines:
            size = float(np.ptp(polygon, axis=0).max()) if len(polygon) else 0.0
            kind, pattern = pattern_for(polygon, display.fill_pattern, max(size / 8, 1.0))
            if kind == "lines":
                for segment in pattern:
                    segments.append(
                        self._to_world(
                            segment if z is None else np.column_stack([np.full(2, z), segment])
                        )
                    )
            elif kind == "points" and len(pattern):
                dots.append(
                    self._to_world(
                        pattern if z is None else np.column_stack([np.full(len(pattern), z), pattern])
                    )
                )
        if segments:
            return self.viewer.add_shapes(
                segments,
                shape_type="line",
                edge_color=display.fill_color,
                edge_width=width / 2,
                opacity=display.opacity,
                name=f"{LAYER_PREFIX}: {name} fill",
                **placement,
            )
        if dots:
            return self.viewer.add_points(
                np.vstack(dots),
                size=width * 1.5,
                face_color=display.fill_color,
                opacity=display.opacity,
                name=f"{LAYER_PREFIX}: {name} fill",
                **placement,
            )
        return None

    # -- previewing a radius ----------------------------------------------

    def _on_radius_changed(self, value: float) -> None:
        self.radius_slider.blockSignals(True)
        self.radius_slider.setValue(round(min(max(value, 1), self.radius_slider.maximum())))
        self.radius_slider.blockSignals(False)
        self.preview()

    def preview(self) -> dict | None:
        """What the radius on the form would make of the current level: how
        many members a neighbourhood gets, and the outline of the one nearest
        the middle of the field - one, so the size is visible and nothing
        overlaps."""
        level = self.current()
        method = self.method_combo.currentData()
        if level is None or method not in (RADIUS, GRID):
            self.preview_label.setText("")
            return None
        try:
            neighborhoods = build_neighborhoods(
                level.table,
                method=RADIUS,
                radius=float(self.radius_spin.value()),
                spacing=self.session.spacing,
                id_column=level.id_column,
            )
        except ValueError as error:
            self.preview_label.setText(str(error))
            return None
        sizes = neighborhoods.sizes()
        if not len(sizes):
            self.preview_label.setText("")
            return None
        centres = neighborhoods.centers()
        middle = int(np.argmin(np.linalg.norm(centres - centres.mean(axis=0), axis=1)))
        chosen = int(neighborhoods.ids()[middle])
        summary = {
            "median": int(np.median(sizes)),
            "min": int(sizes.min()),
            "max": int(sizes.max()),
            "example": chosen,
        }
        self.preview_label.setText(
            f"Members per neighbourhood: median {summary['median']} "
            f"({summary['min']}-{summary['max']}); outlined: the one around {chosen}"
        )
        self._draw_preview(level, neighborhoods, chosen)
        return summary

    def _draw_preview(self, level, neighborhoods, chosen: int) -> None:
        if self.viewer is None:
            return
        import pandas as pd
        from vtea_core.context import ContextGraph, Level

        self._remove_layer_group(PREVIEW_LAYER)
        scratch = ContextGraph()
        scratch.add_level(
            Level(level.name, level.tier, level.table, level.id_column, rank=level.rank)
        )
        preview = scratch.add_level(
            Level(
                "preview",
                NEIGHBORHOOD,
                pd.DataFrame({"neighborhood_id": neighborhoods.ids()}),
                "neighborhood_id",
                rank=level.rank + 1,
            )
        )
        preview.neighborhoods = neighborhoods
        membership = neighborhoods.membership().rename(
            columns={neighborhoods.member_column: "child", "neighborhood_id": "parent"}
        )
        scratch.link(level.name, "preview", membership)
        self._draw_outlines(
            "preview",
            [chosen],
            graph=scratch,
            key=PREVIEW_LAYER,
            display=LevelDisplay(fill_pattern="none", outline_color="#ffffff"),
        )


__all__ = ["ContextWidget"]
