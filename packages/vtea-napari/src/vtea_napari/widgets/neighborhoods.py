"""The Neighborhoods pane: a second level of objects, on the same viewer.

Neighbourhoods are objects made of segmented objects, and this pane is where
both levels are looked at together. It builds them from any table the
protocol produced (objects or cells), measures and types them, and hands
the result back to the members - then draws both levels on the one napari
viewer the segmentation is already on:

- a **Points layer**, one point per neighbourhood at its centre, sized to
  its reach and coloured by any neighbourhood feature (its type, its
  fraction of one class). Clicking a point selects that neighbourhood.
- a **Labels layer** painting every object by what it took on from its
  neighbourhoods - its neighbourhood type, by default - so the tissue reads
  as regions of one kind of neighbourhood or another.
- a **members layer** showing only the objects of the selected
  neighbourhood.

The neighbourhood table joins the Object Explorer's table menu, where it is
plotted and gated like any other; a gate on neighbourhoods centred on cells
lights up those cells. And the columns reflected onto the objects join the
object table, so a cell can be gated - or a protocol class written - on the
kind of neighbourhood it lives in.

What this pane does lives in vtea_core.neighborhoods; `run_neighborhood_
analysis` below is the whole computation with no Qt in it, and the protocol
builder's "neighborhoods" steps are the same functions for a protocol to
carry.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from vtea_core.classification import class_map
from vtea_core.data import Spacing
from vtea_core.neighborhoods import (
    GRID,
    MISSING_CATEGORY,
    NEAREST,
    NEIGHBORHOOD_ID,
    NEIGHBORHOOD_TYPE,
    RADIUS,
    NeighborhoodSet,
    build_neighborhoods,
    classify_neighborhoods,
    composition_columns,
    neighborhood_features,
    reflect_neighborhoods,
)

from vtea_napari.session import AnalysisSession, NeighborhoodResult, session_for
from vtea_napari.widgets.explorer import (
    align_to,
    find_source_layer,
    highlight_array,
    placement_over,
)

logger = logging.getLogger(__name__)

POINTS_LAYER_PREFIX = "Neighborhoods"
TYPES_LAYER_PREFIX = "Neighborhood types"
MEMBERS_LAYER_NAME = "Neighborhood members"

# What the method combo shows, and the Java names they replace.
METHOD_LABELS = {
    RADIUS: "Around each object, within a radius (Java: Spatial by cell)",
    NEAREST: "Each object and its k nearest (Java: Nearest-k)",
    GRID: "Grid points, within a radius (Java: Spatial by point)",
}
CLASSIFY_METHODS = ("kmeans", "gaussian_mixture", "hierarchical", "leiden", "louvain")
NORMALIZE_CHOICES = ("none", "zscore", "robust", "minmax")
RELATIONS = ("auto", "own", "member")

# A column with this few distinct integers is coloured as categories.
MAX_CATEGORIES = 20


@dataclass
class NeighborhoodSettings:
    """Everything a neighbourhood analysis is built from - recorded with the
    result, so what a composition number means can be read back later."""

    method: str = RADIUS
    radius: float = 50.0
    k: int = 10
    interval: float = 0.0
    randomize: bool = False
    random_state: int = 0
    class_column: str = ""
    features: str = ""
    aggregations: str = "mean"
    classify: bool = True
    classify_method: str = "kmeans"
    n_types: int = 5
    normalize: str = "none"
    relation: str = "auto"


def run_neighborhood_analysis(
    frame: pd.DataFrame,
    settings: NeighborhoodSettings,
    *,
    id_column: str = "object_id",
    spacing: Spacing | None = None,
    source: str = "",
    prefix: str = "",
) -> tuple[NeighborhoodSet, pd.DataFrame, pd.DataFrame]:
    """(neighbourhoods, their table, what each member took on from them).

    Types the neighbourhoods only when there is something to type them on - a
    class composition or named features - and says nothing otherwise: a
    neighbourhood analysis without classes is still an analysis of density.
    """
    neighborhoods = build_neighborhoods(
        frame,
        method=settings.method,
        radius=settings.radius,
        k=settings.k,
        interval=settings.interval,
        spacing=spacing,
        randomize=settings.randomize,
        random_state=settings.random_state,
        id_column=id_column,
        source=source,
    )
    table = neighborhood_features(
        neighborhoods,
        frame,
        class_column=settings.class_column,
        features=settings.features,
        aggregations=settings.aggregations,
    )
    # Typed on their composition where there is one - "the same mix of
    # cells" - and otherwise on the member features summarised above.
    classify_on = "" if composition_columns(table) else ",".join(
        f"{reduction.strip()}_{feature.strip()}"
        for reduction in settings.aggregations.split(",")
        for feature in settings.features.split(",")
        if reduction.strip() and feature.strip()
    )
    can_classify = bool(composition_columns(table)) or bool(classify_on)
    if settings.classify and can_classify and len(table) > 1:
        table = classify_neighborhoods(
            table,
            method=settings.classify_method,
            n_clusters=settings.n_types,
            features=classify_on,
            normalize=settings.normalize,
            random_state=settings.random_state,
        )
    relation = settings.relation
    if relation == "own" and not neighborhoods.centred:
        relation = "member"
    reflected = reflect_neighborhoods(
        neighborhoods, table, frame, relation=relation, prefix=prefix
    )
    return neighborhoods, table, reflected


def _is_categorical(values: pd.Series) -> bool:
    """Whether a column reads as categories: booleans, or a few distinct
    whole numbers (a type, a cluster, a class) - missing values aside, since
    an object in no neighbourhood has no type."""
    if pd.api.types.is_bool_dtype(values):
        return True
    if not pd.api.types.is_numeric_dtype(values):
        return False
    present = values.dropna()
    if present.empty or present.nunique() > MAX_CATEGORIES:
        return False
    return bool(np.all(np.equal(np.mod(present.to_numpy(dtype=float), 1), 0)))


class NeighborhoodWidget(QWidget):
    """A napari dock widget for neighbourhood analysis.

    `napari_viewer` is injected by napari; pass None to drive it from a
    script or a test (everything but the layers still works). `session` is
    the analysis shared with the protocol builder and the Object Explorer.
    """

    # A neighbourhood was selected, by id.
    neighborhood_selected = Signal(int)

    def __init__(self, napari_viewer=None, parent=None, session: AnalysisSession | None = None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.session = session if session is not None else session_for(napari_viewer)
        self._points_layers: dict[str, object] = {}
        self._types_layers: dict[str, object] = {}
        self._members_layer = None

        root = QVBoxLayout(self)

        # -- what to build them from ---------------------------------------
        source_box = QGroupBox("Objects")
        source_form = QFormLayout(source_box)
        self.table_combo = QComboBox()
        self.table_combo.setToolTip(
            "The table whose rows become the neighbourhoods' members: the objects of a "
            "segmentation, or the cells built from them"
        )
        self.table_combo.currentTextChanged.connect(self._refresh_columns)
        source_form.addRow("Members from:", self.table_combo)
        self.class_combo = QComboBox()
        self.class_combo.setToolTip(
            "Each member's class - a clustering, a class, a gate. A neighbourhood is "
            "measured by how much of it is each class (Java ClassFraction / ClassSums)"
        )
        source_form.addRow("Class column:", self.class_combo)
        self.features_edit = QLineEdit()
        self.features_edit.setPlaceholderText("optional: mean_ch1, volume")
        self.features_edit.setToolTip(
            "Member features to summarise per neighbourhood, comma-separated"
        )
        source_form.addRow("Member features:", self.features_edit)
        root.addWidget(source_box)

        # -- how to draw them ----------------------------------------------
        build_box = QGroupBox("Neighbourhoods")
        build_form = QFormLayout(build_box)
        self.method_combo = QComboBox()
        for method, label in METHOD_LABELS.items():
            self.method_combo.addItem(label, method)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        build_form.addRow("Method:", self.method_combo)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.01, 1e6)
        self.radius_spin.setDecimals(2)
        self.radius_spin.setValue(50.0)
        build_form.addRow("Radius:", self.radius_spin)
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 10000)
        self.k_spin.setValue(10)
        build_form.addRow("Neighbours (k):", self.k_spin)
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.0, 1e6)
        self.interval_spin.setDecimals(2)
        self.interval_spin.setSpecialValueText("= radius")
        build_form.addRow("Grid interval:", self.interval_spin)
        self.units_label = QLabel()
        build_form.addRow("Units:", self.units_label)
        self.randomize_check = QCheckBox("Randomise positions (null model)")
        self.randomize_check.setToolTip(
            "Shuffle which object sits at which position before building - the same "
            "density, no spatial arrangement. Compare against it to see what is not chance."
        )
        build_form.addRow(self.randomize_check)
        root.addWidget(build_box)

        # -- types, and what the members take on ---------------------------
        type_box = QGroupBox("Neighbourhood types")
        type_form = QFormLayout(type_box)
        self.classify_check = QCheckBox("Cluster neighbourhoods into types")
        self.classify_check.setChecked(True)
        type_form.addRow(self.classify_check)
        self.classify_combo = QComboBox()
        self.classify_combo.addItems(CLASSIFY_METHODS)
        type_form.addRow("Method:", self.classify_combo)
        self.types_spin = QSpinBox()
        self.types_spin.setRange(1, 100)
        self.types_spin.setValue(5)
        type_form.addRow("Types:", self.types_spin)
        self.normalize_combo = QComboBox()
        self.normalize_combo.addItems(NORMALIZE_CHOICES)
        self.normalize_combo.setToolTip("Rescale features before clustering (Java: Z-scale all data)")
        type_form.addRow("Normalise:", self.normalize_combo)
        self.relation_combo = QComboBox()
        self.relation_combo.addItems(RELATIONS)
        self.relation_combo.setToolTip(
            "What each object takes on: its own neighbourhood's values (own), or the "
            "average / most common over every neighbourhood it is in (member). "
            "auto: own where there is one"
        )
        type_form.addRow("Objects take on:", self.relation_combo)
        root.addWidget(type_box)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setToolTip(
            "The neighbourhood table's name, and the prefix of the columns it adds to "
            "the objects (name.neighborhood_type)"
        )
        name_row.addWidget(self.name_edit)
        self.build_button = QPushButton("Build")
        self.build_button.clicked.connect(self.build)
        name_row.addWidget(self.build_button)
        root.addLayout(name_row)

        # -- looking at them -----------------------------------------------
        view_box = QGroupBox("View")
        view_form = QFormLayout(view_box)
        self.result_combo = QComboBox()
        self.result_combo.currentTextChanged.connect(self._on_result_changed)
        view_form.addRow("Analysis:", self.result_combo)
        self.color_combo = QComboBox()
        self.color_combo.setToolTip("Colour the neighbourhood points by one of their features")
        self.color_combo.currentTextChanged.connect(lambda _text: self.show_neighborhoods())
        view_form.addRow("Colour points by:", self.color_combo)
        self.paint_combo = QComboBox()
        self.paint_combo.setToolTip("Paint every object by what it took on from its neighbourhoods")
        self.paint_combo.currentTextChanged.connect(lambda _text: self.paint_objects())
        view_form.addRow("Paint objects by:", self.paint_combo)
        select_row = QHBoxLayout()
        self.select_spin = QSpinBox()
        self.select_spin.setRange(0, 2**31 - 1)
        self.select_spin.setToolTip("A neighbourhood id - or click its point on the viewer")
        select_row.addWidget(self.select_spin)
        self.members_button = QPushButton("Show members")
        self.members_button.clicked.connect(
            lambda: self.select_neighborhood(self.select_spin.value())
        )
        select_row.addWidget(self.members_button)
        view_form.addRow("Neighbourhood:", select_row)
        buttons = QHBoxLayout()
        self.explore_button = QPushButton("Plot in Object Explorer")
        self.explore_button.clicked.connect(self.open_in_explorer)
        buttons.addWidget(self.explore_button)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_current)
        buttons.addWidget(self.remove_button)
        view_form.addRow(buttons)
        root.addWidget(view_box)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        root.addStretch()

        self.session.data_changed.connect(self.refresh)
        self.session.neighborhoods_changed.connect(self._refresh_results)
        self._on_method_changed()
        self.refresh()

    # -- reading the session ----------------------------------------------

    def member_tables(self) -> list[str]:
        """The tables whose rows can be members: anything but a neighbourhood
        table (neighbourhoods of neighbourhoods are not offered - yet)."""
        return [
            name
            for name in self.session.table_names()
            if self.session.tables[name].id_column != NEIGHBORHOOD_ID
        ]

    def refresh(self) -> None:
        """Re-read the tables on offer, keeping the current choices."""
        current = self.table_combo.currentText()
        self.table_combo.blockSignals(True)
        self.table_combo.clear()
        self.table_combo.addItems(self.member_tables())
        position = self.table_combo.findText(current)
        self.table_combo.setCurrentIndex(max(position, 0))
        self.table_combo.blockSignals(False)
        self._refresh_columns()
        self._refresh_units()
        self._refresh_results()

    def _source(self) -> tuple[str, pd.DataFrame | None, str]:
        name = self.table_combo.currentText()
        view = self.session.tables.get(name)
        if view is None:
            return name, None, "object_id"
        # Built from the table as the builder made it, not with another
        # analysis's reflected columns already on it.
        frame = self.session.source_frame(name)
        return name, view.frame if frame is None else frame, view.id_column

    def _refresh_columns(self, *_args) -> None:
        name, frame, id_column = self._source()
        current = self.class_combo.currentData()
        self.class_combo.clear()
        self.class_combo.addItem("(none - density only)", "")
        view = self.session.tables.get(name)
        decorated = view.frame if view is not None else frame
        if decorated is not None:
            categorical = self.session.categorical_columns(decorated)
            candidates = [
                column
                for column in decorated.columns
                if column != id_column
                and (column in categorical or _is_categorical(decorated[column]))
            ]
            for column in candidates:
                self.class_combo.addItem(str(column), str(column))
        position = self.class_combo.findData(current)
        self.class_combo.setCurrentIndex(max(position, 0))
        if not self.name_edit.text():
            self.name_edit.setText(self._next_name())

    def _refresh_units(self) -> None:
        spacing = self.session.spacing
        if spacing is not None and spacing.is_known:
            self.units_label.setText(f"{spacing.unit} (voxel size {spacing.describe()})")
        else:
            self.units_label.setText("voxels (no voxel size set in the Protocol Builder)")

    def _next_name(self) -> str:
        index = 1
        while f"nbhd_{index}" in self.session.tables or f"nbhd_{index}" in (
            self.session.neighborhood_results
        ):
            index += 1
        return f"nbhd_{index}"

    def _on_method_changed(self, *_args) -> None:
        method = self.method_combo.currentData()
        self.k_spin.setEnabled(method == NEAREST)
        self.radius_spin.setEnabled(method in (RADIUS, GRID))
        self.interval_spin.setEnabled(method == GRID)

    # -- building ---------------------------------------------------------

    def settings(self) -> NeighborhoodSettings:
        return NeighborhoodSettings(
            method=self.method_combo.currentData(),
            radius=float(self.radius_spin.value()),
            k=int(self.k_spin.value()),
            interval=float(self.interval_spin.value()),
            randomize=self.randomize_check.isChecked(),
            class_column=self.class_combo.currentData() or "",
            features=self.features_edit.text().strip(),
            classify=self.classify_check.isChecked(),
            classify_method=self.classify_combo.currentText(),
            n_types=int(self.types_spin.value()),
            normalize=self.normalize_combo.currentText(),
            relation=self.relation_combo.currentText(),
        )

    def build(self) -> NeighborhoodResult | None:
        """Build, measure, type and reflect - then publish and draw."""
        source, frame, id_column = self._source()
        if frame is None or frame.empty:
            self.status_label.setText(
                "Nothing to build from: run a protocol with a measurement step first."
            )
            return None
        name = self.name_edit.text().strip() or self._next_name()
        if name in self.session.tables and name not in self.session.neighborhood_results:
            self.status_label.setText(f"'{name}' is already a table name - pick another.")
            return None
        settings = self.settings()
        try:
            neighborhoods, table, reflected = run_neighborhood_analysis(
                frame,
                settings,
                id_column=id_column,
                spacing=self.session.spacing,
                source=source,
                prefix=f"{name}.",
            )
        except (ValueError, KeyError, ImportError) as exc:
            self.status_label.setText(f"Could not build neighbourhoods: {exc}")
            return None

        view = self.session.tables.get(source)
        labels_key = view.labels_key if view is not None and neighborhoods.centred else ""
        result = NeighborhoodResult(
            neighborhoods=neighborhoods,
            table=table,
            source_table=source,
            labels_key=labels_key,
            reflected=reflected,
        )
        self._record_catalog(name, settings, table, reflected, id_column)
        self.session.set_neighborhood_result(name, result)
        self._select_result(name)
        self.show_neighborhoods()
        self.paint_objects()
        summary = neighborhoods.summary()
        if NEIGHBORHOOD_TYPE in table.columns:
            summary += f"; {table[NEIGHBORHOOD_TYPE].nunique()} types"
        self.status_label.setText(
            f"{name}: {summary}. '{name}' is in the Object Explorer's table menu, and "
            f"{len(reflected.columns) - 1} '{name}.' columns were added to {source}."
        )
        self.name_edit.setText(self._next_name())
        return result

    def _record_catalog(self, name, settings, table, reflected, id_column) -> None:
        """Say what the new columns are, so a neighbourhood type is coloured
        as a category and exported with its provenance."""
        catalog = self.session.feature_catalog
        params = asdict(settings)
        catalog.record_derived(
            [column for column in reflected.columns if column != id_column],
            produced_by=name,
            function="neighborhoods.reflect_neighborhoods",
            params=params,
        )
        if NEIGHBORHOOD_TYPE in table.columns:
            catalog.record_derived(
                [NEIGHBORHOOD_TYPE],
                produced_by=name,
                function="neighborhoods.classify_neighborhoods",
                params=params,
            )

    # -- results ----------------------------------------------------------

    def current_result(self) -> tuple[str, NeighborhoodResult | None]:
        name = self.result_combo.currentText()
        return name, self.session.neighborhood_results.get(name)

    def _refresh_results(self) -> None:
        current = self.result_combo.currentText()
        self.result_combo.blockSignals(True)
        self.result_combo.clear()
        self.result_combo.addItems(list(self.session.neighborhood_results))
        position = self.result_combo.findText(current)
        self.result_combo.setCurrentIndex(max(position, 0))
        self.result_combo.blockSignals(False)
        self._on_result_changed()
        for stale in [n for n in self._points_layers if n not in self.session.neighborhood_results]:
            self._remove_layer(self._points_layers.pop(stale))
        for stale in [n for n in self._types_layers if n not in self.session.neighborhood_results]:
            self._remove_layer(self._types_layers.pop(stale))

    def _select_result(self, name: str) -> None:
        position = self.result_combo.findText(name)
        if position >= 0:
            self.result_combo.setCurrentIndex(position)
        self._on_result_changed()

    def _on_result_changed(self, *_args) -> None:
        name, result = self.current_result()
        for combo in (self.color_combo, self.paint_combo):
            combo.blockSignals(True)
            combo.clear()
        if result is not None:
            columns = [
                column
                for column in result.table.columns
                if column not in (NEIGHBORHOOD_ID, "seed")
                and not str(column).startswith("centroid-")
            ]
            self.color_combo.addItems(columns)
            default = NEIGHBORHOOD_TYPE if NEIGHBORHOOD_TYPE in columns else "n_objects"
            self.color_combo.setCurrentIndex(max(self.color_combo.findText(default), 0))
            if result.reflected is not None:
                reflected = [c for c in result.reflected.columns if c != result.member_id_column]
                self.paint_combo.addItems(reflected)
                preferred = f"{name}.{NEIGHBORHOOD_TYPE}"
                self.paint_combo.setCurrentIndex(max(self.paint_combo.findText(preferred), 0))
        for combo in (self.color_combo, self.paint_combo):
            combo.blockSignals(False)

    def remove_current(self) -> None:
        name, result = self.current_result()
        if result is None:
            return
        self._remove_layer(self._points_layers.pop(name, None))
        self._remove_layer(self._types_layers.pop(name, None))
        self.session.remove_neighborhood_result(name)
        self.status_label.setText(f"Removed {name}.")

    def open_in_explorer(self):
        """Plot the neighbourhood table in the Object Explorer."""
        name, result = self.current_result()
        if result is None:
            return None
        self.session.set_active_table(name)
        if self.viewer is None:
            return None
        from qtpy.QtWidgets import QApplication

        from vtea_napari.widgets.explorer import ExplorerWidget

        for widget in QApplication.topLevelWidgets():
            for existing in widget.findChildren(ExplorerWidget):
                if existing.session is self.session:
                    existing.show()
                    existing.raise_()
                    return existing
        explorer = ExplorerWidget(napari_viewer=self.viewer, session=self.session)
        self.viewer.window.add_dock_widget(explorer, name="Object Explorer", area="right")
        return explorer

    # -- drawing ----------------------------------------------------------

    def _source_layer(self):
        return find_source_layer(self.viewer, self.session)

    def _remove_layer(self, layer) -> None:
        if layer is not None and self.viewer is not None and layer in self.viewer.layers:
            self.viewer.layers.remove(layer)

    def point_coordinates(self, result: NeighborhoodResult) -> np.ndarray:
        """Neighbourhood centres, with the source image's channel axis put
        back (at channel 0) so the points land on the same world axes."""
        centers = result.neighborhoods.centers()
        channel_axis = self.session.channel_axis
        source = self._source_layer()
        source_ndim = getattr(getattr(source, "data", None), "ndim", None)
        if (
            channel_axis is not None
            and source_ndim is not None
            and centers.ndim == 2
            and centers.shape[1] == source_ndim - 1
        ):
            centers = np.insert(centers, channel_axis, 0.0, axis=1)
        return centers

    def point_size(self, result: NeighborhoodResult) -> float:
        """A neighbourhood's diameter, in voxels of the image's last axis."""
        params = result.neighborhoods.params
        if result.neighborhoods.method == NEAREST:
            reach = float(np.nanmean(result.table["mean_distance"])) if len(result.table) else 1.0
        else:
            reach = float(params.get("radius", 1.0))
        spacing = self.session.spacing
        if spacing is not None and spacing.is_known and params.get("unit") != "voxel":
            reach /= float(spacing.values[-1])
        return max(2.0 * reach, 1.0)

    def show_neighborhoods(self) -> None:
        """One point per neighbourhood, coloured by the chosen feature."""
        name, result = self.current_result()
        if self.viewer is None or result is None or len(result.neighborhoods) == 0:
            return
        self._remove_layer(self._points_layers.pop(name, None))
        coordinates = self.point_coordinates(result)
        placement = placement_over(self._source_layer(), _Shape(coordinates.shape[1]))
        layer = self.viewer.add_points(
            coordinates,
            name=f"{POINTS_LAYER_PREFIX}: {name}",
            size=self.point_size(result),
            features=result.table.reset_index(drop=True),
            opacity=0.6,
            **placement,
        )
        self._color_points(layer, result, self.color_combo.currentText())
        layer.mouse_drag_callbacks.append(self._on_points_clicked)
        self._points_layers[name] = layer

    def _color_points(self, layer, result: NeighborhoodResult, column: str) -> None:
        if not column or column not in result.table.columns:
            return
        try:
            if _is_categorical(result.table[column]):
                layer.face_color = column
            else:
                layer.face_colormap = "viridis"
                layer.face_color = column
        except Exception:  # noqa: BLE001 - display-only; napari's colour API has moved
            logger.debug("could not colour %s by %s", layer.name, column)

    def _on_points_clicked(self, layer, event) -> None:
        name = next((n for n, known in self._points_layers.items() if known is layer), None)
        result = self.session.neighborhood_results.get(name) if name else None
        if result is None:
            return
        try:
            index = layer.get_value(event.position, world=True)
        except Exception:  # noqa: BLE001 - a click napari cannot resolve selects nothing
            return
        if index is None:
            return
        self.select_neighborhood(int(result.neighborhoods.ids()[int(index)]))

    def paint_objects(self) -> None:
        """Every object in the colour of what it took on - by default the type
        of neighbourhood it lives in."""
        name, result = self.current_result()
        column = self.paint_combo.currentText()
        if self.viewer is None or result is None or result.reflected is None or not column:
            return
        labels = self.session.labels(result.source_table)
        if labels is None or column not in result.reflected.columns:
            return
        values = result.reflected[column]
        ids = result.reflected[result.member_id_column].to_numpy()
        if _is_categorical(values):
            # 0 is background in a label image, so category c is drawn as c + 1
            # and an object in no neighbourhood (-1) is not drawn at all.
            codes = values.fillna(MISSING_CATEGORY).astype(np.int64).to_numpy() + 1
        else:
            # A quantity: drawn in ten bands, low to high.
            numeric = values.to_numpy(dtype=float)
            finite = np.isfinite(numeric)
            codes = np.zeros(len(numeric), dtype=np.int64)
            if finite.any():
                edges = np.nanpercentile(numeric[finite], np.linspace(0, 100, 11)[1:-1])
                codes[finite] = np.searchsorted(edges, numeric[finite]) + 1
        painted = class_map(np.asarray(labels), ids, codes)
        painted = align_to(painted, self._source_layer(), self.session.channel_axis)
        self._remove_layer(self._types_layers.pop(name, None))
        layer = self.viewer.add_labels(
            painted,
            name=f"{TYPES_LAYER_PREFIX}: {name}",
            opacity=0.5,
            **placement_over(self._source_layer(), painted),
        )
        self._types_layers[name] = layer

    def select_neighborhood(self, neighborhood_id: int) -> tuple[int, ...]:
        """Show only the members of one neighbourhood; returns them."""
        _name, result = self.current_result()
        if result is None:
            return ()
        neighborhood = result.neighborhoods.get(neighborhood_id)
        if neighborhood is None:
            self.status_label.setText(f"No neighbourhood {neighborhood_id}.")
            return ()
        self.select_spin.setValue(int(neighborhood_id))
        row = result.table[result.table[NEIGHBORHOOD_ID] == neighborhood_id]
        detail = ""
        if NEIGHBORHOOD_TYPE in row.columns and len(row):
            detail = f", type {int(row[NEIGHBORHOOD_TYPE].iloc[0])}"
        self.status_label.setText(
            f"Neighbourhood {neighborhood_id}: {len(neighborhood.members)} members{detail}."
        )
        self.neighborhood_selected.emit(int(neighborhood_id))
        labels = self.session.labels(result.source_table)
        if self.viewer is not None and labels is not None:
            self._remove_layer(self._members_layer)
            data = align_to(
                highlight_array(labels, list(neighborhood.members)),
                self._source_layer(),
                self.session.channel_axis,
            )
            self._members_layer = self.viewer.add_labels(
                data, name=MEMBERS_LAYER_NAME, **placement_over(self._source_layer(), data)
            )
        return neighborhood.members


class _Shape:
    """Stands in for an array of a given dimensionality where only `ndim`
    is read - `placement_over` asks a layer's data how many axes it has."""

    def __init__(self, ndim: int):
        self.ndim = ndim


__all__ = ["NeighborhoodSettings", "NeighborhoodWidget", "run_neighborhood_analysis"]
