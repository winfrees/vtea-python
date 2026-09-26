"""The Neighborhoods pane: neighbourhoods as a second level of objects,
published beside the object table and drawn on the same viewer - and what
they reflect back onto the objects, surviving a re-run of the protocol."""

import numpy as np
import pandas as pd
import pytest
from vtea_core.neighborhoods import NEIGHBORHOOD_ID, NEIGHBORHOOD_TYPE, GRID, NEAREST

from vtea_napari.session import OBJECT_TABLE, AnalysisSession, TableView
from vtea_napari.widgets.neighborhoods import (
    MEMBERS_LAYER_NAME,
    POINTS_LAYER_PREFIX,
    TYPES_LAYER_PREFIX,
    NeighborhoodSettings,
    NeighborhoodWidget,
    run_neighborhood_analysis,
)


def field(n_side=12, spacing=8):
    """A grid of one-pixel objects; the left half class 0, the right class 1."""
    labels = np.zeros((n_side * spacing, n_side * spacing), dtype=np.int32)
    rows = []
    object_id = 0
    for i in range(n_side):
        for j in range(n_side):
            object_id += 1
            y, x = i * spacing + spacing // 2, j * spacing + spacing // 2
            labels[y, x] = object_id
            rows.append(
                {
                    "object_id": object_id,
                    "centroid-0": float(y),
                    "centroid-1": float(x),
                    "mean": float(j),
                    "kmeans_1": int(j >= n_side // 2),
                }
            )
    return pd.DataFrame(rows), labels


def session_with_objects():
    frame, labels = field()
    session = AnalysisSession()
    session.set_context({"labels": labels}, frame)
    session.feature_catalog.record_derived(["kmeans_1"], function="clustering.kmeans")
    return session, frame, labels


def viewer_model():
    from napari.components import ViewerModel

    return ViewerModel()


def build(widget, **settings):
    widget.class_combo.setCurrentIndex(widget.class_combo.findData(settings.pop("class_column", "kmeans_1")))
    method = settings.pop("method", None)
    if method is not None:
        widget.method_combo.setCurrentIndex(widget.method_combo.findData(method))
    widget.radius_spin.setValue(settings.pop("radius", 12.0))
    widget.types_spin.setValue(settings.pop("n_types", 2))
    return widget.build()


class TestRunAnalysis:
    def test_builds_measures_types_and_reflects(self):
        frame, _labels = field()
        settings = NeighborhoodSettings(radius=12, class_column="kmeans_1", n_types=2)
        neighborhoods, table, reflected = run_neighborhood_analysis(frame, settings, prefix="n.")
        assert len(neighborhoods) == len(frame)
        assert NEIGHBORHOOD_TYPE in table.columns
        assert f"n.{NEIGHBORHOOD_TYPE}" in reflected.columns
        types = reflected.set_index("object_id")[f"n.{NEIGHBORHOOD_TYPE}"]
        halves = frame.set_index("object_id")["kmeans_1"]
        agreement = pd.crosstab(halves, types)
        assert (agreement.max(axis=1) / agreement.sum(axis=1)).min() > 0.9

    def test_without_classes_it_measures_density_and_does_not_type(self):
        frame, _labels = field()
        _n, table, reflected = run_neighborhood_analysis(frame, NeighborhoodSettings(radius=12))
        assert NEIGHBORHOOD_TYPE not in table.columns
        assert "n_objects" in reflected.columns

    def test_types_on_member_features_when_there_are_no_classes(self):
        frame, _labels = field()
        settings = NeighborhoodSettings(radius=12, features="mean", n_types=2)
        _n, table, _reflected = run_neighborhood_analysis(frame, settings)
        assert NEIGHBORHOOD_TYPE in table.columns

    def test_a_grid_falls_back_to_membership(self):
        frame, _labels = field()
        settings = NeighborhoodSettings(method=GRID, radius=12, relation="own", class_column="kmeans_1")
        neighborhoods, _table, reflected = run_neighborhood_analysis(frame, settings)
        assert not neighborhoods.centred
        assert len(reflected) == len(frame)


class TestPane:
    def test_offers_the_member_tables_and_their_categories(self, qtbot):
        session, _frame, _labels = session_with_objects()
        widget = NeighborhoodWidget(session=session)
        qtbot.addWidget(widget)
        assert widget.member_tables() == [OBJECT_TABLE]
        assert widget.class_combo.findData("kmeans_1") >= 0

    def test_build_publishes_a_neighbourhood_table(self, qtbot):
        session, frame, _labels = session_with_objects()
        widget = NeighborhoodWidget(session=session)
        qtbot.addWidget(widget)
        result = build(widget)
        assert result is not None
        view = session.tables["nbhd_1"]
        assert view.id_column == NEIGHBORHOOD_ID
        assert view.noun == "neighborhoods"
        # centred on objects, so a gate on neighbourhoods lights up objects
        assert view.labels_key == session.tables[OBJECT_TABLE].labels_key
        assert len(view.frame) == len(frame)

    def test_objects_gain_their_neighbourhoods_characteristics(self, qtbot):
        session, _frame, _labels = session_with_objects()
        widget = NeighborhoodWidget(session=session)
        qtbot.addWidget(widget)
        build(widget)
        objects = session.results_table(OBJECT_TABLE)
        assert f"nbhd_1.{NEIGHBORHOOD_TYPE}" in objects.columns
        assert "nbhd_1.Class_1_ClassFraction" in objects.columns
        # and the type is coloured as a category, not a quantity
        assert f"nbhd_1.{NEIGHBORHOOD_TYPE}" in session.categorical_columns(objects)

    def test_reflected_columns_survive_a_rerun_of_the_protocol(self, qtbot):
        session, frame, labels = session_with_objects()
        widget = NeighborhoodWidget(session=session)
        qtbot.addWidget(widget)
        build(widget)
        # the builder re-publishes, having lost one object
        session.set_context({"labels": labels}, frame.iloc[1:].reset_index(drop=True))
        objects = session.results_table(OBJECT_TABLE)
        assert f"nbhd_1.{NEIGHBORHOOD_TYPE}" in objects.columns
        assert len(objects) == len(frame) - 1
        assert "nbhd_1" in session.tables

    def test_a_second_analysis_is_built_on_the_protocols_table(self, qtbot):
        session, _frame, _labels = session_with_objects()
        widget = NeighborhoodWidget(session=session)
        qtbot.addWidget(widget)
        build(widget)
        build(widget, method=NEAREST)
        second = session.neighborhood_results["nbhd_2"]
        assert not any(str(c).startswith("nbhd_1.") for c in second.reflected.columns)
        objects = session.results_table(OBJECT_TABLE)
        assert "nbhd_1.n_neighborhoods" in objects.columns
        assert "nbhd_2.n_neighborhoods" in objects.columns

    def test_remove_takes_its_table_and_columns_away(self, qtbot):
        session, _frame, _labels = session_with_objects()
        widget = NeighborhoodWidget(session=session)
        qtbot.addWidget(widget)
        build(widget)
        widget.remove_current()
        assert "nbhd_1" not in session.tables
        assert not any(str(c).startswith("nbhd_1.") for c in session.results_table(OBJECT_TABLE).columns)

    def test_refuses_a_name_that_is_already_a_table(self, qtbot):
        session, _frame, _labels = session_with_objects()
        widget = NeighborhoodWidget(session=session)
        qtbot.addWidget(widget)
        widget.name_edit.setText(OBJECT_TABLE)
        assert build(widget) is None
        assert "already a table" in widget.status_label.text()

    def test_says_so_when_there_is_nothing_to_build_from(self, qtbot):
        widget = NeighborhoodWidget(session=AnalysisSession())
        qtbot.addWidget(widget)
        assert widget.build() is None
        assert "Nothing to build from" in widget.status_label.text()

    def test_cells_can_be_the_members(self, qtbot):
        session, frame, labels = session_with_objects()
        cells = frame.rename(columns={"object_id": "cell_id"})
        session.set_context(
            {"labels": labels},
            frame,
            {"cells_1": TableView(cells, id_column="cell_id", labels_key="labels", noun="cells")},
        )
        widget = NeighborhoodWidget(session=session)
        qtbot.addWidget(widget)
        widget.table_combo.setCurrentText("cells_1")
        build(widget)
        assert "nbhd_1.n_neighborhoods" in session.results_table("cells_1").columns
        assert "nbhd_1.n_neighborhoods" not in session.results_table(OBJECT_TABLE).columns

    def test_opening_a_protocol_forgets_the_analyses(self, qtbot):
        session, _frame, _labels = session_with_objects()
        widget = NeighborhoodWidget(session=session)
        qtbot.addWidget(widget)
        build(widget)
        session.clear_results()
        assert session.neighborhood_results == {}
        assert widget.result_combo.count() == 0


class TestOnTheViewer:
    def test_draws_both_levels(self, qtbot):
        session, _frame, _labels = session_with_objects()
        viewer = viewer_model()
        widget = NeighborhoodWidget(napari_viewer=viewer, session=session)
        qtbot.addWidget(widget)
        build(widget)
        names = [layer.name for layer in viewer.layers]
        assert f"{POINTS_LAYER_PREFIX}: nbhd_1" in names
        assert f"{TYPES_LAYER_PREFIX}: nbhd_1" in names
        points = viewer.layers[f"{POINTS_LAYER_PREFIX}: nbhd_1"]
        assert len(points.data) == len(session.tables["nbhd_1"].frame)
        assert NEIGHBORHOOD_TYPE in points.features.columns

    def test_objects_are_painted_by_their_neighbourhood_type(self, qtbot):
        session, frame, labels = session_with_objects()
        viewer = viewer_model()
        widget = NeighborhoodWidget(napari_viewer=viewer, session=session)
        qtbot.addWidget(widget)
        build(widget)
        painted = np.asarray(viewer.layers[f"{TYPES_LAYER_PREFIX}: nbhd_1"].data)
        reflected = session.neighborhood_results["nbhd_1"].reflected.set_index("object_id")
        some = int(frame["object_id"].iloc[5])
        assert painted[labels == some].max() == reflected.loc[some, f"nbhd_1.{NEIGHBORHOOD_TYPE}"] + 1
        assert painted[labels == 0].max() == 0

    def test_selecting_a_neighbourhood_shows_only_its_members(self, qtbot):
        session, _frame, labels = session_with_objects()
        viewer = viewer_model()
        widget = NeighborhoodWidget(napari_viewer=viewer, session=session)
        qtbot.addWidget(widget)
        build(widget)
        members = widget.select_neighborhood(20)
        shown = np.asarray(viewer.layers[MEMBERS_LAYER_NAME].data)
        assert set(np.unique(shown)) - {0} == set(members)
        assert 20 in members

    def test_recolouring_rebuilds_the_points(self, qtbot):
        session, _frame, _labels = session_with_objects()
        viewer = viewer_model()
        widget = NeighborhoodWidget(napari_viewer=viewer, session=session)
        qtbot.addWidget(widget)
        build(widget)
        widget.color_combo.setCurrentText("n_objects")
        assert len([layer for layer in viewer.layers if layer.name.startswith(POINTS_LAYER_PREFIX)]) == 1

    def test_removing_takes_the_layers_with_it(self, qtbot):
        session, _frame, _labels = session_with_objects()
        viewer = viewer_model()
        widget = NeighborhoodWidget(napari_viewer=viewer, session=session)
        qtbot.addWidget(widget)
        build(widget)
        widget.remove_current()
        assert not [layer for layer in viewer.layers if "nbhd_1" in layer.name]

    @pytest.mark.parametrize("method", [GRID])
    def test_a_grid_is_drawn_too(self, qtbot, method):
        session, _frame, _labels = session_with_objects()
        viewer = viewer_model()
        widget = NeighborhoodWidget(napari_viewer=viewer, session=session)
        qtbot.addWidget(widget)
        build(widget, method=method, radius=20)
        assert session.tables["nbhd_1"].labels_key == ""
        assert f"{POINTS_LAYER_PREFIX}: nbhd_1" in [layer.name for layer in viewer.layers]
