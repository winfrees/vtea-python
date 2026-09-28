"""The Context pane and the session's levels: moving between length
scales, defining the next level up, and drawing neighbourhoods as outlines
of the gated ones only."""

import numpy as np
import pandas as pd
import pytest
from vtea_core.context import NEIGHBORHOOD, ContextSpec, LevelSpec
from vtea_core.gates import Gate, rectangle_vertices
from vtea_core.objects import Cell, CellSet, ObjectRef

from vtea_napari.session import AnalysisSession
from vtea_napari.widgets.context import (
    LAYER_PREFIX,
    PREVIEW_LAYER,
    SELECTION_LAYER,
    ContextWidget,
)


def field(n_side=8, step=10):
    """Nuclei on a grid, one lysosome each; left half class 0, right class 1."""
    size = n_side * step
    nuclei_labels = np.zeros((size, size), dtype=np.int32)
    lysosome_labels = np.zeros((size, size), dtype=np.int32)
    nuclei, lysosomes, cells = [], [], []
    object_id = 0
    for i in range(n_side):
        for j in range(n_side):
            object_id += 1
            y, x = i * step + 4, j * step + 4
            nuclei_labels[y - 1 : y + 2, x - 1 : x + 2] = object_id
            lysosome_labels[y + 3, x + 3] = 1000 + object_id
            nuclei.append(
                {"object_id": object_id, "centroid-0": float(y), "centroid-1": float(x),
                 "kmeans_1": int(j >= n_side // 2)}
            )
            lysosomes.append(
                {"object_id": 1000 + object_id, "centroid-0": y + 3.0, "centroid-1": x + 3.0,
                 "mean": float(j)}
            )
            cells.append(
                Cell(object_id, ObjectRef("nuclei", object_id),
                     {"lysosomes": (ObjectRef("lysosomes", 1000 + object_id),)})
            )
    return (
        pd.DataFrame(nuclei),
        pd.DataFrame(lysosomes),
        CellSet(cells, single_roles=["lysosomes"]),
        nuclei_labels,
        lysosome_labels,
    )


def make_session(spec=None):
    nuclei, lysosomes, cells, nuclei_labels, lysosome_labels = field()
    cell_table = pd.DataFrame(
        {
            "cell_id": nuclei["object_id"],
            "nuclei.centroid-0": nuclei["centroid-0"],
            "nuclei.centroid-1": nuclei["centroid-1"],
            "nuclei.kmeans_1": nuclei["kmeans_1"],
        }
    )
    session = AnalysisSession()
    if spec is not None:
        session.context_spec = spec
    session.set_context(
        {"nuclei": nuclei_labels, "lysosomes": lysosome_labels, "cells": cells},
        nuclei,
        context_inputs={
            "measurement_tables": {"nuclei": nuclei, "lysosomes": lysosomes},
            "cells": {"cells": cells},
            "cell_tables": {"cells": cell_table},
        },
    )
    return session


def neighbourhood_spec(**display):
    return ContextSpec(
        [
            LevelSpec("nuclei", "subcellular", "nuclei"),
            LevelSpec("lysosomes", "subcellular", "lysosomes"),
            LevelSpec("cells", "cellular", "cells"),
            LevelSpec(
                "nbhd_1",
                NEIGHBORHOOD,
                "cells",
                build={"radius": 12.0},
                measure={"class_column": "nuclei.kmeans_1"},
                classify={"n_clusters": 2},
            ),
        ]
    )


def viewer_model():
    from napari.components import ViewerModel

    return ViewerModel()


def gate_all(session, table, x="centroid-0", y="centroid-1", upto=1e9):
    view = session.tables[table]
    view.gate_set.add(Gate("g", x, y, rectangle_vertices(-1, -1, upto, 1e9)))
    session.notify_gates_changed()


class TestSessionLevels:
    def test_the_run_defines_its_own_levels(self):
        session = make_session()
        names = [level.name for level in session.context_levels()]
        assert names == ["lysosomes", "nuclei", "cells"]
        assert session.active_level == "lysosomes"
        # Until a level is defined above the cells, each would only repeat a
        # table the builder already published - so none is added.
        assert "cells" not in session.tables

    def test_level_tables_appear_once_a_level_is_defined(self):
        session = make_session(neighbourhood_spec())
        assert session.tables["cells"].noun == "cells"
        assert session.tables["nbhd_1"].noun == "neighborhoods"

    def test_without_cells_each_segmentation_is_the_cellular_level(self):
        nuclei, *_rest, nuclei_labels, _ = field()
        session = AnalysisSession()
        session.set_context(
            {"nuclei": nuclei_labels},
            nuclei,
            context_inputs={"measurement_tables": {"nuclei": nuclei}},
        )
        (level,) = session.context_levels()
        assert level.tier == "cellular" and level.id_column == "object_id"

    def test_defined_levels_are_built_and_reflected_down(self):
        session = make_session(neighbourhood_spec())
        assert "nbhd_1" in session.tables
        lysosomes = session.results_table("lysosomes")
        assert "nbhd_1.neighborhood_type" in lysosomes.columns
        assert "nbhd_1.neighborhood_type" in session.categorical_columns(lysosomes)

    def test_a_level_that_cannot_be_built_is_named_and_the_rest_survive(self):
        spec = neighbourhood_spec()
        spec.add(LevelSpec("ghost", "subcellular", "renamed_segmentation"))
        session = make_session(spec)
        assert "ghost" in session.context_errors()
        assert "cells" in session.tables

    def test_moving_level_moves_the_explorer_table(self):
        session = make_session(neighbourhood_spec())
        session.set_active_level("cells")
        assert session.active_table == "cells"

    def test_related_entities_across_levels(self):
        session = make_session(neighbourhood_spec())
        pieces = session.related_ids("cells", [1, 2], "lysosomes")
        assert set(pieces) == {1001, 1002}
        around = session.related_ids("cells", [1], "nbhd_1")
        assert 1 in around

    def test_gated_ids_follow_visible_gates(self):
        session = make_session(neighbourhood_spec())
        assert len(session.gated_ids("nuclei")) == 0
        gate_all(session, "nuclei", upto=20)
        gated = session.gated_ids("nuclei")
        assert 0 < len(gated) < 64


class TestPane:
    def test_the_slider_runs_through_the_levels(self, qtbot):
        session = make_session(neighbourhood_spec())
        widget = ContextWidget(session=session)
        qtbot.addWidget(widget)
        assert widget.slider.maximum() == 3
        widget.slider.setValue(2)
        assert session.active_level == "cells"
        assert session.active_table == "cells"
        assert "lysosomes ▸ nuclei ▸ cells ▸ nbhd_1" in widget.stack_label.text()

    def test_following_the_explorer_table_menu(self, qtbot):
        session = make_session(neighbourhood_spec())
        widget = ContextWidget(session=session)
        qtbot.addWidget(widget)
        session.set_active_table("nbhd_1")
        assert session.active_level == "nbhd_1"
        assert widget.slider.value() == 3

    def test_defining_a_level_builds_it_and_goes_to_it(self, qtbot):
        session = make_session()
        widget = ContextWidget(session=session)
        qtbot.addWidget(widget)
        widget.go_to("cells")
        widget.radius_spin.setValue(12.0)
        widget.class_combo.setCurrentIndex(widget.class_combo.findData("nuclei.kmeans_1"))
        widget.types_spin.setValue(2)
        assert widget.define_level() == "nbhd_1"
        assert session.active_level == "nbhd_1"
        assert session.context_spec.get("nbhd_1").source == "cells"
        # and a level above that one, from it
        widget.class_combo.setCurrentIndex(widget.class_combo.findData("neighborhood_type"))
        widget.radius_spin.setValue(30.0)
        assert widget.define_level() == "nbhd_2"
        assert "nbhd_2.neighborhood_type" in session.results_table("lysosomes").columns

    def test_a_level_cannot_be_defined_on_its_own_type(self, qtbot):
        session = make_session(neighbourhood_spec())
        widget = ContextWidget(session=session)
        qtbot.addWidget(widget)
        widget.go_to("cells")
        offered = {widget.class_combo.itemData(i) for i in range(widget.class_combo.count())}
        assert "nuclei.kmeans_1" in offered
        assert "nbhd_1.neighborhood_type" not in offered

    def test_removing_a_level_removes_what_was_built_on_it(self, qtbot):
        spec = neighbourhood_spec()
        spec.add(LevelSpec("nbhd_2", NEIGHBORHOOD, "nbhd_1", build={"radius": 30.0}))
        session = make_session(spec)
        widget = ContextWidget(session=session)
        qtbot.addWidget(widget)
        widget.go_to("nbhd_1")
        widget.remove_level()
        assert session.context_spec.get("nbhd_1") is None
        assert session.context_spec.get("nbhd_2") is None
        assert session.active_level == "cells"

    def test_the_display_is_saved_with_the_definition(self, qtbot):
        session = make_session(neighbourhood_spec())
        widget = ContextWidget(session=session)
        qtbot.addWidget(widget)
        widget.go_to("nbhd_1")
        widget.pattern_combo.setCurrentText("crosshatch")
        widget.set_display(outline_color="#ff00ff")
        display = session.context_spec.get("nbhd_1").display
        assert display.fill_pattern == "crosshatch"
        assert display.outline_color == "#ff00ff"


class TestOnTheViewer:
    def layers(self, viewer):
        return {layer.name: layer for layer in viewer.layers}

    def test_nothing_is_drawn_until_something_is_gated(self, qtbot):
        session = make_session(neighbourhood_spec())
        viewer = viewer_model()
        widget = ContextWidget(napari_viewer=viewer, session=session)
        qtbot.addWidget(widget)
        widget.go_to("nbhd_1")
        assert f"{LAYER_PREFIX}: nbhd_1" not in self.layers(viewer)
        gate_all(session, "nbhd_1", upto=20)
        outlines = self.layers(viewer)[f"{LAYER_PREFIX}: nbhd_1"]
        assert 0 < len(outlines.data) < 64

    def test_all_can_be_asked_for(self, qtbot):
        session = make_session(neighbourhood_spec())
        viewer = viewer_model()
        widget = ContextWidget(napari_viewer=viewer, session=session)
        qtbot.addWidget(widget)
        widget.go_to("nbhd_1")
        widget.set_display(draw="all")
        assert len(self.layers(viewer)[f"{LAYER_PREFIX}: nbhd_1"].data) == 64

    @pytest.mark.parametrize(
        ("pattern", "fill_type"), [("hatch", "Shapes"), ("crosshatch", "Shapes"), ("dots", "Points")]
    )
    def test_fill_patterns(self, qtbot, pattern, fill_type):
        session = make_session(neighbourhood_spec())
        viewer = viewer_model()
        widget = ContextWidget(napari_viewer=viewer, session=session)
        qtbot.addWidget(widget)
        widget.go_to("nbhd_1")
        widget.set_display(draw="all", fill_pattern=pattern, fill_color="#00ff00")
        fill = self.layers(viewer)[f"{LAYER_PREFIX}: nbhd_1 fill"]
        assert type(fill).__name__ == fill_type
        assert len(fill.data) > 0

    def test_no_fill_draws_outlines_only(self, qtbot):
        session = make_session(neighbourhood_spec())
        viewer = viewer_model()
        widget = ContextWidget(napari_viewer=viewer, session=session)
        qtbot.addWidget(widget)
        widget.go_to("nbhd_1")
        widget.set_display(draw="all", fill_pattern="none")
        assert f"{LAYER_PREFIX}: nbhd_1 fill" not in self.layers(viewer)
        assert f"{LAYER_PREFIX}: nbhd_1" in self.layers(viewer)

    def test_carrying_a_gated_selection_up(self, qtbot):
        session = make_session(neighbourhood_spec())
        viewer = viewer_model()
        widget = ContextWidget(napari_viewer=viewer, session=session)
        qtbot.addWidget(widget)
        widget.go_to("cells")
        gate_all(session, "cells", x="nuclei.centroid-0", y="nuclei.centroid-1", upto=5)
        gated = session.gated_ids("cells")
        widget.go_to("nbhd_1")
        drawn = widget.drawn_ids("nbhd_1")
        assert set(drawn) == set(session.related_ids("cells", gated, "nbhd_1"))
        assert f"{LAYER_PREFIX}: nbhd_1" in self.layers(viewer)

    def test_carrying_a_gated_selection_down_to_the_pieces(self, qtbot):
        session = make_session(neighbourhood_spec())
        viewer = viewer_model()
        widget = ContextWidget(napari_viewer=viewer, session=session)
        qtbot.addWidget(widget)
        widget.go_to("cells")
        gate_all(session, "cells", x="nuclei.centroid-0", y="nuclei.centroid-1", upto=5)
        widget.go_to("lysosomes")
        shown = np.asarray(self.layers(viewer)[SELECTION_LAYER].data)
        expected = {1000 + cell for cell in session.gated_ids("cells")}
        assert set(np.unique(shown)) - {0} == expected

    def test_the_radius_preview_outlines_one_neighbourhood(self, qtbot):
        session = make_session()
        viewer = viewer_model()
        widget = ContextWidget(napari_viewer=viewer, session=session)
        qtbot.addWidget(widget)
        widget.go_to("cells")
        widget.radius_spin.setValue(15.0)
        summary = widget.preview()
        assert summary["min"] >= 1 and summary["median"] >= summary["min"]
        preview = [layer for layer in viewer.layers if layer.name.startswith(LAYER_PREFIX)]
        assert len(preview) == 1 and len(preview[0].data) == 1
        assert PREVIEW_LAYER in widget._layers


class TestProtocol:
    def test_the_builder_saves_and_reopens_the_level_definitions(self, qtbot, tmp_path):
        from vtea_napari.widgets.protocol_builder import ProtocolBuilderWidget

        session = make_session(neighbourhood_spec())
        builder = ProtocolBuilderWidget(session=session)
        qtbot.addWidget(builder)
        path = builder.save_protocol_to(tmp_path / "p.vtea.json")

        fresh = AnalysisSession()
        reopened = ProtocolBuilderWidget(session=fresh)
        qtbot.addWidget(reopened)
        reopened.load_protocol_from(path)
        assert fresh.context_spec.get("nbhd_1").build == {"radius": 12.0}
