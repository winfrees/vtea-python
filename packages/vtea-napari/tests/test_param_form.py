from vtea_napari.widgets.param_form import ParameterForm, editable_parameters


class TestEditableParameters:
    def test_excludes_data_parameters(self):
        specs = editable_parameters("segmentation", "threshold_mask")
        names = {s.name for s in specs}
        assert "volume" not in names
        assert {"method", "value", "percentile"} <= names

    def test_infers_kind_from_default(self):
        specs = {s.name: s for s in editable_parameters("segmentation", "threshold_mask")}
        assert specs["method"].kind == "str"
        assert specs["method"].default == "fixed"
        assert specs["value"].required is False

    def test_required_parameter_with_no_default(self):
        specs = {s.name: s for s in editable_parameters("clustering", "kmeans")}
        assert "data" not in specs  # excluded as a data parameter
        assert specs["n_clusters"].required is True

    def test_two_data_parameters_both_excluded(self):
        specs = {s.name: s for s in editable_parameters("segmentation", "watershed_split")}
        assert "intensity" not in specs
        assert "mask" not in specs
        assert "min_distance" in specs


class TestParameterForm:
    def test_builds_one_field_per_editable_parameter(self, qtbot):
        form = ParameterForm("segmentation", "threshold_mask")
        qtbot.addWidget(form)
        assert set(form._field_widgets) == {"method", "value", "percentile"}

    def test_get_values_returns_defaults_initially(self, qtbot):
        form = ParameterForm("segmentation", "threshold_mask")
        qtbot.addWidget(form)
        values = form.get_values()
        assert values["method"] == "fixed"
        assert values["value"] is None

    def test_set_and_get_values_round_trip(self, qtbot):
        form = ParameterForm("segmentation", "label_components")
        qtbot.addWidget(form)
        form.set_values({"connectivity": 2})
        assert form.get_values() == {"connectivity": 2}

    def test_required_int_field_defaults_to_zero(self, qtbot):
        form = ParameterForm("clustering", "kmeans")
        qtbot.addWidget(form)
        assert form.get_values()["n_clusters"] == 0
