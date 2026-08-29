"""Picking which of dozens of measured features a clustering or reduction
step is built from."""

from vtea_core.measurements import FeatureCatalog

from vtea_napari.widgets.feature_select import FeatureSelectWidget, describe_feature

FEATURES = [
    "count",
    "mean_ch0",
    "mean_ch1",
    "mean_ch2",
    "stddev_ch0",
    "stddev_ch1",
    "stddev_ch2",
    "kmeans_1",
]


def make_catalog():
    catalog = FeatureCatalog()
    catalog.record_measured(
        [name for name in FEATURES if name != "kmeans_1"],
        produced_by="measure_1",
        function="measurements.extract_measurements_by_channel",
        segmentation="watershed_split_1",
    )
    catalog.record_derived(
        ["kmeans_1"],
        produced_by="kmeans_1",
        function="clustering.kmeans",
        params={"n_clusters": 3},
        source_features=["mean_ch0", "mean_ch1"],
    )
    return catalog


def make_widget(qtbot, selected=()):
    widget = FeatureSelectWidget(FEATURES, selected, make_catalog())
    qtbot.addWidget(widget)
    return widget


def checked(widget):
    return widget.selected()


class TestInitialState:
    def test_no_selection_checks_everything(self, qtbot):
        """A step with no recorded selection uses every feature; showing an
        empty list would misrepresent what it will do."""
        widget = make_widget(qtbot)
        assert checked(widget) == FEATURES

    def test_a_recorded_selection_is_restored(self, qtbot):
        widget = make_widget(qtbot, ["mean_ch0", "stddev_ch2"])
        assert checked(widget) == ["mean_ch0", "stddev_ch2"]

    def test_the_count_is_shown(self, qtbot):
        widget = make_widget(qtbot, ["mean_ch0"])
        assert widget.count_label.text() == "1 of 8 selected"

    def test_an_empty_feature_list_says_so(self, qtbot):
        widget = FeatureSelectWidget([], [], None)
        qtbot.addWidget(widget)
        assert "No measured features yet" in widget.count_label.text()


class TestFiltering:
    def test_the_filter_hides_non_matching_rows(self, qtbot):
        widget = make_widget(qtbot)
        widget.filter_edit.setText("stddev")
        visible = [item.text() for item in widget._visible_items()]
        assert visible == ["stddev_ch0", "stddev_ch1", "stddev_ch2"]

    def test_the_filter_matches_a_channel_suffix(self, qtbot):
        widget = make_widget(qtbot)
        widget.filter_edit.setText("_ch2")
        visible = [item.text() for item in widget._visible_items()]
        assert visible == ["mean_ch2", "stddev_ch2"]

    def test_it_is_case_insensitive(self, qtbot):
        widget = make_widget(qtbot)
        widget.filter_edit.setText("MEAN")
        upper = [item.text() for item in widget._visible_items()]
        widget.filter_edit.setText("mean")
        assert [item.text() for item in widget._visible_items()] == upper
        # A plain substring match, so "mean" also finds "kmeans_1".
        assert upper == ["mean_ch0", "mean_ch1", "mean_ch2", "kmeans_1"]

    def test_clearing_the_filter_shows_everything_again(self, qtbot):
        widget = make_widget(qtbot)
        widget.filter_edit.setText("stddev")
        widget.filter_edit.setText("")
        assert len(widget._visible_items()) == len(FEATURES)

    def test_filtering_does_not_change_the_selection(self, qtbot):
        widget = make_widget(qtbot, ["mean_ch0"])
        widget.filter_edit.setText("stddev")
        assert checked(widget) == ["mean_ch0"]


class TestBulkActions:
    def test_none_then_all_round_trips(self, qtbot):
        widget = make_widget(qtbot)
        widget.select_none()
        assert checked(widget) == []
        widget.select_all()
        assert checked(widget) == FEATURES

    def test_bulk_actions_apply_only_to_what_the_filter_shows(self, qtbot):
        """This is what makes "select every _ch2 feature" one gesture."""
        widget = make_widget(qtbot)
        widget.select_none()
        widget.filter_edit.setText("_ch2")
        widget.select_all()
        assert checked(widget) == ["mean_ch2", "stddev_ch2"]

    def test_invert_flips_only_the_visible_rows(self, qtbot):
        widget = make_widget(qtbot, ["mean_ch0", "stddev_ch0"])
        widget.filter_edit.setText("mean_ch")
        widget.invert()
        # mean_ch0 off, mean_ch1/mean_ch2 on; stddev_ch0 untouched.
        assert checked(widget) == ["mean_ch1", "mean_ch2", "stddev_ch0"]

    def test_the_count_follows_a_bulk_action(self, qtbot):
        widget = make_widget(qtbot)
        widget.select_none()
        assert widget.count_label.text() == "0 of 8 selected"

    def test_a_change_is_announced(self, qtbot):
        widget = make_widget(qtbot)
        seen = []
        widget.selection_changed.connect(lambda: seen.append(1))
        widget.select_none()
        assert seen


class TestWhatGetsStored:
    def test_everything_checked_stores_as_an_empty_selection(self, qtbot):
        """So a protocol doesn't pin a list that should grow when a later
        measurement step adds features."""
        widget = make_widget(qtbot)
        assert widget.selected_or_all() == []

    def test_a_partial_selection_stores_the_names(self, qtbot):
        widget = make_widget(qtbot, ["mean_ch1"])
        assert widget.selected_or_all() == ["mean_ch1"]

    def test_nothing_checked_stores_as_nothing_checked(self, qtbot):
        widget = make_widget(qtbot)
        widget.select_none()
        assert widget.selected_or_all() == []


class TestProvenanceTooltips:
    def test_a_measured_feature_says_what_and_where_from(self, qtbot):
        widget = make_widget(qtbot)
        tooltip = widget.list.item(FEATURES.index("mean_ch2")).toolTip()
        assert "mean" in tooltip
        assert "channel 2" in tooltip
        assert "watershed_split_1" in tooltip
        assert "measure_1" in tooltip

    def test_a_derived_feature_says_how_many_features_went_in(self, qtbot):
        widget = make_widget(qtbot)
        tooltip = widget.list.item(FEATURES.index("kmeans_1")).toolTip()
        assert "cluster assignment" in tooltip
        assert "2 feature(s)" in tooltip

    def test_an_undescribed_feature_says_so_rather_than_lying(self, qtbot):
        widget = FeatureSelectWidget(["mystery"], [], FeatureCatalog())
        qtbot.addWidget(widget)
        assert "No recorded provenance" in widget.list.item(0).toolTip()

    def test_describe_feature_handles_a_missing_descriptor(self):
        assert "No recorded provenance" in describe_feature(None)


class TestRebuilding:
    def test_new_features_appear_after_a_re_run(self, qtbot):
        widget = make_widget(qtbot, ["mean_ch0"])
        widget.set_features([*FEATURES, "pca_1_1"], ["mean_ch0", "pca_1_1"])
        assert "pca_1_1" in checked(widget)

    def test_a_selection_pointing_at_a_vanished_feature_is_dropped(self, qtbot):
        widget = make_widget(qtbot, ["mean_ch0", "gone"])
        assert checked(widget) == ["mean_ch0"]
