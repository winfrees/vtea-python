import numpy as np
import pandas as pd
import pytest

from vtea_core.classes import (
    CROSS,
    INTERSECT,
    UNION,
    LabelSet,
    LabelSetCollection,
    ObjectLabel,
    class_from_expression,
    class_from_range,
    class_from_values,
    combine_label_sets,
    combine_labels,
    label_image,
    label_set,
)


def a_table():
    """Five objects: two immune, three epithelial; two of them CD3+."""
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3, 4, 5],
            "mean_ch2": [10.0, 60.0, 120.0, 160.0, 80.0],
            "kmeans_1": [0, 1, 1, 0, 1],
            "immune": [True, True, False, False, False],
            "epithelial": [False, False, True, True, True],
            "cd3": [True, False, False, True, False],
        }
    )


def a_set(name="populations", labels=("immune", "epithelial")):
    frame = a_table()
    return LabelSet(
        name,
        [ObjectLabel(one, frame[one].to_numpy(), definition=one) for one in labels],
        n_objects=len(frame),
        object_ids=frame["object_id"].to_numpy(),
    )


class TestLabelSet:
    def test_an_object_can_carry_several_labels(self):
        frame = a_table()
        both = LabelSet(
            "mixed",
            [
                ObjectLabel("immune", frame["immune"].to_numpy()),
                ObjectLabel("cd3", frame["cd3"].to_numpy()),
            ],
        )
        assert list(both.labels_per_object()) == [2, 1, 0, 1, 0]
        assert both.labels_for(0) == ["immune", "cd3"]

    def test_counts_are_per_label(self):
        assert a_set().counts() == {"immune": 2, "epithelial": 3}

    def test_codes_report_the_first_matching_label(self):
        assert list(a_set().codes()) == [0, 0, 1, 1, 1]

    def test_an_object_with_no_label_gets_minus_one(self):
        frame = a_table()
        sparse = LabelSet("sparse", [ObjectLabel("cd3", frame["cd3"].to_numpy())])
        assert list(sparse.codes()) == [0, -1, -1, 0, -1]

    def test_membership_is_the_whole_answer(self):
        assert a_set().membership().shape == (5, 2)

    def test_labels_from_different_tables_are_refused(self):
        wrong = LabelSet("x", [ObjectLabel("a", np.ones(5, dtype=bool))])
        with pytest.raises(ValueError, match="built against different tables"):
            wrong.add(ObjectLabel("b", np.ones(3, dtype=bool)))

    def test_two_labels_cannot_share_a_name(self):
        with pytest.raises(ValueError, match="already has a label"):
            a_set().add(ObjectLabel("immune", np.zeros(5, dtype=bool)))

    def test_it_renders_as_boolean_columns_plus_a_code(self):
        frame = a_set().to_frame()
        assert list(frame.columns) == [
            "populations.immune",
            "populations.epithelial",
            "populations.code",
        ]

    def test_the_summary_says_what_is_unlabelled_and_what_is_double_labelled(self):
        frame = a_table()
        overlapping = LabelSet(
            "overlapping",
            [
                ObjectLabel("immune", frame["immune"].to_numpy()),
                ObjectLabel("cd3", frame["cd3"].to_numpy()),
            ],
        )
        summary = overlapping.summary()
        assert "2 unlabelled" in summary
        assert "1 with more than one label" in summary

    def test_a_label_keeps_the_definition_it_came_from(self):
        """A column of booleans nobody can reproduce is not a finding."""
        built = a_set()
        assert built.get("immune").definition == "immune"


class TestCombining:
    def test_cross_builds_a_hierarchy(self):
        frame = a_table()
        populations = a_set()
        fine = LabelSet("fine", [ObjectLabel("cd3", frame["cd3"].to_numpy())])

        combined = combine_label_sets(populations, fine, mode=CROSS)

        assert "immune > cd3" in combined.names
        assert combined.get("immune > cd3").count == 1
        assert combined.parent == "populations"

    def test_objects_the_finer_set_says_nothing_about_keep_their_coarse_label(self):
        """"immune, not further typed" is a finding; dropping those objects
        would change every proportion computed afterwards."""
        frame = a_table()
        combined = combine_label_sets(
            a_set(), LabelSet("fine", [ObjectLabel("cd3", frame["cd3"].to_numpy())]), mode=CROSS
        )
        assert combined.get("immune").count == 1  # object 2, immune but not CD3+

    def test_unmatched_objects_can_be_dropped_deliberately(self):
        frame = a_table()
        combined = combine_label_sets(
            a_set(),
            LabelSet("fine", [ObjectLabel("cd3", frame["cd3"].to_numpy())]),
            mode=CROSS,
            keep_unmatched=False,
        )
        assert "immune" not in combined.names

    def test_a_combination_nothing_satisfies_is_not_a_population(self):
        """A cell type nothing is an example of is not a population, and a
        legend full of empty ones is unreadable."""
        frame = a_table()
        # cd19 marks object 3 only, which is epithelial - so "immune > cd19"
        # describes nothing.
        cd19 = np.array([False, False, True, False, False])
        combined = combine_label_sets(
            a_set(), LabelSet("fine", [ObjectLabel("cd19", cd19)]), mode=CROSS
        )
        assert "epithelial > cd19" in combined.names
        assert "immune > cd19" not in combined.names

    def test_union_keeps_both_sets_labels(self):
        frame = a_table()
        combined = combine_label_sets(
            a_set(), LabelSet("fine", [ObjectLabel("cd3", frame["cd3"].to_numpy())]), mode=UNION
        )
        assert combined.names == ["immune", "epithelial", "cd3"]

    def test_intersect_keeps_only_the_labels_both_define(self):
        frame = a_table()
        other = LabelSet(
            "second_opinion",
            [
                ObjectLabel("immune", np.array([True, False, False, False, False])),
                ObjectLabel("cd3", frame["cd3"].to_numpy()),
            ],
        )
        combined = combine_label_sets(a_set(), other, mode=INTERSECT)
        assert combined.names == ["immune"]
        assert combined.get("immune").count == 1

    def test_sets_over_different_tables_cannot_be_combined(self):
        with pytest.raises(ValueError, match="same table"):
            combine_label_sets(
                a_set(), LabelSet("other", [ObjectLabel("x", np.ones(3, dtype=bool))])
            )

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="unknown combine mode"):
            combine_label_sets(a_set(), a_set("other"), mode="sideways")

    def test_three_levels_compose(self):
        """Populations, then a fine type, then a marker - the point of the
        hierarchy being a set operation rather than a special case."""
        frame = a_table()
        fine = LabelSet("fine", [ObjectLabel("cd3", frame["cd3"].to_numpy())])
        bright = LabelSet("bright", [ObjectLabel("hi", frame["mean_ch2"].to_numpy() > 50)])
        first = combine_label_sets(a_set(), fine, mode=CROSS)
        second = combine_label_sets(first, bright, mode=CROSS)
        assert any(name.count(">") == 2 for name in second.names)


class TestLabelImage:
    def test_it_paints_classes_back_onto_the_segmentation(self):
        labels = np.array([[0, 1, 1], [2, 2, 0], [3, 3, 3]])
        painted = label_image(labels, object_ids=[1, 2, 3], codes=[0, 1, 0])
        # +1 so a class code of 0 is distinguishable from background.
        assert painted[0, 1] == 1
        assert painted[1, 0] == 2
        assert painted[2, 0] == 1
        assert painted[0, 0] == 0

    def test_an_unlabelled_object_stays_background(self):
        labels = np.array([[1, 2]])
        painted = label_image(labels, object_ids=[1, 2], codes=[-1, 0])
        assert list(painted[0]) == [0, 1]

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="describe the same objects"):
            label_image(np.zeros((2, 2), dtype=int), object_ids=[1, 2], codes=[0])


class TestClassSteps:
    def test_a_range_class(self):
        mask = class_from_range(a_table(), column="mean_ch2", minimum=50, maximum=150)
        assert list(mask) == [False, True, True, False, True]

    def test_an_open_range(self):
        mask = class_from_range(a_table(), column="mean_ch2", minimum=100)
        assert list(mask) == [False, False, True, True, False]

    def test_an_exclusive_range_excludes_its_bounds(self):
        mask = class_from_range(
            a_table(), column="mean_ch2", minimum=60, maximum=160, inclusive=False
        )
        assert list(mask) == [False, False, True, False, True]

    def test_a_missing_measurement_is_not_silently_included(self):
        frame = a_table()
        frame.loc[0, "mean_ch2"] = np.nan
        assert not class_from_range(frame, column="mean_ch2")[0]

    def test_a_range_needs_a_column(self):
        with pytest.raises(ValueError, match="needs a column"):
            class_from_range(a_table())

    def test_a_class_from_cluster_ids(self):
        mask = class_from_values(a_table(), column="kmeans_1", values="1")
        assert list(mask) == [False, True, True, False, True]

    def test_several_values_at_once(self):
        mask = class_from_values(a_table(), column="kmeans_1", values="0, 1")
        assert mask.all()

    def test_a_class_from_a_gate_column(self):
        mask = class_from_values(a_table(), column="immune", values="True")
        assert list(mask) == [True, True, False, False, False]

    def test_a_class_from_an_expression(self):
        mask = class_from_expression(a_table(), expression="immune AND cd3")
        assert list(mask) == [True, False, False, False, False]

    def test_values_must_match_the_column_s_type(self):
        with pytest.raises(ValueError, match="must be numbers"):
            class_from_values(a_table(), column="kmeans_1", values="immune")


class TestLabelSetStep:
    def test_it_takes_every_boolean_column_by_default(self):
        built = label_set(a_table(), name="populations")
        assert built.names == ["immune", "epithelial", "cd3"]

    def test_a_chosen_subset(self):
        built = label_set(a_table(), classes="immune, epithelial", name="populations")
        assert built.names == ["immune", "epithelial"]

    def test_it_records_the_objects_it_is_about(self):
        built = label_set(a_table())
        assert list(built.object_ids) == [1, 2, 3, 4, 5]

    def test_a_missing_class_column_says_which(self):
        with pytest.raises(ValueError, match="no column\\(s\\) nonsense"):
            label_set(a_table(), classes="nonsense")

    def test_the_combine_step_builds_the_hierarchy(self):
        frame = a_table()
        populations = label_set(frame, classes="immune, epithelial", name="populations")
        fine = label_set(frame, classes="cd3", name="fine")
        combined = combine_labels(populations, fine, mode="cross", name="types")
        assert combined.name == "types"
        assert "immune > cd3" in combined.names


class TestLabelSetCollection:
    def test_it_tracks_the_hierarchy(self):
        collection = LabelSetCollection()
        collection.add(a_set("populations"))
        collection.add(LabelSet("fine", [], n_objects=5, parent="populations"))
        assert [one.name for one in collection.children_of("populations")] == ["fine"]

    def test_every_set_at_once_as_one_table(self):
        collection = LabelSetCollection()
        collection.add(a_set("populations"))
        assert "populations.immune" in collection.to_frame().columns
