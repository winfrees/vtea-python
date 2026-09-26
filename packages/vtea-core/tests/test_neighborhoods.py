"""Neighbourhoods: objects made of objects, measured, typed - and reflected
back onto the objects they are made of."""

import numpy as np
import pandas as pd
import pytest
from vtea_core.data import Spacing
from vtea_core.neighborhoods import (
    MISSING_CATEGORY,
    NEIGHBORHOOD_TYPE,
    Neighborhood,
    NeighborhoodSet,
    build_neighborhoods,
    class_values,
    classify_neighborhoods,
    composition_columns,
    load_neighborhoods,
    neighborhood_features,
    reflect_neighborhoods,
    reflectable_columns,
    save_neighborhoods,
)
from vtea_core.workflow import Pipeline, Step


def line_table():
    """Five objects on a line, 10 apart; classes 0 0 1 1 1."""
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3, 4, 5],
            "centroid-0": [0.0, 0.0, 0.0, 0.0, 0.0],
            "centroid-1": [0.0, 10.0, 20.0, 30.0, 40.0],
            "mean": [1.0, 2.0, 3.0, 4.0, 5.0],
            "kind": [0, 0, 1, 1, 1],
        }
    )


def two_halves(n=300, seed=0):
    """Class 0 on the left half, class 1 on the right - two kinds of
    neighbourhood a typing should find."""
    rng = np.random.default_rng(seed)
    position = rng.uniform(0, 100, (n, 2))
    return pd.DataFrame(
        {
            "object_id": np.arange(1, n + 1),
            "centroid-0": position[:, 0],
            "centroid-1": position[:, 1],
            "kind": (position[:, 1] > 50).astype(int),
        }
    )


class TestBuild:
    def test_radius_is_the_object_and_everything_within_it(self):
        neighborhoods = build_neighborhoods(line_table(), method="radius", radius=10)
        assert neighborhoods.get(1).members == (1, 2)
        assert neighborhoods.get(3).members == (2, 3, 4)
        assert neighborhoods.get(3).seed == 3

    def test_nearest_is_the_object_and_its_k_nearest(self):
        neighborhoods = build_neighborhoods(line_table(), method="nearest", k=2)
        assert neighborhoods.get(3).members == (2, 3, 4)
        assert all(len(n.members) == 3 for n in neighborhoods)

    def test_nearest_with_k_above_the_count_takes_everything(self):
        neighborhoods = build_neighborhoods(line_table(), method="nearest", k=50)
        assert neighborhoods.get(1).members == (1, 2, 3, 4, 5)

    def test_a_centred_neighbourhood_takes_its_seeds_id(self):
        neighborhoods = build_neighborhoods(line_table(), method="radius", radius=5)
        assert list(neighborhoods.ids()) == [1, 2, 3, 4, 5]
        assert neighborhoods.centred

    def test_grid_drops_empty_points_and_numbers_the_rest(self):
        neighborhoods = build_neighborhoods(line_table(), method="grid", radius=6, interval=10)
        assert not neighborhoods.centred
        assert list(neighborhoods.ids()) == list(range(1, len(neighborhoods) + 1))
        assert all(n.members for n in neighborhoods)
        assert all(n.seed is None for n in neighborhoods)

    def test_grid_interval_defaults_to_the_radius(self):
        neighborhoods = build_neighborhoods(two_halves(), method="grid", radius=20)
        assert neighborhoods.params["interval"] == 20

    def test_distances_are_physical_when_the_voxel_size_is_known(self):
        """A 0.5 um pixel: objects 10 pixels apart are 5 um apart, so a
        6 um radius reaches the neighbour it would not in pixels."""
        spacing = Spacing(values=(0.5, 0.5), unit="um")
        in_pixels = build_neighborhoods(line_table(), method="radius", radius=6)
        in_microns = build_neighborhoods(line_table(), method="radius", radius=6, spacing=spacing)
        assert in_pixels.get(3).members == (3,)
        assert in_microns.get(3).members == (2, 3, 4)
        assert in_microns.params["unit"] == "um"

    def test_anisotropic_z_is_accounted_for(self):
        table = pd.DataFrame(
            {
                "object_id": [1, 2],
                "centroid-0": [0.0, 2.0],  # two slices apart
                "centroid-1": [0.0, 0.0],
                "centroid-2": [0.0, 0.0],
            }
        )
        coarse_z = Spacing(values=(5.0, 1.0, 1.0), unit="um")
        assert build_neighborhoods(table, radius=3).get(1).members == (1, 2)
        assert build_neighborhoods(table, radius=3, spacing=coarse_z).get(1).members == (1,)

    def test_the_centre_stays_in_voxels_for_drawing(self):
        spacing = Spacing(values=(0.5, 0.5), unit="um")
        neighborhoods = build_neighborhoods(line_table(), radius=6, spacing=spacing)
        assert neighborhoods.get(2).center == (0.0, 10.0)

    def test_randomize_keeps_the_positions_but_not_who_holds_them(self):
        table = two_halves()
        real = build_neighborhoods(table, radius=10)
        null = build_neighborhoods(table, radius=10, randomize=True, random_state=1)
        assert sorted(n.center for n in real) == sorted(n.center for n in null)
        assert real.get(1).center != null.get(1).center
        assert null.params["randomize"] is True

    def test_mean_distance_is_measured_from_the_centre(self):
        neighborhoods = build_neighborhoods(line_table(), method="radius", radius=10)
        assert neighborhoods.get(3).mean_distance == pytest.approx(20 / 3)

    def test_a_table_without_positions_is_refused(self):
        with pytest.raises(ValueError, match="centroid"):
            build_neighborhoods(pd.DataFrame({"object_id": [1, 2]}))

    def test_bad_parameters_are_refused(self):
        with pytest.raises(ValueError, match="radius"):
            build_neighborhoods(line_table(), radius=0)
        with pytest.raises(ValueError, match="k must"):
            build_neighborhoods(line_table(), method="nearest", k=0)
        with pytest.raises(ValueError, match="unknown neighbourhood method"):
            build_neighborhoods(line_table(), method="voronoi")

    def test_an_empty_table_gives_no_neighbourhoods(self):
        empty = line_table().iloc[:0]
        assert len(build_neighborhoods(empty)) == 0

    def test_cells_can_be_the_members(self):
        cells = line_table().rename(columns={"object_id": "cell_id"})
        neighborhoods = build_neighborhoods(cells, radius=10, id_column="cell_id")
        assert "cell_id" in neighborhoods.membership().columns


class TestModel:
    def test_membership_is_many_to_many(self):
        neighborhoods = build_neighborhoods(line_table(), radius=10)
        membership = neighborhoods.membership()
        assert set(membership.columns) == {"neighborhood_id", "object_id", "is_seed"}
        # object 3 is in its own neighbourhood and in those of 2 and 4
        assert sorted(neighborhoods.neighborhoods_of(3)) == [2, 3, 4]
        assert membership["is_seed"].sum() == 5

    def test_own_neighbourhood_only_exists_for_a_centred_method(self):
        assert build_neighborhoods(line_table(), radius=10).own_neighborhood(4) == 4
        grid = build_neighborhoods(line_table(), method="grid", radius=10)
        assert grid.own_neighborhood(4) is None

    def test_duplicate_ids_are_refused(self):
        with pytest.raises(ValueError, match="unique"):
            NeighborhoodSet([Neighborhood(1, (1,), (0.0,)), Neighborhood(1, (2,), (1.0,))])

    def test_round_trips_through_json(self, tmp_path):
        neighborhoods = build_neighborhoods(line_table(), radius=10, randomize=True)
        loaded = load_neighborhoods(save_neighborhoods(neighborhoods, tmp_path / "n.json"))
        assert loaded.neighborhoods == neighborhoods.neighborhoods
        assert loaded.params == neighborhoods.params
        assert loaded.method == neighborhoods.method

    def test_a_newer_file_is_refused(self):
        with pytest.raises(ValueError, match="newer"):
            NeighborhoodSet.from_dict({"vtea_neighborhood_version": 99})

    def test_summary_mentions_a_null_model(self):
        summary = build_neighborhoods(line_table(), radius=10, randomize=True).summary()
        assert "randomised" in summary


class TestFeatures:
    def test_composition_uses_the_java_names_and_units(self):
        table = line_table()
        features = neighborhood_features(
            build_neighborhoods(table, radius=10), table, class_column="kind"
        ).set_index("neighborhood_id")
        # neighbourhood of 3 is {2, 3, 4}: kinds 0, 1, 1
        assert features.loc[3, "Class_0_ClassSums"] == 1
        assert features.loc[3, "Class_1_ClassSums"] == 2
        assert features.loc[3, "Class_1_ClassFraction"] == pytest.approx(200 / 3)
        assert features.loc[3, "n_objects"] == 3

    def test_classes_run_from_zero_like_the_java(self):
        assert class_values(pd.Series([2, 4])) == [0, 1, 2, 3, 4]
        assert class_values(pd.Series([-1, 1])) == [-1, 0, 1]

    def test_named_classes_are_listed_as_they_are(self):
        assert class_values(pd.Series(["b", "a", "b"])) == ["a", "b"]

    def test_a_gate_can_be_the_class(self):
        table = line_table()
        table["gate_bright"] = table["mean"] > 2.5
        features = neighborhood_features(
            build_neighborhoods(table, radius=10), table, class_column="gate_bright"
        )
        assert {"Class_0_ClassFraction", "Class_1_ClassFraction"} <= set(features.columns)

    def test_fractions_sum_to_a_hundred(self):
        table = two_halves()
        features = neighborhood_features(build_neighborhoods(table, radius=10), table, class_column="kind")
        totals = features[composition_columns(features)].sum(axis=1)
        np.testing.assert_allclose(totals, 100.0)

    def test_member_features_are_aggregated(self):
        table = line_table()
        features = neighborhood_features(
            build_neighborhoods(table, radius=10), table, features="mean", aggregations="mean,max"
        ).set_index("neighborhood_id")
        assert features.loc[3, "mean_mean"] == pytest.approx(3.0)
        assert features.loc[1, "max_mean"] == pytest.approx(2.0)

    def test_carries_its_place_so_it_can_be_drawn_and_plotted(self):
        table = line_table()
        features = neighborhood_features(build_neighborhoods(table, radius=10), table)
        assert {"centroid-0", "centroid-1", "seed", "mean_distance"} <= set(features.columns)

    def test_unknown_columns_and_aggregations_are_refused(self):
        table = line_table()
        neighborhoods = build_neighborhoods(table, radius=10)
        with pytest.raises(ValueError, match="class column"):
            neighborhood_features(neighborhoods, table, class_column="nope")
        with pytest.raises(ValueError, match="no such feature"):
            neighborhood_features(neighborhoods, table, features="nope")
        with pytest.raises(ValueError, match="aggregation"):
            neighborhood_features(neighborhoods, table, features="mean", aggregations="mode")


class TestClassify:
    def test_finds_the_two_kinds_of_neighbourhood(self):
        table = two_halves()
        features = neighborhood_features(build_neighborhoods(table, radius=12), table, class_column="kind")
        typed = classify_neighborhoods(features, n_clusters=2)
        by_side = pd.crosstab(table.set_index("object_id").loc[typed["neighborhood_id"], "kind"].to_numpy(), typed[NEIGHBORHOOD_TYPE])
        assert (by_side.max(axis=1) / by_side.sum(axis=1)).min() > 0.9

    def test_needs_something_to_classify_on(self):
        table = line_table()
        features = neighborhood_features(build_neighborhoods(table, radius=10), table)
        with pytest.raises(ValueError, match="nothing to classify"):
            classify_neighborhoods(features)

    def test_can_use_named_features_and_other_methods(self):
        table = line_table()
        features = neighborhood_features(build_neighborhoods(table, radius=10), table)
        typed = classify_neighborhoods(
            features, method="hierarchical", n_clusters=2, features="n_objects,mean_distance"
        )
        assert set(typed[NEIGHBORHOOD_TYPE]) <= {0, 1}


class TestReflect:
    def typed(self, method="radius"):
        table = two_halves()
        neighborhoods = build_neighborhoods(table, method=method, radius=12)
        features = neighborhood_features(neighborhoods, table, class_column="kind")
        return table, neighborhoods, classify_neighborhoods(features, n_clusters=2)

    def test_each_object_takes_its_own_neighbourhoods_values(self):
        table = line_table()
        neighborhoods = build_neighborhoods(table, radius=10)
        features = neighborhood_features(neighborhoods, table, class_column="kind")
        reflected = reflect_neighborhoods(neighborhoods, features, table).set_index("object_id")
        assert reflected.loc[3, "Class_1_ClassFraction"] == pytest.approx(200 / 3)
        assert reflected.loc[3, "n_neighborhoods"] == 3

    def test_cells_gain_the_type_of_neighbourhood_they_live_in(self):
        table, neighborhoods, typed = self.typed()
        reflected = reflect_neighborhoods(neighborhoods, typed, table)
        agreement = pd.crosstab(table["kind"], reflected[NEIGHBORHOOD_TYPE])
        assert (agreement.max(axis=1) / agreement.sum(axis=1)).min() > 0.9

    def test_grid_membership_votes_on_the_type(self):
        table, neighborhoods, typed = self.typed("grid")
        reflected = reflect_neighborhoods(neighborhoods, typed, table)
        assigned = reflected[reflected[NEIGHBORHOOD_TYPE] != MISSING_CATEGORY]
        kinds = table.set_index("object_id").loc[assigned["object_id"], "kind"].to_numpy()
        agreement = pd.crosstab(kinds, assigned[NEIGHBORHOOD_TYPE])
        assert (agreement.max(axis=1) / agreement.sum(axis=1)).min() > 0.85

    def test_membership_averages_a_quantity(self):
        table = line_table()
        neighborhoods = build_neighborhoods(table, radius=10)
        features = neighborhood_features(neighborhoods, table)
        reflected = reflect_neighborhoods(
            neighborhoods, features, table, relation="member", features="n_objects"
        ).set_index("object_id")
        # object 1 is in neighbourhoods 1 (2 members) and 2 (3 members)
        assert reflected.loc[1, "n_objects"] == pytest.approx(2.5)

    def test_rows_follow_the_table_and_nobody_is_dropped(self):
        table = line_table().iloc[::-1].reset_index(drop=True)
        table.loc[len(table)] = {"object_id": 99, "centroid-0": 500.0, "centroid-1": 500.0, "mean": 0.0, "kind": 0}
        neighborhoods = build_neighborhoods(table.iloc[:-1], method="grid", radius=6, interval=10)
        features = classify_neighborhoods(
            neighborhood_features(neighborhoods, table, class_column="kind"), n_clusters=2
        )
        reflected = reflect_neighborhoods(neighborhoods, features, table)
        assert list(reflected["object_id"]) == list(table["object_id"])
        lonely = reflected.set_index("object_id").loc[99]
        assert lonely["n_neighborhoods"] == 0
        assert lonely[NEIGHBORHOOD_TYPE] == MISSING_CATEGORY
        assert np.isnan(lonely["n_objects"])

    def test_a_grid_has_no_own_neighbourhood(self):
        table = line_table()
        neighborhoods = build_neighborhoods(table, method="grid", radius=10)
        features = neighborhood_features(neighborhoods, table)
        with pytest.raises(ValueError, match="relation='member'"):
            reflect_neighborhoods(neighborhoods, features, table, relation="own")

    def test_position_and_identity_are_not_reflected(self):
        table = line_table()
        neighborhoods = build_neighborhoods(table, radius=10)
        features = neighborhood_features(neighborhoods, table)
        columns = reflectable_columns(features)
        assert "neighborhood_id" not in columns and "seed" not in columns
        assert not any(column.startswith("centroid-") for column in columns)

    def test_prefix(self):
        table = line_table()
        neighborhoods = build_neighborhoods(table, radius=10)
        features = neighborhood_features(neighborhoods, table)
        reflected = reflect_neighborhoods(neighborhoods, features, table, prefix="nb.")
        assert "nb.n_neighborhoods" in reflected.columns


class TestAsSteps:
    def test_a_protocol_builds_measures_types_and_reflects(self):
        table = two_halves()
        pipeline = Pipeline()
        for function, params in (
            ("build_neighborhoods", {"radius": 12.0}),
            ("neighborhood_features", {"class_column": "kind"}),
            ("classify_neighborhoods", {"n_clusters": 2}),
            ("reflect_neighborhoods", {}),
        ):
            pipeline.add_step(
                Step.for_function(
                    "neighborhoods",
                    function,
                    available=pipeline.available_keys({"data", "spacing"}),
                    taken_names=pipeline.step_names(),
                    params=params,
                )
            )
        context = pipeline.run({"data": table, "spacing": None})
        reflected = context["neighborhood_reflection"]
        assert len(reflected) == len(table)
        assert NEIGHBORHOOD_TYPE in reflected.columns


class TestNeighborhoodsOfNeighborhoods:
    def test_members_that_are_neighbourhoods_do_not_collide_with_their_parents(self):
        """Both levels are identified by `neighborhood_id`; the member side is
        `member_id`, so neither overwrites the other."""
        table = two_halves()
        first = build_neighborhoods(table, radius=12)
        first_table = classify_neighborhoods(
            neighborhood_features(first, table, class_column="kind"), n_clusters=2
        )
        second = build_neighborhoods(first_table, radius=25, id_column="neighborhood_id")
        membership = second.membership()
        assert {"neighborhood_id", "member_id"} <= set(membership.columns)
        assert membership["member_id"].isin(first_table["neighborhood_id"]).all()
        second_table = neighborhood_features(second, first_table, class_column=NEIGHBORHOOD_TYPE)
        assert "Class_1_ClassFraction" in second_table.columns
        reflected = reflect_neighborhoods(second, second_table, first_table, relation="member")
        assert list(reflected["neighborhood_id"]) == list(first_table["neighborhood_id"])
        assert reflected["n_neighborhoods"].min() >= 1
