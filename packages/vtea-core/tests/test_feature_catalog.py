"""Every column of the measurement table records what it is and how it was
produced.

`mean_ch2` and `pca_1_1` are opaque on their own; the catalog is what makes
a deposited table self-describing rather than a wall of unlabelled numbers.
"""

import pandas as pd
import pytest

from vtea_core.measurements import (
    DERIVED,
    GEOMETRY,
    IDENTIFIER,
    INTENSITY,
    FeatureCatalog,
    FeatureDescriptor,
    classify_column,
    parse_feature_name,
)
from vtea_core.measurements.catalog import CATALOG_FORMAT_VERSION


class TestParseFeatureName:
    def test_splits_off_the_channel_suffix(self):
        assert parse_feature_name("mean_ch2") == ("mean", 2)

    def test_a_name_without_one_keeps_its_channel_as_none(self):
        assert parse_feature_name("count") == ("count", None)

    def test_a_multipart_measurement_survives(self):
        assert parse_feature_name("threshold_mean_ch10") == ("threshold_mean", 10)

    def test_a_non_numeric_suffix_is_not_a_channel(self):
        assert parse_feature_name("mean_check") == ("mean_check", None)

    def test_a_derived_name_is_left_alone(self):
        assert parse_feature_name("pca_1_1") == ("pca_1_1", None)


class TestClassifyColumn:
    def test_object_id_is_an_identifier(self):
        assert classify_column("object_id")[0] == IDENTIFIER

    def test_centroids_are_geometry_and_name_their_axis(self):
        kind, measurement, channel = classify_column("centroid-1")
        assert kind == GEOMETRY
        assert measurement == "centroid along axis 1"
        assert channel is None

    def test_count_is_geometry(self):
        assert classify_column("count")[0] == GEOMETRY

    def test_a_channel_tagged_measurement_is_intensity(self):
        kind, measurement, channel = classify_column("stddev_ch3")
        assert (kind, measurement, channel) == (INTENSITY, "stddev", 3)


class TestRecordingMeasurements:
    def make_catalog(self):
        catalog = FeatureCatalog()
        catalog.record_measured(
            ["object_id", "centroid-0", "count", "mean_ch0", "mean_ch2"],
            produced_by="extract_measurements_by_channel_1",
            function="measurements.extract_measurements_by_channel",
            segmentation="watershed_split_1",
        )
        return catalog

    def test_every_column_is_described(self):
        catalog = self.make_catalog()
        assert len(catalog) == 5
        assert catalog.names() == [
            "object_id",
            "centroid-0",
            "count",
            "mean_ch0",
            "mean_ch2",
        ]

    def test_the_channel_is_recorded_separately_from_the_name(self):
        descriptor = self.make_catalog().get("mean_ch2")
        assert descriptor.measurement == "mean"
        assert descriptor.channel == 2

    def test_the_segmentation_and_producing_step_are_recorded(self):
        descriptor = self.make_catalog().get("mean_ch0")
        assert descriptor.segmentation == "watershed_split_1"
        assert descriptor.produced_by == "extract_measurements_by_channel_1"
        assert descriptor.function == "measurements.extract_measurements_by_channel"

    def test_units_are_filled_in_where_they_are_known(self):
        catalog = self.make_catalog()
        assert catalog.get("count").units == "voxels"
        assert catalog.get("mean_ch0").units == "a.u."

    def test_an_unknown_feature_has_no_descriptor(self):
        assert self.make_catalog().get("never_measured") is None


class TestRecordingDerived:
    def make_catalog(self):
        catalog = FeatureCatalog()
        catalog.record_measured(
            ["count", "mean_ch0", "mean_ch1"],
            produced_by="measure_1",
            function="measurements.extract_measurements_by_channel",
            segmentation="watershed_split_1",
        )
        catalog.record_derived(
            ["kmeans_1"],
            produced_by="kmeans_1",
            function="clustering.kmeans",
            params={"n_clusters": 4},
            source_features=["mean_ch0", "mean_ch1"],
            segmentation="watershed_split_1",
        )
        return catalog

    def test_the_input_features_are_recorded(self):
        """The record that makes a clustering reproducible: which of the
        measured features were actually fed to it."""
        descriptor = self.make_catalog().get("kmeans_1")
        assert descriptor.source_features == ["mean_ch0", "mean_ch1"]
        assert "count" not in descriptor.source_features

    def test_the_parameters_are_recorded(self):
        assert self.make_catalog().get("kmeans_1").params == {"n_clusters": 4}

    def test_it_is_marked_derived_not_measured(self):
        assert self.make_catalog().get("kmeans_1").kind == DERIVED

    def test_the_measurement_reads_as_what_it_is(self):
        assert self.make_catalog().get("kmeans_1").measurement == "cluster assignment"

    def test_a_reduction_says_so_too(self):
        catalog = FeatureCatalog()
        catalog.record_derived(
            ["pca_1_1", "pca_1_2"], function="reduction.pca", produced_by="pca_1"
        )
        assert catalog.get("pca_1_2").measurement == "reduced dimension"

    def test_the_derived_feature_inherits_the_segmentation(self):
        """Its own step never names one, but its rows are still objects of a
        particular segmentation."""
        assert self.make_catalog().get("kmeans_1").segmentation == "watershed_split_1"


class TestStaleEntries:
    def test_dropping_forgets_features_no_longer_in_the_table(self):
        """A stale entry is worse than a missing one - it looks
        authoritative."""
        catalog = FeatureCatalog()
        catalog.record_measured(["mean_ch0", "mean_ch1"], produced_by="measure_1")
        catalog.drop_missing(["mean_ch0"])
        assert catalog.names() == ["mean_ch0"]

    def test_re_recording_a_column_replaces_its_description(self):
        catalog = FeatureCatalog()
        catalog.record_measured(["mean_ch0"], produced_by="measure_1")
        catalog.record_measured(["mean_ch0"], produced_by="measure_2")
        assert len(catalog) == 1
        assert catalog.get("mean_ch0").produced_by == "measure_2"


class TestDataDictionary:
    def test_one_row_per_feature_with_the_provenance_columns(self):
        catalog = FeatureCatalog()
        catalog.record_measured(
            ["mean_ch2"],
            produced_by="measure_1",
            function="measurements.extract_measurements_by_channel",
            segmentation="watershed_split_1",
        )
        frame = catalog.to_dataframe()
        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["column"] == "mean_ch2"
        assert row["measurement"] == "mean"
        assert row["channel"] == 2
        assert row["segmentation"] == "watershed_split_1"
        assert row["units"] == "a.u."

    def test_params_and_sources_render_as_readable_text(self):
        catalog = FeatureCatalog()
        catalog.record_derived(
            ["kmeans_1"],
            function="clustering.kmeans",
            params={"n_clusters": 3, "random_state": 0},
            source_features=["a", "b"],
        )
        row = catalog.to_dataframe().iloc[0]
        assert row["params"] == "n_clusters=3, random_state=0"
        assert row["source_features"] == "a, b"

    def test_a_feature_with_no_channel_leaves_the_cell_blank(self):
        catalog = FeatureCatalog()
        catalog.record_measured(["count"], produced_by="measure_1")
        assert catalog.to_dataframe().iloc[0]["channel"] == ""

    def test_an_empty_catalog_still_has_the_columns(self):
        frame = FeatureCatalog().to_dataframe()
        assert list(frame.columns)[:3] == ["column", "kind", "measurement"]
        assert frame.empty


class TestJsonRoundTrip:
    def test_descriptors_survive(self):
        catalog = FeatureCatalog()
        catalog.record_measured(
            ["mean_ch2"], produced_by="measure_1", segmentation="nuclei"
        )
        catalog.record_derived(
            ["kmeans_1"],
            produced_by="kmeans_1",
            function="clustering.kmeans",
            params={"n_clusters": 2},
            source_features=["mean_ch2"],
        )
        restored = FeatureCatalog.from_dict(catalog.to_dict())
        assert restored.names() == catalog.names()
        assert restored.get("mean_ch2").channel == 2
        assert restored.get("kmeans_1").source_features == ["mean_ch2"]

    def test_the_payload_is_versioned(self):
        assert FeatureCatalog().to_dict()["vtea_feature_catalog_version"] == (
            CATALOG_FORMAT_VERSION
        )

    def test_a_newer_version_is_refused_clearly(self):
        with pytest.raises(ValueError, match="newer than this VTEA"):
            FeatureCatalog.from_dict(
                {"vtea_feature_catalog_version": CATALOG_FORMAT_VERSION + 1, "features": []}
            )

    def test_unknown_fields_from_a_future_writer_are_ignored(self):
        descriptor = FeatureDescriptor.from_dict(
            {"name": "mean", "kind": INTENSITY, "some_future_field": 1}
        )
        assert descriptor.name == "mean"

    def test_it_is_plain_json_serialisable(self):
        import json

        catalog = FeatureCatalog()
        catalog.record_measured(["count"], produced_by="measure_1")
        assert json.loads(json.dumps(catalog.to_dict()))["features"][0]["name"] == "count"


class TestAgainstARealTable:
    def test_every_measured_column_gets_an_entry(self):
        import numpy as np

        from vtea_core.measurements import extract_measurements_by_channel

        labels = np.zeros((2, 4, 4), dtype=np.int32)
        labels[:, 0:2, 0:2] = 1
        intensity = np.zeros((2, 3, 4, 4), dtype=float)
        for channel in range(3):
            intensity[:, channel] = 10.0 * (channel + 1)
        table = extract_measurements_by_channel(labels, intensity, channel_axis=1)

        catalog = FeatureCatalog()
        catalog.record_measured(list(table.columns), produced_by="measure_1")

        assert set(catalog.names()) == set(table.columns)
        assert catalog.get("mean_ch1").channel == 1
        assert catalog.get("count").kind == GEOMETRY
        assert isinstance(catalog.to_dataframe(), pd.DataFrame)
