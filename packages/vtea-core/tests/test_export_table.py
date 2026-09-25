"""The feature table exports as a file anything can open, and never without
saying what its columns are."""

import duckdb
import pandas as pd
import pytest

from vtea_core.export import data_dictionary, dictionary_path_for, export_table
from vtea_core.measurements import FeatureCatalog


def make_table():
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3],
            "count": [10, 20, 30],
            "mean_ch0": [1.5, 2.5, 3.5],
            "kmeans_1": [0, 1, 0],
            "gate_bright": [True, False, True],
        }
    )


def make_catalog():
    catalog = FeatureCatalog()
    catalog.record_measured(
        ["object_id", "count", "mean_ch0"],
        produced_by="measure_label_components_1",
        function="measurements.extract_measurements_by_channel",
        segmentation="label_components_1",
    )
    catalog.record_derived(
        ["kmeans_1"],
        produced_by="kmeans_1",
        function="clustering.kmeans",
        params={"n_clusters": 2},
        source_features=["count", "mean_ch0"],
    )
    return catalog


class TestExportTable:
    def test_csv_round_trips(self, tmp_path):
        table = make_table()
        written = export_table(table, tmp_path / "objects.csv", catalog=make_catalog())
        assert written == [tmp_path / "objects.csv", tmp_path / "objects.dictionary.csv"]
        pd.testing.assert_frame_equal(pd.read_csv(written[0]), table)

    def test_parquet_round_trips_without_pyarrow(self, tmp_path):
        table = make_table()
        (path, _) = export_table(table, tmp_path / "objects.parquet")
        back = duckdb.sql(f"SELECT * FROM '{path}'").df()
        pd.testing.assert_frame_equal(back, table, check_dtype=False)

    def test_the_dictionary_can_be_left_out(self, tmp_path):
        written = export_table(make_table(), tmp_path / "objects.csv", dictionary=False)
        assert written == [tmp_path / "objects.csv"]
        assert not dictionary_path_for(tmp_path / "objects.csv").exists()

    def test_an_unknown_format_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match=r"\.xlsx"):
            export_table(make_table(), tmp_path / "objects.xlsx")


class TestDataDictionary:
    def test_one_row_per_column_in_table_order(self):
        dictionary = data_dictionary(make_table(), make_catalog())
        assert list(dictionary["column"]) == list(make_table().columns)

    def test_catalogued_columns_carry_their_provenance(self):
        dictionary = data_dictionary(make_table(), make_catalog()).set_index("column")
        assert dictionary.loc["mean_ch0", "segmentation"] == "label_components_1"
        assert dictionary.loc["mean_ch0", "channel"] == 0
        assert dictionary.loc["kmeans_1", "source_features"] == "count, mean_ch0"
        assert dictionary.loc["kmeans_1", "params"] == "n_clusters=2"

    def test_uncatalogued_columns_are_still_listed(self):
        dictionary = data_dictionary(make_table(), make_catalog()).set_index("column")
        assert dictionary.loc["gate_bright", "kind"] == "unrecorded"
        bare = data_dictionary(make_table()).set_index("column")
        assert len(bare) == len(make_table().columns)
        # Names the measurement steps produce are still read from the name.
        assert bare.loc["mean_ch0", "kind"] == "intensity"
        assert bare.loc["mean_ch0", "channel"] == 0
