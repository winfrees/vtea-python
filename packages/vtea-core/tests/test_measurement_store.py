import pandas as pd
import pytest

from vtea_core.measurements import MeasurementStore


@pytest.fixture
def objects_frame():
    return pd.DataFrame({"object_id": [1, 2, 3], "mean": [10.0, 20.0, 30.0], "count": [5, 8, 3]})


def test_register_and_query(objects_frame):
    with MeasurementStore() as store:
        store.register("OBJECTS", objects_frame)
        result = store.query("SELECT * FROM OBJECTS ORDER BY object_id")
    pd.testing.assert_frame_equal(result.reset_index(drop=True), objects_frame.reset_index(drop=True))


def test_query_filters_and_aggregates(objects_frame):
    with MeasurementStore() as store:
        store.register("OBJECTS", objects_frame)
        result = store.query("SELECT COUNT(*) AS n FROM OBJECTS WHERE mean > 15")
    assert result["n"].iloc[0] == 2


def test_multiple_tables(objects_frame):
    other = pd.DataFrame({"object_id": [1, 2, 3], "cluster": [0, 1, 0]})
    with MeasurementStore() as store:
        store.register("OBJECTS", objects_frame)
        store.register("CLUSTERS", other)
        result = store.query(
            "SELECT o.object_id, o.mean, c.cluster FROM OBJECTS o "
            "JOIN CLUSTERS c ON o.object_id = c.object_id ORDER BY o.object_id"
        )
    assert list(result["cluster"]) == [0, 1, 0]


def test_context_manager_closes_connection(objects_frame):
    store = MeasurementStore()
    with store:
        store.register("OBJECTS", objects_frame)
    with pytest.raises(Exception):
        store.query("SELECT * FROM OBJECTS")
