"""Scratch: where an intermediate goes when it is as large as the image."""

import numpy as np
import pytest

import dask.array as da
from vtea_core.blocked import ZarrScratch
from vtea_core.blocked.store import ENV_VAR


def test_an_array_written_in_reads_back_out():
    with ZarrScratch() as scratch:
        array = scratch.create("labels", shape=(1, 4, 16, 16), dtype=np.int32)
        array[0, 0, :8, :8] = 7
        assert scratch.as_dask("labels").sum().compute() == 7 * 64


def test_it_is_written_region_by_region_not_all_at_once():
    # The whole point: a blocked result is produced one tile at a time and
    # each is stored as it is finished.
    with ZarrScratch() as scratch:
        target = scratch.create("out", shape=(1, 1, 32, 32), dtype=np.uint16)
        for row in range(0, 32, 8):
            target[0, 0, row : row + 8, :] = row
        result = np.asarray(scratch.as_dask("out"))
        assert result[0, 0, 8, 0] == 8
        assert result[0, 0, 24, 0] == 24


def test_put_streams_a_dask_array_in():
    source = da.arange(2 * 4 * 8 * 8, chunks=64).reshape(2, 4, 8, 8)
    with ZarrScratch() as scratch:
        scratch.put("volume", source)
        np.testing.assert_array_equal(
            np.asarray(scratch.as_dask("volume")), np.asarray(source)
        )


def test_put_handles_a_numpy_array_too():
    data = np.arange(64, dtype=np.uint16).reshape(1, 1, 8, 8)
    with ZarrScratch() as scratch:
        scratch.put("volume", data)
        np.testing.assert_array_equal(np.asarray(scratch.as_dask("volume")), data)


def test_create_like_copies_the_shape_and_takes_a_new_dtype():
    # A step's output usually has its input's shape and a different dtype -
    # a uint16 volume in, an int32 label image out.
    with ZarrScratch() as scratch:
        volume = scratch.create("volume", shape=(1, 4, 16, 16), dtype=np.uint16)
        labels = scratch.create_like("labels", volume, dtype=np.int32)
        assert labels.shape == volume.shape
        assert labels.dtype == np.int32


def test_names_lists_what_is_there():
    with ZarrScratch() as scratch:
        scratch.create("a", shape=(1, 1, 4, 4), dtype=np.uint8)
        scratch.create("b", shape=(1, 1, 4, 4), dtype=np.uint8)
        assert scratch.names() == ["a", "b"]
        assert "a" in scratch and "c" not in scratch
        assert len(scratch) == 2


def test_an_intermediate_can_be_dropped_when_nothing_needs_it():
    with ZarrScratch() as scratch:
        scratch.create("blurred", shape=(1, 1, 4, 4), dtype=np.uint8)
        scratch.drop("blurred")
        assert "blurred" not in scratch
        scratch.drop("blurred")  # idempotent - dropping twice is not an error


def test_asking_for_something_that_is_not_there_says_what_is():
    with ZarrScratch() as scratch:
        scratch.create("labels", shape=(1, 1, 4, 4), dtype=np.uint8)
        with pytest.raises(KeyError, match="labels"):
            scratch.open("mask")


def test_the_directory_goes_away_on_exit():
    with ZarrScratch() as scratch:
        path = scratch.path
        scratch.create("a", shape=(1, 1, 4, 4), dtype=np.uint8)
        assert path.exists()
    assert not path.exists()


def test_keeping_it_is_what_makes_a_crashed_run_resumable():
    scratch = ZarrScratch(keep=True)
    path = scratch.path
    try:
        scratch.put("volume", np.ones((1, 1, 4, 4), np.uint8))
        scratch.close()
        assert path.exists()
        assert (path / "volume").exists()
    finally:
        import shutil

        shutil.rmtree(path, ignore_errors=True)


def test_a_closed_store_refuses_further_work():
    scratch = ZarrScratch()
    scratch.close()
    with pytest.raises(RuntimeError, match="closed"):
        scratch.create("a", shape=(1, 1, 4, 4), dtype=np.uint8)


def test_closing_twice_is_harmless():
    scratch = ZarrScratch()
    scratch.close()
    scratch.close()


def test_the_root_can_be_pointed_at_a_fast_disk(tmp_path):
    # Scratch on a shared network filesystem can be slower than the
    # computation it exists to serve, so where it goes has to be settable.
    with ZarrScratch(root=tmp_path) as scratch:
        assert scratch.path.parent == tmp_path


def test_the_environment_can_point_it_too(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    with ZarrScratch() as scratch:
        assert scratch.path.parent == tmp_path


def test_nbytes_reports_what_is_actually_on_disk():
    with ZarrScratch() as scratch:
        assert scratch.nbytes >= 0
        scratch.put("volume", np.ones((1, 4, 64, 64), np.uint16))
        assert scratch.nbytes > 0


def test_repr_says_where_it_is_and_what_is_in_it():
    with ZarrScratch() as scratch:
        scratch.create("a", shape=(1, 1, 4, 4), dtype=np.uint8)
        assert "1 arrays" in repr(scratch)
    assert "closed" in repr(scratch)
