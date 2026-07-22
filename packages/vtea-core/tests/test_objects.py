import numpy as np
import pytest

from vtea_core.data import object_ids, object_intensity_values, object_pixel_indices


def make_label_mask():
    # 2x3x3 volume: object 1 occupies z=0, object 2 occupies z=1, background elsewhere
    mask = np.zeros((2, 3, 3), dtype=np.int32)
    mask[0, 0, :] = 1
    mask[0, 1, :] = 1
    mask[1, :, 0] = 2
    return mask


def test_object_ids_excludes_background():
    mask = make_label_mask()
    ids = object_ids(mask)
    np.testing.assert_array_equal(ids, [1, 2])


def test_object_ids_empty_mask():
    mask = np.zeros((2, 2, 2), dtype=np.int32)
    assert object_ids(mask).size == 0


def test_object_pixel_indices_matches_nonzero_count():
    mask = make_label_mask()
    z, y, x = object_pixel_indices(mask, 1)
    assert len(z) == len(y) == len(x) == 6  # 2 rows * 3 cols
    assert set(z.tolist()) == {0}

    z2, y2, x2 = object_pixel_indices(mask, 2)
    assert len(z2) == 3
    assert set(z2.tolist()) == {1}


def test_object_pixel_indices_missing_id_is_empty():
    mask = make_label_mask()
    z, y, x = object_pixel_indices(mask, 999)
    assert len(z) == 0 and len(y) == 0 and len(x) == 0


def test_object_intensity_values():
    mask = make_label_mask()
    intensity = np.arange(mask.size, dtype=np.float32).reshape(mask.shape)
    values = object_intensity_values(mask, intensity, 1)
    expected = intensity[mask == 1]
    np.testing.assert_array_equal(np.sort(values), np.sort(expected))
    assert values.shape[0] == 6


def test_object_intensity_values_shape_mismatch_raises():
    mask = make_label_mask()
    bad_intensity = np.zeros((1, 1, 1))
    with pytest.raises(ValueError, match="shape"):
        object_intensity_values(mask, bad_intensity, 1)
