"""NephNet3D's data (no torch): Z x Y x X crops at a model's voxel size, its training
augmentations, and VTEA's per-class crop exports."""

import numpy as np
import pandas as pd
import pytest

from vtea_core.classification import extract_crops
from vtea_core.classification.augment import (
    augment,
    augment_batch,
    contrast,
    flip,
    rot90_xy,
    translate,
)
from vtea_core.classification.crop_sets import NEPHNET_CROP, read_vtea_crops, write_vtea_crops


def nucleus_volume():
    """A 20 x 64 x 64 volume with one bright ellipsoid nucleus (label 1) at the centre."""
    z, y, x = np.mgrid[:20, :64, :64]
    inside = ((z - 10) / 3) ** 2 + ((y - 32) / 8) ** 2 + ((x - 32) / 6) ** 2 <= 1
    labels = inside.astype(np.int32)
    image = np.where(inside, 200.0, 10.0)
    return labels, image


def test_crops_of_nephnet_shape():
    labels, image = nucleus_volume()
    crops, ids = extract_crops(labels, image, size=NEPHNET_CROP, normalize="none")
    assert crops.shape == (1, 1, 7, 32, 32) and ids.tolist() == [1]
    assert crops[0, 0, 3, 16, 16] == 200  # the centre of the crop is the nucleus's
    assert crops[0, 0, 3, 0, 0] == 10
    with pytest.raises(ValueError, match="has 2 axes; the image has 3"):
        extract_crops(labels, image, size=(32, 32))


def test_crops_are_resampled_to_the_models_voxel_size():
    labels, image = nucleus_volume()
    # This image's voxels are twice as fine in Y and X as the model's: the crop covers
    # twice the voxels, so the nucleus looks as large as in the model's training crops.
    same, _ = extract_crops(labels, image, size=NEPHNET_CROP, normalize="none")
    resampled, _ = extract_crops(
        labels,
        image,
        size=NEPHNET_CROP,
        normalize="none",
        spacing=(1.0, 0.25, 0.25),
        target_spacing=(1.0, 0.5, 0.5),
    )
    assert resampled.shape == same.shape

    def width(crop):
        return int((crop[0, 0, 3, 16] > 100).sum())

    assert width(same) == 13 and width(resampled) in (6, 7, 8)  # half the width
    # The same voxel size: identical to no resampling.
    again, _ = extract_crops(
        labels,
        image,
        size=NEPHNET_CROP,
        normalize="none",
        spacing=(1.0, 0.5, 0.5),
        target_spacing=(1.0, 0.5, 0.5),
    )
    np.testing.assert_array_equal(again, same)


def test_augmentations_keep_the_shape_and_respect_their_probabilities():
    rng = np.random.default_rng(0)
    crop = np.random.default_rng(1).uniform(0, 255, size=(7, 32, 32)).astype(np.float32)
    for preset in ("original", "strong"):
        for _ in range(5):
            out = augment(crop, rng, preset)
            assert out.shape == crop.shape and out.dtype == np.float32
            assert np.isfinite(out).all()
    assert augment(crop, rng, "none") is crop
    batch = augment_batch(np.stack([crop, crop]), rng)
    assert batch.shape == (2, 7, 32, 32) and not np.array_equal(batch[0], batch[1])
    with pytest.raises(ValueError, match="unknown augmentation"):
        augment(crop, rng, "wild")
    # Flips and 90-degree rotations only move voxels; probability 0 changes nothing.
    assert sorted(flip(crop, rng).ravel()) == sorted(crop.ravel())
    assert rot90_xy(crop, rng).shape == crop.shape
    np.testing.assert_array_equal(translate(crop, rng, probability=0.0), crop)
    np.testing.assert_array_equal(contrast(crop, rng, probability=0.0), crop)
    brighter = contrast(crop, rng, factor=2.0, probability=1.0)
    assert brighter.max() <= 255 and brighter.mean() > crop.mean()
    # Channels are augmented alike: a (C, Z, Y, X) crop keeps its channels aligned.
    two = np.stack([crop, crop])
    out = augment(two, np.random.default_rng(3), "original")
    np.testing.assert_allclose(out[0], out[1])


def test_vtea_crop_exports_round_trip_with_signed_bytes(tmp_path):
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 256, size=(6, *NEPHNET_CROP)).astype(np.uint8)
    classes = np.array([1, 1, 2, 2, 2, 3])
    ids = np.arange(101, 107)
    folder = write_vtea_crops(tmp_path / "export", crops, classes, ids)
    assert sorted(p.name for p in folder.iterdir()) == ["class_1.csv", "class_2.csv", "class_3.csv"]
    raw = pd.read_csv(folder / "class_2.csv", header=None)
    assert (
        raw.shape == (3, 2 + 7168) and raw.iloc[:, 2:].to_numpy().min() < 0
    )  # signed bytes on disk
    back, back_classes, back_ids = read_vtea_crops(folder)
    order = np.argsort(back_ids)
    np.testing.assert_array_equal(back[order], crops)
    assert back_classes[order].tolist() == classes.tolist() and back.dtype == np.uint8
    one, _, _ = read_vtea_crops(folder / "class_3.csv")
    assert one.shape == (1, *NEPHNET_CROP)
    pd.DataFrame([[1, 7, 0, 0, 0]]).to_csv(tmp_path / "short.csv", header=False, index=False)
    with pytest.raises(ValueError, match="3 voxels"):
        read_vtea_crops(tmp_path / "short.csv")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no CSV"):
        read_vtea_crops(tmp_path / "empty")
