"""Augmenting 3D object crops for training (NumPy / SciPy, no torch).

A model trained on a few thousand nuclei sees each many times; random flips,
rotations, shifts and contrast changes make each pass a little different, so
it learns the cell type and not the orientation or brightness of particular
nuclei. The transforms are those of NephNet's training
(awoloshuk/NephNet `utils/transforms3d.py`, MIT), on crops shaped (Z, Y, X)
or (C, Z, Y, X) in their raw (e.g. 0-255) intensities:

| preset | NephNet's | transforms |
| --- | --- | --- |
| `"original"` | awoloshuk/NephNet, `hdf5_3d_dataloader` | flips on every axis, a 90° rotation in XY, a rotation up to ±25° in XY, a shift up to 5 voxels (p 0.2), a contrast change (factor 0.8, p 0.2) |
| `"strong"` | ntalukde/NephNet (the extension) | shot noise (p 0.3), a downsampling and back (p 0.2), the flips and 90° rotation, a rotation up to ±35°, a shift up to 8 voxels (p 0.3), a contrast change (p 0.3) |
| `"none"` | | none |

Two details differ from NephNet's code, on purpose: the area a rotation
brings in from outside the crop is 0 (NephNet used -1, i.e. 0 after its
/ 255), and the downsampling factor is drawn between 1.5 and 3 (NephNet's
configured factor of 999 shrank every crop to a single voxel, which looks
unintended).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

PRESETS = ("original", "strong", "none")


def _channels(crop: np.ndarray, fn) -> np.ndarray:
    """Apply `fn` to a (Z, Y, X) crop, or to each channel of a (C, Z, Y, X) one."""
    if crop.ndim == 3:
        return fn(crop)
    if crop.ndim == 4:
        return np.stack([fn(c) for c in crop])
    raise ValueError(f"a crop is (Z, Y, X) or (C, Z, Y, X), got shape {crop.shape}")


def flip(crop, rng):
    """Flip along each spatial axis with probability 1/2."""
    axes = [a for a in range(3) if rng.uniform() > 0.5]
    return _channels(crop, lambda m: np.flip(m, axes) if axes else m)


def rot90_xy(crop, rng):
    """Rotate by 0, 90, 180 or 270 degrees in the Y-X plane."""
    k = int(rng.integers(0, 4))
    return _channels(crop, lambda m: np.rot90(m, k, axes=(1, 2)))


def rotate_xy(crop, rng, max_angle=25, order=1):
    """Rotate by a random whole angle in (-max_angle, max_angle) in the Y-X plane."""
    angle = int(rng.integers(-max_angle, max_angle)) if max_angle > 0 else 0
    if angle == 0:
        return crop
    return _channels(
        crop,
        lambda m: ndi.rotate(
            m, angle, axes=(1, 2), reshape=False, order=order, mode="constant", cval=0.0
        ),
    )


def translate(crop, rng, pixels=5, probability=0.2):
    """With `probability`, shift by up to `pixels` in Y and X and the same fraction
    of the crop's depth in Z (cubic interpolation, 0 brought in)."""
    if rng.uniform() >= probability:
        return crop
    shape = crop.shape[-3:]
    y = int(rng.uniform(-pixels, pixels))
    x = int(rng.uniform(-pixels, pixels))
    dz = pixels / shape[1] * shape[0]
    z = int(rng.uniform(-dz, dz))
    return _channels(crop, lambda m: ndi.shift(m, (z, y, x), order=3, mode="constant", cval=0.0))


def contrast(crop, rng, factor=0.8, probability=0.2, max_value=255.0):
    """With `probability`, scale the intensities by factor + U(0, 0.5), clipped to
    0-max_value."""
    if rng.uniform() >= probability:
        return crop
    return np.clip(crop * (factor + rng.uniform(0, 0.5)), 0, max_value)


def shot_noise(crop, rng, alpha=0.8, probability=0.3):
    """With `probability`, dim the signal by a factor in (alpha, 1) and add Poisson
    noise of the original intensities."""
    if rng.uniform() >= probability:
        return crop
    a = rng.uniform(alpha, 1.0)
    clean = np.clip(crop, 0, None)
    return clean * a + (rng.poisson(clean) - clean)


def downsample(crop, rng, probability=0.2, factors=(1.5, 3.0)):
    """With `probability`, shrink in Y and X by a random factor and enlarge back
    (blur by resolution loss)."""
    if rng.uniform() >= probability:
        return crop
    f = rng.uniform(*factors)

    def one(m):
        small = ndi.zoom(m, (1, 1 / f, 1 / f), order=1)
        back = ndi.zoom(small, [n / s for n, s in zip(m.shape, small.shape)], order=1)
        out = np.zeros_like(m)
        region = tuple(slice(0, min(a, b)) for a, b in zip(m.shape, back.shape))
        out[region] = back[region]
        return out

    return _channels(crop, one)


def augment(crop: np.ndarray, rng: np.random.Generator, preset: str = "original") -> np.ndarray:
    """One random augmentation of a crop by a NephNet preset (see the module docstring)."""
    if preset == "none":
        return crop
    crop = np.asarray(crop, dtype=np.float32)
    if preset == "original":
        crop = flip(crop, rng)
        crop = rot90_xy(crop, rng)
        crop = rotate_xy(crop, rng, max_angle=25)
        crop = translate(crop, rng, pixels=5, probability=0.2)
        return contrast(crop, rng, factor=0.8, probability=0.2).astype(np.float32)
    if preset == "strong":
        crop = shot_noise(crop, rng, alpha=0.8, probability=0.3)
        crop = downsample(crop, rng, probability=0.2)
        crop = flip(crop, rng)
        crop = rot90_xy(crop, rng)
        crop = rotate_xy(crop, rng, max_angle=35, order=0)
        crop = translate(crop, rng, pixels=8, probability=0.3)
        return contrast(crop, rng, factor=0.8, probability=0.3).astype(np.float32)
    raise ValueError(f"unknown augmentation {preset!r}, expected one of {PRESETS}")


def augment_batch(
    crops: np.ndarray, rng: np.random.Generator, preset: str = "original"
) -> np.ndarray:
    """`augment` each crop of a batch (N, ...) independently."""
    if preset == "none":
        return crops
    return np.stack([augment(c, rng, preset) for c in crops]).astype(np.float32)
