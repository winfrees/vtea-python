import numpy as np
import pytest

from vtea_core.imageprocessing import enhance_contrast, gaussian_blur, median_filter, subtract_background


class TestGaussianBlur:
    def test_smooths_an_impulse(self):
        volume = np.zeros((11, 11))
        volume[5, 5] = 100.0
        blurred = gaussian_blur(volume, sigma=1.0)
        assert blurred[5, 5] < 100.0
        assert blurred[5, 6] > 0.0

    def test_preserves_shape(self):
        volume = np.random.default_rng(0).random((4, 6, 6))
        blurred = gaussian_blur(volume, sigma=(1.0, 1.0, 1.0))
        assert blurred.shape == volume.shape


class TestMedianFilter:
    def test_removes_salt_and_pepper_noise(self):
        volume = np.full((9, 9), 50.0)
        volume[4, 4] = 255.0  # single outlier spike
        filtered = median_filter(volume, radius=1)
        assert filtered[4, 4] == pytest.approx(50.0)

    def test_scalar_radius_matches_denoise_special_case(self):
        volume = np.random.default_rng(0).random((3, 8, 8))
        scalar = median_filter(volume, radius=1)
        per_axis = median_filter(volume, radius=(1, 1, 1))
        np.testing.assert_array_equal(scalar, per_axis)

    def test_per_axis_radius_shape_preserved(self):
        volume = np.random.default_rng(0).random((5, 10, 10))
        filtered = median_filter(volume, radius=(1, 2, 2))
        assert filtered.shape == volume.shape


class TestEnhanceContrast:
    def test_normalize_stretches_to_unit_range(self):
        volume = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = enhance_contrast(volume, method="normalize")
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            enhance_contrast(np.zeros((3, 3)), method="bogus")


class TestSubtractBackground:
    def test_result_is_nonnegative(self):
        volume = np.random.default_rng(0).integers(0, 255, size=(20, 20)).astype(np.uint8)
        result = subtract_background(volume, radius=5)
        assert result.min() >= 0

    def test_removes_smooth_gradient_background(self):
        y, x = np.mgrid[0:30, 0:30]
        gradient = (x + y).astype(np.float64)  # smooth background
        spike = gradient.copy()
        spike[15, 15] += 50  # a real foreground feature on top
        result = subtract_background(spike, radius=10)
        assert result[15, 15] > result[0, 0]
