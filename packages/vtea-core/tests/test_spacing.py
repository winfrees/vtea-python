"""Physical voxel size, and the difference between knowing it and assuming it.

The awkward case this exists for: napari fills `layer.scale` with ones when
a file carries no scale, so "isotropic, one unit per voxel" and "nobody
recorded it" are the same array. Treating the second as the first is how
anisotropy goes unnoticed.
"""

import numpy as np
import pytest

from vtea_core.data import FROM_METADATA, FROM_USER, UNKNOWN, Spacing, physical_volume
from vtea_core.data.spacing import spacing_from_scale


class TestSpacing:
    def test_it_reports_its_own_voxel_volume(self):
        assert Spacing((1.0, 0.25, 0.25)).voxel_volume == pytest.approx(0.0625)

    def test_anisotropy_is_visible(self):
        assert Spacing((1.0, 0.25, 0.25)).is_isotropic is False
        assert Spacing((0.25, 0.25, 0.25)).is_isotropic is True

    def test_a_zero_or_negative_size_is_refused(self):
        for values in ((1.0, 0.0, 1.0), (1.0, -0.5, 1.0)):
            with pytest.raises(ValueError, match="finite and positive"):
                Spacing(values)

    def test_a_nan_size_is_refused(self):
        with pytest.raises(ValueError, match="finite and positive"):
            Spacing((1.0, float("nan"), 1.0))

    def test_no_axes_is_refused(self):
        with pytest.raises(ValueError, match="at least one axis"):
            Spacing(())

    def test_it_describes_itself_for_a_label(self):
        assert Spacing((1.0, 0.3, 0.3), unit="µm").describe() == "1 × 0.3 × 0.3 µm"

    def test_an_unknown_spacing_says_so_rather_than_showing_ones(self):
        assert Spacing.unknown(3).describe() == "voxel size unknown"


class TestKnownVersusAssumed:
    def test_unknown_is_not_known_even_though_it_has_values(self):
        spacing = Spacing.unknown(3)
        assert spacing.values == (1.0, 1.0, 1.0)
        assert spacing.is_known is False

    def test_an_all_ones_scale_reads_as_unknown(self):
        """napari's default when a file carries no scale."""
        assert spacing_from_scale([1.0, 1.0, 1.0]).is_known is False

    def test_a_real_scale_reads_as_metadata(self):
        spacing = spacing_from_scale([1.0, 0.28, 0.28])
        assert spacing.source == FROM_METADATA
        assert spacing.is_known is True
        assert spacing.values == (1.0, 0.28, 0.28)

    def test_an_isotropic_but_non_unit_scale_is_believed(self):
        """0.5 everywhere is a real calibration; 1.0 everywhere is a default."""
        assert spacing_from_scale([0.5, 0.5, 0.5]).is_known is True

    def test_a_nonsense_scale_reads_as_unknown_rather_than_raising(self):
        assert spacing_from_scale([0.0, 1.0, 1.0]).is_known is False
        assert spacing_from_scale([np.nan, 1.0, 1.0]).is_known is False

    def test_an_empty_scale_reads_as_unknown(self):
        assert spacing_from_scale([]).is_known is False


class TestForNdim:
    def test_extra_leading_axes_get_a_unit_size(self):
        """A channel axis has no physical extent, and the trailing axes are
        the pairing that is never ambiguous."""
        assert Spacing((1.0, 0.25, 0.25)).for_ndim(4) == (1.0, 1.0, 0.25, 0.25)

    def test_a_longer_spacing_is_trimmed_from_the_front(self):
        assert Spacing((1.0, 0.5, 0.25, 0.25)).for_ndim(2) == (0.25, 0.25)

    def test_a_matching_ndim_is_unchanged(self):
        assert Spacing((1.0, 0.25, 0.25)).for_ndim(3) == (1.0, 0.25, 0.25)

    def test_a_nonsense_ndim_is_refused(self):
        with pytest.raises(ValueError, match="ndim must be positive"):
            Spacing((1.0,)).for_ndim(0)


class TestPhysicalVolume:
    def test_a_voxel_count_becomes_a_volume(self):
        assert physical_volume(8, Spacing((1.0, 0.5, 0.5))) == pytest.approx(2.0)

    def test_an_unknown_spacing_gives_none_rather_than_the_count(self):
        """Returning the voxel count would claim a calibration nobody gave."""
        assert physical_volume(8, Spacing.unknown(3)) is None

    def test_no_spacing_gives_none(self):
        assert physical_volume(8, None) is None


class TestRoundTrip:
    def test_it_survives_json(self):
        import json

        spacing = Spacing((1.0, 0.28, 0.28), unit="nm", source=FROM_USER)
        restored = Spacing.from_dict(json.loads(json.dumps(spacing.to_dict())))
        assert restored == spacing

    def test_the_source_survives(self):
        for source in (FROM_METADATA, FROM_USER, UNKNOWN):
            spacing = Spacing((1.0, 1.0, 1.0), source=source)
            assert Spacing.from_dict(spacing.to_dict()).source == source
