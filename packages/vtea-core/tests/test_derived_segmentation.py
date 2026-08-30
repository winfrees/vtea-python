"""Segmentations built from another by morphology.

Every function here preserves label identity, which is what makes the
association between a derived segmentation and its parent exact rather than
inferred. Thicknesses are physical when a spacing is given and in voxels
when it is not - the case these tests care most about, because a band of
"5" is a sphere in index space and a flattened disc in an anisotropic
specimen.
"""

import numpy as np
import pytest
from vtea_core.data import Spacing
from vtea_core.segmentation import (
    expand_labels,
    label_ring,
    label_shell,
    restrict_labels_to,
    subtract_labels,
)

ISOTROPIC = Spacing((1.0, 1.0, 1.0), source="user")
# A z-step four times the lateral pixel size - an ordinary confocal stack.
ANISOTROPIC = Spacing((4.0, 1.0, 1.0), source="user")


def two_nuclei():
    labels = np.zeros((11, 11), dtype=np.int32)
    labels[2:4, 2:4] = 1
    labels[7:9, 7:9] = 2
    return labels


def one_voxel_volume():
    volume = np.zeros((9, 11, 11), dtype=np.int32)
    volume[4, 5, 5] = 1
    return volume


class TestExpandLabels:
    def test_it_keeps_the_original_objects(self):
        labels = two_nuclei()
        grown = expand_labels(labels, 1)
        assert (grown[labels != 0] == labels[labels != 0]).all()

    def test_it_grows_into_the_background(self):
        labels = two_nuclei()
        assert (expand_labels(labels, 1) != 0).sum() > (labels != 0).sum()

    def test_it_preserves_identity(self):
        """Which is what makes the association exact."""
        labels = two_nuclei()
        assert set(np.unique(expand_labels(labels, 2))) == {0, 1, 2}

    def test_zero_distance_changes_nothing(self):
        labels = two_nuclei()
        np.testing.assert_array_equal(expand_labels(labels, 0), labels)

    def test_two_objects_do_not_grow_through_each_other(self):
        """The nearer label takes a contested voxel, so neighbouring nuclei
        meet at a boundary instead of overwriting one another."""
        labels = np.zeros((3, 11), dtype=np.int32)
        labels[1, 2] = 1
        labels[1, 8] = 2
        grown = expand_labels(labels, 10)
        assert (grown[:, :5] != 2).all()
        assert (grown[:, 6:] != 1).all()

    def test_a_negative_distance_is_refused(self):
        with pytest.raises(ValueError, match="must not be negative"):
            expand_labels(two_nuclei(), -1)

    def test_a_non_label_image_is_refused(self):
        with pytest.raises(TypeError, match="label image"):
            expand_labels(np.zeros((4, 4), dtype=float), 1)

    def test_a_boolean_mask_is_accepted_as_one_object(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        assert (expand_labels(mask, 1) != 0).sum() > 1


class TestAnisotropy:
    """The reason Spacing exists."""

    def test_an_isotropic_spacing_reaches_equally_in_every_axis(self):
        grown = expand_labels(one_voxel_volume(), 2.0, spacing=ISOTROPIC)
        z, y, x = (np.unique(axis).size for axis in np.nonzero(grown))
        assert z == y == x

    def test_a_tall_z_step_reaches_fewer_slices(self):
        """A 2 unit band crosses 2 slices at 1 unit each and none at 4."""
        grown = expand_labels(one_voxel_volume(), 2.0, spacing=ANISOTROPIC)
        z_extent = np.unique(np.nonzero(grown)[0]).size
        isotropic_extent = np.unique(
            np.nonzero(expand_labels(one_voxel_volume(), 2.0, spacing=ISOTROPIC))[0]
        ).size
        assert z_extent < isotropic_extent
        assert z_extent == 1  # only the object's own slice is within 2 units

    def test_lateral_reach_is_unaffected_by_the_z_step(self):
        flat = expand_labels(one_voxel_volume(), 2.0, spacing=ISOTROPIC)
        tall = expand_labels(one_voxel_volume(), 2.0, spacing=ANISOTROPIC)
        assert np.unique(np.nonzero(flat)[2]).size == np.unique(np.nonzero(tall)[2]).size

    def test_no_spacing_means_voxels(self):
        in_voxels = expand_labels(one_voxel_volume(), 2.0)
        as_isotropic = expand_labels(one_voxel_volume(), 2.0, spacing=ISOTROPIC)
        np.testing.assert_array_equal(in_voxels, as_isotropic)

    def test_an_unknown_spacing_falls_back_to_voxels(self):
        """Rather than silently claiming the volume is isotropic."""
        unknown = expand_labels(one_voxel_volume(), 2.0, spacing=Spacing.unknown(3))
        np.testing.assert_array_equal(unknown, expand_labels(one_voxel_volume(), 2.0))


class TestLabelRing:
    """A cytosol band around a nucleus."""

    def test_it_excludes_the_original_object(self):
        labels = two_nuclei()
        ring = label_ring(labels, 2)
        assert (ring[labels != 0] == 0).all()

    def test_each_ring_carries_its_parent_s_id(self):
        labels = two_nuclei()
        assert set(np.unique(label_ring(labels, 2))) == {0, 1, 2}

    def test_the_ring_is_adjacent_to_its_own_object(self):
        labels = np.zeros((11, 11), dtype=np.int32)
        labels[5, 5] = 1
        ring = label_ring(labels, 1)
        assert ring[5, 4] == 1
        assert ring[5, 5] == 0

    def test_two_nuclei_do_not_share_cytosol(self):
        labels = np.zeros((3, 11), dtype=np.int32)
        labels[1, 2] = 1
        labels[1, 8] = 2
        ring = label_ring(labels, 10)
        assert not ((ring == 1) & (ring == 2)).any()
        assert (ring[:, :5] != 2).all()

    def test_thickness_is_physical_when_the_spacing_is_known(self):
        thin = label_ring(one_voxel_volume(), 2.0, spacing=ANISOTROPIC)
        assert np.unique(np.nonzero(thin)[0]).size == 1


class TestLabelShell:
    """A nuclear envelope: a band straddling the boundary."""

    def test_it_reaches_both_sides(self):
        labels = np.zeros((11, 11), dtype=np.int32)
        labels[3:8, 3:8] = 1
        shell = label_shell(labels, inward=1, outward=1)
        assert (shell[labels != 0] != 0).any()
        assert (shell[labels == 0] != 0).any()

    def test_it_leaves_the_object_s_interior_alone(self):
        labels = np.zeros((13, 13), dtype=np.int32)
        labels[3:10, 3:10] = 1
        shell = label_shell(labels, inward=1, outward=1)
        assert shell[6, 6] == 0  # the middle is not envelope

    def test_outward_only_is_a_band_outside(self):
        labels = np.zeros((11, 11), dtype=np.int32)
        labels[3:8, 3:8] = 1
        shell = label_shell(labels, inward=0, outward=1)
        assert (shell[labels != 0] == 0).all()

    def test_inward_only_is_a_rim_inside(self):
        labels = np.zeros((11, 11), dtype=np.int32)
        labels[3:8, 3:8] = 1
        shell = label_shell(labels, inward=1, outward=0)
        assert (shell[labels == 0] == 0).all()
        assert (shell[labels != 0] != 0).any()

    def test_touching_nuclei_each_keep_an_envelope_on_the_shared_face(self):
        """A plain distance-to-background would miss it, because the shared
        face is not background."""
        labels = np.zeros((7, 10), dtype=np.int32)
        labels[2:5, 1:5] = 1
        labels[2:5, 5:9] = 2
        shell = label_shell(labels, inward=1, outward=0)
        assert shell[3, 4] == 1
        assert shell[3, 5] == 2

    def test_it_preserves_identity(self):
        labels = two_nuclei()
        assert set(np.unique(label_shell(labels, inward=1, outward=1))) == {0, 1, 2}

    def test_negative_thickness_is_refused(self):
        with pytest.raises(ValueError, match="must not be negative"):
            label_shell(two_nuclei(), inward=-1)


class TestSubtractAndRestrict:
    def test_subtract_removes_the_other_s_footprint(self):
        cytoplasm = np.zeros((7, 7), dtype=np.int32)
        cytoplasm[1:6, 1:6] = 1
        nuclei = np.zeros((7, 7), dtype=np.int32)
        nuclei[3:5, 3:5] = 1

        cytosol = subtract_labels(cytoplasm, nuclei)
        assert (cytosol[nuclei != 0] == 0).all()
        assert cytosol[1, 1] == 1

    def test_subtract_preserves_identity(self):
        labels = two_nuclei()
        other = np.zeros_like(labels)
        other[2, 2] = 5  # a different id, overlapping object 1
        assert set(np.unique(subtract_labels(labels, other))) == {0, 1, 2}

    def test_restrict_keeps_only_what_is_inside(self):
        puncta = np.zeros((7, 7), dtype=np.int32)
        puncta[1, 1] = 1
        puncta[5, 5] = 2
        inside = np.zeros((7, 7), dtype=bool)
        inside[4:7, 4:7] = True

        kept = restrict_labels_to(puncta, inside)
        assert set(np.unique(kept)) == {0, 2}

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="shapes differ"):
            subtract_labels(np.zeros((4, 4), dtype=np.int32), np.zeros((5, 5), dtype=np.int32))
        with pytest.raises(ValueError, match="shapes differ"):
            restrict_labels_to(np.zeros((4, 4), dtype=np.int32), np.zeros((5, 5), dtype=bool))
