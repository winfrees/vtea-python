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
from vtea_core.objects import associate_by_identity
from vtea_core.segmentation import (
    expand_labels,
    label_ring,
    label_shell,
    restrict_labels_to,
    subtract_labels,
    watershed_ownership,
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


class TestWatershedOwnership:
    """Dividing one region among the objects inside it - the deterministic
    answer to which cell owns a contested voxel."""

    @staticmethod
    def two_nuclei_in_one_blob():
        region = np.zeros((12, 30), dtype=bool)
        region[2:10, 2:28] = True
        nuclei = np.zeros((12, 30), dtype=np.int32)
        nuclei[5:7, 6:8] = 1
        nuclei[5:7, 22:24] = 2
        return nuclei, region

    def test_every_voxel_of_the_region_gets_an_owner(self):
        nuclei, region = self.two_nuclei_in_one_blob()
        owned = watershed_ownership(nuclei, region)
        assert (owned[region] != 0).all()

    def test_nothing_outside_the_region_is_claimed(self):
        nuclei, region = self.two_nuclei_in_one_blob()
        assert (watershed_ownership(nuclei, region)[~region] == 0).all()

    def test_each_territory_carries_its_owner_s_id(self):
        """Which is what lets associate_by_identity link the two exactly."""
        nuclei, region = self.two_nuclei_in_one_blob()
        assert set(np.unique(watershed_ownership(nuclei, region))) == {0, 1, 2}

    def test_a_marker_keeps_the_area_around_itself(self):
        nuclei, region = self.two_nuclei_in_one_blob()
        owned = watershed_ownership(nuclei, region)
        assert owned[6, 7] == 1
        assert owned[6, 23] == 2

    def test_the_two_territories_are_disjoint(self):
        nuclei, region = self.two_nuclei_in_one_blob()
        owned = watershed_ownership(nuclei, region)
        assert (owned[:, :10] != 2).all()
        assert (owned[:, 20:] != 1).all()

    def test_it_splits_at_the_waist_rather_than_between_the_markers(self):
        """The reason to watershed the region's own shape: a dumbbell parts
        at its neck even when the neck is nowhere near halfway between the
        markers. Splitting on proximity alone would cut the large lobe and
        hand a third of it to the cell in the small one."""
        region = np.zeros((14, 30), dtype=bool)
        region[4:10, 2:8] = True    # a small lobe
        region[6:8, 8:10] = True    # the neck, far off-centre
        region[2:12, 10:28] = True  # a large lobe
        nuclei = np.zeros((14, 30), dtype=np.int32)
        nuclei[6:8, 4:6] = 1    # in the small lobe
        nuclei[6:8, 24:26] = 2  # in the large one

        owned = watershed_ownership(nuclei, region)
        large_lobe = owned[2:12, 10:28][region[2:12, 10:28]]
        assert set(np.unique(large_lobe)) == {2}
        # x=12 is 7 voxels from marker 1 and 13 from marker 2, so proximity
        # to a marker would have given it to the wrong cell.
        assert owned[7, 12] == 2

    def test_a_region_with_no_marker_is_left_as_background(self):
        """Cytoplasm with no nucleus in it is a finding, not something to
        hand to the nearest cell in the next blob."""
        nuclei, region = self.two_nuclei_in_one_blob()
        region[11, 29] = True  # a separate speck, no marker
        assert watershed_ownership(nuclei, region)[11, 29] == 0

    def test_a_marker_outside_the_region_gets_no_territory(self):
        nuclei, region = self.two_nuclei_in_one_blob()
        nuclei[0, 0] = 3
        assert 3 not in np.unique(watershed_ownership(nuclei, region))

    def test_a_label_image_may_stand_in_for_the_mask(self):
        """Only membership matters, so the cytoplasm segmentation itself is
        an acceptable region."""
        nuclei, region = self.two_nuclei_in_one_blob()
        as_labels = np.where(region, 7, 0).astype(np.int32)
        np.testing.assert_array_equal(
            watershed_ownership(nuclei, as_labels), watershed_ownership(nuclei, region)
        )

    def test_an_anisotropic_spacing_still_divides_the_whole_region(self):
        """A smoke test rather than a claim about where the boundary lands:
        what must not happen is a stack whose z-step confuses the distance
        transform into leaving part of the region unowned."""
        region = np.zeros((7, 5, 20), dtype=bool)
        region[1:6, 1:4, 1:19] = True
        nuclei = np.zeros((7, 5, 20), dtype=np.int32)
        nuclei[3, 2, 3] = 1
        nuclei[3, 2, 16] = 2

        isotropic = watershed_ownership(nuclei, region, spacing=ISOTROPIC)
        anisotropic = watershed_ownership(nuclei, region, spacing=ANISOTROPIC)
        assert (isotropic[region] != 0).all()
        assert (anisotropic[region] != 0).all()
        assert set(np.unique(anisotropic)) == {0, 1, 2}

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="shapes differ"):
            watershed_ownership(np.zeros((4, 4), dtype=np.int32), np.zeros((5, 5), dtype=bool))

    def test_the_territories_associate_back_to_their_owners(self):
        """The pairing this exists for: split the region, then say which
        cell each piece belongs to."""
        nuclei, region = self.two_nuclei_in_one_blob()
        territories = watershed_ownership(nuclei, region)
        links = associate_by_identity(
            territories, nuclei, child_name="cytoplasm_1", parent_name="nuclei_1"
        )
        assert len(links) == 2
        assert all(link.is_certain for link in links)
