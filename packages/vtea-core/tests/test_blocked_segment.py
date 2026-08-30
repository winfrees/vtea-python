"""Objects across tile boundaries.

The acceptance criterion for Phase L3 is here: the same volume segmented at
several tile sizes must give the same objects, voxel for voxel, under the
default strategy. If that fails the reconciliation is wrong, and everything
built on top of it - measurements, association, cells - is wrong with it.

The negative controls matter as much: without matching, the same data
fragments, which is what proves the tests above are testing something.
"""

import numpy as np
import pytest

from vtea_core.blocked import (
    ABUTTING,
    MERGE,
    OVERLAPPING,
    OWN,
    Fragment,
    HaloExceeded,
    LabelLedger,
    MemoryBudget,
    SeamPolicy,
    SeamPolicyError,
    ZarrScratch,
    filter_by_size_blocked,
    load_ledger,
    plan_tiles,
    save_ledger,
    segment_blocked,
)
from vtea_core.blocked.reconcile import (
    UNCUT,
    UnionFind,
    centroid_pairs,
    group_fragments,
    overlap_pairs,
    touching_pairs,
)
from vtea_core.segmentation import label_components

RADIUS = 5


def spheres(shape=(32, 96, 96), count=24, radius=RADIUS, seed=0):
    """Well-separated balls, so "the right answer" is not in doubt."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, bool)
    grid = np.ogrid[-radius : radius + 1, -radius : radius + 1, -radius : radius + 1]
    ball = sum(axis**2 for axis in grid) <= radius**2
    placed, attempts = 0, 0
    while placed < count and attempts < 10000:
        attempts += 1
        centre = [int(rng.integers(radius + 2, size - radius - 2)) for size in shape]
        window = tuple(slice(value - radius - 1, value + radius + 2) for value in centre)
        if mask[window].any():
            continue
        mask[tuple(slice(v - radius, v + radius + 1) for v in centre)] |= ball
        placed += 1
    return mask


@pytest.fixture(scope="module")
def mask():
    return spheres()


@pytest.fixture(scope="module")
def whole(mask):
    return label_components(mask)


def plan_for(shape, tiles_wanted, halo):
    """A plan with roughly `tiles_wanted` tiles, with the halo paid for."""
    core = int(np.prod(shape)) / max(tiles_wanted, 1)
    edge = core ** (1 / len(shape))
    factor = ((edge + 2 * halo) / edge) ** len(shape)
    return plan_tiles(
        shape,
        budget=MemoryBudget(int(core * factor * 8 / 0.6) + 8192),
        bytes_per_voxel=8,
        halo=halo,
    )


def segment(mask, *, tiles, policy=None, halo=2 * RADIUS + 2):
    policy = policy or SeamPolicy.overlap_match()
    if not policy.is_overlapping:
        halo = 0
    plan = plan_for(mask.shape, tiles, halo)
    with ZarrScratch() as scratch:
        result = segment_blocked(
            label_components, {"mask": mask}, plan=plan, scratch=scratch, policy=policy
        )
        return plan.n_tiles, result.ledger, np.asarray(result.array)


def identical_objects(result, reference):
    """How many objects in `result` are voxel-for-voxel an object of
    `reference` - not merely the same count, which any two segmentations of
    the same mask could manage by luck."""
    matched = 0
    for object_id in np.unique(result[result > 0]):
        found = result == object_id
        overlapping = np.unique(reference[found])
        overlapping = overlapping[overlapping > 0]
        if len(overlapping) == 1 and np.array_equal(found, reference == overlapping[0]):
            matched += 1
    return matched


class TestSeamPolicy:
    def test_the_default_is_overlap_matching(self):
        from vtea_core.blocked import DEFAULT_POLICY, OVERLAP

        assert DEFAULT_POLICY.tiles == OVERLAPPING
        assert DEFAULT_POLICY.matching == OVERLAP
        assert DEFAULT_POLICY.resolution == OWN
        assert SeamPolicy() == DEFAULT_POLICY

    def test_the_four_presets(self):
        assert SeamPolicy.overlap_match().resolution == OWN
        assert SeamPolicy.overlap_match(merge=True).resolution == MERGE
        assert SeamPolicy.centroid_match().tiles == OVERLAPPING
        assert SeamPolicy.touching_merge().tiles == ABUTTING
        assert SeamPolicy.no_merge().tiles == ABUTTING

    def test_no_merge_drops_seam_objects_unless_told_otherwise(self):
        # Without the exclusion the object count is inflated, so the honest
        # default is to exclude.
        assert SeamPolicy.no_merge().drop_seam_objects
        assert not SeamPolicy.no_merge(drop_seam_objects=False).drop_seam_objects

    def test_overlap_matching_needs_an_overlap(self):
        with pytest.raises(SeamPolicyError, match="needs overlapping tiles"):
            SeamPolicy(tiles=ABUTTING, matching="overlap")

    def test_touching_matching_needs_tiles_that_touch(self):
        with pytest.raises(SeamPolicyError, match="only means something when tiles abut"):
            SeamPolicy(tiles=OVERLAPPING, matching="touching")

    def test_owning_a_copy_needs_a_copy_to_exist(self):
        with pytest.raises(SeamPolicyError, match="no tile has one"):
            SeamPolicy(tiles=ABUTTING, matching="touching", resolution=OWN)

    def test_owning_needs_to_know_what_matches(self):
        with pytest.raises(SeamPolicyError, match="which fragments are the same"):
            SeamPolicy(matching="none", resolution=OWN)

    def test_it_survives_a_round_trip(self):
        policy = SeamPolicy.centroid_match(max_centroid_distance=12.0, min_overlap=0.25)
        assert SeamPolicy.from_dict(policy.to_dict()) == policy

    def test_a_segmentation_does_not_mirror_its_own_border(self):
        # Padding a border tile by reflection invents objects at the edge of
        # the specimen and fuses them with the real ones they reflect. The
        # right boundary condition for a filter, a fabricated cell here.
        assert SeamPolicy().pad_mode is None


class TestMatching:
    def test_overlap_pairs_scores_by_iou(self):
        left = np.zeros((6, 6), int)
        left[1:4, 1:4] = 5
        right = np.zeros((6, 6), int)
        right[1:4, 1:5] = 9
        assert overlap_pairs(left, right, min_overlap=0.5) == [(5, 9, pytest.approx(0.75))]

    def test_overlap_pairs_respects_the_threshold(self):
        left = np.zeros((10,), int)
        left[:5] = 1
        right = np.zeros((10,), int)
        right[4:] = 2
        assert overlap_pairs(left, right, min_overlap=0.5) == []
        assert overlap_pairs(left, right, min_overlap=0.1)

    def test_overlap_pairs_on_disjoint_labellings(self):
        assert overlap_pairs(np.zeros((4,), int), np.zeros((4,), int)) == []

    def test_touching_pairs_scores_by_contact(self):
        lower = np.array([1, 1, 1, 0])
        upper = np.array([7, 7, 0, 0])
        # Two of the smaller face's two voxels are in contact.
        assert touching_pairs(lower, upper) == [(1, 7, pytest.approx(1.0))]

    def test_centroid_pairs_only_link_across_tiles(self):
        def fragment(tile, provisional_id, centre):
            return Fragment(
                tile=tile,
                local_id=provisional_id,
                provisional_id=provisional_id,
                core_voxels=10,
                block_voxels=10,
                centroid=centre,
                bbox=((0, 1), (0, 1), (0, 1)),
                faces=frozenset({"+z"}),
            )

        same_tile = [fragment((0,), 1, (0.0, 0.0, 0.0)), fragment((0,), 2, (1.0, 0.0, 0.0))]
        assert centroid_pairs(same_tile, max_distance=5.0) == []
        across = [fragment((0,), 1, (0.0, 0.0, 0.0)), fragment((1,), 2, (1.0, 0.0, 0.0))]
        assert len(centroid_pairs(across, max_distance=5.0)) == 1

    def test_centroid_pairs_ignore_fragments_no_seam_touched(self):
        interior = Fragment(
            tile=(0,),
            local_id=1,
            provisional_id=1,
            core_voxels=10,
            block_voxels=10,
            centroid=(0.0, 0.0, 0.0),
            bbox=((0, 1), (0, 1), (0, 1)),
        )
        assert centroid_pairs([interior, interior], max_distance=5.0) == []

    def test_centroid_distances_are_physical(self):
        from vtea_core.data import Spacing

        def fragment(tile, provisional_id, centre):
            return Fragment(
                tile=tile,
                local_id=provisional_id,
                provisional_id=provisional_id,
                core_voxels=1,
                block_voxels=1,
                centroid=centre,
                bbox=((0, 1), (0, 1), (0, 1)),
                faces=frozenset({"+z"}),
            )

        # Four voxels apart along z is 8 um at a 2 um z-step, and 0.8 um in x
        # at 0.2 um pixels. A threshold in physical units has to tell them
        # apart; one in voxels cannot.
        along_z = [fragment((0,), 1, (0.0, 0, 0)), fragment((1,), 2, (4.0, 0, 0))]
        along_x = [fragment((0,), 1, (0, 0, 0.0)), fragment((1,), 2, (0, 0, 4.0))]
        spacing = Spacing((2.0, 0.2, 0.2))
        assert centroid_pairs(along_z, max_distance=5.0, spacing=spacing) == []
        assert centroid_pairs(along_x, max_distance=5.0, spacing=spacing)

    def test_union_find_closes_chains(self):
        # A long object crosses three tiles and is matched pairwise; all
        # three fragments have to end up as one object.
        assignment, _ = group_fragments([1, 2, 3], [(1, 2, 0.9), (2, 3, 0.8)])
        assert len({assignment[i] for i in (1, 2, 3)}) == 1

    def test_group_ids_do_not_depend_on_the_order_of_the_pairs(self):
        forward, _ = group_fragments([1, 2, 3], [(1, 2, 0.9), (2, 3, 0.8)])
        backward, _ = group_fragments([1, 2, 3], [(2, 3, 0.8), (1, 2, 0.9)])
        assert forward == backward

    def test_evidence_is_the_weakest_link(self):
        # A chain joined by one confident match and one marginal one is only
        # as trustworthy as the marginal one.
        _, weakest = group_fragments([1, 2, 3], [(1, 2, 0.95), (2, 3, 0.55)])
        assert weakest[1] == pytest.approx(0.55)

    def test_union_find_is_stable_under_repeated_unions(self):
        union = UnionFind([1, 2, 3])
        union.union(1, 2)
        union.union(2, 1)
        assert union.find(1) == union.find(2) == 1


class TestInvariance:
    """The acceptance criterion for Phase L3."""

    def test_one_tile_equals_the_whole_image(self, mask, whole):
        n_tiles, ledger, result = segment(mask, tiles=1)
        assert n_tiles == 1
        np.testing.assert_array_equal(result, whole)
        assert ledger.seam_exposed_fraction == 0.0

    @pytest.mark.parametrize("tiles_wanted", [1, 8, 27])
    def test_the_same_objects_at_any_tile_size(self, mask, whole, tiles_wanted):
        expected = int(whole.max())
        n_tiles, ledger, result = segment(mask, tiles=tiles_wanted)
        found = np.unique(result[result > 0])
        assert len(found) == expected, f"{n_tiles} tiles gave {len(found)} objects, not {expected}"
        assert identical_objects(result, whole) == expected

    def test_seams_driven_through_objects_change_nothing(self, mask, whole):
        # Seams in the easy places prove less than seams in the hard ones.
        # At 27 tiles of this volume most objects straddle one.
        _n_tiles, ledger, result = segment(mask, tiles=27)
        assert ledger.seam_exposed_fraction > 0.4
        assert ledger.n_reconciled > 0
        assert identical_objects(result, whole) == int(whole.max())

    def test_ids_are_contiguous_from_one(self, mask):
        _n_tiles, ledger, result = segment(mask, tiles=8)
        found = np.unique(result[result > 0])
        np.testing.assert_array_equal(found, np.arange(1, len(found) + 1))
        assert ledger.object_ids == list(range(1, ledger.n_objects + 1))

    def test_a_rerun_gives_the_same_numbers(self, mask):
        _a, _la, first = segment(mask, tiles=8)
        _b, _lb, second = segment(mask, tiles=8)
        np.testing.assert_array_equal(first, second)

    def test_without_matching_the_same_data_fragments(self, mask, whole):
        # The negative control. Without it the tests above could be passing
        # because the tiling never cut anything.
        _n_tiles, ledger, result = segment(
            mask, tiles=27, policy=SeamPolicy.no_merge(drop_seam_objects=False)
        )
        found = np.unique(result[result > 0])
        assert len(found) > int(whole.max())
        assert identical_objects(result, whole) < int(whole.max())


class TestStrategies:
    @pytest.mark.parametrize(
        "policy",
        [
            SeamPolicy.overlap_match(),
            SeamPolicy.overlap_match(merge=True),
            SeamPolicy.centroid_match(max_centroid_distance=RADIUS),
            SeamPolicy.touching_merge(),
        ],
        ids=["overlap/own", "overlap/merge", "centroid/merge", "touching/merge"],
    )
    def test_every_merging_strategy_recovers_well_separated_objects(
        self, mask, whole, policy
    ):
        # Well-separated spheres are the case all four should agree on. Where
        # they differ is packed tissue, which is a judgement about data
        # rather than something a unit test settles.
        _n_tiles, ledger, result = segment(mask, tiles=27, policy=policy)
        assert len(np.unique(result[result > 0])) == int(whole.max())

    def test_an_abutting_tiling_reads_no_voxel_twice(self, mask):
        plan = plan_for(mask.shape, 27, 0)
        assert plan.overlap_ratio == pytest.approx(1.0)

    def test_no_merge_inflates_the_count_by_the_seam_exposed_fraction(self, mask, whole):
        _n_tiles, ledger, result = segment(
            mask, tiles=27, policy=SeamPolicy.no_merge(drop_seam_objects=False)
        )
        fragments = len(np.unique(result[result > 0]))
        # Every extra object is a piece of one a seam cut, so the inflation
        # and the seam-exposed fraction have to tell the same story.
        assert fragments > int(whole.max())
        assert ledger.seam_exposed_fraction > 0
        assert len(ledger.dropped) == 0

    def test_no_merge_plus_dropping_seam_objects_is_honest_instead(self, mask, whole):
        _n_tiles, ledger, result = segment(mask, tiles=27, policy=SeamPolicy.no_merge())
        kept = np.unique(result[result > 0])
        assert len(kept) < int(whole.max())
        assert len(ledger.dropped) > 0
        # What survives is whole objects, not pieces.
        assert identical_objects(result, whole) == len(kept)

    def test_dropping_dataset_border_objects_is_a_different_axis(self, whole):
        # A ball sitting on the edge of the volume is a truncated specimen;
        # one sitting on a tile boundary is not. The policy must not confuse
        # them.
        touching = np.zeros((16, 32, 32), bool)
        touching[:4, 4:12, 4:12] = True  # against the z face
        touching[6:12, 20:28, 20:28] = True  # interior
        policy = SeamPolicy.overlap_match(border_objects="drop")
        _n, ledger, result = segment(touching, tiles=1, policy=policy)
        assert ledger.n_objects == 2
        assert len(ledger.dropped) == 1
        assert len(np.unique(result[result > 0])) == 1

    def test_resegment_says_which_phase_it_belongs_to(self, mask):
        with pytest.raises(NotImplementedError, match="L5"):
            segment(mask, tiles=8, policy=SeamPolicy(resolution="resegment"))


class TestHaloVerification:
    def test_an_object_bigger_than_its_halo_is_flagged(self, mask):
        # Radius-5 balls have a 11-voxel extent, so a 2-voxel halo cannot
        # contain one and the check has to notice.
        _n_tiles, ledger, _result = segment(mask, tiles=27, halo=2)
        assert ledger.exceeded()

    def test_a_sufficient_halo_raises_no_alarm(self, mask):
        _n_tiles, ledger, _result = segment(mask, tiles=27)
        assert ledger.exceeded() == []

    def test_a_partial_view_in_one_tile_is_not_an_alarm(self, mask):
        # Every seam-crossing object is truncated in *some* tile - that is
        # what a halo is for. The object is only in trouble when no tile
        # contained it, so this must not fire on ordinary reconciliation.
        _n_tiles, ledger, _result = segment(mask, tiles=27)
        assert ledger.n_reconciled > 0
        assert all(ledger.confidence(i) > 0 for i in ledger.object_ids)

    def test_it_can_be_made_fatal(self, mask):
        with pytest.raises(HaloExceeded, match="truncated"):
            segment(
                mask,
                tiles=27,
                halo=2,
                policy=SeamPolicy.overlap_match(on_halo_exceeded="raise"),
            )


class TestLedger:
    def test_size_is_the_whole_object_not_one_tiles_piece(self, mask, whole):
        _n_tiles, ledger, _result = segment(mask, tiles=27)
        _ids, counts = np.unique(whole[whole > 0], return_counts=True)
        assert sorted(ledger.sizes().values()) == sorted(counts.tolist())

    def test_an_uncut_object_is_certain(self, mask):
        _n_tiles, ledger, _result = segment(mask, tiles=1)
        assert all(ledger.confidence(i) == 1.0 for i in ledger.object_ids)
        assert all(ledger.decided_by[i] == UNCUT for i in ledger.object_ids)

    def test_a_reconciled_object_carries_its_evidence(self, mask):
        _n_tiles, ledger, _result = segment(mask, tiles=27)
        reconciled = [i for i in ledger.object_ids if ledger.n_fragments(i) > 1]
        assert reconciled
        for object_id in reconciled:
            assert ledger.decided_by[object_id] == OWN
            assert 0 < ledger.confidence(object_id) <= 1.0

    def test_it_joins_the_measurement_table(self, mask):
        _n_tiles, ledger, _result = segment(mask, tiles=27)
        frame = ledger.to_frame()
        assert list(frame.columns) == [
            "object_id",
            "n_fragments",
            "seam_rule",
            "seam_confidence",
            "touches_seam",
            "at_dataset_border",
            "exceeded_halo",
        ]
        assert len(frame) == ledger.n_objects
        # The review workflow: gate on low confidence, open the gallery.
        assert frame["seam_confidence"].between(0, 1).all()

    def test_flagging_links_rather_than_merges(self, mask):
        policy = SeamPolicy(matching="centroid", resolution="flag", max_centroid_distance=RADIUS)
        _n_tiles, ledger, _result = segment(mask, tiles=27, policy=policy)
        assert ledger.links, "flag has to record what it declined to merge"
        for object_id, others in ledger.links.items():
            assert object_id not in others

    def test_it_survives_a_round_trip(self, mask, tmp_path):
        _n_tiles, ledger, _result = segment(mask, tiles=8)
        path = save_ledger(ledger, tmp_path / "ledger.json")
        restored = load_ledger(path)
        assert restored.n_objects == ledger.n_objects
        assert restored.policy == ledger.policy
        assert restored.sizes() == ledger.sizes()
        assert restored.describe() == ledger.describe()

    def test_a_newer_ledger_is_refused(self, tmp_path):
        data = LabelLedger().to_dict()
        data["version"] = 99
        with pytest.raises(ValueError, match="format version"):
            LabelLedger.from_dict(data)

    def test_describe_reports_what_a_user_needs_to_judge_the_run(self, mask):
        _n_tiles, ledger, _result = segment(mask, tiles=27)
        text = ledger.describe()
        assert "objects" in text
        assert "reconciled" in text
        assert "seam-exposed" in text


class TestFilterBySize:
    def test_it_filters_on_whole_object_sizes(self):
        # The step L2 had to refuse. Two balls of different sizes, a tiling
        # that cuts both: no tile holds either object's real size, and the
        # filter has to get it from the ledger.
        volume = np.zeros((24, 64, 64), bool)
        grid = np.ogrid[-3:4, -3:4, -3:4]
        volume[9:16, 9:16, 9:16] |= sum(a**2 for a in grid) <= 9
        grid = np.ogrid[-7:8, -7:8, -7:8]
        volume[5:20, 33:48, 33:48] |= sum(a**2 for a in grid) <= 49

        whole = label_components(volume)
        _ids, counts = np.unique(whole[whole > 0], return_counts=True)
        small, large = sorted(counts.tolist())
        cutoff = (small + large) // 2

        plan = plan_for(volume.shape, 8, 16)
        with ZarrScratch() as scratch:
            result = segment_blocked(
                label_components, {"mask": volume}, plan=plan, scratch=scratch
            )
            assert result.n_objects == 2
            assert sorted(result.ledger.sizes().values()) == [small, large]

            filtered = filter_by_size_blocked(result, scratch=scratch, min_size=cutoff)
            assert filtered.n_objects == 1
            kept = np.asarray(filtered.array)
            assert np.count_nonzero(kept) == large
            np.testing.assert_array_equal(np.unique(kept[kept > 0]), [1])

    def test_no_bounds_is_a_no_op(self, mask):
        plan = plan_for(mask.shape, 4, 2 * RADIUS + 2)
        with ZarrScratch() as scratch:
            result = segment_blocked(
                label_components, {"mask": mask}, plan=plan, scratch=scratch
            )
            assert filter_by_size_blocked(result, scratch=scratch) is result


class TestRefusals:
    def test_a_function_that_does_not_return_labels_is_refused(self, mask):
        plan = plan_for(mask.shape, 4, 0)
        with ZarrScratch() as scratch:
            with pytest.raises(TypeError, match="integer label image"):
                segment_blocked(
                    lambda mask: mask.astype(float),
                    {"mask": mask},
                    plan=plan,
                    scratch=scratch,
                    policy=SeamPolicy.touching_merge(),
                )


@pytest.fixture(scope="module")
def volume():
    rng = np.random.default_rng(0)
    data = rng.normal(800, 200, (24, 80, 80)).clip(0, 4000).astype(np.uint16)
    grid = np.ogrid[-5:6, -5:6, -5:6]
    ball = sum(axis**2 for axis in grid) <= 25
    for centre in [(6, 15, 15), (6, 15, 55), (16, 55, 15), (16, 55, 55), (11, 40, 40)]:
        data[tuple(slice(v - 5, v + 6) for v in centre)][ball] = 3500
    return data


class TestPipelineIntegration:
    """A whole protocol, out of core: blur, threshold, label, filter."""

    def protocol(self):
        from vtea_core.workflow import Pipeline, Step

        return Pipeline(
            [
                Step.for_function("imageprocessing", "gaussian_blur", params={"sigma": 1.0}),
                Step.for_function(
                    "segmentation",
                    "threshold_mask",
                    params={"method": "otsu"},
                    available={"volume"},
                ),
                Step.for_function("segmentation", "label_components", available={"mask"}),
                Step.for_function(
                    "segmentation",
                    "filter_by_size",
                    params={"min_size": 200},
                    available={"labels"},
                ),
            ]
        )

    def run_blocked(self, volume, budget, **kwargs):
        from vtea_core.blocked import BlockedPipeline

        plan = plan_tiles(
            volume.shape, budget=MemoryBudget(budget), bytes_per_voxel=8, halo=12
        )
        with BlockedPipeline(self.protocol(), plan=plan, **kwargs) as blocked:
            context = blocked.run({"volume": volume})
            return plan, blocked, np.asarray(context["labels"])

    def test_the_whole_protocol_matches_the_in_memory_run(self, volume):
        expected = self.protocol().run({"volume": volume})["labels"]
        for budget in (10**9, 700_000):
            plan, blocked, result = self.run_blocked(volume, budget)
            assert result.max() == expected.max()
            assert identical_objects(result, expected) == expected.max()

    def test_it_actually_tiles(self, volume):
        plan, _blocked, _result = self.run_blocked(volume, 700_000)
        assert plan.n_tiles > 1

    def test_the_ledger_is_available_per_step_and_per_key(self, volume):
        _plan, blocked, _result = self.run_blocked(volume, 700_000)
        assert "labels" in blocked.ledgers
        assert blocked.ledgers["labels"].n_reconciled > 0
        assert any(name.startswith("label_components") for name in blocked.ledgers)

    def test_filter_by_size_reads_the_ledger_of_the_labels_it_was_pointed_at(self, volume):
        _plan, blocked, _result = self.run_blocked(volume, 700_000)
        card = blocked.results["filter_by_size_1"]
        assert card.ledger is not None
        assert card.ledger.n_objects == blocked.ledgers["labels"].n_objects

    def test_filtering_labels_nothing_segmented_says_so(self, volume):
        from vtea_core.blocked import BlockedPipeline
        from vtea_core.workflow import Pipeline, Step

        step = Step.for_function(
            "segmentation", "filter_by_size", params={"min_size": 10}, available={"labels"}
        )
        plan = plan_tiles(volume.shape, budget=MemoryBudget(10**9), bytes_per_voxel=8)
        with BlockedPipeline(Pipeline([step]), plan=plan) as blocked:
            with pytest.raises(KeyError, match="ledger"):
                blocked.run({"labels": (volume > 3000).astype(np.int32)})

    def test_a_learned_segmenter_is_still_refused_until_resegment_exists(self, volume):
        # cellpose is not translation-invariant, so every tile's copy of a
        # seam-crossing object is shaped by a boundary that is not real.
        # Picking between them keeps a wrong mask, so this waits for L5.
        from vtea_core.blocked import BlockedPipeline, NotBlockableYet
        from vtea_core.workflow import Pipeline, Step

        step = Step.for_function("segmentation", "cellpose_segmentation")
        plan = plan_tiles(volume.shape, budget=MemoryBudget(10**9), bytes_per_voxel=8)
        with BlockedPipeline(Pipeline([step]), plan=plan) as blocked:
            with pytest.raises(NotBlockableYet, match="resegment"):
                blocked.run({"volume": volume})

    def test_the_policy_is_selectable_on_the_pipeline(self, volume):
        policy = SeamPolicy.no_merge(drop_seam_objects=False)
        _plan, blocked, _result = self.run_blocked(volume, 700_000, policy=policy)
        assert blocked.ledgers["labels"].policy == policy
        # The segmentation itself fragments, which is what the strategy says
        # it will do.
        segmentation = next(
            ledger
            for name, ledger in blocked.ledgers.items()
            if name.startswith("label_components")
        )
        expected = self.protocol().run({"volume": volume})["labels"]
        assert segmentation.n_objects > int(expected.max())

    def test_no_merge_and_a_size_filter_delete_pieces_of_real_objects(self, volume):
        """A trap worth pinning rather than discovering.

        A size filter is set for whole objects. Under a no-merge strategy
        the things being filtered are *fragments* of whole objects, each
        smaller than the object it came from - so a threshold that would
        have kept every object deletes parts of several, and what is left
        is neither the objects nor the fragments. The count can even come
        out *lower* than the correct answer, which is the shape of error
        nobody notices.
        """
        merged_plan, merged, merged_result = self.run_blocked(volume, 700_000)
        _plan, _blocked, fragmented = self.run_blocked(
            volume, 700_000, policy=SeamPolicy.no_merge(drop_seam_objects=False)
        )
        assert merged_plan.n_tiles > 1
        assert int(fragmented.max()) < int(merged_result.max())
        # And the voxels are gone, not merely renumbered.
        assert np.count_nonzero(fragmented) < np.count_nonzero(merged_result)
