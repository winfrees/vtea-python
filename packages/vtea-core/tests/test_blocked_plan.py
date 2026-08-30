"""A tile plan is only useful if its tiles actually cover the data exactly
once, and if the halo arithmetic around the edges is right. Both are
checked here against every tile, not a sample: an off-by-one at one face of
one tile is a stripe through a figure."""

import math

import numpy as np
import pytest

from vtea_core.blocked import (
    BudgetTooSmall,
    HaloTooLarge,
    MemoryBudget,
    Tile,
    TilePlan,
    plan_tiles,
)

GB = 1024**3


def budget(total=8 * GB, **kwargs):
    return MemoryBudget(total, **kwargs)


def test_data_that_fits_gets_one_tile():
    plan = plan_tiles((64, 64, 64), budget=budget(), bytes_per_voxel=8)
    assert plan.is_single_tile
    assert plan.n_tiles == 1
    assert plan.tile == (64, 64, 64)
    assert "the whole dataset fits" in plan.describe()


def test_a_single_tile_covers_the_whole_array():
    plan = plan_tiles((5, 7, 9), budget=budget(), bytes_per_voxel=1)
    (tile,) = plan.tiles()
    assert tile.core == (slice(0, 5), slice(0, 7), slice(0, 9))
    assert tile.pad_width == ((0, 0), (0, 0), (0, 0))


def test_tiles_partition_the_array_exactly_once():
    # The property that matters: every voxel is some tile's responsibility,
    # and no voxel is two tiles' responsibility.
    shape = (37, 53, 61)
    plan = plan_tiles(shape, budget=budget(64 * 1024), bytes_per_voxel=8)
    assert not plan.is_single_tile
    seen = np.zeros(shape, dtype=int)
    for tile in plan.tiles():
        seen[tile.core] += 1
    assert seen.min() == 1 and seen.max() == 1


def test_the_number_of_tiles_matches_the_grid():
    plan = plan_tiles((100, 100), budget=budget(4096), bytes_per_voxel=8)
    assert len(list(plan.tiles())) == plan.n_tiles == math.prod(plan.splits)


def test_the_padded_region_is_the_core_grown_by_the_halo():
    # Three tiles, so there is a genuinely interior one - with two, both
    # touch the dataset border and the interesting case never arises.
    plan = plan_tiles((64,), budget=budget(400), bytes_per_voxel=8, halo=3)
    tiles = list(plan.tiles())
    assert len(tiles) == 3
    middle = tiles[1]
    assert middle.padded[0].start == middle.core[0].start - 3
    assert middle.padded[0].stop == middle.core[0].stop + 3
    assert middle.pad_width == ((0, 0),)


def test_the_halo_is_synthesized_only_at_the_dataset_border():
    plan = plan_tiles((64,), budget=budget(400), bytes_per_voxel=8, halo=3)
    tiles = list(plan.tiles())
    assert len(tiles) == 3
    first, last = tiles[0], tiles[-1]
    assert first.pad_width == ((3, 0),)
    assert first.padded[0].start == 0
    assert last.pad_width == ((0, 3),)
    assert last.padded[0].stop == 64
    assert first.at_dataset_border and last.at_dataset_border
    assert not tiles[1].at_dataset_border


def test_inner_recovers_the_core_from_a_padded_block():
    # The round trip a blocked executor performs: read the padded block,
    # pad the part that fell off the edge, run the step, slice `inner` back
    # out, and it must be the core.
    data = np.arange(64, dtype=np.int32)
    plan = plan_tiles((64,), budget=budget(512), bytes_per_voxel=8, halo=3)
    rebuilt = np.zeros_like(data)
    for tile in plan.tiles():
        block = np.pad(data[tile.padded], tile.pad_width, mode="reflect")
        assert block.shape == tile.padded_shape
        rebuilt[tile.core] = block[tile.inner]
    np.testing.assert_array_equal(rebuilt, data)


def test_inner_round_trip_in_three_dimensions():
    data = np.arange(12 * 14 * 16, dtype=np.int32).reshape(12, 14, 16)
    plan = plan_tiles(data.shape, budget=budget(8192), bytes_per_voxel=8, halo=2)
    assert not plan.is_single_tile
    rebuilt = np.zeros_like(data)
    for tile in plan.tiles():
        block = np.pad(data[tile.padded], tile.pad_width, mode="reflect")
        assert block.shape == tile.padded_shape
        rebuilt[tile.core] = block[tile.inner]
    np.testing.assert_array_equal(rebuilt, data)


def test_a_thin_axis_is_not_split_pointlessly():
    # A 24-slice slab should tile in XY and stay whole in Z, which is both
    # what the optics gave us and what the disk wants.
    plan = plan_tiles((24, 2048, 2048), budget=budget(2 * GB), bytes_per_voxel=35)
    assert not plan.is_single_tile
    assert plan.tile[0] == 24
    assert plan.tile[1] < 2048 and plan.tile[2] < 2048


def test_an_axis_can_be_held_out_of_tiling():
    # A channel or time axis is taken whole by every tile.
    plan = plan_tiles((3, 256, 256), budget=budget(1024**2), bytes_per_voxel=8, tiled_axes=(1, 2))
    assert plan.tile[0] == 3
    assert plan.splits[0] == 1
    for tile in plan.tiles():
        assert tile.core[0] == slice(0, 3)


def test_a_held_out_axis_still_costs_memory():
    small = plan_tiles((1, 512, 512), budget=budget(1024**2), bytes_per_voxel=8, tiled_axes=(1, 2))
    large = plan_tiles((8, 512, 512), budget=budget(1024**2), bytes_per_voxel=8, tiled_axes=(1, 2))
    assert large.n_tiles > small.n_tiles


def test_a_held_out_axis_gets_no_halo():
    plan = plan_tiles((3, 64, 64), budget=budget(4096), bytes_per_voxel=8, halo=2, tiled_axes=(1, 2))
    assert plan.halo[0] == 0
    assert plan.halo[1:] == (2, 2)


def test_tiles_land_on_chunk_boundaries():
    plan = plan_tiles(
        (2000, 2048, 2048), budget=budget(), bytes_per_voxel=35, halo=64, chunks=(128, 128, 128)
    )
    assert all(size % 128 == 0 for size in plan.tile)


def test_chunk_snapping_rounds_up_rather_than_wasting_a_chunk():
    # Snapping down turned a 342-voxel tile into 256 and threw away a
    # quarter of the budget with it.
    unsnapped = plan_tiles((2000, 2048, 2048), budget=budget(), bytes_per_voxel=35, halo=64)
    snapped = plan_tiles(
        (2000, 2048, 2048), budget=budget(), bytes_per_voxel=35, halo=64, chunks=(128, 128, 128)
    )
    assert math.prod(snapped.tile) >= 0.75 * math.prod(unsnapped.tile)
    assert snapped.tile_bytes <= budget().usable_bytes


def test_a_plan_never_exceeds_its_budget():
    for total in (4 * 1024**2, 64 * 1024**2, GB, 8 * GB):
        plan = plan_tiles((512, 512, 512), budget=budget(total), bytes_per_voxel=35, halo=8)
        assert plan.tile_bytes <= budget(total).usable_bytes


def test_a_budget_that_cannot_hold_one_voxel_is_refused():
    with pytest.raises(BudgetTooSmall, match="bytes/voxel"):
        plan_tiles((512, 512), budget=budget(64), bytes_per_voxel=1024)


def test_an_impossible_halo_says_which_knob_to_turn():
    with pytest.raises(HaloTooLarge) as excinfo:
        plan_tiles((512, 512, 512), budget=budget(1024**2), bytes_per_voxel=35, halo=64)
    message = str(excinfo.value)
    assert "memory budget" in message
    assert "64" in message


def test_overlap_ratio_reports_the_cost_of_the_halo():
    plan = plan_tiles((64,), budget=budget(512), bytes_per_voxel=8, halo=3)
    assert plan.overlap_ratio > 1.0
    whole = plan_tiles((64,), budget=budget(), bytes_per_voxel=8)
    assert whole.overlap_ratio == pytest.approx(1.0)


def test_a_wasteful_halo_is_reported_rather_than_hidden():
    # A 32-voxel core with a 16-voxel halo reads nine times the data it
    # writes. That runs, but nobody should discover it from the clock.
    plan = plan_tiles((256, 256), budget=budget(56 * 1024), bytes_per_voxel=8, halo=16)
    assert plan.tile == (32, 32)
    assert not plan.is_efficient
    assert any("halo is large" in note for note in plan.warnings())


def test_an_unmeasured_budget_is_flagged_on_the_plan():
    from vtea_core.blocked import FALLBACK

    plan = plan_tiles((8, 8), budget=MemoryBudget(GB, source=FALLBACK), bytes_per_voxel=1)
    assert any("fallback" in note for note in plan.warnings())


def test_tile_at_matches_iteration():
    plan = plan_tiles((40, 40), budget=budget(2048), bytes_per_voxel=8)
    for tile in plan.tiles():
        assert plan.tile_at(tile.index) == tile


def test_tile_at_rejects_an_index_off_the_grid():
    plan = plan_tiles((40, 40), budget=budget(2048), bytes_per_voxel=8)
    with pytest.raises(IndexError):
        plan.tile_at((plan.splits[0], 0))
    with pytest.raises(ValueError):
        plan.tile_at((0,))


def test_describe_is_one_readable_line_plus_its_caveats():
    plan = plan_tiles(
        (2000, 2048, 2048),
        budget=budget(),
        bytes_per_voxel=35,
        halo=64,
        chunks=(128, 128, 128),
        bound_by="watershed_split_1",
    )
    text = plan.describe()
    assert "tiles of" in text
    assert "halo" in text
    assert "watershed_split_1" in text


def test_shapes_and_costs_are_validated():
    with pytest.raises(ValueError):
        plan_tiles((), budget=budget(), bytes_per_voxel=8)
    with pytest.raises(ValueError):
        plan_tiles((0, 8), budget=budget(), bytes_per_voxel=8)
    with pytest.raises(ValueError):
        plan_tiles((8, 8), budget=budget(), bytes_per_voxel=0)
    with pytest.raises(ValueError):
        plan_tiles((8, 8), budget=budget(), bytes_per_voxel=8, halo=(1, 2, 3))
    with pytest.raises(ValueError):
        plan_tiles((8, 8), budget=budget(), bytes_per_voxel=8, tiled_axes=(5,))


def test_a_tile_reports_its_own_shapes():
    tile = Tile(
        index=(0,),
        core=(slice(0, 4),),
        padded=(slice(0, 6),),
        pad_width=((2, 0),),
    )
    assert tile.ndim == 1
    assert tile.core_shape == (4,)
    assert tile.padded_shape == (8,)
    assert tile.inner == (slice(2, 6),)


def test_a_plan_is_frozen():
    plan = plan_tiles((8, 8), budget=budget(), bytes_per_voxel=1)
    assert isinstance(plan, TilePlan)
    with pytest.raises(AttributeError):
        plan.tile = (4, 4)
