"""Associating two segmentations out of core, and the claim it changes nothing.

The value of the design is that the numbers come out the same, so nearly
every test here compares a tiled run against `vtea_core.objects.scoring` on
the same data held whole - at one tile, where the two must be identical by
construction, and at many, where they must be identical because the
arithmetic composes.

The three methods compose for three different reasons, and each is worth
pinning separately: containment because overlap counts are sums, centroid
distance because it never touches a voxel, and boundary distance because a
parent's window already holds everything that could have scored against it.
"""

import numpy as np
import pandas as pd
import pytest

from vtea_core.blocked import MemoryBudget, plan_tiles
from vtea_core.blocked.associate import (
    associate_objects_blocked,
    boundary_distance_blocked,
    boxes_blocked,
    boxes_from_ledger,
    centroid_distance_table,
    centroids_blocked,
    containment_blocked,
    object_ids_blocked,
    score_candidates_blocked,
)
from vtea_core.data import Spacing
from vtea_core.measurements import extract_measurements
from vtea_core.objects import associate_objects, boundary_distance, centroid_distance, containment

SPACING = Spacing((2.0, 0.5, 0.5))


def scene(shape=(16, 64, 64), step=16):
    """Nuclei inside cytoplasms on a grid, so a tiling cuts plenty of both.

    Sized from the shape rather than fixed, so a small scene is the same
    scene with fewer cells in it - a test that quietly produced no objects
    at all would pass every comparison it made.
    """
    parents = np.zeros(shape, dtype=np.int32)
    children = np.zeros(shape, dtype=np.int32)
    depth = max(shape[0] // 2, 2)
    object_id = 0
    for z in range(1, max(shape[0] - depth, 2), depth + 2):
        for y in range(2, shape[1] - step + 3, step):
            for x in range(2, shape[2] - step + 3, step):
                object_id += 1
                parents[z : z + depth, y : y + step - 4, x : x + step - 4] = object_id
                children[
                    z + 1 : z + depth - 1, y + 3 : y + step - 8, x + 3 : x + step - 8
                ] = object_id
    assert object_id, f"a {shape} scene at step {step} holds no objects"
    return children, parents


def plan_for(shape, n_tiles_at_least=1, halo=0):
    """A plan that actually divides the data, rather than one that says it
    would if the budget were smaller."""
    return plan_tiles(
        shape,
        budget=MemoryBudget(int(np.prod(shape) * 8 / max(n_tiles_at_least, 1) / 0.6) + 8192),
        bytes_per_voxel=8,
        halo=halo,
    )


def as_dict(scores):
    return {child: dict(row) for child, row in scores.scores.items()}


def by_id(links):
    """Child id -> parent id, so two runs are compared on the links rather
    than on what each happened to call the two segmentations."""
    return {link.child.object_id: link.parent.object_id for link in links}


def flat(scores):
    """Pair -> affinity, because pytest.approx does not nest."""
    return {
        (child, parent): value
        for child, row in scores.scores.items()
        for parent, value in row.items()
    }


class TestContainment:
    """A bincount over paired label arrays: the per-pair overlaps and the
    per-child totals are both sums, so a tiled run adds up exactly."""

    def test_one_tile_is_the_whole_image(self):
        children, parents = scene()
        plan = plan_for(children.shape)
        assert plan.is_single_tile
        blocked = containment_blocked(children, parents, plan=plan)
        assert as_dict(blocked) == as_dict(containment(children, parents))

    def test_many_tiles_are_the_whole_image(self):
        children, parents = scene()
        plan = plan_for(children.shape, n_tiles_at_least=27)
        assert plan.n_tiles > 1
        blocked = containment_blocked(children, parents, plan=plan)
        expected = containment(children, parents)
        assert as_dict(blocked) == as_dict(expected)

    def test_it_is_exact_rather_than_close(self):
        """A fraction assembled from four tiles is the same double as one
        computed in a single division - the numerator and the denominator
        are integer counts, and integers add without loss."""
        children, parents = scene()
        plan = plan_for(children.shape, n_tiles_at_least=8)
        blocked = containment_blocked(children, parents, plan=plan)
        expected = containment(children, parents)
        for child in expected.child_ids:
            assert blocked.for_child(int(child)) == expected.for_child(int(child))

    def test_a_child_straddling_two_parents_scores_against_both(self):
        children = np.zeros((4, 8, 8), dtype=np.int32)
        parents = np.zeros((4, 8, 8), dtype=np.int32)
        children[:, 2:6, 2:6] = 1
        parents[:, 0:4, :] = 5
        parents[:, 4:8, :] = 7
        plan = plan_for(children.shape, n_tiles_at_least=4)
        assert containment_blocked(children, parents, plan=plan).for_child(1) == pytest.approx(
            {5: 0.5, 7: 0.5}
        )

    def test_ids_that_do_not_cover_the_labels_are_refused(self):
        """Rather than silently scored against whichever object happens to
        sit at that slot - the two lists come from different places and only
        have to agree."""
        children, parents = scene(shape=(8, 32, 32))
        plan = plan_for(children.shape)
        assert len(object_ids_blocked(children, plan=plan)) > 1
        with pytest.raises(ValueError, match="different runs"):
            containment_blocked(children, parents, plan=plan, child_ids=[1])


class TestCentroidDistance:
    """The one that needs no image at all."""

    def test_the_table_gives_the_same_answer_as_the_labels(self):
        children, parents = scene()
        child_table = extract_measurements(children, children)
        parent_table = extract_measurements(parents, parents)
        from_table = centroid_distance_table(
            child_table, parent_table, spacing=SPACING, max_distance=12.0
        )
        expected = centroid_distance(children, parents, spacing=SPACING, max_distance=12.0)
        assert flat(from_table) == pytest.approx(flat(expected))

    def test_it_reads_no_voxels(self):
        """The claim that makes it the cheapest method on a large dataset:
        a 200 GB pair of segmentations costs two table reads."""
        children, parents = scene(shape=(8, 32, 32))
        child_table = extract_measurements(children, children)
        parent_table = extract_measurements(parents, parents)
        scores = centroid_distance_table(child_table, parent_table, max_distance=8.0)
        assert scores.n_candidates > 0

    def test_centroids_accumulated_over_tiles_match_the_whole_image(self):
        children, _parents = scene()
        plan = plan_for(children.shape, n_tiles_at_least=27)
        ids = object_ids_blocked(children, plan=plan)
        points = centroids_blocked(children, plan=plan, ids=ids)
        table = extract_measurements(children, children)
        expected = table[["centroid-0", "centroid-1", "centroid-2"]].to_numpy()
        np.testing.assert_allclose(points, expected)

    def test_without_a_table_it_falls_back_to_the_labels(self):
        children, parents = scene()
        plan = plan_for(children.shape, n_tiles_at_least=8)
        blocked = score_candidates_blocked(
            children,
            parents,
            plan=plan,
            method="centroid_distance",
            spacing=SPACING,
            max_distance=12.0,
        )
        expected = centroid_distance(children, parents, spacing=SPACING, max_distance=12.0)
        assert flat(blocked) == pytest.approx(flat(expected))

    def test_a_table_with_no_centroids_says_so(self):
        frame = pd.DataFrame({"object_id": [1, 2], "mean": [1.0, 2.0]})
        with pytest.raises(KeyError, match="centroid"):
            centroid_distance_table(frame, frame)


class TestBoundaryDistance:
    """Object-local: one window per parent, and the window already holds
    everything that could have scored against it."""

    def test_it_matches_the_whole_image_version(self):
        children, parents = scene()
        plan = plan_for(children.shape, n_tiles_at_least=27)
        blocked = boundary_distance_blocked(
            children, parents, plan=plan, spacing=SPACING, max_distance=6.0
        )
        expected = boundary_distance(children, parents, spacing=SPACING, max_distance=6.0)
        assert flat(blocked) == pytest.approx(flat(expected))

    def test_the_boxes_are_the_union_of_what_each_tile_saw(self):
        children, parents = scene()
        plan = plan_for(children.shape, n_tiles_at_least=27)
        ids = object_ids_blocked(parents, plan=plan)
        boxes = boxes_blocked(parents, plan=plan, ids=ids)
        for object_id, box in boxes.items():
            where = np.nonzero(parents == object_id)
            for axis, part in enumerate(box):
                assert part.start == where[axis].min()
                assert part.stop == where[axis].max() + 1

    def test_a_ledger_answers_without_reading_anything(self):
        """The point of keeping the ledger: a blocked segmentation already
        knows every object's box, so the window comes from a lookup."""
        from vtea_core.blocked.reconcile import Fragment, LabelLedger

        ledger = LabelLedger()
        ledger.add(
            1,
            [
                Fragment(
                    tile=(0,),
                    local_id=1,
                    provisional_id=1,
                    core_voxels=8,
                    block_voxels=8,
                    centroid=(1.0, 1.0, 1.0),
                    bbox=((0, 4), (0, 4), (0, 4)),
                ),
                Fragment(
                    tile=(1,),
                    local_id=1,
                    provisional_id=1,
                    core_voxels=8,
                    block_voxels=8,
                    centroid=(5.0, 1.0, 1.0),
                    bbox=((4, 9), (1, 3), (0, 2)),
                ),
            ],
        )
        assert boxes_from_ledger(ledger)[1] == (slice(0, 9), slice(0, 4), slice(0, 4))

    def test_supplied_boxes_are_used_rather_than_rescanned(self):
        children, parents = scene(shape=(8, 32, 32))
        plan = plan_for(children.shape, n_tiles_at_least=8)
        ids = object_ids_blocked(parents, plan=plan)
        boxes = boxes_blocked(parents, plan=plan, ids=ids)
        blocked = boundary_distance_blocked(
            children, parents, plan=plan, max_distance=4.0, boxes=boxes, parent_ids=ids
        )
        expected = boundary_distance(children, parents, max_distance=4.0)
        assert flat(blocked) == pytest.approx(flat(expected))


class TestTheWholeStep:
    def test_a_blocked_association_matches_an_in_memory_one(self):
        children, parents = scene()
        plan = plan_for(children.shape, n_tiles_at_least=27)
        blocked = associate_objects_blocked(
            children, parents, plan=plan, child_name="nuclei", parent_name="cells"
        )
        expected = associate_objects(
            children, parents, child_name="nuclei", parent_name="cells"
        )
        assert {link.child: link.parent for link in blocked} == {
            link.child: link.parent for link in expected
        }

    def test_the_posterior_survives_the_tiling(self):
        children, parents = scene()
        plan = plan_for(children.shape, n_tiles_at_least=8)
        blocked = associate_objects_blocked(children, parents, plan=plan)
        expected = associate_objects(children, parents)
        for link, other in zip(blocked, expected):
            assert link.probability == pytest.approx(other.probability)

    def test_it_records_that_it_was_blocked(self):
        """So a table computed one way can be told from one computed the
        other, which is the same reason every other scaled estimator says
        what it did."""
        children, parents = scene(shape=(8, 32, 32))
        plan = plan_for(children.shape, n_tiles_at_least=8)
        blocked = associate_objects_blocked(children, parents, plan=plan)
        assert next(iter(blocked)).params["blocked"] is True

    def test_an_unknown_method_is_refused(self):
        children, parents = scene(shape=(8, 16, 16))
        plan = plan_for(children.shape)
        with pytest.raises(ValueError, match="unknown scoring method"):
            score_candidates_blocked(children, parents, plan=plan, method="vibes")


class TestThroughThePipeline:
    """The step as a protocol runs it, which is where the wiring is - a
    scorer that is right and never reached is not much use."""

    def protocol(self, method="containment", **params):
        from vtea_core.workflow import Pipeline, Step

        return Pipeline(
            [
                Step.for_function(
                    "association",
                    "associate_objects",
                    params={"method": method, **params},
                    available={"child_labels", "parent_labels"},
                )
            ]
        )

    def run_blocked(self, protocol, context, shape, n_tiles=8):
        from vtea_core.blocked import BlockedPipeline

        plan = plan_for(shape, n_tiles_at_least=n_tiles)
        with BlockedPipeline(protocol, plan=plan, spacing=SPACING) as blocked:
            return plan, blocked.run(context)

    def test_a_protocol_associates_out_of_core(self):
        children, parents = scene()
        protocol = self.protocol()
        plan, context = self.run_blocked(
            protocol,
            {"child_labels": children, "parent_labels": parents},
            children.shape,
        )
        assert plan.n_tiles > 1
        links = context["associations"]
        expected = associate_objects(children, parents)
        assert {link.child: link.parent for link in links} == {
            link.child: link.parent for link in expected
        }

    def test_it_used_to_refuse_and_now_does_not(self):
        """Association was an OBJECT_LOCAL step the executor turned away.
        The refusal was honest and the point of this item is to remove it."""
        from vtea_core.blocked.executor import NotBlockableYet

        children, parents = scene(shape=(8, 32, 32))
        protocol = self.protocol()
        try:
            self.run_blocked(
                protocol,
                {"child_labels": children, "parent_labels": parents},
                children.shape,
            )
        except NotBlockableYet as exc:  # pragma: no cover - the thing being fixed
            raise AssertionError(f"association is still refused: {exc}") from exc

    def test_identity_association_needs_no_voxels_either(self):
        from vtea_core.workflow import Pipeline, Step

        children, parents = scene()
        protocol = Pipeline(
            [
                Step.for_function(
                    "association",
                    "associate_by_identity",
                    available={"child_labels", "parent_labels"},
                )
            ]
        )
        _plan, context = self.run_blocked(
            protocol,
            {"child_labels": children, "parent_labels": parents},
            children.shape,
        )
        assert len(list(context["associations"])) == len(np.unique(children)) - 1

    def test_a_measured_run_scores_centroids_from_the_table(self, monkeypatch):
        """The table is already there and the centroids in it are the same
        centroids; reading the image again for them was habit. Pinned by
        making the fallback fail - if the voxels are read, the test does."""
        from vtea_core.blocked import BlockedPipeline, associate as associate_module
        from vtea_core.workflow import Pipeline, Step

        children, parents = scene()
        protocol = Pipeline(
            [
                Step.for_function(
                    "measurements",
                    "extract_measurements",
                    available={"labels", "intensity"},
                ),
                Step.for_function(
                    "measurements",
                    "extract_measurements",
                    name="parent_measurements",
                    input_keys={"labels": "parent_labels", "intensity": "parent_labels"},
                    output_key="parent_table",
                ),
                Step.for_function(
                    "association",
                    "associate_objects",
                    params={"method": "centroid_distance", "max_distance": 12.0},
                    input_keys={"child_labels": "labels", "parent_labels": "parent_labels"},
                    output_key="associations",
                ),
            ]
        )

        def refuse(*_args, **_kwargs):
            raise AssertionError("the centroids were read from the image, not the table")

        monkeypatch.setattr(associate_module, "_centroid_distance_from_labels", refuse)

        plan = plan_for(children.shape, n_tiles_at_least=8)
        with BlockedPipeline(protocol, plan=plan, spacing=SPACING) as blocked:
            context = blocked.run(
                {
                    "labels": children,
                    "intensity": children,
                    "parent_labels": parents,
                }
            )
            assert set(blocked.tables) == {"labels", "parent_labels"}
        expected = associate_objects(
            children,
            parents,
            method="centroid_distance",
            max_distance=12.0,
            spacing=SPACING,
        )
        assert by_id(context["associations"]) == by_id(expected)
