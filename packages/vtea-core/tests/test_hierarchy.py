"""The spatial hierarchy: levels of objects built on levels of objects."""

import numpy as np
import pandas as pd
import pytest
from vtea_core.data import Spacing
from vtea_core.hierarchy import REGIONS, WINDOWS, Hierarchy, ObjectLevel
from vtea_core.neighborhoods import build_neighborhoods, neighborhood_features


def grid_table(nx=8, ny=6, step=10.0, split=4):
    """Cells on a grid `step` apart; kind "A" left of column `split`, "B" right."""
    ys, xs = np.meshgrid(np.arange(ny) * step, np.arange(nx) * step, indexing="ij")
    x, y = xs.ravel(), ys.ravel()
    return pd.DataFrame(
        {
            "object_id": np.arange(1, len(x) + 1),
            "centroid-0": y,
            "centroid-1": x,
            "kind": np.where(x < split * step, "A", "B"),
            "mean": x / step,
        }
    )


def two_rows(n=20, spacing=10.0, gap=16.0):
    """Two parallel rows of cells - two tubules side by side - and which pairs
    touch: neighbours along each row, and the cells facing each other across."""
    x = np.arange(n) * spacing
    table = pd.DataFrame(
        {
            "object_id": np.arange(1, 2 * n + 1),
            "centroid-0": np.r_[np.zeros(n), np.full(n, gap)],
            "centroid-1": np.r_[x, x],
            "kind": "epithelium",
        }
    )
    along = [(i, i + 1) for i in range(1, n)] + [(n + i, n + i + 1) for i in range(1, n)]
    across = [(i, n + i) for i in range(1, n + 1)]
    return table, np.array(along + across)


class TestLevels:
    def test_the_root_is_the_segmentation(self):
        h = Hierarchy(grid_table())
        assert h.root == "Objects" and h["Objects"].kind == "objects" and h["Objects"].depth == 0
        assert len(h) == 1 and h.tree() == [(0, "Objects")]

    def test_a_table_without_positions_is_refused(self):
        with pytest.raises(ValueError, match="centroid"):
            Hierarchy(pd.DataFrame({"object_id": [1, 2]}))

    def test_duplicate_ids_are_refused(self):
        table = grid_table()
        table.loc[1, "object_id"] = 1
        with pytest.raises(ValueError, match="unique"):
            Hierarchy(table)

    def test_names_must_be_new(self):
        h = Hierarchy(grid_table())
        h.add_windows("Objects", "W", radius=10)
        with pytest.raises(ValueError, match="already"):
            h.add_windows("Objects", "W", radius=10)

    def test_selections(self):
        level = Hierarchy(grid_table())["Objects"]
        assert level.selection(None).all()
        assert level.selection([1, 3]).sum() == 2
        assert level.selection(level.table["kind"] == "A").sum() == 24
        with pytest.raises(ValueError, match="no object"):
            level.selection([999])


class TestWindows:
    def test_windows_overlap(self):
        h = Hierarchy(grid_table())
        w = h.add_windows("Objects", "W", radius=10)
        assert w.kind == WINDOWS and len(w) == 48 and w.depth == 1
        # an inner cell is in its own window and its four side neighbours'
        assert w.memberships().max() == 5 and w.memberships().min() == 3
        assert (w.memberships() == w.sizes()).all()  # radius windows are symmetric

    def test_matches_the_neighbourhoods_module(self):
        """At level 1 windows are the neighbourhoods of vtea_core.neighborhoods."""
        table = grid_table()
        h = Hierarchy(table)
        w = h.add_windows("Objects", "W", radius=15, classes="kind")
        ref = neighborhood_features(build_neighborhoods(table, radius=15), table, class_column="kind")
        for column in ("n_objects", "Class_A_ClassFraction", "Class_B_ClassSums", "mean_distance"):
            np.testing.assert_allclose(w.table[column], ref[column])
        assert (w.table["seed"].to_numpy() == table["object_id"].to_numpy()).all()

    def test_nearest_gives_exactly_k_others(self):
        h = Hierarchy(grid_table())
        w = h.add_windows("Objects", "W", method="nearest", k=4)
        assert (w.sizes() == 5).all()

    def test_seeds_and_members_can_be_populations(self):
        table = grid_table()
        h = Hierarchy(table)
        a = table["kind"] == "A"
        w = h.add_windows("Objects", "W", radius=10, seeds=a, classes="kind")
        assert len(w) == int(a.sum())
        assert set(w.table["seed"]) == set(table.loc[a, "object_id"])
        # members are everyone: windows at the border count B cells
        assert w.table["Class_B_ClassSums"].max() == 1
        only = h.add_windows("Objects", "W2", radius=10, seeds=a, members=a, classes="kind")
        assert only.table["Class_B_ClassSums"].max() == 0

    def test_a_seed_is_always_in_its_own_window(self):
        table = grid_table()
        h = Hierarchy(table)
        b = (table["kind"] == "B").to_numpy()
        w = h.add_windows("Objects", "W", method="nearest", k=3, members=b)
        m = w.membership.toarray()
        assert all(m[i, w.seeds[i]] == 1 for i in range(len(w)))
        assert (w.sizes() == 4).all()

    def test_features_are_aggregated_over_members(self):
        h = Hierarchy(grid_table())
        w = h.add_windows("Objects", "W", radius=10, features="mean", aggregations="mean,max")
        corner = w.table.iloc[0]
        assert corner["mean_mean"] == pytest.approx(1 / 3) and corner["max_mean"] == 1

    def test_distances_use_the_spacing(self):
        h = Hierarchy(grid_table(), spacing=Spacing(values=(0.5, 0.5), unit="um"))
        w = h.add_windows("Objects", "W", radius=6)  # 10 px = 5 um apart
        assert w.sizes().max() == 5 and w.provenance["unit"] == "um"

    def test_windows_on_windows(self):
        table = grid_table()
        h = Hierarchy(table)
        w1 = h.add_windows("Objects", "W1", radius=10, classes="kind")
        w1.table["type"] = np.where(w1.table["Class_A_ClassFraction"] > 50, 1, 2)
        w2 = h.add_windows("W1", "W2", radius=10, classes="type")
        assert w2.depth == 2 and w2.parent == "W1"
        assert set(w2.table.columns) >= {"Class_1_ClassFraction", "Class_2_ClassFraction"}
        expanded = h.expand("W2", "Objects").toarray()
        # a level-2 window reaches the cells within two steps of its seed
        manual = (w2.membership.toarray() @ w1.membership.toarray()) > 0
        assert (expanded == manual).all()
        assert expanded.sum(axis=1).max() == 13  # the diamond of radius 2 on a grid
        assert h.tree() == [(0, "Objects"), (1, "W1"), (2, "W2")]


class TestRegions:
    def test_regions_of_touching_cells(self):
        table, contacts = two_rows()
        h = Hierarchy(table, contacts=contacts)
        r = h.add_regions("Objects", "R", types="kind")
        assert r.kind == REGIONS and len(r) == 1 and r.table["n_objects"].iloc[0] == 40

    def test_regions_keep_types_apart_and_drop_small_ones(self):
        table, contacts = two_rows(n=10)
        table.loc[table["object_id"].isin([3, 4]), "kind"] = "immune"
        h = Hierarchy(table, contacts=contacts)
        r = h.add_regions("Objects", "R", types="kind", values=["epithelium"], min_size=3)
        # immune cells cut the first row; its piece 1-2 is joined across to row 2
        assert sorted(r.table["n_objects"]) == [18]
        r2 = h.add_regions("Objects", "R2", types="kind")
        assert sorted(r2.table["n_objects"]) == [2, 18]
        assert r2.table["n_objects"].iloc[0] == 18  # region 1 is the largest

    def test_touching_needs_the_masks_or_a_distance(self):
        h = Hierarchy(grid_table())
        with pytest.raises(ValueError, match="masks"):
            h.add_regions("Objects", "R", types="kind")
        r = h.add_regions("Objects", "R", types="kind", touching=False, d=10)
        assert sorted(r.table["n_objects"]) == [24, 24]

    def test_contacts_from_the_label_image(self):
        labels = np.zeros((4, 8), dtype=np.int32)
        labels[1:3, 0:2], labels[1:3, 2:4], labels[1:3, 5:7] = 1, 2, 3
        table = pd.DataFrame(
            {"object_id": [1, 2, 3], "centroid-0": [1.5] * 3, "centroid-1": [0.5, 2.5, 5.5], "kind": "x"}
        )
        r = Hierarchy(table, labels=labels).add_regions("Objects", "R", types="kind")
        assert sorted(r.table["n_objects"]) == [1, 2]

    def test_windows_join_by_seed_touch_and_member_weight(self):
        """Two rows touching across their whole length: as windows their seeds
        touch across the rows too, but members along a row sit close and
        members across the rows do not, so the weight keeps the rows apart."""
        table, contacts = two_rows()
        h = Hierarchy(table, contacts=contacts)
        w = h.add_windows("Objects", "W", radius=12)
        w.table["type"] = 1
        apart = h.add_regions("W", "R", types="type", tau=0.5)
        assert sorted(apart.table["n_objects"]) == [20, 20]
        assert apart.provenance["sigma"] == pytest.approx(10.0)
        cells = h.expand("R", "Objects").toarray()
        rows = table["centroid-0"].to_numpy()
        assert all(len(set(rows[c.astype(bool)])) == 1 for c in cells)
        together = h.add_regions("W", "R0", types="type", tau=0.0)
        assert list(together.table["n_objects"]) == [40]

    def test_sharing_members_alone_never_joins(self):
        """Windows overlap everywhere; without touching seeds or a distance,
        nothing joins."""
        table, _ = two_rows()
        h = Hierarchy(table, contacts=np.zeros((0, 2), dtype=np.int64))
        h.add_windows("Objects", "W", radius=25).table["type"] = 1
        r = h.add_regions("W", "R", types="type")
        assert len(r) == 40 and (r.table["n_objects"] == 1).all()

    def test_a_distance_bridges_gaps(self):
        table, _ = two_rows()
        h = Hierarchy(table, contacts=np.zeros((0, 2), dtype=np.int64))
        h.add_windows("Objects", "W", radius=12).table["type"] = 1
        r = h.add_regions("W", "R", types="type", d=10)
        assert sorted(r.table["n_objects"]) == [20, 20]

    def test_region_shape(self):
        table, contacts = two_rows()
        h = Hierarchy(table, contacts=contacts)
        h.add_windows("Objects", "W", radius=12).table["type"] = 1
        r = h.add_regions("W", "R", types="type")
        assert (r.table["elongation"] > 10).all()  # a row is a strand

    def test_shape_matches_a_direct_computation(self):
        from scipy import sparse
        from vtea_core.hierarchy import shape

        rng = np.random.default_rng(3)
        positions = rng.normal(size=(60, 3)) * [1, 4, 9] + 5000.0  # far from the origin
        groups = [rng.choice(60, size=s, replace=False) for s in (1, 2, 5, 20, 60)]
        line = np.c_[np.zeros(6), np.zeros(6), np.arange(6.0)] + 5000.0
        positions = np.vstack([positions, line])
        groups.append(np.arange(60, 66))
        rows = np.concatenate([[i] * len(g) for i, g in enumerate(groups)])
        m = sparse.csr_matrix((np.ones(len(rows)), (rows, np.concatenate(groups))), shape=(len(groups), 66))
        out = shape(m, positions)
        for i, g in enumerate(groups[:-1]):
            centred = positions[g] - positions[g].mean(axis=0)
            assert out["radius_of_gyration"][i] == pytest.approx(np.sqrt((centred**2).sum(1).mean()), rel=1e-6)
            if len(g) >= 3:
                eig = np.sort(np.linalg.eigvalsh(centred.T @ centred / len(g)))[::-1]
                assert out["elongation"][i] == pytest.approx(np.sqrt(eig[0] / eig[1]), rel=1e-5)
        assert np.isnan(out["elongation"][0]) and np.isinf(out["elongation"].iloc[-1])

    def test_regions_of_regions_touch_through_their_members(self):
        table, contacts = two_rows(n=10)
        table.loc[table["object_id"] == 5, "kind"] = "immune"
        h = Hierarchy(table, contacts=contacts)
        r1 = h.add_regions("Objects", "R1", types="kind")
        r1.table["unit"] = "u"
        top = h.add_regions("R1", "R2", types="unit", tau=0.0)
        assert len(top) == 1 and h.expand("R2").sum() == 20


class TestMovingBetweenLevels:
    def setup_method(self):
        self.table = grid_table()
        self.h = Hierarchy(self.table)
        w1 = self.h.add_windows("Objects", "W1", radius=10, classes="kind")
        w1.table["type"] = np.where(w1.table["Class_A_ClassFraction"] > 50, 1, 2)
        w1.table["name"] = np.where(w1.table["type"] == 1, "left", "right")
        self.h.add_windows("W1", "W2", radius=10, classes="type")

    def test_reflect_member(self):
        out = self.h.reflect("W1", "Objects", ["Class_A_ClassFraction", "type", "name"],
                             categorical=["type", "name"])
        assert len(out) == len(self.table)
        assert (out["n_containers"] == self.h["W1"].memberships()).all()
        assert set(out["type"]) == {1, 2} and set(out["name"]) == {"left", "right"}

    def test_reflect_own_follows_seeds_down(self):
        own = self.h.own("W2", "Objects")
        w1, w2 = self.h["W1"], self.h["W2"]
        assert (w1.seeds[w2.seeds[own]] == np.arange(len(self.table))).all()
        out = self.h.reflect("W2", "Objects", "n_objects", how="own")
        np.testing.assert_array_equal(out["n_objects"], w2.table["n_objects"].to_numpy()[own])

    def test_own_needs_windows(self):
        table, contacts = two_rows()
        h = Hierarchy(table, contacts=contacts)
        h.add_regions("Objects", "R", types="kind")
        with pytest.raises(ValueError, match="only windows"):
            h.own("R", "Objects")

    def test_aggregate(self):
        out = self.h.aggregate("W2", "Objects", "mean", "max")
        assert out["max_mean"].max() == 7

    def test_members_and_containers(self):
        corner = np.zeros(len(self.h["W2"]), dtype=bool)
        corner[0] = True
        cells = self.h.members_of("W2", corner, "Objects")
        assert cells.sum() == 6  # the corner's diamond of radius 2, clipped
        back = self.h.containers_of("Objects", cells, "W2")
        assert back[0] and back.sum() > 1


class TestBookkeeping:
    def test_stale_when_the_parent_changes(self):
        table = grid_table()
        h = Hierarchy(table)
        w1 = h.add_windows("Objects", "W1", radius=10, classes="kind")
        w1.table["type"] = 1
        h.add_windows("W1", "W2", radius=10, classes="type")
        assert not h.is_stale("W1") and not h.is_stale("W2")
        w1.table["unused"] = 5  # a column the child was not built from
        assert not h.is_stale("W2")
        h["Objects"].table.loc[0, "kind"] = "B"
        assert h.is_stale("W1") and h.is_stale("W2")

    def test_remove_takes_descendants(self):
        h = Hierarchy(grid_table())
        h.add_windows("Objects", "W1", radius=10).table["t"] = 1
        h.add_windows("W1", "W2", radius=10)
        h.add_windows("Objects", "W3", radius=10)
        assert h.remove("W1") == ["W1", "W2"] and list(h.levels) == ["Objects", "W3"]
        with pytest.raises(ValueError, match="root"):
            h.remove("Objects")

    def test_round_trip(self, tmp_path):
        table, contacts = two_rows()
        h = Hierarchy(table, contacts=contacts, spacing=Spacing(values=(1.0, 1.0), unit="um"))
        h.add_windows("Objects", "W", radius=12, classes="kind").table["type"] = 1
        h.add_regions("W", "R", types="type")
        loaded = Hierarchy.load(h.save(tmp_path / "h"))
        assert list(loaded.levels) == ["Objects", "W", "R"] and loaded.spacing.unit == "um"
        for name in h.levels:
            pd.testing.assert_frame_equal(loaded[name].table, h[name].table, check_dtype=False)
            if h[name].membership is not None:
                assert (loaded[name].membership != h[name].membership).nnz == 0
            assert loaded[name].provenance == h[name].provenance
            assert not loaded.is_stale(name)
        assert (loaded.contacts == h.contacts).all()
        assert loaded.adjacency("W").nnz == h.adjacency("W").nnz

    def test_a_newer_file_is_refused(self, tmp_path):
        h = Hierarchy(grid_table())
        folder = h.save(tmp_path / "h")
        manifest = (folder / "hierarchy.json").read_text().replace(
            '"vtea_hierarchy_version": 1', '"vtea_hierarchy_version": 99'
        )
        (folder / "hierarchy.json").write_text(manifest)
        with pytest.raises(ValueError, match="newer"):
            Hierarchy.load(folder)

    def test_level_checks(self):
        table = grid_table()
        with pytest.raises(ValueError, match="parent"):
            ObjectLevel("x", WINDOWS, table)
        with pytest.raises(ValueError, match="kind"):
            ObjectLevel("x", "blobs", table)


class TestProtocol:
    def test_a_protocol_builds_two_levels_types_them_and_reflects(self):
        from vtea_core.workflow import Pipeline, Step

        table, _ = two_rows()
        table["kind"] = np.where(table["centroid-1"] < 100, "A", "B")
        labels = np.zeros((40, 200), dtype=np.int32)  # masks tiling the rows, so neighbours touch
        for oid, y, x in table[["object_id", "centroid-0", "centroid-1"]].to_numpy():
            labels[int(y) : int(y) + 16, int(x) : int(x) + 10] = oid
        pipeline = Pipeline()
        for function, params in (
            ("hierarchy_from_objects", {"connectivity": 2}),
            ("add_windows_level", {"name": "W1", "radius": 12.0, "classes": "kind"}),
            ("classify_level", {"n_clusters": 2}),
            ("add_windows_level", {"name": "W2", "radius": 12.0, "classes": "neighborhood_type"}),
            ("add_regions_level", {"parent": "W1", "name": "R", "touching": False, "d": 10.0}),
            ("reflect_level", {"level": "W1", "prefix": "W1."}),
        ):
            pipeline.add_step(
                Step.for_function(
                    "hierarchy",
                    function,
                    available=pipeline.available_keys({"data", "spacing", "labels"}),
                    taken_names=pipeline.step_names(),
                    params=params,
                )
            )
        context = pipeline.run({"data": table, "spacing": None, "labels": labels})
        h = context["hierarchy"]
        assert [name for _, name in h.tree()] == ["Objects", "W1", "W2", "R"]
        assert h.contacts is not None and len(h.contacts)
        reflected = context["level_reflection"]
        assert len(reflected) == len(table) and "W1.neighborhood_type" in reflected.columns
        assert set(reflected["W1.neighborhood_type"]) <= {0, 1}
        # running it again replaces the levels rather than failing on their names
        again = pipeline.run({"data": table, "spacing": None, "labels": labels})["hierarchy"]
        assert len(again) == 4
