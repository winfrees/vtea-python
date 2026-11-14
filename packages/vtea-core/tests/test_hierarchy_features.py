"""Neighbourhood feature families beyond composition."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from vtea_core.hierarchy import FAMILIES, Hierarchy, transform_composition
from vtea_core.hierarchy import features as F
from vtea_core.spatial import membership_type_enrichment, type_enrichment


def windows(rows):
    """A membership matrix from lists of member rows."""
    n = 1 + max(max(r) for r in rows)
    data = [(i, j) for i, r in enumerate(rows) for j in r]
    i, j = zip(*data)
    return sparse.csr_matrix((np.ones(len(data)), (i, j)), shape=(len(rows), n))


def line(n=9):
    pos = np.c_[np.zeros(n), np.arange(n) * 10.0]
    return pos


class TestFamilies:
    def test_diversity(self):
        m = windows([[0, 1, 2, 3], [0, 1], [4]])
        classes = ["A", "A", "B", "B", None]
        out = F.diversity(m, classes)
        assert out["shannon"][0] == pytest.approx(np.log(2)) and out["shannon"][1] == 0
        assert out["simpson"][0] == pytest.approx(0.5) and out["richness"].tolist() == [2, 1, 0]
        assert out["dominant_class"][1] == "A" and out["dominant_fraction"][1] == 1
        assert np.isnan(out["shannon"][2]) and out["dominant_class"][2] is None

    def test_weighted_composition_favours_near_members(self):
        pos = line(3)
        m = windows([[0, 1, 2]])
        out = F.weighted_composition(m, pos, pos[[0]], ["A", "B", "B"], sigma=5.0)
        # A at 0, B at 10 and 20: weights 1, e^-2, e^-8
        w = np.array([1, np.exp(-2), np.exp(-8)])
        assert out["Class_A_WeightedFraction"][0] == pytest.approx(100 * w[0] / w.sum())

    def test_multiscale(self):
        pos = line(5)
        out = F.multiscale_composition(pos, pos[[2]], list("AABBB"), [5, 15])
        assert out["Class_B_ClassFraction_r5"][0] == 100
        assert out["Class_A_ClassFraction_r15"][0] == pytest.approx(100 / 3)

    def test_arrangement(self):
        pos = line(5)
        m = windows([[0, 1, 2, 3, 4], [0, 1, 2]])
        out = F.arrangement(m, pos, pos[[2, 0]], radius=20)
        assert np.isinf(out["elongation"][0])  # members on a line
        assert out["seed_offset"][0] == 0 and out["seed_offset"][1] > 1  # centred vs at the end
        assert out["member_nn_distance"][0] == 10
        assert out["density"][0] == pytest.approx(5 / (np.pi * 400))

    def test_internal_contacts(self):
        # a chain 0-1-2-3, classes A A B B: within the window, 3 edges, 2 same-class
        a = sparse.csr_matrix((np.ones(6), ([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])), shape=(4, 4))
        m = windows([[0, 1, 2, 3], [1, 2]])
        out = F.internal_contacts(m, a, list("AABB"))
        assert out["internal_edges"].tolist() == [3, 1]
        assert out["homophily"][0] == pytest.approx(2 / 3) and out["homophily"][1] == 0

    def test_asymmetry(self):
        pos = line(5)
        m = windows([[0, 1, 2, 3, 4]])
        assert F.asymmetry(m, pos, pos[[2]], list("AAXBB"))["asymmetry"][0] == 1
        assert F.asymmetry(m, pos, pos[[2]], list("ABXAB"))["asymmetry"][0] == 0

    def test_interface_distance(self):
        out = F.interface_distance(line(5), [1, 1, 1, 2, None])
        assert out["distance_to_other_type"].tolist()[:4] == [30, 20, 10, 10]
        assert np.isnan(out["distance_to_other_type"][4])

    def test_clr(self):
        frame = pd.DataFrame({"Class_A_ClassFraction": [50.0, 100.0], "Class_B_ClassFraction": [50.0, 0.0]})
        out = transform_composition(frame)
        assert out.iloc[0].tolist() == pytest.approx([0, 0])
        assert out.iloc[1].sum() == pytest.approx(0) and out.iloc[1, 0] > 0
        assert transform_composition(frame, "sqrt").iloc[1, 0] == 1
        with pytest.raises(ValueError, match="unknown"):
            transform_composition(frame, "log")


class TestDescribe:
    def make(self):
        ys, xs = np.meshgrid(np.arange(6) * 10.0, np.arange(10) * 10.0, indexing="ij")
        table = pd.DataFrame(
            {
                "object_id": np.arange(1, 61),
                "centroid-0": ys.ravel(),
                "centroid-1": xs.ravel(),
                "kind": np.where(xs.ravel() < 50, "A", "B"),
            }
        )
        pairs = [(i + 1, j + 1) for i in range(60) for j in range(60)
                 if abs(ys.ravel()[i] - ys.ravel()[j]) + abs(xs.ravel()[i] - xs.ravel()[j]) == 10]
        h = Hierarchy(table, contacts=np.array(pairs))
        h.add_windows("Objects", "W", radius=15, classes="kind")
        return h

    def test_every_family(self):
        h = self.make()
        w = h["W"]
        w.table["type"] = np.where(w.table["Class_A_ClassFraction"] > 50, 1, 2)
        out = h.describe("W", [f for f in FAMILIES], radii=[10, 25], types="type")
        for column in ("Class_A_WeightedFraction", "Class_A_ClassFraction_r25", "shannon", "elongation",
                       "homophily", "asymmetry", "distance_to_other_type"):
            assert column in out.columns and column in w.table.columns
        border = (w.table["centroid-1"] == 40) | (w.table["centroid-1"] == 50)
        assert (out.loc[border, "asymmetry"] > out.loc[~border, "asymmetry"].max() - 1e-9).all()
        assert out.loc[~border, "shannon"].max() == 0 and (out.loc[border, "shannon"] > 0).all()
        # adding again replaces the columns rather than duplicating them
        h.describe("W", "diversity")
        assert w.table.columns.is_unique

    def test_needs_what_it_needs(self):
        h = self.make()
        with pytest.raises(ValueError, match="segmentation"):
            h.describe("Objects", "diversity")
        with pytest.raises(ValueError, match="type column"):
            h.describe("W", "interface")
        with pytest.raises(ValueError, match="unknown"):
            h.describe("W", "colour")

    def test_composition_through_levels(self):
        h = self.make()
        h["W"].table["type"] = 1
        h.add_windows("W", "W2", radius=15, classes="type")
        out = h.composition("W2", "Objects", "kind")
        expanded = h.expand("W2").toarray()
        a = (h["Objects"].table["kind"] == "A").to_numpy()
        np.testing.assert_allclose(out["Class_A_ClassSums"], expanded @ a)


def test_membership_enrichment_counts_every_membership():
    m = windows([[0, 1, 2], [2, 3]])
    out = membership_type_enrichment(m, ["A", "A", "B", "B"], [1, 2]).set_index(["class", "type"])
    assert out.loc[("A", 1), "observed"] == 2 and out.loc[("B", 2), "observed"] == 2
    seed_only = type_enrichment(["A", "B"], [1, 2]).set_index(["class", "type"])
    assert seed_only["observed"].sum() == 2 and out["observed"].sum() == 5


def test_the_benchmark_runs_and_recovers_an_easy_structure():
    from vtea_core.hierarchy.benchmark import SCENARIOS, make_tissue, run_benchmark, summarise

    tissue = make_tissue("bands", ndim=3, size=120, seed=0)
    assert set(tissue.truth) == {0, 1, 2} and "centroid-2" in tissue.table
    with pytest.raises(ValueError, match="unknown scenario"):
        make_tissue("stripes")
    results = run_benchmark(
        scenarios=["halves"], feature_sets=["composition", "weighted (sigma r/2)"], methods=("kmeans",),
        ndims=(2,), size=200, tissue_seeds=(0,), cluster_seeds=(0, 1),
    )
    assert len(results) == 2 and (results["ari"] > 0.5).all() and len(SCENARIOS) == 4
    assert list(summarise(results).columns) == ["mean_ari", "worst_ari", "stability", "seconds"]
