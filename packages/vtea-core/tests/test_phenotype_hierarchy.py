import numpy as np
import pandas as pd
import pytest

from vtea_core.classes import UNLABELLED, Phenotype, PhenotypeTree, load_phenotypes, save_phenotypes


@pytest.fixture
def frame():
    # 8 objects: CD45 (immune), CD3 (T), CD8, CD4, panCK (epithelial)
    return pd.DataFrame({
        "object_id": np.arange(1, 9),
        "CD45": [1, 1, 1, 1, 1, 0, 0, 0],
        "CD3":  [1, 1, 1, 0, 0, 0, 0, 1],   # object 8 is CD3+ but not immune: not a T cell
        "CD8":  [1, 0, 1, 0, 0, 0, 0, 0],
        "CD4":  [0, 1, 1, 0, 0, 0, 0, 0],   # object 3 is CD4+ and CD8+: an overlap
        "panCK": [0, 0, 0, 0, 0, 1, 1, 0],
        "cluster": [3, 3, 7, 1, 1, 2, 2, 5],
    })


@pytest.fixture
def tree():
    return PhenotypeTree([
        Phenotype("Immune", "CD45 > 0", color="#ff0000"),
        Phenotype("T cell", "CD3 > 0", parent="Immune"),
        Phenotype("CD8 T", "CD8 > 0", parent="T cell"),
        Phenotype("CD4 T", "CD4 > 0", parent="T cell"),
        Phenotype("Other immune", "", parent="Immune"),
        Phenotype("Epithelial", "panCK > 0"),
    ])


def test_structure(tree):
    assert [p.name for p in tree] == ["Immune", "T cell", "CD8 T", "CD4 T", "Other immune", "Epithelial"]
    assert tree.path("CD8 T") == "Immune > T cell > CD8 T"
    assert tree.depth("CD8 T") == 3 and tree.depth("Epithelial") == 1
    assert [c.name for c in tree.children("T cell")] == ["CD8 T", "CD4 T"]
    with pytest.raises(KeyError):
        tree.add(Phenotype("x", parent="nope"))
    with pytest.raises(ValueError):
        tree.add(Phenotype("Immune"))
    with pytest.raises(ValueError):
        tree.add(Phenotype("a > b"))


def test_members_are_within_the_parent(tree, frame):
    assert list(np.flatnonzero(tree.members("T cell", frame)) + 1) == [1, 2, 3]  # not object 8
    assert list(np.flatnonzero(tree.members("CD4 T", frame)) + 1) == [2, 3]
    assert list(np.flatnonzero(tree.members("CD4 T", frame, exclusive=True)) + 1) == [2]  # 3 went to CD8 T
    # An empty rule is "the rest of the parent" once earlier siblings have claimed theirs.
    assert list(np.flatnonzero(tree.members("Other immune", frame, exclusive=True)) + 1) == [4, 5]


def test_assign_gives_every_level(tree, frame):
    a = tree.assign(frame)
    assert list(a["phenotype"]) == ["CD8 T", "CD4 T", "CD8 T", "Other immune", "Other immune",
                                    "Epithelial", "Epithelial", UNLABELLED]
    assert a.loc[0, "phenotype path"] == "Immune > T cell > CD8 T"
    assert list(a["phenotype level 1"]) == ["Immune"] * 5 + ["Epithelial"] * 2 + [UNLABELLED]
    assert list(a["phenotype level 2"])[:5] == ["T cell", "T cell", "T cell", "Other immune", "Other immune"]
    assert a.loc[5, "phenotype level 3"] == UNLABELLED


def test_counts_report_overlaps(tree, frame):
    c = tree.counts(frame).set_index("phenotype")
    assert c.loc["Immune", "count"] == 5 and c.loc["Immune", "% of all"] == pytest.approx(62.5)
    assert c.loc["T cell", "% of parent"] == pytest.approx(60.0)
    assert c.loc["CD4 T", "count"] == 1 and c.loc["CD4 T", "overlap"] == 1
    assert c.loc["CD8 T", "overlap"] == 0
    assert c.loc["Other immune", "overlap"] == 0  # "the rest of the parent": not an overlap


def test_rules_can_use_clusters_gates_and_backticks(frame):
    frame = frame.assign(**{"gate high": [True, False] * 4})
    tree = PhenotypeTree([Phenotype("A", "cluster in [3, 7]"), Phenotype("B", "`gate high` AND NOT CD3 > 0")])
    assert list(np.flatnonzero(tree.members("A", frame)) + 1) == [1, 2, 3]
    assert list(np.flatnonzero(tree.members("B", frame)) + 1) == [5, 7]
    assert tree.missing_columns(frame) == {}
    assert tree.missing_columns(frame.drop(columns="cluster")) == {"A": {"cluster"}}


def test_remove_rename_and_round_trip(tree, frame, tmp_path):
    tree.rename("T cell", "T")
    assert tree.path("CD4 T") == "Immune > T > CD4 T"
    path = save_phenotypes(tree, tmp_path / "p.json")
    back = load_phenotypes(path)
    assert [(p.name, p.rule, p.parent, p.color) for p in back] == \
        [(p.name, p.rule, p.parent, p.color) for p in tree]
    pd.testing.assert_frame_equal(back.assign(frame), tree.assign(frame))
    assert tree.remove("T") == ["T", "CD8 T", "CD4 T"]
    assert "CD8 T" not in tree and len(tree) == 3
    with pytest.raises(ValueError):
        PhenotypeTree.from_dict({"format": "other"})
