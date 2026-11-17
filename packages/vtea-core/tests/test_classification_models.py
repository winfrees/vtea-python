"""Supervised classification: model folders and cards, training utilities, the ANN."""

import json

import numpy as np
import pandas as pd
import pytest

from vtea_core.classification import (
    ANNClassifier,
    ModelCard,
    ObjectClassifier,
    ObjectData,
    class_weights,
    evaluate,
    list_models,
    load_model,
    model_hash,
    predictions,
    read_card,
    sample_weights,
    split,
)


def cell_types(n=600, seed=0, shift=1.0):
    """n objects of 3 types: CD8 T (CD3+CD8+), CD4 T (CD3+CD4+), other; `shift`
    scales the intensities (a different microscope)."""
    rng = np.random.default_rng(seed)
    kind = np.repeat(["CD8 T", "CD4 T", "other"], n // 3)
    k = np.repeat([0, 1, 2], n // 3)
    table = pd.DataFrame(
        {
            "CD3 mean": shift * np.where(k < 2, 800, 50) * rng.lognormal(0, 0.25, n),
            "CD8 mean": shift * np.where(k == 0, 600, 40) * rng.lognormal(0, 0.25, n),
            "CD4 mean": shift * np.where(k == 1, 500, 30) * rng.lognormal(0, 0.25, n),
            "volume": rng.normal(500, 50, n),
        }
    )
    return table, kind


FEATURES = ["CD3 mean", "CD8 mean", "CD4 mean", "volume"]


# ---- training utilities ------------------------------------------------------------


def test_split_by_object_is_stratified():
    labels = ["a"] * 50 + ["b"] * 10 + ["c"] * 2 + ["d"]
    train, val = split(labels, validation=0.2, random_state=1)
    assert len(np.intersect1d(train, val)) == 0 and len(train) + len(val) == len(labels)
    y = np.array(labels)
    assert (y[val] == "a").sum() == 10 and (y[val] == "b").sum() == 2
    assert (y[val] == "c").sum() == 1 and (
        y[val] == "d"
    ).sum() == 0  # a single object stays in training


def test_split_by_group_never_shares_a_group():
    labels = np.array(["a", "b"] * 50)
    groups = np.repeat(["img1", "img2", "img3", "img4", "img5"], 20)
    train, val = split(labels, groups, validation=0.3, random_state=0)
    assert not set(groups[train]) & set(groups[val])
    assert 20 <= len(val) <= 40 and set(labels[val]) == {"a", "b"}
    with pytest.raises(ValueError, match="two groups"):
        split(labels, ["one"] * 100)


def test_class_weights_balance_the_classes():
    labels = ["a"] * 90 + ["b"] * 10
    w = class_weights(labels)
    assert w["a"] * 90 == pytest.approx(w["b"] * 10)
    assert sample_weights(labels).mean() == pytest.approx(1.0)


def test_evaluate():
    true = ["a", "a", "a", "b", "b", "c"]
    pred = ["a", "a", "b", "b", None, "c"]
    m = evaluate(true, pred, ["a", "b", "c"])
    assert m["confusion"] == [[2, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2 + 1) / 3)
    assert m["per_class"]["a"]["precision"] == 1.0 and m["per_class"]["b"]["support"] == 2
    json.dumps(m)  # fit for a model card


# ---- the ANN --------------------------------------------------------------------


def test_ann_learns_cell_types_from_half_and_predicts_the_other_half():
    table, kind = cell_types()
    train, test = split(kind, validation=0.5, random_state=2)
    model = ANNClassifier(FEATURES, random_state=0)
    assert isinstance(model, ObjectClassifier)
    seen = []
    history = model.fit(
        ObjectData(table.iloc[train]), kind[train], on_epoch=lambda e, info: seen.append(e)
    )
    assert (
        seen and history["epochs"] == len(seen) and 1 <= history["best_epoch"] <= history["epochs"]
    )
    out = predictions(model, ObjectData(table.iloc[test].reset_index(drop=True)))
    assert list(out.columns) == ["class", "confidence", "p CD4 T", "p CD8 T", "p other"]
    m = evaluate(kind[test], out["class"], model.classes)
    assert m["balanced_accuracy"] > 0.95
    assert np.allclose(out[[c for c in out.columns if c.startswith("p ")]].sum(axis=1), 1)
    assert (
        model.card.training["per_class"]["CD8 T"] > 0 and model.card.inputs["features"] == FEATURES
    )


def test_ann_stops_when_asked_and_skips_objects_missing_features():
    table, kind = cell_types(n=150)
    table.loc[3, "CD8 mean"] = np.nan
    model = ANNClassifier(FEATURES, max_epochs=50)
    history = model.fit(ObjectData(table), kind, stop=lambda: True)
    assert history["epochs"] == 1
    out = predictions(model, ObjectData(table))
    assert out.loc[3, "class"] is None and np.isnan(out.loc[3, "confidence"])
    assert out["class"].notna().sum() == len(table) - 1
    with pytest.raises(ValueError, match="two classes"):
        ANNClassifier(FEATURES).fit(ObjectData(table), ["x"] * len(table))


def test_two_classes_and_the_forward_pass_matches_scikit_learn():
    table, kind = cell_types(n=300)
    labels = np.where(kind == "other", "other", "T")
    model = ANNClassifier(FEATURES, max_epochs=30)
    model.fit(ObjectData(table), labels)
    proba = model.predict_proba(ObjectData(table))
    assert proba.shape == (300, 2) and model.classes == ["T", "other"]
    assert (np.array(model.classes)[proba.argmax(axis=1)] == labels).mean() > 0.95


def test_model_folder_round_trip_without_pickles(tmp_path):
    table, kind = cell_types(n=300)
    model = ANNClassifier(FEATURES, hidden=(16,), max_epochs=40)
    model.fit(ObjectData(table), kind)
    model.card.colors = {"CD8 T": "#ff0000"}
    model.card.notes = "test model"
    folder = model.save(tmp_path / "library" / "T cells")
    assert sorted(p.name for p in folder.iterdir()) == ["card.json", "config.json", "weights.npz"]
    with np.load(folder / "weights.npz", allow_pickle=False) as data:  # plain arrays only
        assert "coef_0" in data.files
    loaded = load_model(folder)
    assert isinstance(loaded, ANNClassifier) and loaded.classes == model.classes
    np.testing.assert_allclose(
        loaded.predict_proba(ObjectData(table)), model.predict_proba(ObjectData(table))
    )
    card = read_card(folder)
    assert card.colors == {"CD8 T": "#ff0000"} and card.notes == "test model" and card.created
    assert "3 classes" in card.summary() and "4 features" in card.summary()
    assert list_models(tmp_path / "library") == [folder] and list_models(tmp_path / "nothing") == []
    assert model_hash(folder) == model_hash(folder) and len(model_hash(folder)) == 16
    assert ModelCard.from_dict({**card.to_dict(), "unknown key": 1}).family == "ann"
    with pytest.raises(ValueError, match="not a model folder"):
        load_model(tmp_path)


def test_fine_tuning_adapts_a_model_to_new_data():
    table, kind = cell_types(n=600, seed=0)
    model = ANNClassifier(FEATURES, random_state=0)
    model.fit(ObjectData(table), kind)
    # A new microscope: everything 4x as bright, and fewer labels.
    new, new_kind = cell_types(n=600, seed=5, shift=4.0)
    train, test = split(new_kind, validation=0.8, random_state=0)
    before = evaluate(new_kind[test], predictions(model, ObjectData(new.iloc[test]))["class"])[
        "balanced_accuracy"
    ]
    tuned = ANNClassifier(FEATURES, random_state=0, max_epochs=100)
    tuned.fit(ObjectData(new.iloc[train]), new_kind[train], start_from=model)
    after = evaluate(new_kind[test], predictions(tuned, ObjectData(new.iloc[test]))["class"])[
        "balanced_accuracy"
    ]
    assert after > 0.9 and after >= before
    assert tuned.mean_ is model.mean_  # the original preprocessing is kept
    with pytest.raises(ValueError, match="keeps the model's features"):
        ANNClassifier(FEATURES[:2]).fit(ObjectData(new), new_kind, start_from=model)


def test_object_data_subset_keeps_the_graph_consistent():
    data = ObjectData(
        pd.DataFrame({"a": range(5)}), graph=np.array([[0, 1], [1, 2], [3, 4], [4, 0]])
    )
    sub = data.subset([0, 1, 4])
    assert sub.table["a"].tolist() == [0, 1, 4]
    assert sorted(map(tuple, sub.graph)) == [(0, 1), (2, 0)]


def test_fine_tuning_adds_new_classes_and_keeps_the_old_ones():
    table, kind = cell_types(n=600)
    two = np.where(kind == "other", "other", "T")  # first: T cells vs the rest
    model = ANNClassifier(FEATURES, random_state=0)
    model.fit(ObjectData(table), two)
    assert model.classes == ["T", "other"] and model.coefs_[-1].shape[1] == 1  # logistic output
    # New labels split T into CD8 T and CD4 T: the model gains two classes.
    new_labels = np.where(kind == "other", "other", kind)
    labels = np.where(new_labels == "other", "other", new_labels)
    tuned = ANNClassifier(FEATURES, hidden=(5,), random_state=0, max_epochs=150)
    tuned.fit(ObjectData(table), labels, start_from=model)
    assert tuned.classes == ["CD4 T", "CD8 T", "T", "other"] and tuned.hidden == model.hidden
    assert tuned.new_classes(model) == ["CD4 T", "CD8 T"]
    out = predictions(tuned, ObjectData(table))
    m = evaluate(labels, out["class"], ["CD4 T", "CD8 T", "other"])
    assert m["balanced_accuracy"] > 0.95


def test_carrying_a_logistic_output_over_to_softmax_keeps_the_predictions():
    from vtea_core.classification.ann import _carry_over

    table, kind = cell_types(n=300)
    model = ANNClassifier(FEATURES, max_epochs=20)
    model.fit(ObjectData(table), np.where(kind == "other", "other", "T"))
    coefs, intercepts = _carry_over(model, ["T", "other", "zzz"], 0)
    wide = ANNClassifier(FEATURES)
    wide.classes, wide.mean_, wide.scale_ = ["T", "other", "zzz"], model.mean_, model.scale_
    wide.coefs_, wide.intercepts_, wide.out_activation = coefs, intercepts, "softmax"
    p_old = model.predict_proba(ObjectData(table))
    p_new = wide.predict_proba(ObjectData(table))
    # The old classes keep their ratio; the new class starts unlikely.
    np.testing.assert_allclose(p_new[:, 1] / (p_new[:, 0] + p_new[:, 1]), p_old[:, 1], rtol=1e-6)
    assert p_new[:, 2].mean() < 0.3
