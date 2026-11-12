"""Training sets: splitting them, weighting classes, and judging a model.

- `split` holds out a validation set, either by object (stratified by class) or
  **by group** - an image, a specimen, a region - so that no group has objects
  on both sides. Neighbouring cells of one image are alike; a split by object
  lets them leak between training and validation and overstates how well a
  model will do on a new image.
- `class_weights` / `sample_weights`: rare classes count as much as common ones
  (each class's weights sum to the same total).
- `evaluate`: accuracy, balanced accuracy, per-class precision / recall / F1 /
  support and the confusion matrix, as JSON-safe values for a model card.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _labels(labels) -> np.ndarray:
    return np.asarray(pd.Series(labels, dtype=object).to_numpy())


def labelled(labels) -> np.ndarray:
    """Which objects have a label (not None or NaN)."""
    return pd.notna(pd.Series(labels, dtype=object)).to_numpy()


def split(
    labels,
    groups=None,
    *,
    validation: float = 0.2,
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """(training rows, validation rows) of a labelled set.

    Without `groups`: per class, `validation` of its objects (at least one, when
    the class has two or more) go to validation. With `groups` (one value per
    object): whole groups go to validation until about `validation` of the
    objects are there, preferring groups that bring classes validation does not
    have yet; no group is on both sides. Raises ValueError when a split is
    impossible (fewer than two groups)."""
    y = _labels(labels)
    n = len(y)
    if not 0 < validation < 1:
        raise ValueError(f"validation must be between 0 and 1, got {validation}")
    rng = np.random.default_rng(random_state)
    if groups is None:
        val = []
        for value in pd.unique(y):
            rows = np.flatnonzero(y == value)
            rng.shuffle(rows)
            k = round(validation * len(rows))
            if len(rows) >= 2:
                k = min(max(k, 1), len(rows) - 1)
            else:
                k = 0
            val.extend(rows[:k])
        val = np.sort(np.asarray(val, dtype=np.int64))
        train = np.setdiff1d(np.arange(n), val)
        return train, val
    g = _labels(groups)
    if len(g) != n:
        raise ValueError(f"{len(g)} groups for {n} objects")
    names = list(pd.unique(g))
    if len(names) < 2:
        raise ValueError(
            "a split by group needs at least two groups (images, specimens or regions)"
        )
    order = [names[i] for i in rng.permutation(len(names))]
    target = validation * n
    chosen, covered, size = [], set(), 0
    # First pass: groups that bring classes validation does not have yet.
    for name in order:
        if size >= target or len(chosen) == len(names) - 1:
            break
        new = set(y[g == name]) - covered
        if new:
            chosen.append(name)
            covered |= set(y[g == name])
            size += int((g == name).sum())
    for name in order:
        if size >= target or len(chosen) == len(names) - 1:
            break
        if name not in chosen:
            chosen.append(name)
            size += int((g == name).sum())
    in_val = np.isin(g, np.asarray(chosen, dtype=object))
    return np.flatnonzero(~in_val), np.flatnonzero(in_val)


def class_weights(labels) -> dict:
    """{class: weight} so that every class's weights sum to the same total (the
    mean weight over the objects is 1)."""
    y = _labels(labels)
    values, counts = np.unique(y.astype(str), return_counts=True)
    original = {str(v): v for v in pd.unique(y)}
    k = len(values)
    return {original[v]: len(y) / (k * c) for v, c in zip(values, counts)}


def sample_weights(labels) -> np.ndarray:
    """One weight per object, from `class_weights`."""
    y = _labels(labels)
    weights = class_weights(y)
    return np.array([weights[v] for v in y], dtype=float)


def evaluate(true, predicted, classes: Sequence | None = None) -> dict:
    """Metrics of predicted against true classes (None in `predicted` counts as
    wrong): accuracy, balanced accuracy (mean recall over the classes present),
    per class precision / recall / F1 / support, and the confusion matrix
    (rows: true, columns: predicted, in `classes` order)."""
    t, p = _labels(true), _labels(predicted)
    if len(t) != len(p):
        raise ValueError(f"{len(t)} true classes for {len(p)} predictions")
    classes = (
        list(classes) if classes is not None else list(pd.unique(np.concatenate([t, p[p != None]])))
    )
    index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for a, b in zip(t, p):
        if a in index and b in index:
            matrix[index[a], index[b]] += 1
    support = np.array([(t == c).sum() for c in classes])
    tp = np.diag(matrix)
    predicted_n = matrix.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(support > 0, tp / np.maximum(support, 1), np.nan)
        precision = np.where(predicted_n > 0, tp / np.maximum(predicted_n, 1), np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    present = support > 0
    per_class = {
        str(c): {
            "precision": _num(precision[i]),
            "recall": _num(recall[i]),
            "f1": _num(f1[i]),
            "support": int(support[i]),
        }
        for i, c in enumerate(classes)
    }
    return {
        "n": len(t),
        "accuracy": float((t == p).mean()) if len(t) else None,
        "balanced_accuracy": float(np.nanmean(recall[present])) if present.any() else None,
        "per_class": per_class,
        "classes": [str(c) for c in classes],
        "confusion": matrix.tolist(),
    }


def _num(value) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)
