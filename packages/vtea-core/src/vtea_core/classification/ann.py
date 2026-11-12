"""A generic artificial neural network (multilayer perceptron) over measurements.

The simplest supervised object classifier: each object is its row of
measurements (channel intensities, size, shape; spatial and neighbourhood
features too), and a small fully connected network learns the classes.
CPU-only and fast - seconds for tens of thousands of objects - so it is the
first family of the classification workbench.

- **Preprocessing** is part of the model: the optional asinh(x / cofactor) and
  the standardisation (mean and scale of each feature, fitted on the training
  objects) are saved with the weights, so new data are scaled exactly as the
  training data were.
- **Training** (scikit-learn's `MLPClassifier`, Adam, ReLU, L2 penalty
  `alpha`) runs one pass over the data at a time, so it can report progress
  (`on_epoch`), be stopped (`stop`), and keep the weights of the epoch with
  the lowest validation loss (early stopping after `patience` epochs without
  improvement). Classes are weighted so that rare classes count as much as
  common ones.
- **Fine-tuning** (`start_from=` a trained model with the same classes and
  features) continues from its weights and keeps its preprocessing.
- **Prediction** is a NumPy forward pass from the saved weights: a model folder
  holds plain arrays, never a pickle. Objects missing a feature get no
  prediction (NaN probabilities).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from vtea_core.classification.models import (
    ModelCard,
    ObjectData,
    read_arrays,
    read_card,
    read_config,
    register_family,
    write_folder,
)
from vtea_core.classification.training import labelled, sample_weights, split

ASINH_COFACTOR = 150.0
TRANSFORMS = ("none", "asinh")


@register_family("ann")
class ANNClassifier:
    """Multilayer perceptron over the columns `features` of `ObjectData.table`."""

    family = "ann"

    def __init__(
        self,
        features,
        *,
        hidden=(64, 32),
        transform: str = "asinh",
        cofactor: float = ASINH_COFACTOR,
        alpha: float = 1e-4,
        learning_rate: float = 1e-3,
        batch_size: int = 200,
        max_epochs: int = 200,
        patience: int = 15,
        random_state: int | None = 0,
    ):
        if transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {transform!r}, expected {TRANSFORMS}")
        self.features = [str(f) for f in features]
        if not self.features:
            raise ValueError("choose at least one feature")
        self.hidden = tuple(int(h) for h in hidden)
        self.transform, self.cofactor = transform, float(cofactor)
        self.alpha, self.learning_rate = float(alpha), float(learning_rate)
        self.batch_size, self.max_epochs, self.patience = (
            int(batch_size),
            int(max_epochs),
            int(patience),
        )
        self.random_state = random_state
        self.classes: list[str] = []
        self.mean_ = self.scale_ = None
        self.coefs_: list[np.ndarray] = []
        self.intercepts_: list[np.ndarray] = []
        self.out_activation = "softmax"
        self.card = ModelCard(family=self.family, classes=[])

    # ---- inputs ---------------------------------------------------------------------

    def inputs(self) -> dict[str, Any]:
        return {
            "kind": "features",
            "features": list(self.features),
            "transform": self.transform,
            "cofactor": self.cofactor,
        }

    def _raw(self, table) -> np.ndarray:
        missing = [f for f in self.features if f not in table.columns]
        if missing:
            raise ValueError(f"the table has no feature(s) {missing}")
        values = table[self.features].to_numpy(dtype=float)
        if self.transform == "asinh":
            values = np.arcsinh(values / self.cofactor)
        return values

    def _scaled(self, table) -> np.ndarray:
        return (self._raw(table) - self.mean_) / self.scale_

    @property
    def trained(self) -> bool:
        return bool(self.coefs_)

    # ---- training ---------------------------------------------------------------------

    def fit(
        self,
        data: ObjectData,
        labels,
        *,
        sample_weight=None,
        start_from=None,
        validation=None,
        on_epoch=None,
        stop=None,
    ) -> dict[str, Any]:
        """Train on the labelled objects (labels None / NaN are left out, as are objects
        missing a feature). `validation`: (ObjectData, labels) to stop early on; by
        default 15% of the objects (stratified) are held out when there are enough.
        `start_from`: a trained ANNClassifier to fine-tune. `on_epoch(epoch, info)` is
        called after each pass; `stop()` returning True ends training. Returns the
        history (losses per epoch, best epoch)."""
        from sklearn.neural_network import MLPClassifier

        x_all = self._raw(data.table) if start_from is None else start_from._raw(data.table)
        y_all = np.asarray(labels, dtype=object)
        if len(y_all) != len(x_all):
            raise ValueError(f"{len(y_all)} labels for {len(x_all)} objects")
        usable = np.isfinite(x_all).all(axis=1) & labelled(y_all)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, float)[usable]
        x_all, y_all = x_all[usable], y_all[usable].astype(str)
        classes = sorted(set(y_all))
        if len(classes) < 2:
            raise ValueError("training needs objects of at least two classes")
        if start_from is not None:
            if not start_from.trained:
                raise ValueError("the model to fine-tune is not trained")
            if sorted(start_from.classes) != classes or start_from.features != self.features:
                raise ValueError(
                    "fine-tuning keeps the model's classes and features; train a new model for "
                    f"different ones (model: {start_from.classes}, labels: {classes})"
                )
            classes = list(start_from.classes)
        if validation is None and len(y_all) >= 50:
            train, val = split(y_all, validation=0.15, random_state=self.random_state)
            x_val, y_val = x_all[val], y_all[val]
            x_tr, y_tr = x_all[train], y_all[train]
            w_tr = sample_weight[train] if sample_weight is not None else None
        elif validation is not None:
            vdata, vlabels = validation
            x_val = self._raw(vdata.table) if start_from is None else start_from._raw(vdata.table)
            y_val = np.asarray(vlabels, dtype=object)
            keep = np.isfinite(x_val).all(axis=1) & labelled(y_val)
            x_val, y_val = x_val[keep], y_val[keep].astype(str)
            x_tr, y_tr, w_tr = x_all, y_all, sample_weight
        else:
            x_val = y_val = None
            x_tr, y_tr, w_tr = x_all, y_all, sample_weight
        if start_from is not None:
            self.mean_, self.scale_ = start_from.mean_, start_from.scale_
        else:
            self.mean_ = x_tr.mean(axis=0)
            scale = x_tr.std(axis=0)
            self.scale_ = np.where(scale > 0, scale, 1.0)
        x_tr = (x_tr - self.mean_) / self.scale_
        if x_val is not None:
            x_val = (x_val - self.mean_) / self.scale_
        weights = sample_weights(y_tr) if w_tr is None else w_tr
        mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate,
            batch_size=min(self.batch_size, len(y_tr)),
            random_state=self.random_state,
            shuffle=True,
        )
        self.classes = list(classes)
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_balanced_accuracy": [],
            "best_epoch": None,
        }
        best, best_loss, waited = None, np.inf, 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for epoch in range(1, self.max_epochs + 1):
                mlp.partial_fit(x_tr, y_tr, sample_weight=weights, classes=self.classes)
                if epoch == 1 and start_from is not None:
                    for mine, theirs in zip(
                        mlp.coefs_ + mlp.intercepts_, start_from.coefs_ + start_from.intercepts_
                    ):
                        mine[...] = theirs  # in place: the optimiser keeps references
                self._take(mlp)
                history["train_loss"].append(float(mlp.loss_))
                info = {"train_loss": float(mlp.loss_)}
                if x_val is not None and len(y_val):
                    loss, bacc = self._val_scores(x_val, y_val)
                    history["val_loss"].append(loss)
                    history["val_balanced_accuracy"].append(bacc)
                    info.update(val_loss=loss, val_balanced_accuracy=bacc)
                    if loss < best_loss - 1e-6:
                        best_loss, waited = loss, 0
                        best = (
                            [c.copy() for c in self.coefs_],
                            [b.copy() for b in self.intercepts_],
                            epoch,
                        )
                    else:
                        waited += 1
                if on_epoch is not None:
                    on_epoch(epoch, info)
                if (x_val is not None and waited >= self.patience) or (stop is not None and stop()):
                    break
        if best is not None:
            self.coefs_, self.intercepts_, history["best_epoch"] = best[0], best[1], best[2]
        else:
            history["best_epoch"] = len(history["train_loss"])
        history["epochs"] = len(history["train_loss"])
        self.card = ModelCard(
            family=self.family,
            classes=list(self.classes),
            inputs=self.inputs(),
            training={
                "objects": len(y_tr),
                "validation_objects": int(0 if y_val is None else len(y_val)),
                "per_class": {c: int((y_all == c).sum()) for c in self.classes},
                "epochs": history["epochs"],
                "best_epoch": history["best_epoch"],
                "hidden": list(self.hidden),
                "alpha": self.alpha,
                "learning_rate": self.learning_rate,
            },
            fine_tuned_from=(start_from.card.notes or start_from.card.created)
            if start_from is not None
            else None,
        )
        return history

    def _take(self, mlp) -> None:
        self.coefs_ = [c.copy() for c in mlp.coefs_]
        self.intercepts_ = [b.copy() for b in mlp.intercepts_]
        self.out_activation = mlp.out_activation_

    def _val_scores(self, x, y) -> tuple[float, float]:
        from vtea_core.classification.training import evaluate

        proba = self._forward(x)
        index = {c: i for i, c in enumerate(self.classes)}
        known = np.array([v in index for v in y])
        rows = np.flatnonzero(known)
        cols = np.array([index[v] for v in y[known]], dtype=int)
        w = sample_weights(y[known])
        loss = (
            float(-(w * np.log(np.clip(proba[rows, cols], 1e-12, 1))).sum() / w.sum())
            if len(rows)
            else np.inf
        )
        predicted = np.array(self.classes, dtype=object)[proba.argmax(axis=1)]
        bacc = evaluate(y, predicted, self.classes)["balanced_accuracy"]
        return loss, float(bacc) if bacc is not None else float("nan")

    # ---- prediction ---------------------------------------------------------------------

    def _forward(self, x: np.ndarray) -> np.ndarray:
        a = x
        for k, (w, b) in enumerate(zip(self.coefs_, self.intercepts_)):
            a = a @ w + b
            if k < len(self.coefs_) - 1:
                a = np.maximum(a, 0)
        if self.out_activation == "logistic" or a.shape[1] == 1:
            p = 1 / (1 + np.exp(-np.clip(a[:, 0], -500, 500)))
            return np.column_stack([1 - p, p])
        a = a - a.max(axis=1, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, data: ObjectData) -> np.ndarray:
        """Objects x classes probabilities (rows of NaN where a feature is missing)."""
        if not self.trained:
            raise ValueError("the model is not trained")
        x = self._scaled(data.table)
        ok = np.isfinite(x).all(axis=1)
        out = np.full((len(x), len(self.classes)), np.nan)
        if ok.any():
            out[ok] = self._forward(x[ok])
        return out

    # ---- saving ---------------------------------------------------------------------

    def save(self, folder) -> Path:
        if not self.trained:
            raise ValueError("train the model before saving it")
        config = {
            "family": self.family,
            "classes": self.classes,
            **self.inputs(),
            "hidden": list(self.hidden),
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "random_state": self.random_state,
            "out_activation": self.out_activation,
            "layers": len(self.coefs_),
        }
        arrays = {"mean": self.mean_, "scale": self.scale_}
        for k, (w, b) in enumerate(zip(self.coefs_, self.intercepts_)):
            arrays[f"coef_{k}"], arrays[f"intercept_{k}"] = w, b
        self.card.family, self.card.classes, self.card.inputs = (
            self.family,
            list(self.classes),
            self.inputs(),
        )
        return write_folder(folder, config, arrays, self.card)

    @classmethod
    def load(cls, folder) -> ANNClassifier:
        config = read_config(folder)
        if config.get("family") != "ann":
            raise ValueError(f"{folder} is a {config.get('family')!r} model, not an ANN")
        model = cls(
            config["features"],
            hidden=config.get("hidden", (64, 32)),
            transform=config.get("transform", "none"),
            cofactor=config.get("cofactor", ASINH_COFACTOR),
            alpha=config.get("alpha", 1e-4),
            learning_rate=config.get("learning_rate", 1e-3),
            batch_size=config.get("batch_size", 200),
            max_epochs=config.get("max_epochs", 200),
            patience=config.get("patience", 15),
            random_state=config.get("random_state", 0),
        )
        arrays = read_arrays(folder)
        model.classes = [str(c) for c in config["classes"]]
        model.mean_, model.scale_ = arrays["mean"], arrays["scale"]
        n = int(config["layers"])
        model.coefs_ = [arrays[f"coef_{k}"] for k in range(n)]
        model.intercepts_ = [arrays[f"intercept_{k}"] for k in range(n)]
        model.out_activation = config.get("out_activation", "softmax")
        model.card = read_card(folder)
        return model
