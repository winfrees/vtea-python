"""Object classifiers: one interface for every model family, and model folders.

A supervised object classifier learns classes from labelled objects and predicts
them for every object. Families differ in what they see - a row of measurements
(`ann`), an image crop around the object (NephNet3D), the object and its
neighbourhood graph (STELLAR) - but the workbench drives them all the same way,
through `ObjectClassifier`:

- `fit(data, labels, ...)` / `predict_proba(data)` on an `ObjectData`;
- `save(folder)` / `<Family>.load(folder)`, with `save_model` / `load_model`
  choosing the family from the folder.

**A model is a folder**, never a pickle, so a shared model library can be opened
without running anything it contains:

```
<model>/
  config.json    family, classes, inputs, architecture, preprocessing parameters
  weights.npz    the weights, as plain arrays (or a family's own non-pickle format)
  card.json      the model card: what it was trained on, how well it validated
```

The **model card** (`ModelCard`) is what makes a model reusable and reviewable:
its classes and colours, its inputs, its training data, its validation metrics,
what it was fine-tuned from, and free notes.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

MODEL_FORMAT_VERSION = 1
CONFIG, WEIGHTS, CARD = "config.json", "weights.npz", "card.json"


@dataclass
class ObjectData:
    """What a classifier may look at, for a set of objects (rows of `table`):
    their measurements, and - for the families that need them - image crops
    (objects x channels x *spatial) and a neighbour graph ((edges, 2) row pairs)."""

    table: pd.DataFrame
    crops: np.ndarray | None = None
    graph: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.table)

    def subset(self, rows) -> ObjectData:
        rows = np.asarray(rows)
        graph = None
        if self.graph is not None:
            keep = np.zeros(len(self.table), dtype=bool)
            keep[rows] = True
            new = np.full(len(self.table), -1)
            new[rows] = np.arange(len(rows))
            edges = self.graph[keep[self.graph[:, 0]] & keep[self.graph[:, 1]]]
            graph = new[edges]
        return ObjectData(
            self.table.iloc[rows].reset_index(drop=True),
            None if self.crops is None else self.crops[rows],
            graph,
        )


@dataclass
class ModelCard:
    """What a model is, what it was trained on and how well it does (card.json)."""

    family: str
    classes: list[str]
    colors: dict[str, str] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    fine_tuned_from: str | None = None
    created: str = ""
    author: str = ""
    notes: str = ""
    vtea_core_version: str = ""
    format_version: int = MODEL_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCard:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def summary(self) -> str:
        """One line: family, classes, inputs and validated balanced accuracy."""
        bacc = self.validation.get("balanced_accuracy")
        features = self.inputs.get("features") or []
        inputs = f"{len(features)} features" if features else self.inputs.get("kind", "?")
        text = f"{self.family}: {len(self.classes)} classes ({', '.join(map(str, self.classes))}); {inputs}"
        return text + (f"; validated balanced accuracy {bacc:.3f}" if bacc is not None else "")


@runtime_checkable
class ObjectClassifier(Protocol):
    """The interface every model family implements."""

    family: str
    classes: list[str]
    card: ModelCard

    def inputs(self) -> dict[str, Any]: ...

    def fit(
        self,
        data: ObjectData,
        labels,
        *,
        sample_weight=None,
        start_from=None,
        on_epoch=None,
        stop=None,
    ) -> dict[str, Any]: ...

    def predict_proba(self, data: ObjectData) -> np.ndarray: ...

    def save(self, folder) -> Path: ...


FAMILIES: dict[str, type] = {}


def register_family(name: str):
    """Class decorator: make a model family loadable by `load_model`."""

    def wrap(cls):
        FAMILIES[name] = cls
        cls.family = name
        return cls

    return wrap


def save_model(model: ObjectClassifier, folder) -> Path:
    """Write the model folder (config, weights, card)."""
    return model.save(folder)


def load_model(folder) -> ObjectClassifier:
    """Read a model folder, whichever family it is (no pickles are read)."""
    folder = Path(folder)
    config = read_config(folder)
    family = config.get("family")
    if family not in FAMILIES:
        _import_families()
    if family not in FAMILIES:
        raise ValueError(f"{folder}: unknown model family {family!r} (known: {sorted(FAMILIES)})")
    return FAMILIES[family].load(folder)


def read_config(folder) -> dict[str, Any]:
    path = Path(folder) / CONFIG
    if not path.exists():
        raise ValueError(f"{folder} is not a model folder (no {CONFIG})")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("format_version", 1) > MODEL_FORMAT_VERSION:
        raise ValueError(
            f"{folder} was saved by a newer vtea-core (format {config['format_version']})"
        )
    return config


def read_card(folder) -> ModelCard:
    """The model card alone (cheap: for listing a model library)."""
    path = Path(folder) / CARD
    if path.exists():
        return ModelCard.from_dict(json.loads(path.read_text(encoding="utf-8")))
    config = read_config(folder)
    return ModelCard(family=config.get("family", "?"), classes=list(config.get("classes", [])))


def list_models(library) -> list[Path]:
    """Model folders directly inside `library` (or `library` itself), sorted by name."""
    library = Path(library)
    if not library.is_dir():
        return []
    if (library / CONFIG).exists():
        return [library]
    return sorted(
        (p for p in library.iterdir() if (p / CONFIG).exists()), key=lambda p: p.name.lower()
    )


def model_hash(folder) -> str:
    """A short hash of a model's config and weights (to tell which model made a
    result, without keeping the model)."""
    digest = hashlib.sha256()
    for name in (CONFIG, WEIGHTS):
        path = Path(folder) / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def write_folder(folder, config: dict, arrays: dict[str, np.ndarray], card: ModelCard) -> Path:
    """Write config.json, weights.npz (no object arrays) and card.json."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = {**_plain(config), "format_version": MODEL_FORMAT_VERSION}
    (folder / CONFIG).write_text(json.dumps(config, indent=2), encoding="utf-8")
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise ValueError(f"weights '{key}' must be numeric arrays, not objects")
    np.savez(folder / WEIGHTS, **{k: np.asarray(v) for k, v in arrays.items()})
    if not card.created:
        card.created = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    if not card.vtea_core_version:
        card.vtea_core_version = _version()
    (folder / CARD).write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
    return folder


def read_arrays(folder) -> dict[str, np.ndarray]:
    with np.load(Path(folder) / WEIGHTS, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def predictions(model: ObjectClassifier, data: ObjectData) -> pd.DataFrame:
    """Per object: `class` (name; None where it could not be classified),
    `confidence` (top probability) and `p <class>` for each class."""
    proba = np.asarray(model.predict_proba(data), dtype=float)
    out = pd.DataFrame(index=range(len(data)))
    ok = np.isfinite(proba).all(axis=1)
    top = np.where(ok, np.nanargmax(np.where(np.isfinite(proba), proba, -1), axis=1), -1)
    out["class"] = pd.Series([model.classes[t] if t >= 0 else None for t in top], dtype=object)
    out["confidence"] = np.where(ok, proba.max(axis=1, initial=0), np.nan)
    for k, name in enumerate(model.classes):
        out[f"p {name}"] = proba[:, k]
    return out


def _import_families() -> None:
    import importlib

    for module in ("vtea_core.classification.ann",):
        try:
            importlib.import_module(module)
        except ImportError:
            pass


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("vtea-core")
    except (ImportError, ValueError):
        return ""


def _plain(value: Any) -> Any:
    """JSON-safe copy (numpy scalars and arrays to Python)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
