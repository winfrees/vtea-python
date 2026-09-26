"""Neighbourhoods as objects: a second level of objects, made of the first.

A segmented nucleus is an object; the dozen cells around it are a
neighbourhood, and a neighbourhood is an object too - it has members, a
place, and measurements of its own (how much of it is each cell type), and
it can be clustered, gated and plotted exactly as a cell can. That is the
first half of the model.

The second half is that the relationship runs both ways. A cell *belongs
to* neighbourhoods, and so it carries what they are: the composition around
it, and - once neighbourhoods are clustered into types - which kind of
neighbourhood it lives in. See `vtea_core.neighborhoods.reflect`.

Why this is not an `AssociationSet`: an association gives a child one
parent. Neighbourhoods overlap by design - a cell is a member of its own
neighbourhood and of the neighbourhood of every cell near it - so
membership is many-to-many, and it is held as a membership table rather
than forced into a shape that would have to pick one.

Ports `vteaobjects.MicroNeighborhoodObject` (a neighbourhood as an object
made of objects) from the Java VTEA; the Java kept no record of which
objects' neighbourhoods a cell sat in, so nothing could flow back to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NEIGHBORHOOD_FORMAT_VERSION = 1

# How a neighbourhood was drawn. The first two are centred on an object -
# its "own" neighbourhood - the third on a point of a regular grid.
NEAREST = "nearest"  # Java "Nearest-k": the object and its k nearest
RADIUS = "radius"  # Java "Spatial by cell": the object and everything within r
GRID = "grid"  # Java "Spatial by point": everything within r of a grid point
METHODS = (NEAREST, RADIUS, GRID)
CENTRED_METHODS = frozenset({NEAREST, RADIUS})

NEIGHBORHOOD_ID = "neighborhood_id"


@dataclass(frozen=True)
class Neighborhood:
    """One neighbourhood: which objects are in it, and where it is.

    `center` is in the same (voxel) coordinates as the table's `centroid-*`
    columns, so a neighbourhood can be drawn over the image it came from.
    `seed` is the object a centred neighbourhood was drawn around, and None
    for a grid point.
    """

    neighborhood_id: int
    members: tuple[int, ...]
    center: tuple[float, ...]
    seed: int | None = None
    # How far the members sit from the centre on average, in the unit the
    # neighbourhood was built in - measured at build time, from the
    # positions actually used, so it stays right for a randomised set.
    mean_distance: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.neighborhood_id),
            "members": [int(member) for member in self.members],
            "center": [float(value) for value in self.center],
            "seed": None if self.seed is None else int(self.seed),
            "mean_distance": None if np.isnan(self.mean_distance) else float(self.mean_distance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Neighborhood:
        seed = data.get("seed")
        return cls(
            neighborhood_id=int(data["id"]),
            members=tuple(int(member) for member in data.get("members", [])),
            center=tuple(float(value) for value in data.get("center", [])),
            seed=None if seed is None else int(seed),
            mean_distance=float("nan")
            if data.get("mean_distance") is None
            else float(data["mean_distance"]),
        )


@dataclass
class NeighborhoodSet:
    """Every neighbourhood of one table, and how they were drawn.

    `source` names what the members are rows of - a segmentation or a cell
    table - and `id_column` the column their ids are read from, so a
    neighbourhood of cells and a neighbourhood of nuclei are not confused.
    `params` records the construction exactly (method, k, radius, whether
    positions were randomised), which is what makes a composition number
    interpretable later: "30% CD3+ within 50 um" is a different fact from
    "30% of the 10 nearest".

    For a centred method, a neighbourhood's id *is* its seed object's id.
    That makes the neighbourhood table and the object table joinable on id,
    and lets a gate drawn on neighbourhoods light up the cells at their
    centres.
    """

    neighborhoods: list[Neighborhood] = field(default_factory=list)
    method: str = RADIUS
    source: str = ""
    id_column: str = "object_id"
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown neighbourhood method {self.method!r}, expected one of {METHODS}")
        self._by_id = {n.neighborhood_id: n for n in self.neighborhoods}
        if len(self._by_id) != len(self.neighborhoods):
            raise ValueError("neighbourhood ids must be unique")

    @property
    def centred(self) -> bool:
        """Whether every neighbourhood belongs to one object - its seed."""
        return self.method in CENTRED_METHODS

    @property
    def ndim(self) -> int:
        return len(self.neighborhoods[0].center) if self.neighborhoods else 0

    def __len__(self) -> int:
        return len(self.neighborhoods)

    def __iter__(self):
        return iter(self.neighborhoods)

    def __contains__(self, neighborhood_id: int) -> bool:
        return int(neighborhood_id) in self._by_id

    def get(self, neighborhood_id: int) -> Neighborhood | None:
        return self._by_id.get(int(neighborhood_id))

    def ids(self) -> np.ndarray:
        return np.array([n.neighborhood_id for n in self.neighborhoods], dtype=np.int64)

    def centers(self) -> np.ndarray:
        """(n_neighbourhoods, ndim) centres, in voxel coordinates."""
        if not self.neighborhoods:
            return np.empty((0, 0))
        return np.array([n.center for n in self.neighborhoods], dtype=float)

    def sizes(self) -> np.ndarray:
        return np.array([len(n.members) for n in self.neighborhoods], dtype=np.int64)

    def membership(self) -> pd.DataFrame:
        """The many-to-many relationship as a table: one row per (neighbourhood,
        member) pair, with `is_seed` marking the object a neighbourhood is
        centred on. Everything that joins neighbourhoods to objects goes
        through this."""
        rows_n: list[int] = []
        rows_o: list[int] = []
        seeds: list[bool] = []
        for neighborhood in self.neighborhoods:
            for member in neighborhood.members:
                rows_n.append(neighborhood.neighborhood_id)
                rows_o.append(member)
                seeds.append(member == neighborhood.seed)
        return pd.DataFrame(
            {
                NEIGHBORHOOD_ID: np.asarray(rows_n, dtype=np.int64),
                self.id_column: np.asarray(rows_o, dtype=np.int64),
                "is_seed": np.asarray(seeds, dtype=bool),
            }
        )

    def neighborhoods_of(self, object_id: int) -> list[int]:
        """Every neighbourhood `object_id` is a member of."""
        object_id = int(object_id)
        return [n.neighborhood_id for n in self.neighborhoods if object_id in n.members]

    def own_neighborhood(self, object_id: int) -> int | None:
        """The neighbourhood centred on `object_id`, for a centred method."""
        if not self.centred:
            return None
        return int(object_id) if int(object_id) in self._by_id else None

    def summary(self) -> str:
        if not self.neighborhoods:
            return "no neighbourhoods"
        sizes = self.sizes()
        text = (
            f"{len(self)} neighbourhoods ({self.method}), "
            f"{sizes.min()}-{sizes.max()} objects each, median {int(np.median(sizes))}"
        )
        if self.params.get("randomize"):
            text += "; positions randomised (null model)"
        return text

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "vtea_neighborhood_version": NEIGHBORHOOD_FORMAT_VERSION,
            "method": self.method,
            "source": self.source,
            "id_column": self.id_column,
            "params": _plain(self.params),
            "neighborhoods": [n.to_dict() for n in self.neighborhoods],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NeighborhoodSet:
        version = data.get("vtea_neighborhood_version")
        if version is not None and version > NEIGHBORHOOD_FORMAT_VERSION:
            raise ValueError(
                f"neighbourhood file version {version} is newer than this VTEA understands "
                f"({NEIGHBORHOOD_FORMAT_VERSION}); upgrade vtea-core to read it"
            )
        return cls(
            neighborhoods=[Neighborhood.from_dict(entry) for entry in data.get("neighborhoods", [])],
            method=data.get("method", RADIUS),
            source=data.get("source", ""),
            id_column=data.get("id_column", "object_id"),
            params=dict(data.get("params", {})),
        )


def _plain(value: Any) -> Any:
    """JSON-safe copy of a parameter record (numpy scalars, tuples)."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_neighborhoods(neighborhoods: NeighborhoodSet, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(neighborhoods.to_dict(), indent=2), encoding="utf-8")
    return path


def load_neighborhoods(path: str | Path) -> NeighborhoodSet:
    return NeighborhoodSet.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def centroid_columns(frame: pd.DataFrame) -> list[str]:
    """The `centroid-0`, `centroid-1`, ... columns of a table, in axis order."""
    columns = [column for column in frame.columns if str(column).startswith("centroid-")]
    return sorted(columns, key=lambda name: int(str(name).rsplit("-", 1)[1]))

