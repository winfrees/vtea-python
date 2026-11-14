"""A spatial hierarchy: segmented objects, and levels built on them, to any depth.

```
Objects (cells)                 level 0: the segmentation
├── Neighbourhoods 1            windows on cells
│   ├── Neighbourhoods 2        windows on those windows
│   └── Regions 2               regions of contiguous same-type windows
└── Regions 1                   regions of touching cells of one type
```

Levels branch: one level can be the parent of several. Every level is an
`ObjectLevel`; this class holds them, builds new ones, and moves information
between them:

- `expand(level, to)` - which objects of an ancestor level each object
  contains, through every level in between (a product of memberships).
- `reflect(level, onto, ...)` - down: each object of a descendant level
  takes on what the objects containing it are (their mean, or most common
  category), or what its *own* window is.
- `aggregate(level, of, ...)` - up: each object summarises a descendant
  level's objects it contains.
- `members_of` / `containers_of` - follow a selection down or up.

Positions are physical (the `spacing`'s unit) for every distance, and every
level's `centroid-*` stays in voxels, so any level can be drawn on the image.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from vtea_core.data.spacing import Spacing
from vtea_core.hierarchy.build import build_regions, build_windows, voxel_positions
from vtea_core.hierarchy.level import OBJECTS, REGIONS, WINDOWS, ObjectLevel, one_hot
from vtea_core.hierarchy.measure import composition
from vtea_core.neighborhoods.reflect import MISSING_CATEGORY, _most_frequent
from vtea_core.spatial.features import touching_pairs

HIERARCHY_FORMAT_VERSION = 1


class Hierarchy:
    """Levels of objects, each built on one below it.

    Start from a segmentation's object table (`Hierarchy(table, ...)`), give
    the label image (or `contacts`, pairs of touching ids) if regions are to
    follow touching masks, then add levels with `add_windows` and
    `add_regions`.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        name: str = "Objects",
        id_column: str = "object_id",
        spacing: Spacing | None = None,
        labels: np.ndarray | None = None,
        contacts: np.ndarray | None = None,
        connectivity: int = 1,
    ) -> None:
        self.spacing = spacing
        root = ObjectLevel(name=name, kind=OBJECTS, table=table.copy(), id_column=id_column)
        voxel_positions(root)  # refuse a table without positions now, not at the first build
        self.levels: dict[str, ObjectLevel] = {name: root}
        self.root = name
        if labels is not None and contacts is not None:
            raise ValueError("give the label image or the touching pairs, not both")
        if labels is not None:
            contacts = touching_pairs(np.asarray(labels), connectivity)
        self.contacts = (
            None if contacts is None else np.asarray(contacts, dtype=np.int64).reshape(-1, 2)
        )
        self._adjacency: dict[str, sparse.csr_matrix] = {}

    # -- the tree ----------------------------------------------------------

    def __getitem__(self, name: str) -> ObjectLevel:
        if name not in self.levels:
            raise KeyError(f"no level {name!r} (levels: {list(self.levels)})")
        return self.levels[name]

    def __contains__(self, name: str) -> bool:
        return name in self.levels

    def __iter__(self):
        return iter(self.levels.values())

    def __len__(self) -> int:
        return len(self.levels)

    def children(self, name: str) -> list[str]:
        return [level.name for level in self if level.parent == name]

    def descendants(self, name: str) -> list[str]:
        out = []
        for child in self.children(name):
            out += [child, *self.descendants(child)]
        return out

    def path(self, name: str) -> list[str]:
        """From the root down to `name`."""
        out = [name]
        while self[out[0]].parent is not None:
            out.insert(0, self[out[0]].parent)
        return out

    def tree(self) -> list[tuple[int, str]]:
        """(depth, name) of every level, depth-first, children in the order
        they were made - what a hierarchy map draws."""
        out = []

        def walk(name, depth):
            out.append((depth, name))
            for child in self.children(name):
                walk(child, depth + 1)

        walk(self.root, 0)
        return out

    def remove(self, name: str) -> list[str]:
        """Delete a level and every level built on it; returns their names."""
        if name == self.root:
            raise ValueError("the segmentation is the hierarchy's root and cannot be removed")
        gone = [name, *self.descendants(name)]
        for item in gone:
            del self.levels[item]
            self._adjacency.pop(item, None)
        return gone

    def _new_name(self, name: str) -> str:
        if not name or not name.strip():
            raise ValueError("a level needs a name")
        if name in self.levels:
            raise ValueError(f"there is already a level named {name!r}")
        return name

    # -- positions and adjacency --------------------------------------------

    def positions(self, name: str) -> np.ndarray:
        """Physical positions of a level's objects (voxels x the spacing)."""
        voxels = voxel_positions(self[name])
        if self.spacing is None or not self.spacing.is_known:
            return voxels
        return voxels * np.asarray(self.spacing.for_ndim(voxels.shape[1]), dtype=float)

    def unit(self) -> str:
        return self.spacing.unit if self.spacing is not None and self.spacing.is_known else "voxel"

    def adjacency(self, name: str) -> sparse.csr_matrix:
        """Which objects of a level touch, as a symmetric 0/1 matrix over rows.

        Level 0: their masks touch (from the label image or `contacts`).
        Windows: their seeds touch, at the level below. Regions: any of their
        members touch, at the level below. So touching always comes down to
        touching masks.
        """
        if name in self._adjacency:
            return self._adjacency[name]
        level = self[name]
        n = len(level)
        if level.kind == OBJECTS:
            if self.contacts is None:
                raise ValueError(
                    "touching needs the segmentation's masks: give the label image (or the touching "
                    "pairs) when making the hierarchy, or join by distance instead"
                )
            ids = pd.Series(np.arange(n), index=level.ids)
            known = np.isin(self.contacts, level.ids).all(axis=1)
            pairs = self.contacts[known]
            rows = ids.loc[pairs.ravel()].to_numpy().reshape(-1, 2) if len(pairs) else pairs
            matrix = sparse.coo_matrix((np.ones(len(rows)), (rows[:, 0], rows[:, 1])), shape=(n, n))
        else:
            below = self.adjacency(level.parent)
            n_parent = len(self[level.parent])
            link = one_hot(level.seeds, n_parent) if level.kind == WINDOWS else level.membership
            link = link.astype(np.float64)
            matrix = link @ below @ link.T
        matrix = sparse.csr_matrix(matrix)
        matrix = ((matrix + matrix.T) > 0).astype(np.int8).tolil()
        matrix.setdiag(0)
        matrix = matrix.tocsr()
        matrix.eliminate_zeros()
        self._adjacency[name] = matrix
        return matrix

    # -- building ------------------------------------------------------------

    def add_windows(
        self,
        parent: str,
        name: str,
        *,
        method: Literal["radius", "nearest"] = "radius",
        radius: float = 50.0,
        k: int = 10,
        classes: str | None = None,
        seeds=None,
        members=None,
        features: str = "",
        aggregations: str = "mean",
    ) -> ObjectLevel:
        """Windows on `parent`: one around each seed object, overlapping.
        `classes`: the parent column (phenotype, cluster, type) the
        composition is measured over. `seeds` / `members`: optional
        populations of the parent - see `vtea_core.hierarchy.build`."""
        self._new_name(name)
        base = self[parent]
        level = build_windows(
            base,
            self.positions(parent),
            name=name,
            method=method,
            radius=radius,
            k=k,
            classes=classes,
            seeds=seeds,
            members=members,
            features=features,
            aggregations=aggregations,
            unit=self.unit(),
        )
        return self._register(level, base, [classes] if classes else [])

    def add_regions(
        self,
        parent: str,
        name: str,
        *,
        types: str,
        values=None,
        touching: bool = True,
        d: float = 0.0,
        sigma: float | None = None,
        tau: float = 0.5,
        min_size: int = 1,
        classes: str | None = None,
        features: str = "",
        aggregations: str = "mean",
    ) -> ObjectLevel:
        """Regions on `parent`: contiguous objects with the same `types` value
        merged into one (see `vtea_core.hierarchy.build` for the rule).
        `touching`: objects that touch are candidates (needs the masks);
        `d`: so are objects whose centres are within d (physical units)."""
        self._new_name(name)
        base = self[parent]
        member_positions = None
        if base.membership is not None:
            member_positions = self.positions(base.parent)
        level = build_regions(
            base,
            self.positions(parent),
            name=name,
            types=types,
            values=values,
            adjacency=self.adjacency(parent) if touching else None,
            d=d,
            member_positions=member_positions,
            sigma=sigma,
            tau=tau,
            min_size=min_size,
            classes=classes,
            features=features,
            aggregations=aggregations,
            unit=self.unit(),
        )
        return self._register(level, base, [types] + ([classes] if classes else []))

    def _register(self, level: ObjectLevel, parent: ObjectLevel, columns: list[str]) -> ObjectLevel:
        level.provenance["parent_columns"] = list(columns)
        level.provenance["parent_fingerprint"] = parent.fingerprint(columns)
        self.levels[level.name] = level
        return level

    def is_stale(self, name: str) -> bool:
        """Whether the level's parent (its objects, members, or the columns it
        was built from) changed since it was built - or any ancestor did."""
        level = self[name]
        if level.parent is None:
            return False
        parent = self[level.parent]
        columns = level.provenance.get("parent_columns", [])
        if parent.fingerprint(columns) != level.provenance.get("parent_fingerprint"):
            return True
        return self.is_stale(parent.name)

    # -- describing ------------------------------------------------------------

    def centers(self, name: str) -> np.ndarray:
        """Physical centre of each object: a window's seed, a region's members'
        mean, an object's own position."""
        level = self[name]
        if level.kind == OBJECTS:
            return self.positions(name)
        below = self.positions(level.parent)
        if level.kind == WINDOWS:
            return below[level.seeds]
        m = sparse.csr_matrix(level.membership, dtype=np.float64)
        counts = np.maximum(np.asarray(m.sum(axis=1)).ravel(), 1)
        return (m @ below) / counts[:, None]

    def composition(self, name: str, of: str, classes: str) -> pd.DataFrame:
        """Composition of each object of `name` over the classes of any level
        below it (`of`), through every level in between - a level-2 window
        described by the cells it reaches, not only by its parts."""
        table = self[of].table
        if classes not in table.columns:
            raise ValueError(f"level {of!r} has no column {classes!r}")
        return composition(self.expand(name, of), table[classes])

    def describe(
        self,
        name: str,
        families=("diversity", "arrangement"),
        *,
        classes: str | None = None,
        sigma: float | None = None,
        radii=(),
        types: str | None = None,
        add: bool = True,
    ) -> pd.DataFrame:
        """Feature families (see `vtea_core.hierarchy.features`) for a level,
        measured from its members in the parent. `classes`: the parent column
        for the class-based families (default: the one the level was built
        on). `sigma` for `weighted` (default: half the window radius, or one
        member spacing). `radii` for `multiscale`. `types`: this level's own
        type column, for `interface`. With `add`, the columns are also added
        to the level's table (replacing any of the same name)."""
        from vtea_core.hierarchy import features as F

        families = [families] if isinstance(families, str) else list(families)
        unknown = set(families) - set(F.FAMILIES)
        if unknown:
            raise ValueError(f"unknown feature famil(ies) {sorted(unknown)}, expected {list(F.FAMILIES)}")
        level = self[name]
        blocks = []
        if "interface" in families:
            column = types or ""
            if column not in level.table.columns:
                raise ValueError(f"'interface' needs this level's type column (types=...), not {column!r}")
            blocks.append(F.interface_distance(self.positions(name), level.table[column]))
        member_families = [f for f in families if f != "interface"]
        if member_families:
            if level.membership is None:
                raise ValueError(f"{name!r} is the segmentation: its objects have no members to describe")
            parent = self[level.parent]
            positions, centers = self.positions(level.parent), self.centers(name)
            classes = classes or level.provenance.get("classes") or level.provenance.get("types")
            needs_classes = {"weighted", "multiscale", "diversity", "contacts", "asymmetry"}
            if needs_classes & set(member_families):
                if not classes or classes not in parent.table.columns:
                    raise ValueError(f"these families need a class column of {parent.name!r} (classes=...)")
                labels = parent.table[classes]
            radius = level.provenance.get("radius") if level.provenance.get("method") == "radius" else None
            for family in member_families:
                if family == "weighted":
                    if sigma is None:
                        sigma = radius / 2 if radius else _spacing(positions)
                    blocks.append(F.weighted_composition(level.membership, positions, centers, labels, sigma))
                elif family == "multiscale":
                    blocks.append(F.multiscale_composition(positions, centers, labels, radii))
                elif family == "diversity":
                    blocks.append(F.diversity(level.membership, labels))
                elif family == "arrangement":
                    block = F.arrangement(level.membership, positions, centers, radius=radius)
                    blocks.append(block)
                elif family == "contacts":
                    blocks.append(F.internal_contacts(level.membership, self.adjacency(parent.name), labels))
                elif family == "asymmetry":
                    blocks.append(F.asymmetry(level.membership, positions, centers, labels))
        out = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=range(len(level)))
        out = out.loc[:, ~out.columns.duplicated(keep="last")]
        if add:
            table = level.table.drop(columns=[c for c in out.columns if c in level.table.columns])
            level.table = pd.concat([table, out.set_index(table.index)], axis=1)
        return out

    # -- moving between levels ---------------------------------------------

    def expand(self, name: str, to: str | None = None) -> sparse.csr_matrix:
        """(objects of `name`) x (objects of `to`) 0/1: which objects of the
        ancestor level `to` (the root by default) each object contains,
        through every level in between."""
        to = self.root if to is None else to
        chain = self.path(name)
        if to not in chain:
            raise ValueError(f"{to!r} is not below {name!r} in the hierarchy")
        if to == name:
            return sparse.identity(len(self[name]), dtype=np.int8, format="csr")
        result = None
        for level_name in reversed(chain[chain.index(to) + 1 :]):
            m = self[level_name].membership.astype(np.int64)
            result = m if result is None else result @ m
        result = sparse.csr_matrix(result)
        result.data = np.ones_like(result.data, dtype=np.int64)
        return result.astype(np.int8)

    def own(self, name: str, onto: str) -> np.ndarray:
        """For each object of the descendant level `onto`, the row of its own
        window at `name` (-1 if it seeds none): the window centred on it, or
        on its own window at the level in between. Every level from `onto`
        up to `name` must be windows."""
        chain = self.path(name)
        if onto not in chain or onto == name:
            raise ValueError(f"{onto!r} is not below {name!r} in the hierarchy")
        steps = chain[chain.index(onto) + 1 :]
        mapping = np.arange(len(self[onto]))  # row at the current level, per onto object
        for level_name in steps:
            level = self[level_name]
            if level.kind != WINDOWS:
                raise ValueError(f"{level_name!r} is {level.kind}: only windows have an own one")
            seeded = np.full(len(self[level.parent]), -1, dtype=np.int64)
            seeded[level.seeds] = np.arange(len(level))
            mapping = np.where(mapping >= 0, seeded[np.maximum(mapping, 0)], -1)
        return mapping

    def reflect(
        self,
        name: str,
        onto: str,
        columns,
        *,
        how: Literal["member", "own"] = "member",
        categorical=(),
    ) -> pd.DataFrame:
        """Columns of level `name` handed down to the descendant level `onto`,
        one row per `onto` object in its table's order.

        `how="member"`: over every object of `name` containing it - the mean
        for a quantity, the most frequent value for a category (the columns
        in `categorical`; smallest of a tie). `how="own"`: its own window's
        value (see `own`). Also gives `n_containers`, how many objects of
        `name` contain each one. An object in none gets NaN, or -1 for a
        category."""
        columns = [columns] if isinstance(columns, str) else list(columns)
        source = self[name].table
        missing = [c for c in columns if c not in source.columns]
        if missing:
            raise ValueError(f"level {name!r} has no column(s) {missing}")
        categorical = {categorical} if isinstance(categorical, str) else set(categorical)
        expanded = self.expand(name, onto)
        n_onto = expanded.shape[1]
        out = pd.DataFrame(
            {"n_containers": np.asarray(expanded.sum(axis=0)).ravel().astype(np.int64)}
        )
        if how == "own":
            rows = self.own(name, onto)
            for column in columns:
                values = source[column].to_numpy()
                picked = pd.Series(values[np.maximum(rows, 0)]).where(rows >= 0)
                out[column] = _category(picked) if column in categorical else picked.to_numpy()
            return out
        if how != "member":
            raise ValueError(f"unknown way to reflect {how!r}, expected 'member' or 'own'")
        coo = expanded.tocoo()
        pairs = source[columns].iloc[coo.row].reset_index(drop=True)
        pairs["_onto"] = coo.col
        grouped = pairs.groupby("_onto")
        for column in columns:
            if column in categorical:
                voted = grouped[column].agg(_most_frequent).reindex(range(n_onto))
                out[column] = _category(voted)
            else:
                out[column] = grouped[column].mean().reindex(range(n_onto)).to_numpy()
        return out

    def aggregate(self, name: str, of: str, columns, aggregations: str = "mean") -> pd.DataFrame:
        """Columns of the descendant level `of` summarised over every object of
        level `name` it is contained in, as `<aggregation>_<column>`."""
        from vtea_core.hierarchy.measure import aggregate as _aggregate

        columns = [columns] if isinstance(columns, str) else list(columns)
        table = self[of].table
        missing = [c for c in columns if c not in table.columns]
        if missing:
            raise ValueError(f"level {of!r} has no column(s) {missing}")
        return _aggregate(self.expand(name, of), table[columns], aggregations)

    def members_of(self, name: str, selection, to: str) -> np.ndarray:
        """Rows of the descendant level `to` contained in the selected objects
        of `name` (a boolean mask over `to`)."""
        chosen = self[name].selection(selection).astype(np.int64)
        return np.asarray(self.expand(name, to).T @ chosen).ravel() > 0

    def containers_of(self, name: str, selection, up: str) -> np.ndarray:
        """Rows of the level `up`, built on `name` directly or through levels
        in between, that contain any selected object of `name` (a boolean
        mask over `up`)."""
        chosen = self[name].selection(selection).astype(np.int64)
        return np.asarray(self.expand(up, name) @ chosen).ravel() > 0

    # -- saving ---------------------------------------------------------------

    def save(self, folder: str | Path) -> Path:
        """One sub-folder per level (`table.csv`, `membership.npz`,
        `seeds.npy`) and `hierarchy.json` listing them in build order."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        entries = []
        for index, level in enumerate(self):
            sub = f"level_{index:03d}"
            (folder / sub).mkdir(exist_ok=True)
            level.table.to_csv(folder / sub / "table.csv", index=False)
            if level.membership is not None:
                sparse.save_npz(folder / sub / "membership.npz", level.membership)
            if level.seeds is not None:
                np.save(folder / sub / "seeds.npy", level.seeds)
            entries.append(
                {
                    "name": level.name,
                    "kind": level.kind,
                    "folder": sub,
                    "id_column": level.id_column,
                    "parent": level.parent,
                    "depth": level.depth,
                    "provenance": _plain(level.provenance),
                }
            )
        if self.contacts is not None:
            np.save(folder / "contacts.npy", self.contacts)
        manifest = {
            "vtea_hierarchy_version": HIERARCHY_FORMAT_VERSION,
            "root": self.root,
            "spacing": None if self.spacing is None else self.spacing.to_dict(),
            "levels": entries,
        }
        (folder / "hierarchy.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> Hierarchy:
        folder = Path(folder)
        manifest = json.loads((folder / "hierarchy.json").read_text(encoding="utf-8"))
        version = manifest.get("vtea_hierarchy_version")
        if version is not None and version > HIERARCHY_FORMAT_VERSION:
            raise ValueError(
                f"hierarchy version {version} is newer than this VTEA understands "
                f"({HIERARCHY_FORMAT_VERSION}); upgrade vtea-core to read it"
            )
        entries = manifest["levels"]
        root = entries[0]
        contacts = folder / "contacts.npy"
        hierarchy = cls(
            pd.read_csv(folder / root["folder"] / "table.csv"),
            name=root["name"],
            id_column=root["id_column"],
            spacing=None
            if manifest.get("spacing") is None
            else Spacing.from_dict(manifest["spacing"]),
            contacts=np.load(contacts) if contacts.exists() else None,
        )
        for entry in entries[1:]:
            sub = folder / entry["folder"]
            seeds = sub / "seeds.npy"
            hierarchy.levels[entry["name"]] = ObjectLevel(
                name=entry["name"],
                kind=entry["kind"],
                table=pd.read_csv(sub / "table.csv"),
                id_column=entry["id_column"],
                parent=entry["parent"],
                depth=entry["depth"],
                membership=sparse.load_npz(sub / "membership.npz"),
                seeds=np.load(seeds) if seeds.exists() else None,
                provenance=entry.get("provenance", {}),
            )
        return hierarchy


def _spacing(positions: np.ndarray) -> float:
    """The median nearest-neighbour distance (1.0 for fewer than two)."""
    if len(positions) < 2:
        return 1.0
    gaps, _ = cKDTree(positions).query(positions, k=2)
    return float(np.median(gaps[:, 1])) or 1.0


def _category(values: pd.Series) -> np.ndarray:
    """Integer categories with -1 for none; named ones (phenotypes) with None."""
    values = pd.Series(values)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == values.notna().sum():
        return numeric.fillna(MISSING_CATEGORY).astype(np.int64).to_numpy()
    return values.astype(object).where(values.notna(), None).to_numpy()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = ["HIERARCHY_FORMAT_VERSION", "OBJECTS", "REGIONS", "WINDOWS", "Hierarchy"]
