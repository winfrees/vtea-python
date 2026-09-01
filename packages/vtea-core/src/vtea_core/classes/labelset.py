"""Label sets: the n labels an object carries, and the hierarchies they build.

A segmented object is not one thing. The same nucleus is *a cell*, *an
immune cell*, *a CD3+ T cell*, *inside the tubule ROI*, and *cluster 7* -
five true statements at five levels of precision, and an analysis that
forces a choice between them is throwing away the structure the tissue
actually has. So an object carries as many labels as apply, grouped into
named sets:

    populations = LabelSet("populations", [immune, epithelial, stromal])
    fine        = LabelSet("fine", [t_cell, b_cell], parent="populations")

and sets combine into hierarchies (`combine_label_sets`), which is what
turns "immune" and "CD3+" into the population "immune > CD3+" that can be
counted, mapped back onto the image, and modelled.

A label is a boolean mask over the rows of one measurement table plus the
definition it came from. Keeping the definition on the label - not only the
mask - is what makes a label set re-computable against a re-run rather than
a set of numbers nobody can reproduce, and is what
`vtea_core.classes.expression` exists to write.

Nothing here decides *what* the labels are: that is the protocol's job (see
steps.py). This is the container, its arithmetic, and its answer to "how
many of each, and which objects".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# How two label sets are put together - see combine_label_sets.
#
# CROSS is the hierarchy builder: every combination of a parent label and a
#   child label that any object actually satisfies becomes a label of its
#   own ("immune > CD3+"), which is what a finer level of cell typing is.
# UNION keeps both sets' labels side by side, so an object simply carries
#   more labels.
# INTERSECT keeps the labels the two sets agree on by name, ANDed - two
#   independent definitions of the same population, kept only where they
#   agree.
CROSS = "cross"
UNION = "union"
INTERSECT = "intersect"
COMBINE_MODES = (CROSS, UNION, INTERSECT)

# What a combined label reads as. A separator rather than a bare space, so
# "immune > CD3+" says which half is the coarser one.
HIERARCHY_SEPARATOR = " > "

# What an object with no label in a set is called, where the set has to name
# it (a code of -1 elsewhere). "Unlabelled" and not "other": nothing has been
# said about these objects, which is different from having said they are
# something else.
UNLABELLED = "unlabelled"


@dataclass
class ObjectLabel:
    """One label: which objects have it, and what made it so.

    `mask` is per row of the table the set was built against. `definition`
    is the class expression, range or gate it came from - kept because a
    label whose rule was lost is a column of booleans nobody can reproduce
    or defend. `color` is optional and display-only; a label that came from
    a napari ROI or a gate carries that thing's colour so the same
    population looks the same everywhere.
    """

    name: str
    mask: np.ndarray
    definition: str = ""
    color: str = ""
    source: str = ""  # the step, gate or layer that produced it

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)

    @property
    def count(self) -> int:
        return int(self.mask.sum())

    def to_dict(self) -> dict[str, Any]:
        """The label without its mask: what it *is*, not what it currently
        selects. The mask follows from re-running the definition."""
        return {
            "name": self.name,
            "definition": self.definition,
            "color": self.color,
            "source": self.source,
            "count": self.count,
        }


class LabelSet:
    """A named group of labels over one population of objects.

    Ordered, because the order is the reading order of a legend and decides
    which label a `codes()` array reports for an object carrying several.
    `object_ids` records which objects the masks are about, so a set built
    on one table can be checked against another rather than silently
    lining up the wrong rows.
    """

    def __init__(
        self,
        name: str,
        labels: Iterable[ObjectLabel] | None = None,
        *,
        n_objects: int | None = None,
        object_ids: Sequence[int] | None = None,
        parent: str = "",
    ):
        self.name = name
        # The set this one refines, for a hierarchy that can be walked back
        # up - "fine types, within populations".
        self.parent = parent
        self.object_ids = None if object_ids is None else np.asarray(object_ids)
        self._labels: list[ObjectLabel] = []
        self._n_objects = n_objects
        for label in labels or []:
            self.add(label)

    # -- building ---------------------------------------------------------

    def add(self, label: ObjectLabel) -> ObjectLabel:
        if self._n_objects is None:
            self._n_objects = int(label.mask.size)
        elif label.mask.size != self._n_objects:
            raise ValueError(
                f"label {label.name!r} covers {label.mask.size} objects but this set is over "
                f"{self._n_objects}; the two were built against different tables"
            )
        if label.name in self.names:
            raise ValueError(f"this set already has a label called {label.name!r}")
        self._labels.append(label)
        return label

    def remove(self, name: str) -> None:
        self._labels = [label for label in self._labels if label.name != name]

    # -- reading ----------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return [label.name for label in self._labels]

    @property
    def n_objects(self) -> int:
        return int(self._n_objects or 0)

    def get(self, name: str) -> ObjectLabel:
        for label in self._labels:
            if label.name == name:
                return label
        raise KeyError(f"no label {name!r} in {self.name!r} (have: {', '.join(self.names)})")

    def __iter__(self):
        return iter(self._labels)

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def membership(self) -> np.ndarray:
        """(n_objects, n_labels) boolean - the whole set at once."""
        if not self._labels:
            return np.zeros((self.n_objects, 0), dtype=bool)
        return np.column_stack([label.mask for label in self._labels])

    def counts(self) -> dict[str, int]:
        return {label.name: label.count for label in self._labels}

    def labels_per_object(self) -> np.ndarray:
        """How many labels each object carries. Zero is a normal answer, and
        more than one is the point of the whole arrangement."""
        return self.membership().sum(axis=1).astype(int)

    def codes(self) -> np.ndarray:
        """One label per object as an index into `names`, -1 for none.

        The *first* matching label wins, which is why the set is ordered.
        This is what a discrete colour map and a label image are drawn from;
        `membership()` is the honest full answer for anything that can take
        it.
        """
        membership = self.membership()
        if membership.size == 0:
            return np.full(self.n_objects, -1, dtype=int)
        any_label = membership.any(axis=1)
        return np.where(any_label, membership.argmax(axis=1), -1).astype(int)

    def labels_for(self, index: int) -> list[str]:
        """Every label the object at row `index` carries."""
        return [label.name for label in self._labels if bool(label.mask[index])]

    def name_of(self, index: int) -> str:
        """The single name an object goes by - its first label, or
        "unlabelled"."""
        names = self.labels_for(index)
        return names[0] if names else UNLABELLED

    def mask_for(self, name: str) -> np.ndarray:
        return self.get(name).mask

    # -- output -----------------------------------------------------------

    def to_frame(self, prefix: str = "") -> pd.DataFrame:
        """One boolean column per label, plus the set's own code column.

        The prefix keeps two sets' labels apart in one table - `populations`
        and `fine` may both have a `CD3+` - and defaults to the set's name
        for exactly that reason.
        """
        prefix = f"{prefix or self.name}." if (prefix or self.name) else ""
        columns = {f"{prefix}{label.name}": label.mask for label in self._labels}
        columns[f"{prefix}code"] = self.codes()
        return pd.DataFrame(columns)

    def summary(self) -> str:
        if not self._labels:
            return f"{self.name}: no labels"
        counted = ", ".join(f"{name} {count}" for name, count in self.counts().items())
        unlabelled = int((self.codes() < 0).sum())
        multiple = int((self.labels_per_object() > 1).sum())
        parts = [f"{self.name}: {len(self._labels)} label(s) over {self.n_objects} objects"]
        parts.append(counted)
        if unlabelled:
            parts.append(f"{unlabelled} unlabelled")
        if multiple:
            parts.append(f"{multiple} with more than one label")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "n_objects": self.n_objects,
            "labels": [label.to_dict() for label in self._labels],
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<LabelSet {self.name!r}: {len(self._labels)} label(s)>"


def combine_label_sets(
    first: LabelSet,
    second: LabelSet,
    *,
    name: str = "",
    mode: str = CROSS,
    keep_unmatched: bool = True,
) -> LabelSet:
    """Put two label sets together - see CROSS / UNION / INTERSECT above.

    `CROSS` is the one that builds a hierarchy: the labels of `second`
    refine those of `first`, and only the combinations some object actually
    satisfies become labels, because a cell type nothing is an example of is
    not a population. With `keep_unmatched`, objects that have a coarse
    label and no fine one keep the coarse label alone rather than falling
    out of the analysis - "immune, not further typed" is a finding, and
    losing those objects would quietly change every proportion computed
    afterwards.
    """
    if mode not in COMBINE_MODES:
        raise ValueError(f"unknown combine mode {mode!r}, expected one of {COMBINE_MODES}")
    if first.n_objects != second.n_objects:
        raise ValueError(
            f"{first.name!r} is over {first.n_objects} objects and {second.name!r} over "
            f"{second.n_objects}; label sets can only be combined over the same table"
        )

    combined = LabelSet(
        name or f"{first.name}{HIERARCHY_SEPARATOR}{second.name}",
        n_objects=first.n_objects,
        object_ids=first.object_ids,
        parent=first.name,
    )
    if mode == UNION:
        for label in list(first) + list(second):
            if label.name not in combined:
                combined.add(_copy(label))
        return combined

    if mode == INTERSECT:
        for label in first:
            if label.name not in second:
                continue
            other = second.get(label.name)
            combined.add(
                ObjectLabel(
                    name=label.name,
                    mask=label.mask & other.mask,
                    definition=f"({label.definition or label.name}) AND "
                    f"({other.definition or other.name})",
                    color=label.color or other.color,
                    source=f"{first.name} INTERSECT {second.name}",
                )
            )
        return combined

    for parent_label in first:
        matched = np.zeros(first.n_objects, dtype=bool)
        for child_label in second:
            mask = parent_label.mask & child_label.mask
            if not mask.any():
                continue
            matched |= mask
            combined.add(
                ObjectLabel(
                    name=f"{parent_label.name}{HIERARCHY_SEPARATOR}{child_label.name}",
                    mask=mask,
                    definition=f"({parent_label.definition or parent_label.name}) AND "
                    f"({child_label.definition or child_label.name})",
                    color=child_label.color or parent_label.color,
                    source=f"{first.name} x {second.name}",
                )
            )
        remainder = parent_label.mask & ~matched
        if keep_unmatched and remainder.any():
            combined.add(
                ObjectLabel(
                    name=parent_label.name,
                    mask=remainder,
                    definition=parent_label.definition or parent_label.name,
                    color=parent_label.color,
                    source=f"{first.name} (not further typed)",
                )
            )
    return combined


def _copy(label: ObjectLabel) -> ObjectLabel:
    return ObjectLabel(
        name=label.name,
        mask=label.mask.copy(),
        definition=label.definition,
        color=label.color,
        source=label.source,
    )


def label_image(labels: np.ndarray, object_ids, codes, *, background: int = 0) -> np.ndarray:
    """Paint a label set back onto the segmentation.

    `codes` is one small integer per object (a LabelSet's `codes()`, or any
    per-object class), `object_ids` the object each code belongs to. The
    result is the label image remapped so every voxel of an object carries
    its class instead of its identity - which is what makes a population
    something you can see in the viewer rather than a column in a table.

    Done as a lookup table rather than a loop over objects: one pass over
    the voxels regardless of how many objects there are, which is the
    difference between interactive and unusable at ten million objects.
    """
    object_ids = np.asarray(object_ids)
    codes = np.asarray(codes)
    if object_ids.shape != codes.shape:
        raise ValueError(
            f"{object_ids.size} object ids and {codes.size} codes; they describe the same objects"
        )
    if object_ids.size == 0:
        return np.full_like(labels, background)
    highest = int(max(int(labels.max()), int(object_ids.max()))) + 1
    table = np.full(highest, background, dtype=np.int32)
    # +1 so a class code of 0 is distinguishable from background, which is
    # what a label image means by 0.
    keep = (object_ids >= 0) & (object_ids < highest) & (codes >= 0)
    table[object_ids[keep].astype(int)] = codes[keep].astype(int) + 1
    return table[np.clip(labels, 0, highest - 1)]


@dataclass
class LabelSetCollection:
    """Every label set an analysis has, in the order they were built.

    Held together because a hierarchy is a relationship *between* sets, and
    because the explorer offers them as a list to colour and gate by.
    """

    sets: dict[str, LabelSet] = field(default_factory=dict)

    def add(self, label_set: LabelSet) -> LabelSet:
        self.sets[label_set.name] = label_set
        return label_set

    def get(self, name: str) -> LabelSet:
        return self.sets[name]

    def names(self) -> list[str]:
        return list(self.sets)

    def children_of(self, name: str) -> list[LabelSet]:
        return [one for one in self.sets.values() if one.parent == name]

    def to_frame(self) -> pd.DataFrame:
        """Every set's labels as one table of boolean columns - the join
        between "which populations exist" and "which objects are in them"."""
        frames = [one.to_frame() for one in self.sets.values()]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    def __len__(self) -> int:
        return len(self.sets)

    def __iter__(self):
        return iter(self.sets.values())

    def __contains__(self, name: str) -> bool:
        return name in self.sets
