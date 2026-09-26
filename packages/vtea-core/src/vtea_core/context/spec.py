"""Level definitions: what a protocol's `context` section says.

A protocol records *how* each level of the hierarchy is made, not the
levels themselves: re-run on the next acquisition, the same definitions
rebuild the same kinds of level from that image's objects. Three tiers,
fixed rather than open-ended:

- **subcellular** - the objects of one segmentation that are pieces of a
  cell: a nucleus, a cytoplasm, a lysosome.
- **cellular** - cells, as composed by a `build_cells` step from those
  pieces and the associations between them.
- **neighborhood** - neighbourhoods built from a lower level: from cells (or
  a subcellular level) at depth 1, from neighbourhoods at depth 2, and so on
  for as many levels as a protocol defines.

A level at a higher tier is always made from one at a lower tier or depth,
so a definition list is a stack that can be built top to bottom in order.

Display settings travel with the definition (outline colour, fill pattern)
because they are part of how a level is read. Which neighbourhoods are
drawn is *not* a display setting: by default only those selected by a gate
in the Object Explorer are, and drawing all of them has to be asked for
(`draw="all"`) - neighbourhoods overlap by design, and a screen of
overlapping hulls shows nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SUBCELLULAR = "subcellular"
CELLULAR = "cellular"
NEIGHBORHOOD = "neighborhood"
TIERS = (SUBCELLULAR, CELLULAR, NEIGHBORHOOD)

CONTEXT_FORMAT_VERSION = 1

FILL_PATTERNS = ("none", "solid", "hatch", "crosshatch", "dots")
DRAW_MODES = ("gated", "all")


@dataclass
class LevelDisplay:
    """How a level is drawn over the image. Used by neighbourhood levels,
    whose members are drawn as a hull outline with a patterned fill."""

    outline_color: str = "#ffd400"
    fill_pattern: str = "hatch"
    fill_color: str = "#ffd400"
    opacity: float = 0.6
    draw: str = "gated"

    def __post_init__(self) -> None:
        if self.fill_pattern not in FILL_PATTERNS:
            raise ValueError(f"unknown fill pattern {self.fill_pattern!r}, expected {FILL_PATTERNS}")
        if self.draw not in DRAW_MODES:
            raise ValueError(f"unknown draw mode {self.draw!r}, expected {DRAW_MODES}")
        if not 0.0 <= float(self.opacity) <= 1.0:
            raise ValueError(f"opacity must be between 0 and 1, got {self.opacity}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "outline_color": self.outline_color,
            "fill_pattern": self.fill_pattern,
            "fill_color": self.fill_color,
            "opacity": float(self.opacity),
            "draw": self.draw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LevelDisplay:
        return cls(**{key: data[key] for key in cls().to_dict() if key in data})


@dataclass
class LevelSpec:
    """One level of the hierarchy, as a protocol defines it.

    `source` is what the level is made from:

    - subcellular: the segmentation whose objects it is (a step name);
    - cellular: the `build_cells` step that composes the cells;
    - neighborhood: the level (by name) the neighbourhoods are built from.

    `build` holds a neighbourhood level's `build_neighborhoods` settings
    (method, radius, k, interval, randomize) and `measure` its
    `neighborhood_features` ones (class_column, features, aggregations).
    `classify`, when not empty, types the level (`classify_neighborhoods`
    settings: method, n_clusters, features, normalize). `reflect` names the
    columns handed down to every level below it; empty means the type, if
    the level is typed, and otherwise its composition.
    """

    name: str
    tier: str
    source: str
    build: dict[str, Any] = field(default_factory=dict)
    measure: dict[str, Any] = field(default_factory=dict)
    classify: dict[str, Any] = field(default_factory=dict)
    reflect: list[str] = field(default_factory=list)
    display: LevelDisplay = field(default_factory=LevelDisplay)

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"level '{self.name}': unknown tier {self.tier!r}, expected {TIERS}")
        if not self.name:
            raise ValueError("a level needs a name")
        if not self.source:
            raise ValueError(f"level '{self.name}' needs a source to be made from")
        if self.tier != NEIGHBORHOOD and (self.build or self.measure or self.classify):
            raise ValueError(
                f"level '{self.name}' is {self.tier}: only a neighborhood level is built, "
                f"measured and classified by the context"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "source": self.source,
            "build": dict(self.build),
            "measure": dict(self.measure),
            "classify": dict(self.classify),
            "reflect": list(self.reflect),
            "display": self.display.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LevelSpec:
        return cls(
            name=data["name"],
            tier=data["tier"],
            source=data["source"],
            build=dict(data.get("build", {})),
            measure=dict(data.get("measure", {})),
            classify=dict(data.get("classify", {})),
            reflect=list(data.get("reflect", [])),
            display=LevelDisplay.from_dict(data.get("display", {})),
        )


@dataclass
class ContextSpec:
    """The ordered level definitions of one protocol."""

    levels: list[LevelSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return len(self.levels)

    def __iter__(self):
        return iter(self.levels)

    def __bool__(self) -> bool:
        return bool(self.levels)

    def get(self, name: str) -> LevelSpec | None:
        return next((level for level in self.levels if level.name == name), None)

    def depth(self, name: str) -> int:
        """0 for a subcellular or cellular level; 1 for neighbourhoods of
        either, 2 for neighbourhoods of those, and so on."""
        level = self.get(name)
        if level is None or level.tier != NEIGHBORHOOD:
            return 0
        return 1 + self.depth(level.source)

    def rank(self, name: str) -> int:
        """Where a level sits in the stack: subcellular 0, cellular 1,
        neighbourhoods 1 + depth. What the circularity rule compares."""
        level = self.get(name)
        if level is None:
            return 0
        if level.tier == SUBCELLULAR:
            return 0
        if level.tier == CELLULAR:
            return 1
        return 1 + self.depth(name)

    def validate(self) -> None:
        """Names are unique and every neighbourhood level is built from a level
        defined before it - so the list can be built in order."""
        seen: set[str] = set()
        for level in self.levels:
            if level.name in seen:
                raise ValueError(f"two levels are called '{level.name}'")
            if level.tier == NEIGHBORHOOD and level.source not in seen:
                raise ValueError(
                    f"neighborhood level '{level.name}' is built from '{level.source}', which "
                    f"is not a level defined before it"
                )
            seen.add(level.name)

    def add(self, level: LevelSpec) -> LevelSpec:
        self.levels.append(level)
        try:
            self.validate()
        except ValueError:
            self.levels.pop()
            raise
        return level

    def to_dict(self) -> dict[str, Any]:
        return {
            "vtea_context_version": CONTEXT_FORMAT_VERSION,
            "levels": [level.to_dict() for level in self.levels],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ContextSpec:
        if not data:
            return cls()
        version = data.get("vtea_context_version", CONTEXT_FORMAT_VERSION)
        if version > CONTEXT_FORMAT_VERSION:
            raise ValueError(
                f"context version {version} is newer than this VTEA understands "
                f"({CONTEXT_FORMAT_VERSION}); upgrade vtea-core to read it"
            )
        return cls([LevelSpec.from_dict(entry) for entry in data.get("levels", [])])
