"""Contexts: the hierarchy of levels an analysis moves between.

Pieces of cells (subcellular), cells (cellular), and neighbourhoods built
from them - and from each other, to any depth. A protocol's `context`
section defines the levels (`ContextSpec`); `build_context` builds them from
a run's results into a `ContextGraph`, whose two operators move features
along it: aggregate up (a parent from its children) and reflect down (a
child takes on its parents' characteristics, through any number of links).
See docs/CONTEXTS.md.
"""

from vtea_core.context.build import build_context
from vtea_core.context.graph import (
    AGGREGATIONS,
    LINK_KINDS,
    MEMBER_OF,
    MISSING_CATEGORY,
    PART_OF,
    ContextGraph,
    Level,
    Link,
)
from vtea_core.context.spec import (
    CELLULAR,
    CONTEXT_FORMAT_VERSION,
    DRAW_MODES,
    FILL_PATTERNS,
    NEIGHBORHOOD,
    SUBCELLULAR,
    TIERS,
    ContextSpec,
    LevelDisplay,
    LevelSpec,
)

__all__ = [
    "AGGREGATIONS",
    "CELLULAR",
    "CONTEXT_FORMAT_VERSION",
    "DRAW_MODES",
    "FILL_PATTERNS",
    "LINK_KINDS",
    "MEMBER_OF",
    "MISSING_CATEGORY",
    "NEIGHBORHOOD",
    "PART_OF",
    "SUBCELLULAR",
    "TIERS",
    "ContextGraph",
    "ContextSpec",
    "Level",
    "LevelDisplay",
    "LevelSpec",
    "Link",
    "build_context",
]
