"""Classes and label sets: the labels an object carries, and their hierarchy.

Replaces the `gates` protocol category. Drawing a polygon is an Object
Explorer gesture (it stays there, in vtea_core.gates); what a *protocol*
needs is the rule - a range of a feature, a cluster id, a napari ROI, a
gate, or any boolean combination of them - written down so it re-runs on the
next acquisition. See steps.py for the protocol steps, expression.py for the
little language a rule is written in, and labelset.py for the sets those
labels are grouped into and the hierarchies they combine into.
"""

from vtea_core.classes.expression import (
    BOOLEAN_OPERATORS,
    ExpressionError,
    evaluate,
    parse,
    referenced_columns,
)
from vtea_core.classes.labelset import (
    COMBINE_MODES,
    CROSS,
    HIERARCHY_SEPARATOR,
    INTERSECT,
    UNION,
    UNLABELLED,
    LabelSet,
    LabelSetCollection,
    ObjectLabel,
    combine_label_sets,
    label_image,
)
from vtea_core.classes.steps import (
    class_columns,
    class_from_expression,
    class_from_range,
    class_from_values,
    combine_labels,
    label_set,
)

__all__ = [
    "BOOLEAN_OPERATORS",
    "COMBINE_MODES",
    "CROSS",
    "HIERARCHY_SEPARATOR",
    "INTERSECT",
    "UNION",
    "UNLABELLED",
    "ExpressionError",
    "LabelSet",
    "LabelSetCollection",
    "ObjectLabel",
    "class_columns",
    "class_from_expression",
    "class_from_range",
    "class_from_values",
    "combine_label_sets",
    "combine_labels",
    "evaluate",
    "label_image",
    "label_set",
    "parse",
    "referenced_columns",
]
