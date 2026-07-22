"""Supervised classification of segmented objects, and mapping predictions onto label images.

Ports vtea.deeplearning's classification-oriented pieces (Generic3DCNN,
NephNet3D) from the Java codebase - not as a separate "deep learning"
module, but as its own domain parallel to clustering/reduction, since
classification (supervised, trained per-object labeling) is conceptually
distinct from both. See PORT_PLAN.md's "Why deep learning isn't a separate
module".

class_map() has no extra dependencies. Cell3DClassifier/train_classifier/
predict require the `deeplearning` extra (torch) and are only exposed here
if it's installed.
"""

from vtea_core.classification.class_map import class_map

__all__ = ["class_map"]

try:
    from vtea_core.classification.cnn import Cell3DClassifier, predict, train_classifier

    __all__ += ["Cell3DClassifier", "train_classifier", "predict"]
except ImportError:
    pass
