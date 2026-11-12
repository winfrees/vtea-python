"""Supervised classification of segmented objects, and mapping predictions onto label images.

Ports vtea.deeplearning's classification-oriented pieces (Generic3DCNN,
NephNet3D) from the Java codebase - not as a separate "deep learning"
module, but as its own domain parallel to clustering/reduction, since
classification (supervised, trained per-object labeling) is conceptually
distinct from both. See PORT_PLAN.md's "Why deep learning isn't a separate
module".

Supervised object classifiers share one interface (`models.ObjectClassifier`),
model folders with model cards (`models`), and training utilities - splits by
object or by group, class weights, metrics (`training`). The generic ANN
(`ann.ANNClassifier`, a multilayer perceptron over measurements) needs only
scikit-learn. class_map() and extract_crops() have no extra dependencies. The CNN
(Cell3DClassifier/train_classifier/predict) and the variational autoencoders
(vae.py - the Java VAE plugins) require the `deeplearning` extra (torch)
and are only exposed here if it's installed.
"""

from vtea_core.classification.ann import ANNClassifier
from vtea_core.classification.class_map import class_map
from vtea_core.classification.crops import extract_crops, object_ids
from vtea_core.classification.models import (
    FAMILIES,
    ModelCard,
    ObjectClassifier,
    ObjectData,
    list_models,
    load_model,
    model_hash,
    predictions,
    read_card,
    register_family,
    save_model,
)
from vtea_core.classification.training import class_weights, evaluate, sample_weights, split

__all__ = [
    "FAMILIES",
    "ANNClassifier",
    "ModelCard",
    "ObjectClassifier",
    "ObjectData",
    "class_map",
    "class_weights",
    "evaluate",
    "extract_crops",
    "list_models",
    "load_model",
    "model_hash",
    "object_ids",
    "predictions",
    "read_card",
    "register_family",
    "sample_weights",
    "save_model",
    "split",
]

try:
    from vtea_core.classification.cnn import Cell3DClassifier, predict, train_classifier

    __all__ += ["Cell3DClassifier", "predict", "train_classifier"]
except ImportError:
    pass

try:
    from vtea_core.classification.vae import (
        ARCHITECTURES,
        VAEConfig,
        VariationalAutoencoder,
        encode,
        fit_vae,
        load_vae,
        reconstruction_errors,
        save_vae,
        train_vae,
        vae_anomaly,
        vae_clustering,
        vae_features,
        vae_loss,
        vae_reduction,
    )

    __all__ += [
        "ARCHITECTURES",
        "VAEConfig",
        "VariationalAutoencoder",
        "encode",
        "fit_vae",
        "load_vae",
        "reconstruction_errors",
        "save_vae",
        "train_vae",
        "vae_anomaly",
        "vae_clustering",
        "vae_features",
        "vae_loss",
        "vae_reduction",
    ]
except ImportError:
    pass
