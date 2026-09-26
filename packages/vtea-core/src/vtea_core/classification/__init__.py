"""Supervised classification of segmented objects, and mapping predictions onto label images.

Ports vtea.deeplearning's classification-oriented pieces (Generic3DCNN,
NephNet3D) from the Java codebase - not as a separate "deep learning"
module, but as its own domain parallel to clustering/reduction, since
classification (supervised, trained per-object labeling) is conceptually
distinct from both. See PORT_PLAN.md's "Why deep learning isn't a separate
module".

class_map() and extract_crops() have no extra dependencies. The CNN
(Cell3DClassifier/train_classifier/predict) and the variational autoencoders
(vae.py - the Java VAE plugins) require the `deeplearning` extra (torch)
and are only exposed here if it's installed.
"""

from vtea_core.classification.class_map import class_map
from vtea_core.classification.crops import extract_crops, object_ids

__all__ = ["class_map", "extract_crops", "object_ids"]

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
