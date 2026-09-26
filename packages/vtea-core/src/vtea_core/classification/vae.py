"""Variational autoencoders over object crops: unsupervised features for cells.

Replaces the Java VTEA's 3D VAE stack (`vtea.deeplearning.models.VAE*`,
`VAELoss`, `VAETrainer`, over JavaCPP's PyTorch bindings) and its four
FeatureProcessing plugins with plain PyTorch:

| Java plugin | step | per-object output |
| --- | --- | --- |
| `VAEFeatureExtraction` | `vae_features` | the latent mean, `latent_dim` columns |
| `VAEDimensionalityReduction` | `vae_reduction` | 2 or 3 coordinates |
| `VAEClustering` | `vae_clustering` | a cluster id, k-means in latent space |
| `VAEAnomalyDetection` | `vae_anomaly` | reconstruction error, score or flag |

plus `train_vae`, the Java training dialog's job, as a step of its own.

The architecture is the Java one: each encoder block is conv (stride 2
except the last block) -> BatchNorm -> LeakyReLU -> conv -> BatchNorm ->
LeakyReLU, flattened into two linear heads for mu and log sigma^2; the
decoder mirrors it with transposed convolutions. The loss is reconstruction
MSE plus beta times the KL divergence, with the KL weight ramped up
linearly over `warmup_epochs` so the latent space is not collapsed before
the decoder has learned anything. The `small`/`medium`/`large` presets are
the Java `VAEArchitecture` ones. 2D images get the same network in 2D.

Differences from the Java, made deliberately:

- **No sigmoid on the decoder output.** The Java decoder ends in a sigmoid,
  whose output lies in [0, 1], while its inputs are z-scored crops with
  negative values - it can never reconstruct its own input, so its
  reconstruction error (and every anomaly score built on it) is dominated by
  an error it cannot reduce. The output here is linear.
- **`vae_reduction` is a real PCA** of the latent means. The Java took the
  first two or three latent dimensions ("PCA-like"; its own guide lists
  full PCA as future work), and a VAE's latent dimensions are not ordered
  by how much they explain.
- **A checkpoint is a directory** holding `model.pt`, `config.json` and
  `metadata.json`, as the Java's is, but the weights are a PyTorch state
  dict: a Java checkpoint cannot be loaded, and has to be retrained.

Requires torch (the `deeplearning` extra).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from vtea_core.classification.crops import extract_crops

VAE_FORMAT_VERSION = 1

# The Java VAEConfig.VAEArchitecture presets: (crop size, latent dim, channels).
ARCHITECTURES: dict[str, tuple[int, int, tuple[int, ...]]] = {
    "small": (32, 16, (16, 32, 64, 128)),
    "medium": (64, 32, (32, 64, 128, 256)),
    "large": (128, 64, (32, 64, 128, 256, 512)),
}


@dataclass
class VAEConfig:
    """Everything that decides the network's shape and how it is trained."""

    crop_size: int = 32
    latent_dim: int = 16
    channels: tuple[int, ...] = (16, 32, 64, 128)
    in_channels: int = 1
    spatial_dims: int = 3
    beta: float = 1.0
    warmup_epochs: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 50

    def __post_init__(self) -> None:
        self.channels = tuple(int(value) for value in self.channels)
        if self.spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {self.spatial_dims}")
        if not self.channels:
            raise ValueError("the encoder needs at least one block")
        factor = 2 ** (len(self.channels) - 1)
        if self.crop_size % factor:
            raise ValueError(
                f"crop_size {self.crop_size} must be divisible by {factor}: the encoder halves it "
                f"{len(self.channels) - 1} time(s) and the decoder has to double it back exactly"
            )

    @property
    def bottleneck(self) -> int:
        return self.crop_size // 2 ** (len(self.channels) - 1)

    @classmethod
    def preset(cls, architecture: str, **overrides: Any) -> VAEConfig:
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {architecture!r}, expected {list(ARCHITECTURES)}")
        size, latent, channels = ARCHITECTURES[architecture]
        values = {"crop_size": size, "latent_dim": latent, "channels": channels}
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["channels"] = list(self.channels)
        return data


def _layers(spatial_dims: int):
    if spatial_dims == 3:
        return nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d
    return nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d


def _block(conv, norm, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        norm(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
        conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        norm(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class VariationalAutoencoder(nn.Module):
    """The Java VariationalAutoencoder3D, in 2D or 3D."""

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        conv, deconv, norm = _layers(config.spatial_dims)
        channels = config.channels

        blocks = []
        previous = config.in_channels
        for index, width in enumerate(channels):
            stride = 2 if index < len(channels) - 1 else 1
            blocks.append(_block(conv, norm, previous, width, stride))
            previous = width
        self.encoder = nn.Sequential(*blocks)

        self._shape = (channels[-1],) + (config.bottleneck,) * config.spatial_dims
        flat = int(np.prod(self._shape))
        self.fc_mu = nn.Linear(flat, config.latent_dim)
        self.fc_logvar = nn.Linear(flat, config.latent_dim)
        self.fc_decode = nn.Linear(config.latent_dim, flat)

        ups = []
        reversed_channels = list(reversed(channels))
        for index in range(len(reversed_channels) - 1):
            ups.append(
                nn.Sequential(
                    deconv(
                        reversed_channels[index],
                        reversed_channels[index + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    norm(reversed_channels[index + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.decoder = nn.Sequential(*ups)
        # Linear output: the crops are z-scored - see the module docstring.
        self.head = conv(channels[0], config.in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x).flatten(1)
        return self.fc_mu(features), self.fc_logvar(features)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.fc_decode(z).view(-1, *self._shape)
        return self.head(self.decoder(features))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(total, reconstruction, KL), each averaged over the batch.

    Reconstruction is the squared error summed over a crop's voxels, and KL
    is summed over the latent dimensions, so the two are on the scales the
    beta-VAE literature (and the Java VAELoss) weigh against each other.
    """
    batch = target.shape[0]
    recon = torch.sum((reconstruction - target) ** 2) / batch
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch
    return recon + kl_weight * kl, recon, kl


def kl_weight_for(epoch: int, config: VAEConfig) -> float:
    """beta, ramped up linearly over the warm-up epochs."""
    if config.warmup_epochs <= 0:
        return float(config.beta)
    return float(config.beta) * min(1.0, (epoch + 1) / config.warmup_epochs)


def _device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_vae(
    crops: np.ndarray,
    config: VAEConfig | None = None,
    *,
    device: str | None = None,
    random_state: int | None = 0,
    progress=None,
) -> tuple[VariationalAutoencoder, list[dict[str, float]]]:
    """Train a VAE on (N, C, *spatial) crops. Returns (model, history).

    `progress(epoch, n_epochs, losses)` is called after each epoch.
    """
    crops = np.asarray(crops, dtype=np.float32)
    if crops.ndim not in (4, 5):
        raise ValueError(f"crops must be (N, C, H, W) or (N, C, D, H, W), got {crops.shape}")
    spatial = crops.ndim - 2
    if config is None:
        config = VAEConfig(crop_size=crops.shape[-1], spatial_dims=spatial, in_channels=crops.shape[1])
    if config.spatial_dims != spatial or config.in_channels != crops.shape[1]:
        raise ValueError(
            f"the crops are {spatial}D with {crops.shape[1]} channel(s), the model is configured "
            f"for {config.spatial_dims}D with {config.in_channels}"
        )
    if any(extent != config.crop_size for extent in crops.shape[2:]):
        raise ValueError(f"crops are {crops.shape[2:]}, the model expects {config.crop_size} a side")
    if len(crops) < 2:
        raise ValueError("a VAE needs at least two objects to train on (batch normalisation)")

    if random_state is not None:
        torch.manual_seed(random_state)
    generator = np.random.default_rng(random_state)
    target = _device(device)
    model = VariationalAutoencoder(config).to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    data = torch.as_tensor(crops)

    history: list[dict[str, float]] = []
    batch_size = max(2, int(config.batch_size))
    for epoch in range(int(config.epochs)):
        model.train()
        weight = kl_weight_for(epoch, config)
        order = generator.permutation(len(data))
        totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
        for begin in range(0, len(order), batch_size):
            index = order[begin : begin + batch_size]
            if len(index) < 2:
                continue  # BatchNorm cannot normalise a batch of one
            batch = data[index].to(target)
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(batch)
            loss, recon, kl = vae_loss(reconstruction, batch, mu, logvar, kl_weight=weight)
            loss.backward()
            optimizer.step()
            totals["loss"] += loss.item() * len(index)
            totals["reconstruction"] += recon.item() * len(index)
            totals["kl"] += kl.item() * len(index)
        record = {key: value / len(data) for key, value in totals.items()}
        record["kl_weight"] = weight
        history.append(record)
        if progress is not None:
            progress(epoch + 1, int(config.epochs), record)
    model.eval()
    return model, history


@torch.no_grad()
def encode(model: VariationalAutoencoder, crops: np.ndarray, *, batch_size: int = 64) -> np.ndarray:
    """The latent mean of every crop, (N, latent_dim) - the deterministic
    code, not a sample, as the Java feature extraction used."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for begin in range(0, len(crops), batch_size):
        batch = torch.as_tensor(np.asarray(crops[begin : begin + batch_size], dtype=np.float32))
        mu, _logvar = model.encode(batch.to(device))
        out.append(mu.cpu().numpy())
    if not out:
        return np.empty((0, model.config.latent_dim), dtype=np.float32)
    return np.concatenate(out)


@torch.no_grad()
def reconstruction_errors(
    model: VariationalAutoencoder, crops: np.ndarray, *, batch_size: int = 64
) -> np.ndarray:
    """Mean squared reconstruction error per crop, decoding the latent mean."""
    model.eval()
    device = next(model.parameters()).device
    errors = []
    for begin in range(0, len(crops), batch_size):
        batch = torch.as_tensor(np.asarray(crops[begin : begin + batch_size], dtype=np.float32))
        batch = batch.to(device)
        mu, _logvar = model.encode(batch)
        reconstruction = model.decode(mu)
        errors.append(((reconstruction - batch) ** 2).flatten(1).mean(1).cpu().numpy())
    return np.concatenate(errors) if errors else np.empty(0, dtype=np.float32)


# -- checkpoints ------------------------------------------------------------


def save_vae(
    model: VariationalAutoencoder,
    directory: str | Path,
    *,
    history: list[dict[str, float]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint directory: model.pt, config.json, metadata.json."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model.pt")
    (directory / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2))
    record = {
        "vtea_vae_version": VAE_FORMAT_VERSION,
        "torch": torch.__version__,
        "history": history or getattr(model, "history", []),
    }
    record.update(metadata or getattr(model, "metadata", {}) or {})
    (directory / "metadata.json").write_text(json.dumps(record, indent=2, default=str))
    return directory


def load_vae(directory: str | Path, *, device: str | None = "cpu") -> VariationalAutoencoder:
    directory = Path(directory)
    config_path, weights = directory / "config.json", directory / "model.pt"
    if not config_path.exists() or not weights.exists():
        raise FileNotFoundError(
            f"{directory} is not a VAE checkpoint: it needs model.pt and config.json "
            f"(a Java VTEA checkpoint cannot be read - retrain it with train_vae)"
        )
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        version = json.loads(metadata_path.read_text()).get("vtea_vae_version")
        if version is not None and version > VAE_FORMAT_VERSION:
            raise ValueError(
                f"VAE checkpoint version {version} is newer than this VTEA understands "
                f"({VAE_FORMAT_VERSION}); upgrade vtea-core to read it"
            )
    config = VAEConfig(**json.loads(config_path.read_text()))
    model = VariationalAutoencoder(config)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.to(_device(device)).eval()
    return model


# -- protocol steps ---------------------------------------------------------


def _crops_for(model_or_size, labels, intensity, channel_axis, channel, mask_outside) -> np.ndarray:
    """One crop per object of `labels`, ascending id - measurement-table order."""
    size = model_or_size.config.crop_size if isinstance(model_or_size, nn.Module) else model_or_size
    crops, _ids = extract_crops(
        labels,
        intensity,
        size=int(size),
        channel_axis=channel_axis,
        channel=channel,
        mask_outside=mask_outside,
    )
    return crops


def _resolve_model(vae_model, model_path: str) -> VariationalAutoencoder:
    if vae_model is not None:
        return vae_model
    if model_path:
        return load_vae(model_path, device=None)
    raise ValueError(
        "no VAE to use: add a train_vae step before this one, or set model_path to a "
        "checkpoint directory saved by train_vae"
    )


def train_vae(
    labels: np.ndarray,
    intensity: np.ndarray,
    channel_axis: int | None = None,
    *,
    channel: int | None = None,
    architecture: Literal["small", "medium", "large", "custom"] = "small",
    crop_size: int = 32,
    latent_dim: int = 16,
    epochs: int = 50,
    beta: float = 1.0,
    warmup_epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
    mask_outside: bool = False,
    save_to: str = "",
    random_state: int | None = 0,
    device: str = "",
) -> VariationalAutoencoder:
    """Train a VAE on a crop around every object of `labels` - the Java VAE
    training dialog, as a step.

    `architecture` picks a Java preset (small: 32-voxel crops, 16 latent
    dimensions; medium: 64, 32; large: 128, 64); `custom` uses `crop_size`
    and `latent_dim` with the small preset's channel widths. `channel` picks
    one channel of a multi-channel image, or None to train on all of them.
    `save_to`, when set, writes a checkpoint directory other protocols can
    load with `model_path`.
    """
    spatial = np.asarray(labels).ndim
    in_channels = 1
    if channel is None and channel_axis is not None and np.asarray(intensity).ndim > spatial:
        in_channels = int(np.asarray(intensity).shape[channel_axis])
    shared = {
        "in_channels": in_channels,
        "spatial_dims": spatial,
        "beta": beta,
        "warmup_epochs": warmup_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
    }
    if architecture == "custom":
        config = VAEConfig(
            crop_size=crop_size, latent_dim=latent_dim, channels=ARCHITECTURES["small"][2], **shared
        )
    else:
        config = VAEConfig.preset(architecture, **shared)
    crops = _crops_for(config.crop_size, labels, intensity, channel_axis, channel, mask_outside)
    model, history = fit_vae(crops, config, device=device or None, random_state=random_state)
    model.history = history
    model.metadata = {"n_objects": len(crops), "mask_outside": bool(mask_outside)}
    if save_to:
        save_vae(model, save_to, history=history, metadata=model.metadata)
    return model


def vae_features(
    labels: np.ndarray,
    intensity: np.ndarray,
    channel_axis: int | None = None,
    vae_model: VariationalAutoencoder | None = None,
    *,
    channel: int | None = None,
    model_path: str = "",
    mask_outside: bool = False,
) -> np.ndarray:
    """The latent mean of every object, (n_objects, latent_dim), in
    measurement-table order - Java `VAEFeatureExtraction`."""
    model = _resolve_model(vae_model, model_path)
    return encode(model, _crops_for(model, labels, intensity, channel_axis, channel, mask_outside))


def vae_reduction(
    labels: np.ndarray,
    intensity: np.ndarray,
    channel_axis: int | None = None,
    vae_model: VariationalAutoencoder | None = None,
    *,
    channel: int | None = None,
    model_path: str = "",
    n_components: int = 2,
    mask_outside: bool = False,
) -> np.ndarray:
    """The latent means projected onto their first `n_components` principal
    components - Java `VAEDimensionalityReduction`, with a real PCA."""
    from vtea_core.reduction import pca

    latent = vae_features(
        labels,
        intensity,
        channel_axis,
        vae_model,
        channel=channel,
        model_path=model_path,
        mask_outside=mask_outside,
    )
    n = max(1, min(int(n_components), latent.shape[1], len(latent)))
    return pca(latent, n)


def vae_clustering(
    labels: np.ndarray,
    intensity: np.ndarray,
    channel_axis: int | None = None,
    vae_model: VariationalAutoencoder | None = None,
    *,
    channel: int | None = None,
    model_path: str = "",
    n_clusters: int = 5,
    max_iterations: int = 100,
    random_state: int | None = 0,
    mask_outside: bool = False,
) -> np.ndarray:
    """k-means cluster ids in the latent space - Java `VAEClustering`."""
    from sklearn.cluster import KMeans

    latent = vae_features(
        labels,
        intensity,
        channel_axis,
        vae_model,
        channel=channel,
        model_path=model_path,
        mask_outside=mask_outside,
    )
    if len(latent) == 0:
        return np.empty(0, dtype=np.int64)
    k = max(1, min(int(n_clusters), len(latent)))
    return KMeans(
        n_clusters=k, max_iter=int(max_iterations), n_init="auto", random_state=random_state
    ).fit_predict(latent)


def vae_anomaly(
    labels: np.ndarray,
    intensity: np.ndarray,
    channel_axis: int | None = None,
    vae_model: VariationalAutoencoder | None = None,
    *,
    channel: int | None = None,
    model_path: str = "",
    output: Literal["error", "score", "binary"] = "error",
    threshold_sd: float = 2.0,
    mask_outside: bool = False,
) -> np.ndarray:
    """How badly the VAE reconstructs each object - Java `VAEAnomalyDetection`.

    `output`: `"error"`, the mean squared reconstruction error; `"score"`,
    that rescaled to [0, 1] across the objects; `"binary"`, 1 for an error
    more than `threshold_sd` (population) standard deviations above the mean
    and 0 otherwise. An object the model reconstructs badly is one unlike
    what it was trained on: debris, a mis-segmentation, or a rare cell.
    """
    model = _resolve_model(vae_model, model_path)
    crops = _crops_for(model, labels, intensity, channel_axis, channel, mask_outside)
    errors = reconstruction_errors(model, crops).astype(float)
    if output == "error" or len(errors) == 0:
        return errors
    if output == "score":
        low, high = errors.min(), errors.max()
        return (errors - low) / (high - low) if high > low else np.zeros_like(errors)
    if output == "binary":
        cutoff = errors.mean() + float(threshold_sd) * errors.std()
        return (errors > cutoff).astype(np.int64)
    raise ValueError(f"unknown output {output!r}, expected 'error', 'score' or 'binary'")

