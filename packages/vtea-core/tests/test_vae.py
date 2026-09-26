"""The VAE steps (the Java VAE plugins) and the crops they are fed.

Crop extraction is plain numpy and always runs; everything with a network in
it is skipped without torch (the `deeplearning` extra). The networks are
tiny and train for a few epochs - these check shapes, alignment with the
measurement table, persistence and the step wiring, not what a well-trained
VAE learns.
"""

import importlib.util

import numpy as np
import pytest
from skimage.draw import ellipsoid
from vtea_core.classification import extract_crops, object_ids
from vtea_core.measurements import extract_measurements

torch_available = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not torch_available, reason="torch (deeplearning extra) not installed")


def field_of_nuclei(n_side=4, seed=0):
    """A 3D field of n_side**2 ellipsoids, alternately bright and dim."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((20, 24 * n_side, 24 * n_side), dtype=np.int32)
    image = rng.normal(10, 2, labels.shape).astype(np.float32)
    shape = ellipsoid(3, 5, 5).astype(bool)
    object_id = 0
    for row in range(n_side):
        for column in range(n_side):
            object_id += 1
            region = (
                slice(4, 4 + shape.shape[0]),
                slice(5 + 24 * row, 5 + 24 * row + shape.shape[1]),
                slice(5 + 24 * column, 5 + 24 * column + shape.shape[2]),
            )
            labels[region][shape] = object_id
            image[region][shape] += 100 if object_id % 2 else 30
    return labels, image


class TestCrops:
    def test_one_crop_per_object_in_table_order(self):
        labels, image = field_of_nuclei()
        crops, ids = extract_crops(labels, image, size=16)
        assert crops.shape == (16, 1, 16, 16, 16)
        np.testing.assert_array_equal(ids, extract_measurements(labels, image)["object_id"])

    def test_centred_on_the_object(self):
        labels = np.zeros((1, 21, 21), dtype=np.int32)
        image = np.zeros((1, 21, 21), dtype=np.float32)
        labels[0, 10, 10] = 1
        image[0, 10, 10] = 5
        crops, _ids = extract_crops(labels[0], image[0], size=5, normalize="none")
        assert crops[0, 0, 2, 2] == 5

    def test_replicates_the_edge_where_the_crop_leaves_the_image(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        image = np.arange(100, dtype=np.float32).reshape(10, 10)
        labels[0, 0] = 1
        crops, _ids = extract_crops(labels, image, size=4, normalize="none")
        # a 4-crop centred on (0, 0) starts two pixels above and left of
        # the image: those rows and columns repeat row and column 0
        np.testing.assert_array_equal(crops[0, 0, 0], crops[0, 0, 2])
        np.testing.assert_array_equal(crops[0, 0, :, 0], crops[0, 0, :, 2])
        assert crops[0, 0, 2, 2] == image[0, 0]

    def test_each_crop_is_zscored(self):
        labels, image = field_of_nuclei()
        crops, _ids = extract_crops(labels, image, size=16)
        np.testing.assert_allclose(crops.mean(axis=(2, 3, 4)), 0, atol=1e-4)
        np.testing.assert_allclose(crops.std(axis=(2, 3, 4)), 1, atol=1e-3)

    def test_multichannel_keeps_every_channel_or_one(self):
        labels, image = field_of_nuclei()
        stacked = np.stack([image, image * 2], axis=-1)
        both, _ = extract_crops(labels, stacked, size=8, channel_axis=-1)
        one, _ = extract_crops(labels, stacked, size=8, channel_axis=-1, channel=1)
        assert both.shape[1] == 2 and one.shape[1] == 1

    def test_mask_outside_hides_the_neighbours(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[4:6, 4:6] = 1
        labels[4:6, 7:9] = 2
        image = np.ones((10, 10), dtype=np.float32)
        crops, _ids = extract_crops(labels, image, size=8, normalize="none", mask_outside=True, ids=[1])
        assert crops.sum() == 4

    def test_needs_the_channel_axis_for_a_multichannel_image(self):
        labels, image = field_of_nuclei()
        with pytest.raises(ValueError, match="channel axis"):
            extract_crops(labels, np.stack([image, image]), size=8)

    def test_object_ids_are_ascending(self):
        labels = np.array([[3, 0, 1], [0, 7, 0]])
        np.testing.assert_array_equal(object_ids(labels), [1, 3, 7])


@needs_torch
class TestNetwork:
    def test_config_refuses_a_size_the_decoder_cannot_restore(self):
        from vtea_core.classification import VAEConfig

        with pytest.raises(ValueError, match="divisible"):
            VAEConfig(crop_size=20, channels=(8, 16, 32, 64))

    def test_presets_are_the_java_architectures(self):
        from vtea_core.classification import VAEConfig

        small = VAEConfig.preset("small")
        large = VAEConfig.preset("large")
        assert (small.crop_size, small.latent_dim, small.channels) == (32, 16, (16, 32, 64, 128))
        assert (large.crop_size, large.latent_dim) == (128, 64)

    @pytest.mark.parametrize("spatial_dims", [2, 3])
    def test_reconstructs_the_input_shape(self, spatial_dims):
        import torch
        from vtea_core.classification import VAEConfig, VariationalAutoencoder

        config = VAEConfig(crop_size=16, latent_dim=4, channels=(4, 8), spatial_dims=spatial_dims, in_channels=2)
        model = VariationalAutoencoder(config)
        x = torch.zeros((3, 2) + (16,) * spatial_dims)
        reconstruction, mu, logvar = model(x)
        assert reconstruction.shape == x.shape
        assert mu.shape == logvar.shape == (3, 4)

    def test_kl_warmup_ramps_to_beta(self):
        from vtea_core.classification import VAEConfig
        from vtea_core.classification.vae import kl_weight_for

        config = VAEConfig(beta=2.0, warmup_epochs=4)
        assert [kl_weight_for(epoch, config) for epoch in range(6)] == [0.5, 1.0, 1.5, 2.0, 2.0, 2.0]

    def test_training_reduces_the_loss(self):
        from vtea_core.classification import VAEConfig, fit_vae

        labels, image = field_of_nuclei()
        crops, _ids = extract_crops(labels, image, size=16)
        config = VAEConfig(crop_size=16, latent_dim=4, channels=(4, 8), epochs=15, warmup_epochs=0, batch_size=8)
        _model, history = fit_vae(crops, config, device="cpu")
        assert history[-1]["loss"] < history[0]["loss"]


@needs_torch
class TestSteps:
    @pytest.fixture(scope="class")
    def trained(self):
        from vtea_core.classification import train_vae

        labels, image = field_of_nuclei()
        model = train_vae(
            labels, image, architecture="custom", crop_size=16, latent_dim=4, epochs=3, device="cpu"
        )
        return labels, image, model

    def test_features_line_up_with_the_measurement_table(self, trained):
        from vtea_core.classification import vae_features

        labels, image, model = trained
        latent = vae_features(labels, image, vae_model=model)
        assert latent.shape == (len(extract_measurements(labels, image)), 4)

    def test_reduction_clustering_and_anomaly(self, trained):
        from vtea_core.classification import vae_anomaly, vae_clustering, vae_reduction

        labels, image, model = trained
        assert vae_reduction(labels, image, vae_model=model, n_components=3).shape == (16, 3)
        assert set(vae_clustering(labels, image, vae_model=model, n_clusters=2)) <= {0, 1}
        errors = vae_anomaly(labels, image, vae_model=model)
        scores = vae_anomaly(labels, image, vae_model=model, output="score")
        flags = vae_anomaly(labels, image, vae_model=model, output="binary", threshold_sd=0.0)
        assert errors.shape == (16,) and np.all(errors >= 0)
        assert scores.min() == 0 and scores.max() == 1
        np.testing.assert_array_equal(flags, (errors > errors.mean()).astype(int))

    def test_a_checkpoint_round_trips(self, trained, tmp_path):
        from vtea_core.classification import save_vae, vae_features

        labels, image, model = trained
        save_vae(model, tmp_path / "vae")
        assert {p.name for p in (tmp_path / "vae").iterdir()} == {"model.pt", "config.json", "metadata.json"}
        np.testing.assert_allclose(
            vae_features(labels, image, model_path=str(tmp_path / "vae")),
            vae_features(labels, image, vae_model=model),
            atol=1e-5,
        )

    def test_says_what_is_missing_without_a_model(self, trained):
        from vtea_core.classification import vae_features

        labels, image, _model = trained
        with pytest.raises(ValueError, match="train_vae"):
            vae_features(labels, image)

    def test_a_java_checkpoint_is_not_mistaken_for_one(self, tmp_path):
        from vtea_core.classification import load_vae

        (tmp_path / "config.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="Java"):
            load_vae(tmp_path)

    def test_trains_and_extracts_in_a_protocol(self):
        from vtea_core.workflow import Pipeline, Step

        labels, image = field_of_nuclei()
        pipeline = Pipeline()
        train = Step.for_function(
            "vae",
            "train_vae",
            available={"labels", "intensity", "channel_axis"},
            params={"architecture": "custom", "crop_size": 16, "latent_dim": 4, "epochs": 2, "device": "cpu"},
        )
        pipeline.add_step(train)
        extract = Step.for_function(
            "vae",
            "vae_features",
            available=pipeline.available_keys({"labels", "intensity", "channel_axis"}),
            taken_names=pipeline.step_names(),
        )
        assert extract.input_keys["vae_model"] == "vae_model"
        pipeline.add_step(extract)
        context = pipeline.run({"labels": labels, "intensity": image, "channel_axis": None})
        assert context["vae_latent"].shape == (16, 4)
