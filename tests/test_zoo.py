"""Tests for hcr.zoo parameterised model builders."""

import keras
import numpy as np
import pytest

from hcr.zoo import FAMILIES, build_model, count_params

NUM_CLASSES = 5


def _x(n=8, seed=0):
    return np.random.RandomState(seed).rand(n, 28, 28, 1).astype("float32")


def _dense_count(model):
    return sum(
        1
        for layer in model.layers
        if isinstance(layer, keras.layers.Dense) and layer.units != NUM_CLASSES
    )


def test_families_constant():
    assert FAMILIES == ("mlp", "cnn", "resnet", "cnn_bilstm")


@pytest.mark.parametrize("family", FAMILIES)
def test_output_shape_and_softmax(family):
    model = build_model(family, NUM_CLASSES, units=16, blocks=1, seed=0)
    preds = model.predict(_x(4), verbose=0)
    assert preds.shape == (4, NUM_CLASSES)
    np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("family", FAMILIES)
def test_hidden_layers_control_dense_depth(family):
    shallow = build_model(family, NUM_CLASSES, hidden_layers=0, units=16, blocks=1)
    deep = build_model(family, NUM_CLASSES, hidden_layers=3, units=16, blocks=1)
    assert _dense_count(shallow) == 0
    assert _dense_count(deep) == 3
    assert count_params(deep) > count_params(shallow)


def test_blocks_increase_resnet_params():
    small = build_model("resnet", NUM_CLASSES, blocks=1, units=16)
    big = build_model("resnet", NUM_CLASSES, blocks=3, units=16)
    assert count_params(big) > count_params(small)


@pytest.mark.parametrize("family", FAMILIES)
def test_compiled_and_trainable(family):
    model = build_model(family, NUM_CLASSES, units=16, blocks=1, seed=1)
    y = np.arange(8) % NUM_CLASSES
    hist = model.fit(_x(8), y, epochs=1, batch_size=4, verbose=0)
    assert "accuracy" in hist.history
    assert count_params(model) == model.count_params()


def test_augment_flag_adds_layers_and_is_inference_inert():
    aug = build_model("cnn", NUM_CLASSES, augment=True, seed=3)
    plain = build_model("cnn", NUM_CLASSES, augment=False, seed=3)
    names = [type(layer).__name__ for layer in aug.layers]
    assert "RandomRotation" in names
    assert "RandomTranslation" in names
    assert "RandomZoom" in names
    assert not any(type(layer).__name__.startswith("Random") for layer in plain.layers)
    plain.set_weights(aug.get_weights())
    x = _x(4)
    np.testing.assert_allclose(aug.predict(x, verbose=0), plain.predict(x, verbose=0), atol=1e-5)
    np.testing.assert_allclose(aug.predict(x, verbose=0), aug.predict(x, verbose=0), atol=1e-6)


def test_seed_gives_identical_initial_weights():
    a = build_model("mlp", NUM_CLASSES, seed=7)
    b = build_model("mlp", NUM_CLASSES, seed=7)
    for wa, wb in zip(a.get_weights(), b.get_weights(), strict=True):
        np.testing.assert_array_equal(wa, wb)


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="family"):
        build_model("transformer", NUM_CLASSES)


def test_negative_hidden_layers_raises():
    with pytest.raises(ValueError, match="hidden_layers"):
        build_model("mlp", NUM_CLASSES, hidden_layers=-1)


def test_zero_blocks_raises():
    with pytest.raises(ValueError, match="blocks"):
        build_model("resnet", NUM_CLASSES, blocks=0)


@pytest.mark.parametrize("family", FAMILIES)
def test_keras_save_load_roundtrip(family, tmp_path):
    model = build_model(family, NUM_CLASSES, units=16, blocks=1, seed=2)
    path = tmp_path / "m.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    x = _x(4)
    np.testing.assert_allclose(
        model.predict(x, verbose=0), reloaded.predict(x, verbose=0), atol=1e-5
    )


@pytest.mark.parametrize("family", ["cnn", "resnet"])
def test_batchnorm_momentum_tracks_fast_early_training(family):
    # Default momentum 0.99 left moving statistics stale in short runs, so validation
    # accuracy stayed near chance while training accuracy rose.
    model = build_model(family, num_classes=5)
    momenta = [layer.momentum for layer in model.layers if "batch_norm" in layer.name]
    assert momenta and all(m == 0.9 for m in momenta)


def test_dropout_of_one_is_rejected():
    with pytest.raises(ValueError, match="dropout"):
        build_model("mlp", num_classes=5, dropout=1.0)  # a model that can never learn
