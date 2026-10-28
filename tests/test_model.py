"""Smoke tests for hcr.model and hcr.train.run_training on synthetic data."""

import json

import numpy as np
import pytest

from hcr.model import build_cnn, build_mlp
from hcr.train import run_training


def _synthetic_dataset(num_samples=32, num_classes=5, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(num_samples, 28, 28, 1).astype("float32")
    y = rng.randint(0, num_classes, size=num_samples).astype("int64")
    return x, y


@pytest.mark.parametrize("build_fn", [build_cnn, build_mlp])
def test_build_output_shape(build_fn):
    model = build_fn(5)
    x, _ = _synthetic_dataset(num_samples=4, num_classes=5)

    preds = model.predict(x, verbose=0)

    assert preds.shape == (4, 5)
    np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("build_fn", [build_cnn, build_mlp])
def test_fit_save_and_reload(build_fn, tmp_path):
    import keras

    x, y = _synthetic_dataset(num_samples=32, num_classes=5)
    model = build_fn(5)

    model.fit(x, y, epochs=1, batch_size=8, verbose=0)

    save_path = tmp_path / "model.keras"
    model.save(save_path)
    reloaded = keras.models.load_model(save_path)

    preds = reloaded.predict(x[:4], verbose=0)
    assert preds.shape == (4, 5)
    np.testing.assert_allclose(preds.sum(axis=1), np.ones(4), rtol=1e-4, atol=1e-4)


def test_run_training_writes_contract_artifacts(tmp_path):
    num_classes = 4
    classes = [str(i) for i in range(num_classes)]
    x_train, y_train = _synthetic_dataset(num_samples=32, num_classes=num_classes, seed=1)
    x_test, y_test = _synthetic_dataset(num_samples=8, num_classes=num_classes, seed=2)

    out_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    metadata = run_training(
        x_train,
        y_train,
        x_test,
        y_test,
        classes,
        arch="mlp",
        dataset="mnist",
        epochs=1,
        batch_size=8,
        seed=42,
        val_split=0.25,
        out_dir=out_dir,
        reports_dir=reports_dir,
    )

    name = "mnist_mlp"
    model_path = out_dir / f"{name}.keras"
    json_path = out_dir / f"{name}.json"
    confusion_path = reports_dir / f"{name}_confusion.png"
    history_path = reports_dir / f"{name}_history.png"

    assert model_path.exists()
    assert json_path.exists()
    assert confusion_path.exists()
    assert history_path.exists()

    sidecar = json.loads(json_path.read_text())
    expected_keys = {
        "dataset",
        "arch",
        "classes",
        "input_shape",
        "seed",
        "epochs_requested",
        "epochs_trained",
        "batch_size",
        "train_samples",
        "val_samples",
        "test_samples",
        "test_accuracy",
        "test_loss",
        "params",
        "train_seconds",
        "top_confusions",
        "tensorflow_version",
    }
    assert expected_keys <= sidecar.keys()
    assert sidecar["dataset"] == "mnist"
    assert sidecar["arch"] == "mlp"
    assert sidecar["classes"] == classes
    assert sidecar["test_samples"] == 8
    assert sidecar == metadata
