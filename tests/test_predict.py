"""Tests for hcr.predict."""

import json
from pathlib import Path

import keras
import numpy as np
import pytest

from hcr.predict import Predictor, main

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def _build_tiny_model(tmp_path: Path, name: str = "tiny_model") -> Path:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Flatten(),
            keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    model_path = tmp_path / f"{name}.keras"
    model.save(model_path)

    sidecar = {
        "classes": ["a", "b", "c"],
        "dataset": "synthetic",
        "arch": "tiny",
        "input_shape": [28, 28, 1],
        "test_accuracy": 0.0,
    }
    (tmp_path / f"{name}.json").write_text(json.dumps(sidecar), encoding="utf-8")

    return model_path


def test_predictor_returns_sorted_topk(tmp_path):
    model_path = _build_tiny_model(tmp_path)
    predictor = Predictor(model_path)

    gray = np.zeros((28, 28), dtype=np.uint8)
    gray[10:18, 10:18] = 255

    results = predictor.predict(gray, top_k=3)

    assert len(results) == 3
    probs = [prob for _, prob in results]
    assert probs == sorted(probs, reverse=True)
    for char, prob in results:
        assert char in ("a", "b", "c")
        assert 0.0 <= prob <= 1.0


def test_cli_runs_on_sample_image(tmp_path, capsys):
    model_path = _build_tiny_model(tmp_path)
    sample = SAMPLES_DIR / "digit_3.png"

    exit_code = main([str(sample), "--model", str(model_path), "--top-k", "2"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert str(sample) in captured.out
    assert "->" in captured.out


def test_missing_sidecar_gives_clear_error(tmp_path):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Flatten(),
            keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model_path = tmp_path / "no_sidecar.keras"
    model.save(model_path)

    with pytest.raises(FileNotFoundError, match="train"):
        Predictor(model_path)
