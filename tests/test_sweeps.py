"""Tests for hcr.sweeps experiment runner (tiny synthetic data)."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hcr import sweeps
from hcr.sweeps import (
    best_by,
    depth_sweep_configs,
    epoch_sweep_configs,
    load_results,
    results_frame,
    run_experiment,
)

K = 5


def _data():
    rng = np.random.RandomState(0)

    def part(n):
        return rng.rand(n, 28, 28, 1).astype("float32"), (np.arange(n) % K).astype("int64")

    xtr, ytr = part(64)
    xva, yva = part(20)
    xte, yte = part(20)
    return SimpleNamespace(
        x_train=xtr,
        y_train=ytr,
        x_val=xva,
        y_val=yva,
        x_test=xte,
        y_test=yte,
        classes=[str(i) for i in range(K)],
    )


CFG = dict(
    family="mlp",
    epochs=2,
    hidden_layers=1,
    units=16,
    blocks=1,
    dropout=0.1,
    lr=1e-3,
    batch_size=16,
    seed=0,
    augment=False,
)


def test_run_experiment_result_and_file(tmp_path):
    res = run_experiment(CFG, _data(), out_dir=str(tmp_path), tag="t1")
    assert (tmp_path / "t1.json").exists()
    assert json.loads((tmp_path / "t1.json").read_text())["cfg"] == res["cfg"]
    for key in (
        "val_acc",
        "test_acc",
        "test_f1",
        "params",
        "train_seconds",
        "infer_ms_per_1000",
        "best_epoch",
        "history",
        "cfg",
    ):
        assert key in res
    assert res["cfg"]["family"] == "mlp"
    hist = res["history"]
    assert len(hist["loss"]) == 2
    for k in ("acc", "val_loss", "val_acc"):
        assert len(hist[k]) == 2
    assert res["best_epoch"] in (1, 2)
    assert 0.0 <= res["test_acc"] <= 1.0
    assert 0.0 <= res["test_f1"] <= 1.0


def test_cache_skips_recomputation(tmp_path, monkeypatch):
    data = _data()
    first = run_experiment(CFG, data, out_dir=str(tmp_path), tag="c")

    def boom(*a, **k):
        raise AssertionError("should not retrain")

    monkeypatch.setattr(sweeps, "build_model", boom)
    again = run_experiment(CFG, data, out_dir=str(tmp_path), tag="c")
    assert again == first


def test_cache_invalidated_when_cfg_differs(tmp_path):
    data = _data()
    run_experiment(CFG, data, out_dir=str(tmp_path), tag="c")
    other = run_experiment({**CFG, "epochs": 1}, data, out_dir=str(tmp_path), tag="c")
    assert len(other["history"]["loss"]) == 1


def test_auto_name_deterministic(tmp_path):
    data = _data()
    run_experiment({**CFG, "epochs": 1}, data, out_dir=str(tmp_path))
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    run_experiment({**CFG, "epochs": 1}, data, out_dir=str(tmp_path))
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_train_limit_subsamples(tmp_path):
    seen = {}
    real = sweeps.build_model

    def spy(*a, **k):
        model = real(*a, **k)
        orig_fit = model.fit

        def fit(x, y, **kw):
            seen["n"] = len(x)
            return orig_fit(x, y, **kw)

        model.fit = fit
        return model

    sweeps.build_model = spy
    try:
        run_experiment({**CFG, "epochs": 1, "train_limit": 32}, _data(), out_dir=str(tmp_path))
    finally:
        sweeps.build_model = real
    assert seen["n"] == 32


def test_load_results_and_frame(tmp_path):
    data = _data()
    run_experiment({**CFG, "epochs": 1}, data, out_dir=str(tmp_path), tag="a")
    run_experiment({**CFG, "epochs": 1, "family": "cnn", "units": 8}, data, str(tmp_path), "b")
    results = load_results(str(tmp_path))
    assert len(results) == 2
    df = results_frame(results)
    assert isinstance(df, pd.DataFrame)
    for col in (
        "family",
        "epochs",
        "hidden_layers",
        "units",
        "blocks",
        "val_acc",
        "test_acc",
        "test_f1",
        "params",
        "train_seconds",
    ):
        assert col in df.columns
    assert set(df["family"]) == {"mlp", "cnn"}
    assert load_results(str(tmp_path / "missing")) == []


def test_best_by_uses_filters():
    results = [
        {"cfg": {"family": "mlp"}, "val_acc": 0.5},
        {"cfg": {"family": "mlp"}, "val_acc": 0.7},
        {"cfg": {"family": "cnn"}, "val_acc": 0.9},
    ]
    assert best_by(results)["val_acc"] == 0.9
    assert best_by(results, family="mlp")["val_acc"] == 0.7
    assert best_by(results, key="val_acc", family="nope") is None


def test_sweep_helpers():
    base = {"family": "mlp", "epochs": 1, "hidden_layers": 2, "units": 8}
    eps = list(epoch_sweep_configs(base, [1, 3, 5]))
    assert [c["epochs"] for c in eps] == [1, 3, 5]
    assert base["epochs"] == 1
    assert all(c["units"] == 8 for c in eps)
    grid = list(depth_sweep_configs(base, [0, 1, 2], [16, 32]))
    assert len(grid) == 6
    assert {(c["hidden_layers"], c["units"]) for c in grid} == {
        (d, w) for d in (0, 1, 2) for w in (16, 32)
    }


def test_unknown_family_propagates(tmp_path):
    with pytest.raises(ValueError):
        run_experiment({**CFG, "family": "nope"}, _data(), out_dir=str(tmp_path), tag="x")


def test_model_path_saves_loadable_model(tmp_path):
    import keras

    data = _data()
    path = tmp_path / "m.keras"
    run_experiment({**CFG, "family": "mlp"}, data, out_dir=str(tmp_path), model_path=str(path))
    assert path.exists()
    assert keras.models.load_model(path).output_shape[-1] == len(data.classes)


def test_unknown_cfg_key_is_rejected(tmp_path):
    # A typo such as "unit" used to be ignored silently while still changing the cache name.
    with pytest.raises(ValueError, match="unit"):
        run_experiment({**CFG, "unit": 64}, _data(), out_dir=str(tmp_path))
