"""Config-driven experiment runner with on-disk caching, plus result utilities.

Protocol notes
--------------
* There is deliberately NO early stopping: the number of epochs is an
  experimental variable.  The best-validation epoch is recorded as
  ``best_epoch`` (1-based) and ``best_val_acc`` for analysis.
* ``val_acc`` / ``test_acc`` are those of the model after the FINAL epoch.
* Test data is evaluated exactly once, at the end.  ``test_acc`` and
  ``test_f1`` are for reporting only; model selection (``best_by``) must use
  ``val_acc`` (the default key).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from hcr.evaluate import confusion_matrix, predict_classes
from hcr.zoo import build_model, count_params

DEFAULT_OUT_DIR = "experiments/results"

DEFAULTS = {
    "family": "cnn",
    "epochs": 5,
    "hidden_layers": 2,
    "units": 128,
    "blocks": 2,
    "dropout": 0.3,
    "lr": 1e-3,
    "batch_size": 256,
    "seed": 0,
    "train_limit": None,
    "augment": True,
}


def _full_cfg(cfg: dict) -> dict:
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(
            f"unknown cfg keys {sorted(unknown)}; expected a subset of {sorted(DEFAULTS)}"
        )
    return json.loads(json.dumps({**DEFAULTS, **cfg}, sort_keys=True))


def _auto_name(cfg: dict) -> str:
    digest = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return f"{cfg['family']}_e{cfg['epochs']}_h{cfg['hidden_layers']}_u{cfg['units']}_{digest}"


def _subsample(x, y, limit, seed):
    if limit is None or limit >= len(x):
        return x, y
    idx = np.sort(np.random.RandomState(seed).choice(len(x), size=limit, replace=False))
    return x[idx], y[idx]


def _macro_f1(y_true, y_pred, n):
    cm = confusion_matrix(y_true, y_pred, n)
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    present = denom > 0
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=present)
    return float(f1[present].mean()) if present.any() else 0.0


def _inference_ms_per_1000(model, x) -> float:
    x = x[:1000]
    predict_classes(model, x[:32])  # warm-up
    start = time.perf_counter()
    predict_classes(model, x)
    return (time.perf_counter() - start) * 1000.0 * (1000.0 / len(x))


def run_experiment(
    cfg: dict,
    data,
    out_dir: str = DEFAULT_OUT_DIR,
    tag: str | None = None,
    verbose: int = 0,
    model_path: str | None = None,
) -> dict:
    """Train one configuration and write ``<out_dir>/<tag or hash>.json``.

    If that file already exists and holds the same ``cfg`` it is returned
    without retraining. With ``model_path`` the trained model is also saved there,
    and a cached result is only reused while that model file still exists.
    """
    import keras

    cfg = _full_cfg(cfg)
    out = Path(out_dir)
    path = out / f"{tag or _auto_name(cfg)}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("cfg") == cfg and (model_path is None or Path(model_path).exists()):
            return cached

    keras.utils.set_random_seed(cfg["seed"])
    num_classes = len(data.classes)
    model = build_model(
        cfg["family"],
        num_classes,
        hidden_layers=cfg["hidden_layers"],
        units=cfg["units"],
        blocks=cfg["blocks"],
        dropout=cfg["dropout"],
        lr=cfg["lr"],
        augment=cfg["augment"],
        seed=cfg["seed"],
    )
    x_tr, y_tr = _subsample(data.x_train, data.y_train, cfg["train_limit"], cfg["seed"])

    start = time.perf_counter()
    h = model.fit(
        x_tr,
        y_tr,
        validation_data=(data.x_val, data.y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        shuffle=True,
        verbose=verbose,
    ).history
    train_seconds = time.perf_counter() - start
    if model_path is not None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

    history = {
        "loss": [float(v) for v in h["loss"]],
        "acc": [float(v) for v in h["accuracy"]],
        "val_loss": [float(v) for v in h["val_loss"]],
        "val_acc": [float(v) for v in h["val_accuracy"]],
    }
    best_idx = int(np.argmax(history["val_acc"]))

    y_val_pred = predict_classes(model, data.x_val)
    val_acc = float((y_val_pred == np.asarray(data.y_val)).mean())
    y_test_pred = predict_classes(model, data.x_test)  # test is touched once, at the end
    test_acc = float((y_test_pred == np.asarray(data.y_test)).mean())

    result = {
        "cfg": cfg,
        "history": history,
        "best_epoch": best_idx + 1,
        "best_val_acc": history["val_acc"][best_idx],
        "val_acc": val_acc,
        "test_acc": test_acc,
        "test_f1": _macro_f1(data.y_test, y_test_pred, num_classes),
        "params": count_params(model),
        "train_seconds": train_seconds,
        "infer_ms_per_1000": _inference_ms_per_1000(model, data.x_test),
    }
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


def load_results(out_dir: str = DEFAULT_OUT_DIR) -> list[dict]:
    """All result dicts stored in ``out_dir`` (sorted by file name)."""
    folder = Path(out_dir)
    if not folder.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]


def results_frame(results: list[dict]):
    """Flat pandas DataFrame: cfg fields plus scalar metrics, one row per run."""
    import pandas as pd

    rows = []
    for r in results:
        row = dict(r["cfg"])
        row.update({k: v for k, v in r.items() if k not in ("cfg", "history")})
        row["final_val_loss"] = r["history"]["val_loss"][-1]
        rows.append(row)
    return pd.DataFrame(rows)


def best_by(results: list[dict], key: str = "val_acc", **filters) -> dict | None:
    """Result with the highest ``key`` among those whose cfg matches ``filters``.

    Use ``val_acc`` (default) for model selection, not ``test_acc``.
    """
    pool = [r for r in results if all(r["cfg"].get(k) == v for k, v in filters.items())]
    return max(pool, key=lambda r: r[key]) if pool else None


def epoch_sweep_configs(base: dict, epochs_list):
    """Yield copies of ``base`` with each epoch count."""
    for e in epochs_list:
        yield {**base, "epochs": e}


def depth_sweep_configs(base: dict, depths, widths):
    """Yield copies of ``base`` over the depth x width grid."""
    for d, w in itertools.product(depths, widths):
        yield {**base, "hidden_layers": d, "units": w}
