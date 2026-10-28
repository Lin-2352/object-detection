"""CLI for training and evaluating hcr models with reproducible reports.

Usage:
    python -m hcr.train --dataset mnist --arch cnn --epochs 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from hcr.data import load_dataset  # noqa: E402
from hcr.model import build_cnn, build_mlp  # noqa: E402

_BUILDERS = {"cnn": build_cnn, "mlp": build_mlp}


def _set_seed(seed: int) -> None:
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def _subsample(x, y, limit, seed):
    if limit is None or limit >= len(x):
        return x, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(x), size=limit, replace=False)
    return x[idx], y[idx]


def _split_train_val(x, y, val_split, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    n_val = int(round(len(x) * val_split))
    if n_val <= 0:
        return x, y, x[:0], y[:0]
    return x[n_val:], y[n_val:], x[:n_val], y[:n_val]


def _confusion_matrix(y_true, y_pred, num_classes):
    idx = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def _top_confusions(cm, classes, top_n=10):
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append((classes[i], classes[j], count))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs[:top_n]


def _plot_confusion(cm, classes, test_accuracy, out_path):
    n = len(classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0
    )

    size = max(6.0, n * 0.35)
    fig, ax = plt.subplots(figsize=(size, size), dpi=120)
    im = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
    tick_fontsize = max(4, 10 - n // 10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=tick_fontsize)
    ax.set_yticklabels(classes, fontsize=tick_fontsize)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (row-normalized) - test accuracy {test_accuracy:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    axes[0].plot(history.history.get("accuracy", []), label="train")
    axes[0].plot(history.history.get("val_accuracy", []), label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history.get("loss", []), label="train")
    axes[1].plot(history.history.get("val_loss", []), label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_without_optimizer(model, path):
    # keras.models.clone_model() preserves the compile() config (and thus a
    # freshly-built optimizer with its own slot variables), so rebuilding
    # from get_config()/from_config() instead gives us an uncompiled copy
    # with no optimizer state: the .keras file only holds architecture and
    # weights, which is what keeps it small.
    inference_model = keras.Model.from_config(model.get_config())
    inference_model.set_weights(model.get_weights())
    inference_model.save(path)


def run_training(
    x_train,
    y_train,
    x_test,
    y_test,
    classes,
    arch,
    dataset,
    epochs,
    batch_size=128,
    seed=42,
    val_split=0.1,
    out_dir="models",
    reports_dir="reports",
):
    """Train, evaluate and write reports/artifacts for one (dataset, arch) run.

    Takes arrays directly (as returned by hcr.data.load_dataset) so it can be
    exercised in tests without touching real datasets. Returns the sidecar
    metadata dict that also gets written to <out_dir>/<name>.json.
    """
    _set_seed(seed)

    out_dir = Path(out_dir)
    reports_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    name = f"{dataset.replace('-', '_')}_{arch}"
    num_classes = len(classes)

    x_tr, y_tr, x_val, y_val = _split_train_val(x_train, y_train, val_split, seed)
    validation_data = (x_val, y_val) if len(x_val) > 0 else None

    model = _BUILDERS[arch](num_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    start = time.time()
    history = model.fit(
        x_tr,
        y_tr,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.time() - start

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = _confusion_matrix(np.asarray(y_test), y_pred, num_classes)
    top_confusions = _top_confusions(cm, classes)

    model_path = out_dir / f"{name}.keras"
    _save_without_optimizer(model, model_path)

    confusion_path = reports_dir / f"{name}_confusion.png"
    _plot_confusion(cm, classes, test_accuracy, confusion_path)

    history_path = reports_dir / f"{name}_history.png"
    _plot_history(history, history_path)

    metadata = {
        "dataset": dataset,
        "arch": arch,
        "classes": list(classes),
        "input_shape": [28, 28, 1],
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_trained": len(history.history.get("loss", [])),
        "batch_size": batch_size,
        "train_samples": int(len(x_tr)),
        "val_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "params": int(model.count_params()),
        "train_seconds": train_seconds,
        "top_confusions": [[t, p, c] for t, p, c in top_confusions],
        "tensorflow_version": tf.__version__,
    }

    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(metadata, indent=2))

    print(
        f"{name}: test_accuracy={test_accuracy:.4f} test_loss={test_loss:.4f} "
        f"params={metadata['params']} epochs_trained={metadata['epochs_trained']} "
        f"train_seconds={train_seconds:.1f}"
    )
    print(f"saved model -> {model_path}")
    print(f"saved sidecar -> {json_path}")
    print(f"saved confusion matrix -> {confusion_path}")
    print(f"saved history plot -> {history_path}")

    return metadata


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an hcr model.")
    parser.add_argument(
        "--dataset", required=True, choices=["mnist", "emnist-letters", "emnist-balanced"]
    )
    parser.add_argument("--arch", required=True, choices=["cnn", "mlp"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--limit", type=int, default=None, help="subsample train/test for a quick run"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    (x_train, y_train), (x_test, y_test), classes = load_dataset(args.dataset)

    if args.limit is not None:
        x_train, y_train = _subsample(x_train, y_train, args.limit, args.seed)
        x_test, y_test = _subsample(x_test, y_test, max(1, args.limit // 5), args.seed)

    run_training(
        x_train,
        y_train,
        x_test,
        y_test,
        classes,
        arch=args.arch,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        val_split=args.val_split,
        out_dir=args.out_dir,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
