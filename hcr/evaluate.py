"""Pure-numpy evaluation helpers: predictions, confusion analysis, per-class F1."""

from __future__ import annotations

import numpy as np


def predict_classes(model, x, batch_size: int = 512) -> np.ndarray:
    """Argmax class indices (int64) for ``x``."""
    probs = model.predict(x, batch_size=batch_size, verbose=0)
    return np.argmax(probs, axis=1).astype(np.int64)


def accuracy_by_group(y_true, y_pred, groups: dict[str, list[int]]) -> dict[str, float]:
    """Accuracy over samples whose true label is in each group (NaN if none)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, float] = {}
    for name, members in groups.items():
        mask = np.isin(y_true, list(members))
        out[name] = float((y_true[mask] == y_pred[mask]).mean()) if mask.any() else float("nan")
    return out


def case_insensitive_accuracy(y_true, y_pred, classes, ambiguous_groups: list[list[str]]) -> float:
    """Accuracy where confusions inside an ambiguous group count as correct."""
    index = {c: i for i, c in enumerate(classes)}
    group_of = np.arange(len(classes))
    for gid, group in enumerate(ambiguous_groups):
        for ch in group:
            if ch in index:
                group_of[index[ch]] = len(classes) + gid
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((group_of[y_true] == group_of[y_pred]).mean())


def confusion_matrix(y_true, y_pred, n: int) -> np.ndarray:
    """Counts matrix with rows = true class, columns = predicted class."""
    t, p = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    if np.any((t < 0) | (t >= n) | (p < 0) | (p >= n)):
        raise ValueError(f"labels must be in [0, {n})")
    idx = t * n + p
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def top_confusions(cm, classes, k: int = 15) -> list[tuple[str, str, int]]:
    """The ``k`` most frequent off-diagonal (true, predicted, count) triples."""
    off = np.array(cm, copy=True)
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None, kind="stable")
    result = []
    for f in flat[:k]:
        i, j = divmod(int(f), off.shape[1])
        if off[i, j] <= 0:
            break
        result.append((classes[i], classes[j], int(off[i, j])))
    return result


def per_class_f1(y_true, y_pred, classes) -> dict[str, float]:
    """F1 per class (0.0 when a class has no true and no predicted samples)."""
    cm = confusion_matrix(y_true, y_pred, len(classes))
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return {c: float(f1[i]) for i, c in enumerate(classes)}


def error_examples(x, y_true, y_pred, k: int) -> np.ndarray:
    """Indices of the first ``k`` misclassified samples (``x`` is unused, for symmetry)."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    wrong = np.flatnonzero(np.asarray(y_true) != np.asarray(y_pred))
    return wrong[:k]
