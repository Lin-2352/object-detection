"""Tests for hcr.evaluate (pure numpy utilities, hand-computed expectations)."""

import numpy as np
import pytest

from hcr.evaluate import (
    accuracy_by_group,
    case_insensitive_accuracy,
    confusion_matrix,
    error_examples,
    per_class_f1,
    predict_classes,
    top_confusions,
)

Y_TRUE = np.array([0, 0, 1, 1, 2, 2])
Y_PRED = np.array([0, 1, 1, 1, 2, 0])
CLASSES = ["a", "b", "c"]


def test_predict_classes_batches_and_argmax():
    class Fake:
        def __init__(self):
            self.calls = []

        def predict(self, x, batch_size=None, verbose=0):
            self.calls.append(len(x))
            out = np.zeros((len(x), 3), dtype="float32")
            out[np.arange(len(x)), x[:, 0, 0, 0].astype(int) % 3] = 1.0
            return out

    x = np.zeros((7, 2, 2, 1), dtype="float32")
    x[:, 0, 0, 0] = np.arange(7)
    fake = Fake()
    pred = predict_classes(fake, x, batch_size=512)
    np.testing.assert_array_equal(pred, np.arange(7) % 3)
    assert pred.dtype == np.int64


def test_confusion_matrix():
    cm = confusion_matrix(Y_TRUE, Y_PRED, 3)
    expected = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    np.testing.assert_array_equal(cm, expected)


def test_accuracy_by_group():
    acc = accuracy_by_group(Y_TRUE, Y_PRED, {"low": [0, 1], "high": [2], "none": [9]})
    assert acc["low"] == pytest.approx(3 / 4)
    assert acc["high"] == pytest.approx(1 / 2)
    assert np.isnan(acc["none"])


def test_case_insensitive_accuracy():
    classes = ["c", "C", "x"]
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([1, 0, 2, 0])
    # c<->C counted correct (2), x==x correct (1), x->c wrong
    assert case_insensitive_accuracy(y_true, y_pred, classes, [["c", "C"]]) == pytest.approx(0.75)
    assert case_insensitive_accuracy(y_true, y_pred, classes, []) == pytest.approx(0.25)


def test_top_confusions():
    cm = np.array([[5, 3, 0], [1, 4, 2], [0, 7, 1]])
    top = top_confusions(cm, CLASSES, k=2)
    assert top == [("c", "b", 7), ("a", "b", 3)]
    assert len(top_confusions(cm, CLASSES, k=15)) == 4


def test_per_class_f1():
    f1 = per_class_f1(Y_TRUE, Y_PRED, CLASSES)
    # a: tp1 fp1 fn1 -> 0.5; b: tp2 fp1 fn0 -> 0.8; c: tp1 fp0 fn1 -> 2/3
    assert f1["a"] == pytest.approx(0.5)
    assert f1["b"] == pytest.approx(0.8)
    assert f1["c"] == pytest.approx(2 / 3)


def test_per_class_f1_absent_class_is_zero():
    f1 = per_class_f1(np.array([0, 0]), np.array([0, 0]), CLASSES)
    assert f1["a"] == 1.0
    assert f1["b"] == 0.0


def test_error_examples():
    np.testing.assert_array_equal(error_examples(None, Y_TRUE, Y_PRED, 10), [1, 5])
    np.testing.assert_array_equal(error_examples(None, Y_TRUE, Y_PRED, 1), [1])


def test_confusion_matrix_rejects_out_of_range_prediction():
    import pytest

    with pytest.raises(ValueError):
        confusion_matrix([0], [7], 5)  # used to wrap silently into another cell


def test_error_examples_rejects_negative_k():
    import pytest

    with pytest.raises(ValueError):
        error_examples(None, [0, 1], [1, 0], -1)  # used to return all but the last error
