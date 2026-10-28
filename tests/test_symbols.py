import numpy as np
import pytest

from hcr.symbols import KEYBOARD_SYMBOLS, load_keyboard_symbols


def test_keyboard_symbols_are_reproducible_and_complete():
    first = load_keyboard_symbols(samples_per_class=2, test_samples_per_class=1, seed=7)
    second = load_keyboard_symbols(samples_per_class=2, test_samples_per_class=1, seed=7)

    (x_train, y_train), (x_test, y_test), classes = first
    (x_train_2, y_train_2), (x_test_2, y_test_2), _ = second

    assert classes == list(KEYBOARD_SYMBOLS)
    assert x_train.shape == (len(classes) * 2, 28, 28, 1)
    assert x_test.shape == (len(classes), 28, 28, 1)
    assert y_train.shape == (len(classes) * 2,)
    assert y_test.shape == (len(classes),)
    assert x_train.dtype == np.float32
    assert np.all((x_train >= 0) & (x_train <= 1))
    np.testing.assert_array_equal(x_train, x_train_2)
    np.testing.assert_array_equal(y_train, y_train_2)
    np.testing.assert_array_equal(x_test, x_test_2)
    np.testing.assert_array_equal(y_test, y_test_2)


def test_keyboard_symbol_sample_counts_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        load_keyboard_symbols(samples_per_class=0)