"""Deterministic synthetic keyboard-symbol glyph data.

This is a printed-glyph dataset, not a handwriting dataset. It exists because
EMNIST and Chars74K do not provide the common keyboard punctuation classes
needed by this project. OpenCV's bundled Hershey fonts are used so no font
file or third-party dataset is redistributed.
"""

from __future__ import annotations

import cv2
import numpy as np

KEYBOARD_SYMBOLS = tuple("!@#$%&*()-_=+[]{}\\|;:'\",.<>/?^~`")


def _render_symbol(symbol: str, rng: np.random.RandomState) -> np.ndarray:
    image = np.zeros((28, 28), dtype=np.uint8)
    # Keep the glyph family and placement stable: this benchmark measures
    # whether the classes are separable before a real multi-font dataset is
    # introduced.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (width, height), baseline = cv2.getTextSize(symbol, font, scale, thickness)
    x = max(0, (28 - width) // 2)
    y = max(height + baseline, (28 + height) // 2)
    cv2.putText(image, symbol, (x, y), font, scale, 255, thickness, cv2.LINE_AA)

    angle = 0.0
    matrix = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
    return cv2.warpAffine(image, matrix, (28, 28), borderValue=0)


def load_keyboard_symbols(
    samples_per_class: int = 120,
    test_samples_per_class: int = 30,
    seed: int = 42,
):
    """Return train/test arrays and labels for common printed keyboard symbols."""
    if samples_per_class < 1 or test_samples_per_class < 1:
        raise ValueError("sample counts must be positive")

    rng = np.random.RandomState(seed)
    train_images, train_labels, test_images, test_labels = [], [], [], []
    for label, symbol in enumerate(KEYBOARD_SYMBOLS):
        for images, labels, count in (
            (train_images, train_labels, samples_per_class),
            (test_images, test_labels, test_samples_per_class),
        ):
            for _ in range(count):
                images.append(_render_symbol(symbol, rng))
                labels.append(label)

    def finish(images, labels):
        order = rng.permutation(len(images))
        x = np.asarray(images, dtype=np.float32)[order, ..., np.newaxis] / 255.0
        y = np.asarray(labels, dtype=np.int64)[order]
        return x, y

    return (
        finish(train_images, train_labels),
        finish(test_images, test_labels),
        list(KEYBOARD_SYMBOLS),
    )