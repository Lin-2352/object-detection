"""Tests for hcr.preprocess."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from hcr import preprocess
from hcr.preprocess import load_image, to_model_input

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"
SAMPLE_NAMES = ["digit_1.png", "digit_3.png", "digit_4.png", "digit_5.png", "digit_6.png"]


def _draw_cross(background: int, ink: int, size: int = 200) -> np.ndarray:
    img = np.full((size, size), background, dtype=np.uint8)
    cv2.line(img, (60, 60), (140, 140), ink, 15)
    cv2.line(img, (140, 60), (60, 140), ink, 15)
    return img


def test_dark_on_light_gets_inverted():
    img = _draw_cross(background=255, ink=20)
    inverted = preprocess._auto_invert(img)
    border = np.concatenate([inverted[0, :], inverted[-1, :], inverted[:, 0], inverted[:, -1]])
    assert np.median(border) < 127


def test_light_on_dark_is_not_inverted():
    img = _draw_cross(background=10, ink=240)
    result = preprocess._auto_invert(img)
    assert np.array_equal(result, img)


def test_rgba_transparent_background_handled(tmp_path):
    size = 60
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    cv2.circle(rgba, (30, 30), 15, (0, 0, 0, 255), -1)
    path = tmp_path / "rgba.png"
    cv2.imwrite(str(path), rgba)

    gray = load_image(path)
    assert gray.ndim == 2
    assert gray.dtype == np.uint8
    assert gray[0, 0] > 200  # transparent corner composited over white
    assert gray[30, 30] < 60  # opaque black circle stays dark

    model_input = to_model_input(gray)
    assert model_input.shape == (28, 28)
    assert model_input.max() > 0.0


def test_to_model_input_shape_dtype_range():
    img = _draw_cross(background=10, ink=240)
    out = to_model_input(img)
    assert out.shape == (28, 28)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out.max() > 0.0


def test_ink_bbox_and_center_of_mass_for_off_center_stroke():
    size = 200
    img = np.full((size, size), 10, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (40, 70), 240, -1)  # off-center, near top-left
    out = to_model_input(img)

    ys, xs = np.nonzero(out > 0.05)
    assert ys.max() - ys.min() <= 21
    assert xs.max() - xs.min() <= 21

    total = out.sum()
    row_idx = np.arange(28, dtype=np.float64)
    col_idx = np.arange(28, dtype=np.float64)
    cy = (row_idx[:, None] * out).sum() / total
    cx = (col_idx[None, :] * out).sum() / total
    assert abs(cy - 14) < 1.0
    assert abs(cx - 14) < 1.0


def test_large_non_square_image_works():
    img = np.full((600, 300), 5, dtype=np.uint8)
    cv2.circle(img, (150, 200), 80, 250, -1)
    out = to_model_input(img)
    assert out.shape == (28, 28)
    assert out.max() > 0.0


def test_blank_black_image_raises():
    img = np.full((50, 50), 0, dtype=np.uint8)
    with pytest.raises(ValueError):
        to_model_input(img)


def test_blank_white_image_raises():
    img = np.full((50, 50), 255, dtype=np.uint8)
    with pytest.raises(ValueError):
        to_model_input(img)


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_real_samples_produce_ink(name):
    gray = load_image(SAMPLES_DIR / name)
    assert gray.ndim == 2
    assert gray.dtype == np.uint8

    out = to_model_input(gray)
    assert out.shape == (28, 28)
    assert out.max() > 0.0
