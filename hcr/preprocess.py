"""Image loading and normalization for handwritten character recognition."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

CANVAS_SIZE = 28
INK_TARGET_SIZE = 20


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as a 2D uint8 grayscale array.

    Handles grayscale, RGB and RGBA sources. RGBA images are alpha-composited
    over a white background before conversion to grayscale.
    """
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not read image: {path}")

    if img.ndim == 2:
        return img.astype(np.uint8)

    channels = img.shape[2]
    if channels == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = (img[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        white = np.full_like(bgr, 255.0)
        composited = bgr * alpha + white * (1.0 - alpha)
        return cv2.cvtColor(composited.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if channels == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"unsupported image format for {path}: shape {img.shape}")


def _auto_invert(gray: np.ndarray) -> np.ndarray:
    """Invert to white-ink-on-black if the image background is light."""
    ring = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if np.median(ring) > 127:
        return 255 - gray
    return gray


def _denoise_mask(ink: np.ndarray) -> np.ndarray:
    """Zero out faint background noise, keeping the real ink intensities."""
    blurred = cv2.GaussianBlur(ink, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.where(mask > 0, ink, 0).astype(np.uint8)


def _bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("no ink found")
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _resize_to_ink_target(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape
    if h >= w:
        new_h = INK_TARGET_SIZE
        new_w = max(1, round(w * INK_TARGET_SIZE / h))
    else:
        new_w = INK_TARGET_SIZE
        new_h = max(1, round(h * INK_TARGET_SIZE / w))
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _pad_to_canvas(resized: np.ndarray) -> np.ndarray:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    h, w = resized.shape
    y_off = (CANVAS_SIZE - h) // 2
    x_off = (CANVAS_SIZE - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = resized
    return canvas


def _center_of_mass_shift(canvas: np.ndarray) -> np.ndarray:
    total = canvas.sum()
    if total <= 0:
        raise ValueError("no ink found")

    ys, xs = np.nonzero(canvas > 0)
    min_y, max_y, min_x, max_x = ys.min(), ys.max(), xs.min(), xs.max()

    row_idx = np.arange(CANVAS_SIZE, dtype=np.float32)
    col_idx = np.arange(CANVAS_SIZE, dtype=np.float32)
    cy = float((row_idx[:, None] * canvas).sum() / total)
    cx = float((col_idx[None, :] * canvas).sum() / total)

    center = CANVAS_SIZE / 2.0
    shift_y = round(center - cy)
    shift_x = round(center - cx)

    shift_y = max(-int(min_y), min(shift_y, CANVAS_SIZE - 1 - int(max_y)))
    shift_x = max(-int(min_x), min(shift_x, CANVAS_SIZE - 1 - int(max_x)))

    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(
        canvas, matrix, (CANVAS_SIZE, CANVAS_SIZE), flags=cv2.INTER_NEAREST, borderValue=0
    )


def to_model_input(gray: np.ndarray) -> np.ndarray:
    """Normalize a grayscale image into a 28x28 float32 model input in [0, 1].

    The result has white ink on a black background, upright, with the ink
    bounding box scaled to a 20 px longer side and re-centered on the
    ink's center of mass, following MNIST-style normalization.
    """
    gray = np.asarray(gray)
    if gray.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {gray.shape}")

    ink = _auto_invert(gray.astype(np.uint8))
    cleaned = _denoise_mask(ink)

    y0, y1, x0, x1 = _bounding_box(cleaned)
    crop = cleaned[y0 : y1 + 1, x0 : x1 + 1]

    resized = _resize_to_ink_target(crop)
    canvas = _pad_to_canvas(resized)
    shifted = _center_of_mass_shift(canvas)

    return (shifted.astype(np.float32) / 255.0).clip(0.0, 1.0)


def batch_to_model_input(grays: Sequence[np.ndarray]) -> np.ndarray:
    """Convert a sequence of grayscale images into an (N, 28, 28, 1) float32 batch."""
    stacked = np.stack([to_model_input(gray) for gray in grays], axis=0)
    return stacked[..., None]
