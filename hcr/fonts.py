"""Printed-glyph rendering from system fonts, with font-disjoint train/test splits.

This produces *printed* characters, not handwriting. It exists to give the
keyboard-symbol classes (which EMNIST lacks and HASYv2 only partly covers)
enough varied training examples. Augmentation (rotation, shear, scale, stroke
dilation/erosion, blur, noise) is applied so fonts are not a clean-render
shortcut; the final crop/scale/centre step is shared with the real pipeline
(:func:`hcr.preprocess.to_model_input`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hcr.preprocess import to_model_input

ASCII_CHARS: tuple[str, ...] = tuple(chr(i) for i in range(33, 127))

_SKIP_NAME = re.compile(
    r"webdings|wingding|dingbat|symbol|marlett|mdl2|fluent|emoji|holomdl|bssym|mtextra|"
    r"cmex|cmsy|cmmi|lastresort|icon|seguisym|seguiemj|seguihis|simsunb|simsunext|msyi",
    re.IGNORECASE,
)
_NOTDEF_PROBE = "\U0010fffe"  # a noncharacter, so every font renders its .notdef glyph
_RENDER_SIZE = 56
_CANVAS = 128


@lru_cache(maxsize=256)
def _load_font(path: str, size: int = _RENDER_SIZE) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _mask_bytes(font: ImageFont.FreeTypeFont, char: str) -> bytes | None:
    mask = font.getmask(char)
    if mask.getbbox() is None:
        return None
    return bytes(mask)


def _has_all_glyphs(path: Path, chars: Sequence[str]) -> bool:
    try:
        font = _load_font(str(path))
        notdef = _mask_bytes(font, _NOTDEF_PROBE)
        for char in chars:
            glyph = _mask_bytes(font, char)
            if glyph is None or (notdef is not None and glyph == notdef):
                return False
    except OSError:
        return False
    return True


def discover_fonts(
    font_dir: str | Path = r"C:\Windows\Fonts", chars: Sequence[str] = ASCII_CHARS
) -> list[Path]:
    """Return usable .ttf/.otf files that contain a glyph for every char in ``chars``.

    Symbol/dingbat/icon fonts are skipped by name. The result is sorted by
    lower-cased file name so repeated calls give the same order.
    """
    font_dir = Path(font_dir)
    if not font_dir.is_dir():
        return []
    candidates = sorted(
        (p for p in font_dir.iterdir() if p.suffix.lower() in (".ttf", ".otf")),
        key=lambda p: p.name.lower(),
    )
    return [p for p in candidates if not _SKIP_NAME.search(p.stem) and _has_all_glyphs(p, chars)]


def split_fonts(
    fonts: Sequence[Path], seed: int = 42, test_fraction: float = 0.2
) -> tuple[list[Path], list[Path]]:
    """Split fonts into disjoint (train, test) lists; both are non-empty if len >= 2."""
    ordered = sorted(fonts, key=lambda p: str(p).lower())
    if len(ordered) < 2:
        raise ValueError("need at least two fonts to make a font-disjoint split")
    n_test = min(len(ordered) - 1, max(1, round(test_fraction * len(ordered))))
    order = np.random.RandomState(seed).permutation(len(ordered))
    test_idx = set(order[:n_test].tolist())
    train = [f for i, f in enumerate(ordered) if i not in test_idx]
    test = [f for i, f in enumerate(ordered) if i in test_idx]
    return train, test


def _draw(font: ImageFont.FreeTypeFont, char: str) -> np.ndarray:
    canvas = Image.new("L", (_CANVAS, _CANVAS), 0)
    ImageDraw.Draw(canvas).text((_CANVAS / 2, _CANVAS / 2), char, fill=255, font=font, anchor="mm")
    return np.asarray(canvas, dtype=np.uint8)


def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = np.deg2rad(rng.uniform(-15, 15))
    shear = rng.uniform(-0.25, 0.25)
    scale = rng.uniform(0.85, 1.1)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    lin = scale * rot @ np.array([[1.0, shear], [0.0, 1.0]])
    centre = np.array([_CANVAS / 2, _CANVAS / 2])
    matrix = np.hstack([lin, (centre - lin @ centre)[:, None]]).astype(np.float32)
    out = cv2.warpAffine(img, matrix, (_CANVAS, _CANVAS), flags=cv2.INTER_LINEAR)

    morph = int(rng.integers(0, 3))  # 0 none, 1 dilate (bolder), 2 erode (thinner)
    if morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        iters = int(rng.integers(1, 3))
        changed = (cv2.dilate if morph == 1 else cv2.erode)(out, kernel, iterations=iters)
        if changed.max() > 0:
            out = changed

    sigma = rng.uniform(0.0, 1.5)
    if sigma > 0.2:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    return out


def _add_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Jitter intensities on and next to ink only, keeping the background clean."""
    sigma = rng.uniform(0.0, 12.0)
    near_ink = cv2.dilate((img > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape) * near_ink
    return np.clip(noisy, 0, 255).round().astype(np.uint8)


def _render_one(font: ImageFont.FreeTypeFont, char: str, rng: np.random.Generator) -> np.ndarray:
    base = _draw(font, char)
    for _ in range(8):
        try:
            model_input = to_model_input(_augment(base, rng))
        except ValueError:  # augmentation erased the glyph; draw new parameters
            continue
        return _add_noise((model_input * 255.0).round().astype(np.uint8), rng)
    model_input = to_model_input(base)
    return (model_input * 255.0).round().astype(np.uint8)


def render_font_glyphs(
    chars: Sequence[str],
    font_paths: Sequence[str | Path],
    per_font: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render ``per_font`` augmented samples of every char in every font.

    Returns ``(images, labels, font_idx)``: uint8 (N, 28, 28) white-on-black
    images, a str array of single-character labels, and an int64 array giving
    each sample's index into ``font_paths``. Order is font-major, then char.
    """
    if per_font < 1:
        raise ValueError("per_font must be positive")
    images, labels, font_idx = [], [], []
    for idx, path in enumerate(font_paths):
        font = _load_font(str(path))
        for char in chars:
            for _ in range(per_font):
                images.append(_render_one(font, char, rng))
                labels.append(char)
                font_idx.append(idx)
    if not images:
        return np.zeros((0, 28, 28), np.uint8), np.array([], dtype="<U1"), np.zeros(0, np.int64)
    return np.stack(images), np.array(labels, dtype="<U1"), np.array(font_idx, dtype=np.int64)
