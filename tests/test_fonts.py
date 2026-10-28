"""Tests for hcr.fonts using the DejaVu/STIX fonts that ship with matplotlib."""

import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from hcr.fonts import ASCII_CHARS, discover_fonts, render_font_glyphs, split_fonts

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
GOOD = ["DejaVuSans.ttf", "DejaVuSerif.ttf", "DejaVuSansMono.ttf"]


@pytest.fixture
def font_dir(tmp_path):
    for name in GOOD:
        shutil.copy(FONT_DIR / name, tmp_path / name)
    shutil.copy(FONT_DIR / "STIXSizOneSymReg.ttf", tmp_path / "Partial.ttf")  # 9/94 glyphs
    shutil.copy(FONT_DIR / "DejaVuSans.ttf", tmp_path / "Webdings.ttf")  # symbol font name
    shutil.copy(FONT_DIR / "DejaVuSans.ttf", tmp_path / "wingding.ttf")
    (tmp_path / "notes.txt").write_text("not a font")
    (tmp_path / "Broken.ttf").write_bytes(b"garbage")
    return tmp_path


def test_ascii_chars_are_the_94_printable_characters():
    assert len(ASCII_CHARS) == 94
    assert ASCII_CHARS[0] == "!" and ASCII_CHARS[-1] == "~"
    assert "0" in ASCII_CHARS and " " not in ASCII_CHARS


def test_discover_fonts_filters_and_is_deterministic(font_dir):
    fonts = discover_fonts(font_dir)
    assert [f.name for f in fonts] == sorted(GOOD)
    assert fonts == discover_fonts(font_dir)


def test_discover_fonts_missing_dir_returns_empty(tmp_path):
    assert discover_fonts(tmp_path / "nope") == []


def test_split_fonts_is_font_disjoint_and_deterministic():
    fonts = [Path(f"f{i}.ttf") for i in range(20)]
    train, test = split_fonts(fonts, seed=3)
    assert len(train) == 16 and len(test) == 4
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(fonts)
    assert (train, test) == split_fonts(fonts, seed=3)
    assert test != split_fonts(fonts, seed=4)[1]


def test_split_fonts_always_keeps_both_sides_nonempty():
    train, test = split_fonts([Path("a.ttf"), Path("b.ttf")], seed=0)
    assert len(train) == 1 and len(test) == 1


def test_render_font_glyphs_shapes_labels_and_font_idx():
    fonts = [FONT_DIR / n for n in GOOD[:2]]
    rng = np.random.default_rng(0)
    images, labels, font_idx = render_font_glyphs(list("A8%"), fonts, per_font=4, rng=rng)
    assert images.shape == (2 * 3 * 4, 28, 28) and images.dtype == np.uint8
    assert len(labels) == len(font_idx) == 24
    assert sorted(set(labels.tolist())) == ["%", "8", "A"]
    assert set(font_idx.tolist()) == {0, 1}
    # white ink on a black background
    assert (images.reshape(24, -1).max(axis=1) > 100).all()
    assert (images == 0).mean() > 0.5


def test_render_font_glyphs_is_deterministic_and_augmented():
    fonts = [FONT_DIR / GOOD[0]]
    a = render_font_glyphs(list("Q"), fonts, 6, np.random.default_rng(5))[0]
    b = render_font_glyphs(list("Q"), fonts, 6, np.random.default_rng(5))[0]
    np.testing.assert_array_equal(a, b)
    assert len({img.tobytes() for img in a}) > 1  # augmentation produces variety


def test_render_font_glyphs_handles_punctuation_and_thin_glyphs():
    fonts = [FONT_DIR / GOOD[0]]
    images, labels, _ = render_font_glyphs(list(".,'-`"), fonts, 5, np.random.default_rng(1))
    assert images.shape[0] == 25 and (images.reshape(25, -1).max(axis=1) > 0).all()
