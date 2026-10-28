"""Tests for hcr.hasy using a tiny synthetic HASYv2 layout (no real data)."""

import tarfile

import cv2
import numpy as np
import pytest

from hcr.hasy import (
    COVERAGE,
    KEYBOARD_LATEX_MAP,
    load_hasy_symbols,
    split_writers,
)
from hcr.symbols import KEYBOARD_SYMBOLS

SYMBOLS_CSV = (
    "symbol_id,latex,training_samples,test_samples\n195,-,1,1\n196,+,1,1\n31,A,1,1\n922,/,1,1\n"
)

ROWS = [  # path, symbol_id, user
    ("hasy-data/v2-00000.png", 195, 1),
    ("hasy-data/v2-00001.png", 196, 2),
    ("hasy-data/v2-00002.png", 31, 3),  # letter A: not a keyboard symbol -> dropped
    ("hasy-data/v2-00003.png", 922, 4),
    ("hasy-data/v2-00004.png", 195, 5),
]


def _png(seed: int) -> bytes:
    rng = np.random.RandomState(seed)
    img = np.full((32, 32, 3), 255, np.uint8)
    x = rng.randint(4, 12)
    cv2.line(img, (x, 6), (x + 14, 26), (0, 0, 0), 2)
    return cv2.imencode(".png", img)[1].tobytes()


def _labels_csv() -> str:
    lines = ["path,symbol_id,latex,user_id"]
    lines += [f"{p},{s},x,{u}" for p, s, u in ROWS]
    return "\n".join(lines) + "\n"


def _write_dir(root):
    (root / "hasy-data").mkdir(parents=True)
    (root / "symbols.csv").write_text(SYMBOLS_CSV)
    (root / "hasy-data-labels.csv").write_text(_labels_csv())
    for i, (p, _, _) in enumerate(ROWS):
        (root / p).write_bytes(_png(i))


def test_latex_map_is_subset_of_keyboard_symbols_and_consistent():
    assert set(KEYBOARD_LATEX_MAP) <= set(KEYBOARD_SYMBOLS)
    assert set(COVERAGE["covered"]) == set(KEYBOARD_LATEX_MAP)
    assert set(COVERAGE["covered"]) | set(COVERAGE["missing"]) == set(KEYBOARD_SYMBOLS)
    assert not set(COVERAGE["covered"]) & set(COVERAGE["missing"])
    for char in "+-/<>[]{}|#$%&":
        assert char in KEYBOARD_LATEX_MAP
    for char in "!@()=_":
        assert char in COVERAGE["missing"]


def test_load_from_extracted_directory(tmp_path):
    _write_dir(tmp_path)
    images, labels, writers = load_hasy_symbols(tmp_path)
    assert images.shape == (4, 28, 28) and images.dtype == np.uint8
    assert sorted(labels.tolist()) == ["+", "-", "-", "/"]
    assert sorted(writers.tolist()) == [1, 2, 4, 5]
    # white ink on black background, ink present
    assert images.max() > 200 and (images == 0).mean() > 0.5


def test_load_from_tarball(tmp_path):
    src = tmp_path / "src"
    _write_dir(src)
    raw = tmp_path / "raw"
    raw.mkdir()
    with tarfile.open(raw / "HASYv2.tar.bz2", "w:bz2") as tf:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                tf.add(f, arcname=f.relative_to(src).as_posix())
    images, labels, writers = load_hasy_symbols(raw)
    assert images.shape[0] == 4 and len(labels) == 4 == len(writers)


def test_load_rejects_symbol_id_latex_mismatch(tmp_path):
    _write_dir(tmp_path)
    (tmp_path / "symbols.csv").write_text(
        "symbol_id,latex,training_samples,test_samples\n195,+,1,1\n"
    )
    with pytest.raises(ValueError, match="latex"):
        load_hasy_symbols(tmp_path)


def test_load_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_hasy_symbols(tmp_path)


def test_split_writers_is_disjoint_deterministic_and_pins_dominant_writer():
    writers = np.array([9] * 50 + list(range(1, 41)))
    a = split_writers(writers, test_fraction=0.25, seed=1)
    b = split_writers(writers, test_fraction=0.25, seed=1)
    np.testing.assert_array_equal(a, b)
    assert not set(writers[a]) & set(writers[~a])
    assert not a[:50].any()  # dominant writer goes to train
    assert 5 <= len(set(writers[a])) <= 15
