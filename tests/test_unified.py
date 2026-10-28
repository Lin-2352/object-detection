"""Tests for hcr.unified using tiny synthetic sources (plus one slow real-data test)."""

from pathlib import Path

import matplotlib
import numpy as np
import pytest

from hcr.fonts import ASCII_CHARS
from hcr.symbols import KEYBOARD_SYMBOLS
from hcr.unified import (
    AMBIGUOUS_GROUPS,
    SOURCE_EMNIST,
    SOURCE_FONT,
    SOURCE_HASY,
    Unified,
    ambiguous_merge_map,
    build_unified,
    class_groups,
    load_unified,
    pixel_hashes,
    save_unified,
)

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONT_NAMES = [
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSans-Oblique.ttf",
    "DejaVuSerif.ttf",
    "DejaVuSerif-Bold.ttf",
    "DejaVuSansMono.ttf",
    "DejaVuSansMono-Bold.ttf",
    "cmss10.ttf",
]
EMNIST_CHARS = [*"0123456789", *"ABCDEFGHIJKLMNOPQRSTUVWXYZ", *"abcdefghijklmnopqrstuvwxyz"]
SPLITS = ("train", "val", "test")


def _emnist(rng, per_class, shift):
    labels = np.repeat(np.arange(62), per_class)
    images = rng.integers(0, 256, (len(labels), 28, 28), dtype=np.uint8)
    images[:, 0, 0] = np.arange(len(labels)) % 256 + shift  # keep samples distinct
    images[:, 0, 1] = np.arange(len(labels)) // 256
    return images, labels


def _hasy(rng):
    chars, writers = [], []
    for char in "+-/":
        for writer in range(1, 41):
            chars += [char] * 3
            writers += [writer] * 3
    chars += ["+"] * 200  # one dominant writer, like HASYv2's 16925
    writers += [999] * 200
    images = rng.integers(0, 256, (len(chars), 28, 28), dtype=np.uint8)
    images[:, 0, 0] = np.arange(len(chars)) % 256
    images[:, 0, 1] = np.arange(len(chars)) // 256
    images[:, 1, 1] = 77
    return images, np.array(chars), np.array(writers)


@pytest.fixture(scope="module")
def built():
    rng = np.random.default_rng(0)
    tr = _emnist(rng, 20, 0)
    te = _emnist(rng, 5, 128)
    # EMNIST classes in a different order than ASCII to exercise remapping
    emnist_classes = EMNIST_CHARS
    fonts = [FONT_DIR / n for n in FONT_NAMES]
    unified = build_unified(
        emnist_train=tr,
        emnist_test=te,
        emnist_classes=emnist_classes,
        hasy=_hasy(rng),
        fonts=fonts,
        seed=1,
        fonts_per_class=8,
        symbol_train_target=20,
    )
    return unified, tr


def test_class_list_is_94_ascii_chars(built):
    unified, _ = built
    assert unified.classes == list(ASCII_CHARS)
    assert len(unified.classes) == 94 and set(KEYBOARD_SYMBOLS) <= set(unified.classes)


def test_shapes_dtypes_ranges_and_sources(built):
    unified, _ = built
    for split in SPLITS:
        x, y, s = (getattr(unified, f"{n}_{split}") for n in ("x", "y", "source"))
        assert x.dtype == np.float32 and x.shape[1:] == (28, 28, 1)
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert y.dtype == np.int64 and y.min() >= 0 and y.max() < 94
        assert s.dtype == np.int8 and len(x) == len(y) == len(s)
        assert set(np.unique(s)) <= {SOURCE_EMNIST, SOURCE_HASY, SOURCE_FONT}
    assert {SOURCE_EMNIST, SOURCE_HASY, SOURCE_FONT} == set(np.unique(unified.source_train))


def test_emnist_labels_are_remapped_to_ascii_space(built):
    unified, (tr_x, tr_y) = built
    x_train = (unified.x_train[..., 0] * 255).round().astype(np.uint8)
    lookup = {img.tobytes(): int(unified.y_train[i]) for i, img in enumerate(x_train)}
    lookup.update(
        {
            img.tobytes(): int(unified.y_val[i])
            for i, img in enumerate((unified.x_val[..., 0] * 255).round().astype(np.uint8))
        }
    )
    for i in (0, 5, 400, 1000, 1239):
        expected = unified.classes.index(EMNIST_CHARS[tr_y[i]])
        assert lookup[tr_x[i].tobytes()] == expected


def test_validation_is_stratified_10_percent_of_emnist_train(built):
    unified, _ = built
    val = unified.y_val[unified.source_val == SOURCE_EMNIST]
    counts = np.bincount(val, minlength=94)
    emnist_idx = [unified.classes.index(c) for c in EMNIST_CHARS]
    assert (counts[emnist_idx] == 2).all()  # 10% of 20 per class
    train = unified.y_train[unified.source_train == SOURCE_EMNIST]
    assert (np.bincount(train, minlength=94)[emnist_idx] == 18).all()


def test_symbol_classes_are_balanced_up_to_target(built):
    unified, _ = built
    counts = np.bincount(unified.y_train, minlength=94)
    for char in KEYBOARD_SYMBOLS:
        assert counts[unified.classes.index(char)] >= 20
    # alphanumerics get exactly fonts_per_class printed samples on top of EMNIST
    font_counts = np.bincount(unified.y_train[unified.source_train == SOURCE_FONT], minlength=94)
    assert font_counts[unified.classes.index("A")] == 8
    # a symbol with handwritten data is topped up only to the target
    plus = unified.classes.index("+")
    hasy_plus = ((unified.y_train == plus) & (unified.source_train == SOURCE_HASY)).sum()
    assert counts[plus] == max(20, hasy_plus + 8)


def test_no_duplicate_pixels_between_train_and_test_per_source(built):
    unified, _ = built
    for source in (SOURCE_EMNIST, SOURCE_HASY, SOURCE_FONT):
        train = pixel_hashes(unified.x_train[unified.source_train == source])
        test = pixel_hashes(unified.x_test[unified.source_test == source])
        assert len(test) > 0
        assert not set(train) & set(test)


def test_no_duplicate_pixels_between_any_train_and_eval_split(built):
    unified, _ = built
    train = set(pixel_hashes(unified.x_train))
    assert not train & set(pixel_hashes(unified.x_val))
    assert not train & set(pixel_hashes(unified.x_test))


def test_fonts_and_writers_are_disjoint_across_splits(built):
    unified, _ = built
    fonts = unified.meta["fonts"]
    writers = unified.meta["hasy_writers"]
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(fonts[a]) & set(fonts[b])
        assert not set(writers[a]) & set(writers[b])
    assert fonts["test"] and fonts["val"] and writers["test"]
    assert 999 in writers["train"]  # dominant writer pinned to train


def test_cache_roundtrip_and_reload_without_raw_data(built, tmp_path):
    unified, _ = built
    save_unified(unified, tmp_path / "unified.npz")
    loaded = load_unified(
        processed_dir=tmp_path,
        seed=1,
        fonts_per_class=8,
        symbol_train_target=20,
        hasy_dir=tmp_path / "missing",
        font_dir=tmp_path / "missing",
    )
    assert isinstance(loaded, Unified)
    np.testing.assert_array_equal(loaded.x_test, unified.x_test)
    np.testing.assert_array_equal(loaded.y_train, unified.y_train)
    np.testing.assert_array_equal(loaded.source_val, unified.source_val)
    assert loaded.classes == unified.classes
    assert loaded.meta["fonts"] == unified.meta["fonts"]


def test_cache_with_different_params_triggers_rebuild(built, tmp_path):
    unified, _ = built
    save_unified(unified, tmp_path / "unified.npz")
    with pytest.raises((FileNotFoundError, RuntimeError)):
        load_unified(
            processed_dir=tmp_path,
            seed=2,
            fonts_per_class=8,
            symbol_train_target=20,
            hasy_dir=tmp_path / "missing",
            font_dir=tmp_path / "missing",
        )
    with pytest.raises((FileNotFoundError, RuntimeError)):
        load_unified(
            processed_dir=tmp_path,
            seed=1,
            fonts_per_class=8,
            symbol_train_target=20,
            rebuild=True,
            hasy_dir=tmp_path / "missing",
            font_dir=tmp_path / "missing",
        )


def test_class_groups():
    groups = class_groups(list(ASCII_CHARS))
    assert set(groups) == {"digits", "upper", "lower", "symbols"}
    assert [len(groups[k]) for k in ("digits", "upper", "lower", "symbols")] == [10, 26, 26, 32]
    flat = sorted(i for v in groups.values() for i in v)
    assert flat == list(range(94))
    assert all(ASCII_CHARS[i].isdigit() for i in groups["digits"])
    assert all(ASCII_CHARS[i] in KEYBOARD_SYMBOLS for i in groups["symbols"])


def test_ambiguous_groups_cover_known_confusions():
    as_sets = [set(g) for g in AMBIGUOUS_GROUPS]
    for expected in (
        "cC",
        "kK",
        "mM",
        "oO0",
        "pP",
        "sS",
        "uU",
        "vV",
        "wW",
        "xX",
        "yY",
        "zZ",
        "lI1|",
    ):
        assert set(expected) in as_sets
    members = [c for g in AMBIGUOUS_GROUPS for c in g]
    assert len(members) == len(set(members)) and set(members) <= set(ASCII_CHARS)


@pytest.mark.slow
def test_real_data_is_leak_free():
    root = Path(__file__).resolve().parents[1]
    has_cache = (root / "data" / "processed" / "unified.npz").exists()
    has_raw = (root / "data/raw/emnist/gzip.zip").exists() and (
        root / "data/raw/hasy/HASYv2.tar.bz2"
    ).exists()
    if not (has_cache or has_raw):
        pytest.skip("real EMNIST/HASYv2 data not present")
    unified = load_unified()
    assert len(unified.classes) == 94
    fonts, writers = unified.meta["fonts"], unified.meta["hasy_writers"]
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(fonts[a]) & set(fonts[b])
        assert not set(writers[a]) & set(writers[b])
    for source in (SOURCE_EMNIST, SOURCE_HASY, SOURCE_FONT):
        train = set(pixel_hashes(unified.x_train[unified.source_train == source]))
        for split_x, split_s in (
            (unified.x_val, unified.source_val),
            (unified.x_test, unified.source_test),
        ):
            assert not train & set(pixel_hashes(split_x[split_s == source]))
    counts = np.bincount(unified.y_train, minlength=94)
    for char in KEYBOARD_SYMBOLS:
        assert counts[unified.classes.index(char)] >= 3000


def test_ambiguous_merge_map_collapses_case_and_lookalikes():
    classes = list(ASCII_CHARS)
    mapping = ambiguous_merge_map(classes)
    assert mapping[classes.index("C")] == mapping[classes.index("c")]
    assert len({mapping[classes.index(c)] for c in "lI1|"}) == 1
    assert len({mapping[classes.index(c)] for c in "oO0"}) == 1
    assert mapping[classes.index("a")] == classes.index("a")
    assert mapping[classes.index("C")] != mapping[classes.index("K")]
