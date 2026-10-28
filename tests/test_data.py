"""Tests for IDX/mapping parsing in hcr.data, using synthetic bytes only."""

import struct

import numpy as np

from hcr.data import (
    fix_emnist_orientation,
    parse_balanced_mapping,
    parse_idx_images,
    parse_idx_labels,
)


def _make_idx_images(images: np.ndarray) -> bytes:
    num, rows, cols = images.shape
    header = struct.pack(">4I", 2051, num, rows, cols)
    return header + images.astype(np.uint8).tobytes()


def _make_idx_labels(labels: np.ndarray) -> bytes:
    header = struct.pack(">2I", 2049, len(labels))
    return header + labels.astype(np.uint8).tobytes()


def test_parse_idx_images_roundtrip():
    images = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4) % 255
    buf = _make_idx_images(images)

    parsed = parse_idx_images(buf)

    assert parsed.shape == (2, 3, 4)
    assert parsed.dtype == np.uint8
    np.testing.assert_array_equal(parsed, images)


def test_parse_idx_images_rejects_bad_magic():
    buf = struct.pack(">4I", 1234, 1, 2, 2) + bytes(4)
    try:
        parse_idx_images(buf)
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad magic number")


def test_parse_idx_labels_roundtrip():
    labels = np.array([0, 1, 2, 25], dtype=np.uint8)
    buf = _make_idx_labels(labels)

    parsed = parse_idx_labels(buf)

    assert parsed.dtype == np.uint8
    np.testing.assert_array_equal(parsed, labels)


def test_parse_idx_labels_rejects_bad_magic():
    buf = struct.pack(">2I", 9999, 1) + bytes(1)
    try:
        parse_idx_labels(buf)
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad magic number")


def test_fix_emnist_orientation_transposes_each_image():
    # A single 2x3 image where transposing swaps rows/cols recognisably.
    image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    batch = image[np.newaxis, ...]

    fixed = fix_emnist_orientation(batch)

    assert fixed.shape == (1, 3, 2)
    np.testing.assert_array_equal(fixed[0], image.T)


def test_parse_balanced_mapping():
    text = "0 48\n1 49\n2 65\n"

    mapping = parse_balanced_mapping(text)

    assert mapping == {0: 48, 1: 49, 2: 65}
    assert chr(mapping[2]) == "A"


def test_parse_balanced_mapping_ignores_blank_lines():
    text = "0 48\n\n1 49\n"

    mapping = parse_balanced_mapping(text)

    assert mapping == {0: 48, 1: 49}


def test_emnist_letters_label_shift_convention():
    # Labels in the emnist-letters files are 1..26; hcr.data shifts them to
    # 0..25 so they can index directly into classes = ["A", ..., "Z"].
    raw_labels = np.array([1, 2, 26], dtype=np.uint8)
    shifted = raw_labels.astype(np.int64) - 1

    classes = [chr(ord("A") + i) for i in range(26)]

    assert [classes[i] for i in shifted] == ["A", "B", "Z"]


def test_data_dir_defaults_to_repo_raw_emnist(monkeypatch):
    from pathlib import Path

    import hcr.data as data

    monkeypatch.delenv("HCR_DATA_DIR", raising=False)
    expected = Path(data.__file__).resolve().parents[1] / "data" / "raw" / "emnist"
    assert data._data_dir() == expected


def test_data_dir_env_override(monkeypatch, tmp_path):
    import hcr.data as data

    monkeypatch.setenv("HCR_DATA_DIR", str(tmp_path))
    assert data._data_dir() == tmp_path


def test_emnist_byclass_registered_and_loads_from_synthetic_zip(monkeypatch, tmp_path):
    import gzip
    import zipfile

    import hcr.data as data

    assert "emnist-byclass" in data._LOADERS
    chars = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    mapping = "\n".join(f"{i} {ord(c)}" for i, c in enumerate(chars))
    rng = np.random.RandomState(0)
    tr_x = rng.randint(0, 255, (3, 28, 28)).astype(np.uint8)
    te_x = rng.randint(0, 255, (2, 28, 28)).astype(np.uint8)
    with zipfile.ZipFile(tmp_path / "gzip.zip", "w") as zf:
        zf.writestr("gzip/emnist-byclass-mapping.txt", mapping)
        for split, x, y in (("train", tr_x, [0, 10, 61]), ("test", te_x, [5, 36])):
            zf.writestr(
                f"gzip/emnist-byclass-{split}-images-idx3-ubyte.gz",
                gzip.compress(_make_idx_images(x)),
            )
            zf.writestr(
                f"gzip/emnist-byclass-{split}-labels-idx1-ubyte.gz",
                gzip.compress(_make_idx_labels(np.array(y))),
            )
    monkeypatch.setenv("HCR_DATA_DIR", str(tmp_path))

    (xtr, ytr), (xte, yte), classes = data.load_dataset("emnist-byclass")

    assert classes == chars
    assert xtr.shape == (3, 28, 28, 1) and xtr.dtype == np.float32
    assert xte.shape == (2, 28, 28, 1)
    assert ytr.tolist() == [0, 10, 61] and yte.tolist() == [5, 36]
    np.testing.assert_allclose(xtr[0, :, :, 0], tr_x[0].T / 255.0)
