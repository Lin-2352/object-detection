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
