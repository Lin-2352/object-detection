"""Dataset loading and preprocessing for MNIST and EMNIST.

The official EMNIST IDX files store each image transposed relative to a
normal row-major bitmap (a quirk inherited from the NIST conversion
pipeline); :func:`fix_emnist_orientation` undoes that so images come out
upright, matching MNIST's orientation.
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from keras.datasets import mnist as _keras_mnist

EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

_IDX_IMAGE_MAGIC = 2051
_IDX_LABEL_MAGIC = 2049


def _data_dir() -> Path:
    return Path(os.environ.get("HCR_DATA_DIR", str(Path.home() / ".cache" / "hcr")))


def _archive_path() -> Path:
    return _data_dir() / "gzip.zip"


def _ensure_archive() -> Path:
    """Return the path to the EMNIST archive, downloading it if missing."""
    path = _archive_path()
    if path.exists():
        return path
    data_dir = path.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(EMNIST_URL, path)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear message below
        raise RuntimeError(
            f"EMNIST archive not found at {path} and automatic download failed "
            f"({exc}). Download it manually from {EMNIST_URL} and save it as "
            f"{path}, or point HCR_DATA_DIR at a directory that already "
            "contains gzip.zip."
        ) from exc
    return path


def parse_idx_images(buf: bytes) -> np.ndarray:
    """Parse an uncompressed IDX3 image file into a (N, rows, cols) uint8 array."""
    magic, num, rows, cols = struct.unpack(">4I", buf[:16])
    if magic != _IDX_IMAGE_MAGIC:
        raise ValueError(f"unexpected IDX image magic number: {magic}")
    data = np.frombuffer(buf, dtype=np.uint8, offset=16)
    return data.reshape(num, rows, cols)


def parse_idx_labels(buf: bytes) -> np.ndarray:
    """Parse an uncompressed IDX1 label file into a (N,) uint8 array."""
    magic, num = struct.unpack(">2I", buf[:8])
    if magic != _IDX_LABEL_MAGIC:
        raise ValueError(f"unexpected IDX label magic number: {magic}")
    return np.frombuffer(buf, dtype=np.uint8, offset=8).reshape(num)


def fix_emnist_orientation(images: np.ndarray) -> np.ndarray:
    """Undo the row/column transpose used in the official EMNIST IDX files."""
    return np.transpose(images, (0, 2, 1))


def parse_balanced_mapping(text: str) -> dict[int, int]:
    """Parse an emnist-balanced-mapping.txt file into {label: ascii_code}."""
    mapping: dict[int, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        label_str, ascii_str = line.split()
        mapping[int(label_str)] = int(ascii_str)
    return mapping


def _read_gz_member(zf: zipfile.ZipFile, name: str) -> bytes:
    with zf.open(name) as fh:
        return gzip.decompress(fh.read())


def _load_emnist_split(
    zf: zipfile.ZipFile, dataset: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
    images_buf = _read_gz_member(zf, f"gzip/{dataset}-{split}-images-idx3-ubyte.gz")
    labels_buf = _read_gz_member(zf, f"gzip/{dataset}-{split}-labels-idx1-ubyte.gz")
    images = fix_emnist_orientation(parse_idx_images(images_buf))
    labels = parse_idx_labels(labels_buf).astype(np.int64)
    return images, labels


def _to_float_unit(images: np.ndarray) -> np.ndarray:
    return images.astype("float32")[..., np.newaxis] / 255.0


def _load_mnist():
    (x_train, y_train), (x_test, y_test) = _keras_mnist.load_data()
    x_train = _to_float_unit(x_train)
    x_test = _to_float_unit(x_test)
    classes = [str(i) for i in range(10)]
    return (x_train, y_train.astype(np.int64)), (x_test, y_test.astype(np.int64)), classes


def _load_emnist_letters():
    """Load the EMNIST 'letters' split.

    This split merges upper- and lower-case examples of each letter into a
    single 26-way label, so a classifier trained on it cannot distinguish
    case; it only predicts which of A-Z a glyph represents.
    """
    archive = _ensure_archive()
    with zipfile.ZipFile(archive) as zf:
        x_train, y_train = _load_emnist_split(zf, "emnist-letters", "train")
        x_test, y_test = _load_emnist_split(zf, "emnist-letters", "test")
    # Labels in the file are 1..26; shift to 0..25 to index into classes.
    y_train = y_train - 1
    y_test = y_test - 1
    classes = [chr(ord("A") + i) for i in range(26)]
    return (
        (_to_float_unit(x_train), y_train),
        (_to_float_unit(x_test), y_test),
        classes,
    )


def _load_emnist_balanced():
    archive = _ensure_archive()
    with zipfile.ZipFile(archive) as zf:
        mapping_text = zf.read("gzip/emnist-balanced-mapping.txt").decode("ascii")
        x_train, y_train = _load_emnist_split(zf, "emnist-balanced", "train")
        x_test, y_test = _load_emnist_split(zf, "emnist-balanced", "test")
    mapping = parse_balanced_mapping(mapping_text)
    num_classes = max(mapping) + 1
    classes = [chr(mapping[i]) for i in range(num_classes)]
    return (
        (_to_float_unit(x_train), y_train),
        (_to_float_unit(x_test), y_test),
        classes,
    )


_LOADERS = {
    "mnist": _load_mnist,
    "emnist-letters": _load_emnist_letters,
    "emnist-balanced": _load_emnist_balanced,
}


def load_dataset(name: str):
    """Load a dataset by name: 'mnist', 'emnist-letters' or 'emnist-balanced'.

    Returns ((x_train, y_train), (x_test, y_test), classes) where the x
    arrays are float32 of shape (N, 28, 28, 1) scaled to [0, 1] with white
    strokes on a black background in upright orientation, the y arrays hold
    int labels in [0, len(classes)), and classes is a list of single
    character strings mapping a label index to the character it represents.
    """
    try:
        loader = _LOADERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(_LOADERS)}") from exc
    return loader()
