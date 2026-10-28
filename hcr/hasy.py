"""HASYv2 handwritten-symbol loader restricted to keyboard characters.

HASYv2 (Thoma 2017, arXiv:1701.08380, CC-BY-SA 4.0) holds 168k 32x32 images of
369 LaTeX symbols with a writer id per sample. Only symbols that are genuine
visual matches for an ASCII keyboard symbol are used (see KEYBOARD_LATEX_MAP);
nothing is mapped by loose analogy (e.g. ``\\wedge`` is not ``^`` and
``\\cdot`` is not ``.``).
"""

from __future__ import annotations

import csv
import io
import tarfile
from pathlib import Path

import cv2
import numpy as np

from hcr.preprocess import to_model_input
from hcr.symbols import KEYBOARD_SYMBOLS

# keyboard char -> (HASYv2 symbol_id, latex name), verified against symbols.csv.
KEYBOARD_LATEX_MAP: dict[str, tuple[int, str]] = {
    "-": (195, "-"),
    "+": (196, "+"),
    "<": (187, "<"),
    ">": (188, ">"),
    "$": (254, "\\$"),
    "{": (257, "\\{"),
    "}": (261, "\\}"),
    "&": (265, "\\&"),
    "#": (266, "\\#"),
    "%": (267, "\\%"),
    "[": (918, "["),
    "]": (919, "]"),
    "|": (920, "|"),
    "/": (922, "/"),
    "\\": (949, "\\backslash"),
    "*": (514, "\\ast"),
    "~": (618, "\\sim"),
}

COVERAGE: dict[str, list[str]] = {
    "covered": [c for c in KEYBOARD_SYMBOLS if c in KEYBOARD_LATEX_MAP],
    "missing": [c for c in KEYBOARD_SYMBOLS if c not in KEYBOARD_LATEX_MAP],
}

_TAR_NAME = "HASYv2.tar.bz2"
_WRITER_PIN_FRACTION = 0.05


class _DirPngs:
    def __init__(self, root: Path):
        self.root = root

    def __getitem__(self, key: str) -> bytes:
        return (self.root / key).read_bytes()


def _read_sources(raw_dir: Path):
    """Return (symbols_csv_text, labels_csv_text, mapping path -> png bytes)."""
    tar_path = raw_dir / _TAR_NAME
    if tar_path.exists():
        with tarfile.open(tar_path, "r:bz2") as tf:
            symbols = tf.extractfile("symbols.csv").read().decode("utf8")
            labels = tf.extractfile("hasy-data-labels.csv").read().decode("utf8")
            pngs: dict[str, bytes] = {}
            for member in tf:
                if member.name.startswith("hasy-data/") and member.name.endswith(".png"):
                    pngs[member.name] = tf.extractfile(member).read()
        return symbols, labels, pngs
    if (raw_dir / "hasy-data-labels.csv").exists():
        symbols = (raw_dir / "symbols.csv").read_text(encoding="utf8")
        labels = (raw_dir / "hasy-data-labels.csv").read_text(encoding="utf8")
        return symbols, labels, _DirPngs(raw_dir)
    raise FileNotFoundError(f"no HASYv2 data found in {raw_dir} (expected {_TAR_NAME})")


def load_hasy_symbols(raw_dir: str | Path = "data/raw/hasy"):
    """Load HASYv2 samples of the keyboard symbols in KEYBOARD_LATEX_MAP.

    Returns ``(images, labels, writers)``: uint8 (N, 28, 28) white-ink-on-black
    images normalised exactly like model input (inverted, cropped to ink,
    scaled and centred via :func:`hcr.preprocess.to_model_input`), a str array
    of single-character labels and an int64 array of HASYv2 writer ids.
    """
    symbols_text, labels_text, pngs = _read_sources(Path(raw_dir))
    latex_by_id = {
        int(r["symbol_id"]): r["latex"] for r in csv.DictReader(io.StringIO(symbols_text))
    }
    char_by_id: dict[int, str] = {}
    for char, (sid, latex) in KEYBOARD_LATEX_MAP.items():
        if sid in latex_by_id and latex_by_id[sid] != latex:
            raise ValueError(
                f"HASYv2 symbol {sid} has latex {latex_by_id[sid]!r}, expected {latex!r}"
            )
        char_by_id[sid] = char

    images, labels, writers = [], [], []
    for row in csv.DictReader(io.StringIO(labels_text)):
        sid = int(row["symbol_id"])
        if sid not in char_by_id:
            continue
        buf = np.frombuffer(pngs[row["path"]], dtype=np.uint8)
        gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        try:
            model_input = to_model_input(gray)
        except ValueError:  # blank image
            continue
        images.append((model_input * 255.0).round().astype(np.uint8))
        labels.append(char_by_id[sid])
        writers.append(int(row["user_id"]))
    if not images:
        return (
            np.zeros((0, 28, 28), np.uint8),
            np.array([], dtype="<U1"),
            np.zeros(0, np.int64),
        )
    return np.stack(images), np.array(labels, dtype="<U1"), np.array(writers, dtype=np.int64)


def split_writers(writers: np.ndarray, test_fraction: float = 0.2, seed: int = 42) -> np.ndarray:
    """Boolean test mask such that no writer appears in both train and test.

    Writers owning more than 5% of all samples (HASYv2 has one aggregate id,
    16925, holding most symbols) cannot be split and are pinned to train.
    """
    writers = np.asarray(writers)
    ids, counts = np.unique(writers, return_counts=True)
    pinned = ids[counts > _WRITER_PIN_FRACTION * len(writers)]
    candidates = ids[~np.isin(ids, pinned)]
    rng = np.random.RandomState(seed)
    n_test = int(round(test_fraction * len(candidates)))
    test_writers = rng.permutation(candidates)[:n_test]
    return np.isin(writers, test_writers)
