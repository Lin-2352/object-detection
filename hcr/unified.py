"""Leak-free unified 94-class character dataset (handwritten + printed).

Sources (``source_*`` codes): 0 = EMNIST byclass (handwritten alphanumerics),
1 = HASYv2 (handwritten keyboard symbols it genuinely contains), 2 = rendered
system fonts (printed; fills the symbol classes HASYv2 lacks or has few of).

Leakage control
---------------
* EMNIST keeps its official train/test split; validation is a stratified 10% of
  EMNIST train (EMNIST has no writer ids, so val shares writers with train).
* HASYv2 is split by writer id (test ~20% of writers, then val ~10% of the
  remaining writers). A writer owning more than 5% of the mapped samples
  (HASYv2's aggregate id 16925) cannot be split and is pinned to train.
* Fonts are split by font file: ~20% test fonts, then ~10% of the remaining
  fonts for validation. A font never appears in two splits.
* Finally every val/test sample whose exact pixels also occur anywhere in train
  is dropped (counts are kept in ``meta["removed_duplicates"]``).

Balancing
---------
Alphanumeric classes get ``fonts_per_class`` printed train samples on top of
EMNIST. Symbol classes are topped up with augmented font renders until each has
at least ``symbol_train_target`` train samples counting its HASYv2 samples
(``max(fonts_per_class, target - n_hasy_train)`` renders). Val gets 10% and test
``fonts_per_class // 4`` renders per class, from the held-out fonts. EMNIST
class imbalance is left untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hcr import data as _data
from hcr.fonts import ASCII_CHARS, discover_fonts, render_font_glyphs, split_fonts
from hcr.hasy import load_hasy_symbols, split_writers
from hcr.symbols import KEYBOARD_SYMBOLS

SOURCE_EMNIST, SOURCE_HASY, SOURCE_FONT = 0, 1, 2
CACHE_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SPLITS = ("train", "val", "test")

AMBIGUOUS_GROUPS: tuple[tuple[str, ...], ...] = (
    tuple("cC"),
    tuple("kK"),
    tuple("mM"),
    tuple("oO0"),
    tuple("pP"),
    tuple("sS"),
    tuple("uU"),
    tuple("vV"),
    tuple("wW"),
    tuple("xX"),
    tuple("yY"),
    tuple("zZ"),
    tuple("lI1|"),
)


@dataclass
class Unified:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    classes: list[str]
    source_train: np.ndarray
    source_val: np.ndarray
    source_test: np.ndarray
    meta: dict = field(default_factory=dict)


def class_groups(classes: list[str]) -> dict[str, list[int]]:
    """Indices of digits, upper, lower and keyboard-symbol classes."""
    groups: dict[str, list[int]] = {"digits": [], "upper": [], "lower": [], "symbols": []}
    for i, char in enumerate(classes):
        if char.isdigit():
            groups["digits"].append(i)
        elif char.isupper():
            groups["upper"].append(i)
        elif char.islower():
            groups["lower"].append(i)
        elif char in KEYBOARD_SYMBOLS:
            groups["symbols"].append(i)
    return groups


def ambiguous_merge_map(classes: list[str]) -> np.ndarray:
    """Map each class index to the index of its AMBIGUOUS_GROUPS representative."""
    mapping = np.arange(len(classes), dtype=np.int64)
    for group in AMBIGUOUS_GROUPS:
        members = [classes.index(c) for c in group if c in classes]
        for m in members:
            mapping[m] = members[0]
    return mapping


def _to_uint8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x.reshape(len(x), -1)
    return np.rint(x.reshape(len(x), -1) * 255.0).astype(np.uint8)


def pixel_hashes(x: np.ndarray) -> list[bytes]:
    """Hash every image's 8-bit pixels (accepts uint8 or float32 in [0, 1])."""
    flat = np.ascontiguousarray(_to_uint8(x))
    return [hashlib.blake2b(row.tobytes(), digest_size=16).digest() for row in flat]


def _to_float(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 255.0)[..., np.newaxis]


def _stratified_val_mask(y: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(len(y), dtype=bool)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        n_val = int(round(fraction * len(idx)))
        mask[rng.permutation(idx)[:n_val]] = True
    return mask


def _render_counts(counts: dict[str, int], fonts, rng):
    """Render exactly ``counts[char]`` samples per char, spread over all given fonts."""
    images, labels = [], []
    for char, n in counts.items():
        if n <= 0:
            continue
        per_font = math.ceil(n / len(fonts))
        imgs, _, _ = render_font_glyphs([char], fonts, per_font, rng)
        keep = rng.permutation(len(imgs))[:n]
        images.append(imgs[keep])
        labels += [char] * n
    if not images:
        return np.zeros((0, 28, 28), np.uint8), np.array([], dtype="<U1")
    return np.concatenate(images), np.array(labels, dtype="<U1")


def build_unified(
    emnist_train,
    emnist_test,
    emnist_classes: list[str],
    hasy,
    fonts,
    seed: int = 42,
    fonts_per_class: int = 300,
    symbol_train_target: int = 3000,
) -> Unified:
    """Assemble the unified dataset from already-loaded sources (see module docstring).

    ``emnist_*`` are (uint8 images (N,28,28), int labels indexing ``emnist_classes``),
    ``hasy`` is the output of :func:`hcr.hasy.load_hasy_symbols`, ``fonts`` a list of paths.
    """
    rng = np.random.default_rng(seed)
    classes = list(ASCII_CHARS)
    index = {c: i for i, c in enumerate(classes)}
    lut = np.array([index[c] for c in emnist_classes], dtype=np.int64)

    parts: dict[str, list[tuple[np.ndarray, np.ndarray, int]]] = {s: [] for s in _SPLITS}

    # EMNIST
    tr_x, tr_y = emnist_train
    te_x, te_y = emnist_test
    tr_y, te_y = lut[np.asarray(tr_y)], lut[np.asarray(te_y)]
    val_mask = _stratified_val_mask(tr_y, 0.1, rng)
    parts["train"].append((tr_x[~val_mask], tr_y[~val_mask], SOURCE_EMNIST))
    parts["val"].append((tr_x[val_mask], tr_y[val_mask], SOURCE_EMNIST))
    parts["test"].append((te_x, te_y, SOURCE_EMNIST))

    # HASYv2, writer-disjoint
    h_x, h_chars, h_writers = hasy
    h_y = np.array([index[c] for c in h_chars], dtype=np.int64)
    test_mask = split_writers(h_writers, 0.2, seed)
    rest = np.flatnonzero(~test_mask)
    val_sub = split_writers(h_writers[rest], 0.1, seed + 1)
    val_mask_h = np.zeros(len(h_writers), dtype=bool)
    val_mask_h[rest[val_sub]] = True
    train_mask_h = ~(test_mask | val_mask_h)
    for name, m in (("train", train_mask_h), ("val", val_mask_h), ("test", test_mask)):
        parts[name].append((h_x[m], h_y[m], SOURCE_HASY))
    writers = {
        "train": sorted(set(h_writers[train_mask_h].tolist())),
        "val": sorted(set(h_writers[val_mask_h].tolist())),
        "test": sorted(set(h_writers[test_mask].tolist())),
    }

    # Fonts, font-disjoint
    test_fonts_all_but = split_fonts(fonts, seed)
    fonts_trainval, fonts_test = test_fonts_all_but
    fonts_train, fonts_val = split_fonts(fonts_trainval, seed + 1, 0.1)
    hasy_train_counts = np.bincount(h_y[train_mask_h], minlength=len(classes))
    train_counts, val_counts, test_counts = {}, {}, {}
    for char in classes:
        n = fonts_per_class
        if char in KEYBOARD_SYMBOLS:
            n = max(fonts_per_class, symbol_train_target - int(hasy_train_counts[index[char]]))
        train_counts[char] = n
        val_counts[char] = max(1, n // 10)
        test_counts[char] = max(1, fonts_per_class // 4)
    for name, counts, font_set in (
        ("train", train_counts, fonts_train),
        ("val", val_counts, fonts_val),
        ("test", test_counts, fonts_test),
    ):
        imgs, chars = _render_counts(counts, font_set, rng)
        parts[name].append((imgs, np.array([index[c] for c in chars], dtype=np.int64), SOURCE_FONT))

    # Exact-pixel de-duplication against train, then assemble.
    train_hashes: set[bytes] = set()
    for imgs, _, _ in parts["train"]:
        train_hashes.update(pixel_hashes(imgs))
    removed = {}
    arrays = {}
    for name in _SPLITS:
        xs, ys, ss = [], [], []
        removed[name] = 0
        for imgs, labels, source in parts[name]:
            if name != "train" and len(imgs):
                keep = np.array([h not in train_hashes for h in pixel_hashes(imgs)], dtype=bool)
                removed[name] += int((~keep).sum())
                imgs, labels = imgs[keep], labels[keep]
            xs.append(imgs)
            ys.append(labels)
            ss.append(np.full(len(imgs), source, dtype=np.int8))
        x, y, s = np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)
        order = rng.permutation(len(x))
        arrays[name] = (x[order], y[order].astype(np.int64), s[order])

    def names(paths):
        return [Path(p).name for p in paths]

    meta = {
        "params": {
            "version": CACHE_VERSION,
            "seed": seed,
            "fonts_per_class": fonts_per_class,
            "symbol_train_target": symbol_train_target,
        },
        "fonts": {
            "train": names(fonts_train),
            "val": names(fonts_val),
            "test": names(fonts_test),
        },
        "hasy_writers": writers,
        "removed_duplicates": removed,
    }
    return Unified(
        x_train=_to_float(arrays["train"][0]),
        y_train=arrays["train"][1],
        x_val=_to_float(arrays["val"][0]),
        y_val=arrays["val"][1],
        x_test=_to_float(arrays["test"][0]),
        y_test=arrays["test"][1],
        classes=classes,
        source_train=arrays["train"][2],
        source_val=arrays["val"][2],
        source_test=arrays["test"][2],
        meta=meta,
    )


def save_unified(unified: Unified, path: str | Path) -> None:
    """Write the dataset to an .npz (images stored as uint8, which is lossless here)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"classes": np.array(unified.classes), "meta": np.array(json.dumps(unified.meta))}
    for s in _SPLITS:
        payload[f"x_{s}"] = _to_uint8(getattr(unified, f"x_{s}")).reshape(-1, 28, 28)
        payload[f"y_{s}"] = getattr(unified, f"y_{s}")
        payload[f"source_{s}"] = getattr(unified, f"source_{s}")
    np.savez(path, **payload)


def _read_cache(path: Path) -> Unified:
    with np.load(path, allow_pickle=False) as z:
        kwargs = {}
        for s in _SPLITS:
            kwargs[f"x_{s}"] = _to_float(z[f"x_{s}"])
            kwargs[f"y_{s}"] = z[f"y_{s}"].astype(np.int64)
            kwargs[f"source_{s}"] = z[f"source_{s}"].astype(np.int8)
        return Unified(
            classes=[str(c) for c in z["classes"]],
            meta=json.loads(str(z["meta"])),
            **kwargs,
        )


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def _load_emnist_uint8():
    """EMNIST byclass as uint8 (upright), avoiding a float32 copy of ~800k images."""
    archive = _data._ensure_archive()
    with zipfile.ZipFile(archive) as zf:
        mapping = _data.parse_balanced_mapping(
            zf.read("gzip/emnist-byclass-mapping.txt").decode("ascii")
        )
        train = _data._load_emnist_split(zf, "emnist-byclass", "train")
        test = _data._load_emnist_split(zf, "emnist-byclass", "test")
    classes = [chr(mapping[i]) for i in range(max(mapping) + 1)]
    return train, test, classes


def load_unified(
    processed_dir: str | Path = "data/processed",
    seed: int = 42,
    rebuild: bool = False,
    fonts_per_class: int = 300,
    symbol_train_target: int = 3000,
    hasy_dir: str | Path = "data/raw/hasy",
    font_dir: str | Path = r"C:\Windows\Fonts",
) -> Unified:
    """Load the unified dataset from ``<processed_dir>/unified.npz``, building it if needed.

    The cache is reused only when its recorded seed / fonts_per_class /
    symbol_train_target match; otherwise (or with ``rebuild=True``) it is rebuilt
    from the raw EMNIST + HASYv2 downloads and the fonts in ``font_dir``.
    """
    cache = _resolve(processed_dir) / "unified.npz"
    params = {
        "version": CACHE_VERSION,
        "seed": seed,
        "fonts_per_class": fonts_per_class,
        "symbol_train_target": symbol_train_target,
    }
    if cache.exists() and not rebuild:
        cached = _read_cache(cache)
        if cached.meta.get("params") == params:
            return cached
    hasy = load_hasy_symbols(_resolve(hasy_dir))
    fonts = discover_fonts(font_dir)
    if len(fonts) < 3:
        raise RuntimeError(f"need at least 3 usable fonts in {font_dir}, found {len(fonts)}")
    train, test, emnist_classes = _load_emnist_uint8()
    unified = build_unified(
        train, test, emnist_classes, hasy, fonts, seed, fonts_per_class, symbol_train_target
    )
    save_unified(unified, cache)
    return unified
