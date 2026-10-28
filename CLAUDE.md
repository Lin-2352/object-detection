# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements-dev.txt        # tensorflow 2.20, keras 3.15, pandas, opencv, pytest, ruff, nbconvert
pytest -m "not slow"                       # unit tests (synthetic data only); plain `pytest` also runs the slow real-data test if data/ exists
pytest tests/test_zoo.py::test_name        # single test
ruff check . && ruff format --check .      # CI runs both, then pytest (Python 3.12, CPU); notebooks/ is excluded from ruff

python scripts/run_experiments.py [epochs depth comparison final]   # resumable; results cached in experiments/results/<stage>/
python scripts/bn_momentum_check.py                                  # writes experiments/bn_momentum_check.json (quoted in the notebook)
python scripts/build_notebook.py                                     # regenerates notebooks/OCR_Keras_TF.ipynb (unexecuted)
python -m nbconvert --to notebook --execute --inplace notebooks/OCR_Keras_TF.ipynb   # `python -m jupyter` does not work here

python -m hcr.predict samples/digit_1.png --model models/ocr_best.keras   # 94-class model; default model is the old 47-class EMNIST-balanced one
python -m hcr.train --dataset emnist-byclass --arch cnn --epochs 15       # legacy single-dataset trainer (cnn | mlp only)
```

Ruff: line length 100, rules E, F, I, B, UP. `pyproject.toml` has no `[build-system]`, so the package is not pip-installable; run from the repo root.

## Architecture

There are two pipelines that share `hcr/preprocess.py` and `hcr/predict.py`.

**Legacy per-dataset pipeline** (MNIST, EMNIST balanced/letters/byclass, synthetic keyboard-symbols): `hcr/data.py::load_dataset` -> `hcr/model.py` -> `hcr/train.py::run_training` (writes `models/{dataset}_{arch}.keras` + JSON sidecar and plots in `reports/`). `dataset_names()` feeds the CLI choices, so new loaders appear there automatically.

**94-class pipeline** (what the notebook uses): `hcr/unified.py::load_unified` merges EMNIST ByClass (`hcr/data.py`), HASYv2 handwritten symbols (`hcr/hasy.py`) and rendered system fonts (`hcr/fonts.py`) into one 94-class cache (`data/processed/unified.npz`, ~750 MB) -> `hcr/zoo.py::build_model(family, ...)` builds `mlp | cnn | resnet | cnn_bilstm` from a few hyper-parameters -> `hcr/sweeps.py::run_experiment(cfg, data)` trains one config and writes a JSON (history, val/test accuracy, params, timing) -> `hcr/evaluate.py` (group accuracy, merged look-alike accuracy, confusions, F1) -> `scripts/run_experiments.py` (epoch sweep, hidden-layer sweep, 3-seed comparison, final full-data model saved to `models/ocr_best.keras` + `.json`).

`notebooks/OCR_Keras_TF.ipynb` is **generated**: edit `scripts/build_notebook.py`, never the `.ipynb`. Its prose is computed from `experiments/results/*.json`, so re-run the experiments before re-executing it.

## Things that are easy to get wrong

- Model selection must use validation accuracy only. Test accuracy is stored in every result JSON for reporting but nothing may be chosen with it.
- `run_experiment` caches on `cfg` only. Changing `hcr/zoo.py`, the data or the unified-dataset parameters does **not** invalidate `experiments/results/`; delete the folder (or the affected stage) yourself. Unknown cfg keys raise.
- `load_unified` caches on a code version only. `font_dir`, `hasy_dir`, installed fonts or `seed` do not trigger a rebuild, and `rebuild=True` overwrites the cache. The printed glyphs come from `C:\Windows\Fonts`, so the dataset differs per machine.
- Data locations: EMNIST archive in `data/raw/emnist/gzip.zip` (auto-downloaded from NIST, ~560 MB; `HCR_DATA_DIR` overrides), HASYv2 in `data/raw/hasy/HASYv2.tar.bz2` (never downloaded by code, get it from zenodo.org/records/259444). `data/` is git-ignored except `data/README.md`.
- All BatchNormalization layers in `hcr/zoo.py` use momentum 0.9 on purpose; the Keras default 0.99 left validation accuracy near chance in short runs (see `experiments/bn_momentum_check.json` and `tests/test_zoo.py`).
- Class look-alikes (c/C, o/O/0, l/I/1/|, ...) are listed in `hcr.unified.AMBIGUOUS_GROUPS`; "merged" accuracy only merges those groups.
- The font split is by file, not family, so printed-font test accuracy is slightly optimistic. Only 17 of the 32 symbols have real handwriting (HASYv2), and for several of those printed fonts dominate training.
- The old `keyboard-symbols` dataset draws every symbol in one font, so its 100 % accuracy is meaningless.
- TensorFlow >= 2.11 has no native-Windows GPU support and WSL2/Docker cannot start on this machine (virtualisation is off in firmware), so everything trains on CPU.
- Tests use synthetic data; the only exception is one `@pytest.mark.slow` integration test that skips itself when the real data is absent.
