"""Resumable experiment driver for the OCR notebook (CPU TensorFlow).

Stages (each result is cached as JSON under experiments/results/<stage>/, so the
script can be interrupted and re-run without repeating finished work):

1. epochs      - how accuracy changes with the number of epochs, per model family
2. depth       - number of hidden layers (and widths / residual blocks), per family
3. comparison  - the best configuration of each family, repeated over 3 seeds
4. final       - the winning family trained on the full training set

Model selection uses validation accuracy only; test accuracy is reporting-only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hcr.sweeps import depth_sweep_configs, epoch_sweep_configs, load_results, run_experiment
from hcr.unified import load_unified
from hcr.zoo import FAMILIES

RESULTS = "experiments/results"
SWEEP_TRAIN = 80_000  # subset size for sweeps (keeps CPU time tractable)
COMPARE_TRAIN = 200_000
BASE = {"batch_size": 256, "dropout": 0.3, "lr": 1e-3, "units": 128, "seed": 0, "augment": True}


def _run(cfgs, data, stage, log):
    for i, cfg in enumerate(cfgs, 1):
        t = time.time()
        res = run_experiment(cfg, data, out_dir=f"{RESULTS}/{stage}")
        log(
            f"[{stage} {i}/{len(cfgs)}] {cfg['family']} e={cfg['epochs']} h={cfg['hidden_layers']} "
            f"u={cfg['units']} b={cfg.get('blocks')} seed={cfg['seed']} "
            f"-> val={res['val_acc']:.4f} test={res['test_acc']:.4f} ({time.time() - t:.0f}s)"
        )


def stage_epochs(data, log):
    cfgs = []
    for family in FAMILIES:
        base = {
            **BASE,
            "family": family,
            "hidden_layers": 2,
            "blocks": 2,
            "train_limit": SWEEP_TRAIN,
        }
        cfgs += list(epoch_sweep_configs(base, [2, 5, 10, 20]))
    _run(cfgs, data, "epochs", log)


def stage_depth(data, log):
    base = {**BASE, "epochs": 8, "blocks": 2, "train_limit": SWEEP_TRAIN}
    cfgs = list(
        depth_sweep_configs(
            {**base, "family": "mlp", "epochs": 10}, [0, 1, 2, 3, 4, 5], [64, 128, 256, 512]
        )
    )
    for family in ("cnn", "cnn_bilstm"):
        cfgs += list(depth_sweep_configs({**base, "family": family}, [0, 1, 2, 3], [128]))
    for blocks in (1, 2, 3, 4):
        cfgs.append({**base, "family": "resnet", "hidden_layers": 1, "blocks": blocks})
    _run(cfgs, data, "depth", log)


def best_per_family():
    best = {}
    for r in load_results(f"{RESULTS}/depth"):
        fam = r["cfg"]["family"]
        if fam not in best or r["val_acc"] > best[fam]["val_acc"]:
            best[fam] = r
    return best


def stage_comparison(data, log):
    cfgs = []
    for r in best_per_family().values():
        for seed in (0, 1, 2):
            cfgs.append({**r["cfg"], "epochs": 12, "seed": seed, "train_limit": COMPARE_TRAIN})
    _run(cfgs, data, "comparison", log)


def stage_final(data, log):
    rows = load_results(f"{RESULTS}/comparison")
    by_fam = {}
    for r in rows:
        by_fam.setdefault(r["cfg"]["family"], []).append(r["val_acc"])
    winner = max(by_fam, key=lambda f: sum(by_fam[f]) / len(by_fam[f]))
    cfg = {**best_per_family()[winner]["cfg"], "epochs": 10, "seed": 0, "train_limit": None}
    log(f"final winner (mean val over seeds): {winner}")
    res = run_experiment(cfg, data, out_dir=f"{RESULTS}/final", model_path="models/ocr_best.keras")
    Path("models/ocr_best.json").write_text(json.dumps({"classes": data.classes, "cfg": cfg}))
    # Keep exactly one result in final/: a leftover file from an earlier winner would otherwise be
    # picked up by the notebook and shown next to the newer model.
    for path in (Path(RESULTS) / "final").glob("*.json"):
        if json.loads(path.read_text())["cfg"] != res["cfg"]:
            path.unlink()
    log(f"final {winner}: val={res['val_acc']:.4f} test={res['test_acc']:.4f}")


STAGES = {
    "epochs": stage_epochs,
    "depth": stage_depth,
    "comparison": stage_comparison,
    "final": stage_final,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=list(STAGES))
    args = ap.parse_args()
    data = load_unified()
    for name in args.stages:
        STAGES[name](data, lambda m: print(m, flush=True))
