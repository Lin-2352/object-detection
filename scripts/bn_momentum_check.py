"""Re-run the batch-norm momentum check quoted in the notebook (Section 6).

Trains the plain CNN and the ResNet-style CNN for 5 epochs on 30,000 images with Keras'
default batch-norm momentum (0.99) and with the 0.9 used in ``hcr.zoo``, and stores the
per-epoch validation accuracy in experiments/bn_momentum_check.json.
"""

import json
import tempfile
from pathlib import Path

import keras

from hcr.sweeps import run_experiment
from hcr.unified import load_unified

OUT = Path("experiments/bn_momentum_check.json")
SETUP = {"epochs": 5, "train_limit": 30_000, "batch_size": 256, "seed": 0}


def run(family, momentum, data):
    original = keras.layers.BatchNormalization
    # hcr.zoo asks for momentum=0.9; override it so we can also measure the old 0.99 behaviour.
    keras.layers.BatchNormalization = lambda **kw: original(**{**kw, "momentum": momentum})
    try:
        res = run_experiment({"family": family, **SETUP}, data, out_dir=tempfile.mkdtemp())
    finally:
        keras.layers.BatchNormalization = original
    return {"val_acc": res["history"]["val_acc"], "train_acc": res["history"]["acc"]}


if __name__ == "__main__":
    data = load_unified()
    results = {
        family: {str(m): run(family, m, data) for m in (0.99, 0.9)} for family in ("cnn", "resnet")
    }
    OUT.write_text(json.dumps({"setup": SETUP, "results": results}, indent=2))
    for family, by_m in results.items():
        for m, r in by_m.items():
            print(family, m, [round(v, 3) for v in r["val_acc"]])
