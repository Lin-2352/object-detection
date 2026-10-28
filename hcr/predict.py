"""Predictor class and CLI for running trained handwritten character models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import keras
import numpy as np

from hcr.preprocess import load_image, to_model_input

DEFAULT_MODEL = "models/emnist_balanced_cnn.keras"

_TRAIN_HINT = "Run `python -m hcr.train ...` to train and save a model first."


def _sidecar_path(model_path: Path) -> Path:
    return model_path.with_suffix(".json")


class Predictor:
    """Loads a trained model and its sidecar metadata to predict characters."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL) -> None:
        self.model_path = Path(model_path)
        sidecar_path = _sidecar_path(self.model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}. {_TRAIN_HINT}")
        if not sidecar_path.exists():
            raise FileNotFoundError(f"model metadata not found: {sidecar_path}. {_TRAIN_HINT}")

        with open(sidecar_path, encoding="utf-8") as fh:
            metadata = json.load(fh)
        self.classes: list[str] = metadata["classes"]
        self.model = keras.models.load_model(self.model_path)

    def predict(self, gray_or_path, top_k: int = 3) -> list[tuple[str, float]]:
        if isinstance(gray_or_path, str | Path):
            gray = load_image(gray_or_path)
        else:
            gray = np.asarray(gray_or_path)

        model_input = to_model_input(gray)[None, ..., None]
        probs = np.asarray(self.model.predict(model_input, verbose=0))[0]

        top_k = min(top_k, len(probs))
        order = np.argsort(probs)[::-1][:top_k]
        return [(self.classes[i], float(probs[i])) for i in order]


def _format_result(name: str, results: list[tuple[str, float]]) -> str:
    top_char, top_prob = results[0]
    line = f"{name}  ->  {top_char} ({top_prob:.3f})"
    alternatives = ", ".join(f"{c} ({p:.3f})" for c, p in results[1:])
    if alternatives:
        line += f"   alternatives: {alternatives}"
    return line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict handwritten characters from images.")
    parser.add_argument("images", nargs="+", help="Image file paths to classify.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to a trained .keras model.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of alternatives to show.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    args = parser.parse_args(argv)

    try:
        predictor = Predictor(args.model)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output = []
    for image_path in args.images:
        try:
            results = predictor.predict(image_path, top_k=args.top_k)
        except ValueError as exc:
            print(f"{image_path}: {exc}", file=sys.stderr)
            return 1

        if args.json:
            output.append({"image": image_path, "predictions": results})
        else:
            print(_format_result(image_path, results))

    if args.json:
        print(json.dumps(output, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
