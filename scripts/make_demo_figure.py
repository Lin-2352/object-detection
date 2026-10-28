"""Render a figure of the preprocessing pipeline and predictions for sample images.

Run from the repository root with:
    python -m scripts.make_demo_figure IMAGE [IMAGE ...] \
        --model models/emnist_balanced_cnn.keras --out docs/pipeline.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hcr.predict import Predictor  # noqa: E402
from hcr.preprocess import load_image, to_model_input  # noqa: E402

DEFAULT_OUT = "docs/pipeline.png"


def _prediction_text(results: list[tuple[str, float]]) -> str:
    return "\n".join(f"{char}: {prob:.3f}" for char, prob in results)


def build_figure(images: list[str], predictor: Predictor):
    """One column per image: raw input, 28x28 model input, top-3 predictions."""
    n = len(images)
    fig, axes = plt.subplots(
        3, n, figsize=(1.7 * n, 4.4), squeeze=False, gridspec_kw={"height_ratios": [1, 1, 0.5]}
    )

    for col, image_path in enumerate(images):
        gray = load_image(image_path)
        model_input = to_model_input(gray)
        results = predictor.predict(gray, top_k=3)

        ax_raw, ax_input, ax_text = axes[:, col]

        ax_raw.imshow(gray, cmap="gray")
        ax_raw.set_title(Path(image_path).stem, fontsize=7)
        ax_raw.axis("off")

        ax_input.imshow(model_input, cmap="gray")
        ax_input.axis("off")

        ax_text.axis("off")
        ax_text.text(0.5, 0.95, _prediction_text(results), va="top", ha="center", fontsize=9)

    axes[0, 0].text(-0.15, 0.5, "raw", transform=axes[0, 0].transAxes, rotation=90, va="center")
    axes[1, 0].text(
        -0.15, 0.5, "model input", transform=axes[1, 0].transAxes, rotation=90, va="center"
    )
    axes[2, 0].text(-0.15, 0.7, "top-3", transform=axes[2, 0].transAxes, rotation=90, va="center")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the preprocessing pipeline figure.")
    parser.add_argument("images", nargs="+", help="Image file paths to include in the figure.")
    parser.add_argument("--model", default="models/emnist_balanced_cnn.keras")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    predictor = Predictor(args.model)
    fig = build_figure(args.images, predictor)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
