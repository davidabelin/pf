"""Train a baseline Polyfolds classifier from a unified manifest.

Role
----
CLI entrypoint for the first reproducible Polyfolds classifier training pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polyfolds_ml.training import ClassifierTrainConfig, train_classifier_baseline


def main() -> int:
    """Parse CLI arguments, train the baseline classifier, and print metrics."""

    parser = argparse.ArgumentParser(description="Train a baseline Polyfolds classifier.")
    parser.add_argument("--manifest", required=True, help="Input manifest JSON path.")
    parser.add_argument("--artifact", required=True, help="Output pickle artifact path.")
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "nn"])
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-layers", default="128,64", help="Comma-separated hidden layer sizes for model-type=nn")
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()

    hidden_layers = tuple(int(token) for token in str(args.hidden_layers).split(",") if token.strip())
    metrics = train_classifier_baseline(
        ClassifierTrainConfig(
            manifest_path=str(Path(args.manifest).resolve()),
            artifact_path=str(Path(args.artifact).resolve()),
            model_type=str(args.model_type),
            image_size=int(args.image_size),
            hidden_layer_sizes=hidden_layers or (128, 64),
            max_iter=int(args.max_iter),
        )
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
