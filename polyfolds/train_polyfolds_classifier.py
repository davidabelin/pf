"""Train the shared Polyfolds CNN classifier from a unified manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polyfolds_ml.training import ClassifierTrainConfig, train_classifier_baseline


def main() -> int:
    """Parse CLI arguments, train the shared classifier, and print metrics."""

    parser = argparse.ArgumentParser(description="Train the shared Polyfolds CNN classifier.")
    parser.add_argument("--manifest", required=True, help="Input manifest JSON path.")
    parser.add_argument("--artifact", required=True, help="Output model artifact path (.pt).")
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train_classifier_baseline(
        ClassifierTrainConfig(
            manifest_path=str(Path(args.manifest).resolve()),
            artifact_path=str(Path(args.artifact).resolve()),
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            patience=int(args.patience),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
        )
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
