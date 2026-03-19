"""Print the current Polyfolds model and training specs as JSON."""

from __future__ import annotations

import argparse
import json

from polyfolds_ml.architecture import default_classifier_spec, default_classifier_training_spec, default_repair_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show the current Polyfolds model and training specs.")
    parser.add_argument("--which", choices=("all", "classifier", "repair"), default="all")
    args = parser.parse_args(argv)

    payload: dict[str, object] = {}
    if args.which in {"all", "classifier"}:
        payload["classifier"] = {
            "model": default_classifier_spec().to_dict(),
            "training": default_classifier_training_spec().to_dict(),
        }
    if args.which in {"all", "repair"}:
        payload["repair"] = {"model": default_repair_spec().to_dict()}

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
