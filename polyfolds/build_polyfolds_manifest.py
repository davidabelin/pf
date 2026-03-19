"""Build a unified Polyfolds manifest from legacy or canonical dataset folders.

Role
----
CLI entrypoint for the first normalization step in the offline Polyfolds
workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polyfolds_ml.manifest import build_manifest, summarize_manifest


def main() -> int:
    """Parse CLI arguments, build the manifest, print a compact summary."""

    parser = argparse.ArgumentParser(description="Build a unified manifest from legacy or canonical Polyfolds datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset folder containing labels.jsonl or samples.jsonl. Repeat for multiple roots.",
    )
    parser.add_argument("--output", required=True, help="Output JSON manifest path.")
    parser.add_argument("--name", default="polyfolds_v1", help="Manifest dataset name.")
    args = parser.parse_args()

    payload = build_manifest(
        [Path(item).resolve() for item in args.dataset],
        output_path=Path(args.output).resolve(),
        dataset_name=str(args.name),
    )
    print(json.dumps(summarize_manifest(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
