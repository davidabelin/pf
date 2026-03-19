from __future__ import annotations

import json
from pathlib import Path

from canonical_dataset import CanonicalBuildConfig, build_canonical_dataset
from polyfolds_ml.manifest import build_manifest


def test_canonical_dataset_builds_and_manifest_roundtrips(tmp_path: Path):
    dataset_dir = tmp_path / "canonical_core"
    payload = build_canonical_dataset(
        CanonicalBuildConfig(
            out_dir=str(dataset_dir),
            solids=("hexa",),
            test_mode=True,
            max_attempts_per_solid=5000,
        )
    )

    assert payload["manifest"]["dataset_kind"] == "canonical"
    assert (dataset_dir / "samples.jsonl").exists()
    assert (dataset_dir / "dataset_manifest.json").exists()

    sample_rows = [json.loads(line) for line in (dataset_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(sample_rows) == 9
    assert all(row["source_kind"] == "canonical" for row in sample_rows)
    assert all(row["joint_label"].startswith("hexa:") for row in sample_rows)
    assert all(Path(row["vector_json_path"]).exists() for row in sample_rows)
    assert all(Path(row["canonical_svg_path"]).exists() for row in sample_rows)

    manifest_path = tmp_path / "manifest.json"
    manifest_payload = build_manifest([dataset_dir], output_path=manifest_path, dataset_name="canonical_hexa_smoke")
    assert manifest_payload["manifest"]["dataset_kind"] == "canonical"
    assert manifest_payload["manifest"]["coverage_kind"] == "sampled"
    assert manifest_payload["manifest"]["label_space_version"] == "solid_state_v1"
    assert manifest_payload["manifest"]["sample_count"] == 9
    assert manifest_path.exists()
