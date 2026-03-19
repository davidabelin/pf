from __future__ import annotations

import json
from pathlib import Path

from polyfolds_ml.training import ClassifierTrainConfig, train_classifier_baseline


def _square(offset: float) -> list[tuple[float, float]]:
    return [
        (offset + 0.0, 0.0),
        (offset + 1.0, 0.0),
        (offset + 1.0, 1.0),
        (offset + 0.0, 1.0),
    ]


def test_torch_training_smoke(tmp_path: Path):
    solids = ["tetra", "hexa", "octa", "dodeca", "icosa"]
    states = ["valid", "incomplete", "invalid"]
    samples = []
    label_index = 0
    for solid in solids:
        for state in states:
            for split in ("train", "val", "test"):
                offset = float(label_index) * 0.05
                samples.append(
                    {
                        "sample_id": f"{solid}_{state}_{split}",
                        "split": split,
                        "class_label": state,
                        "solid": solid,
                        "source_dataset": "synthetic_smoke",
                        "state": state,
                        "joint_label": f"{solid}:{state}",
                        "topology_hash": f"{solid}_{state}",
                        "vector_json_path": None,
                        "canonical_svg_path": None,
                        "render_profile_id": "neutral_v1",
                        "source_kind": "canonical",
                        "vector_faces": [
                            {
                                "face_index": 0,
                                "polygon": _square(offset),
                                "present": state != "incomplete",
                                "edge_group": None,
                            }
                        ],
                        "vector_edges": [],
                        "repair_target": None,
                        "metadata": {"generation_mode": "sampled"},
                        "schema_version": 2,
                    }
                )
            label_index += 1

    manifest_path = tmp_path / "manifest.json"
    manifest_payload = {
        "manifest": {
            "dataset_name": "synthetic_smoke",
            "schema_version": 2,
            "created_at": "2026-03-18T00:00:00+00:00",
            "source_roots": [str(tmp_path)],
            "classes": sorted({sample["joint_label"] for sample in samples}),
            "solids": solids,
            "sample_count": len(samples),
            "dataset_kind": "canonical",
            "coverage_kind": "sampled",
            "label_space_version": "solid_state_v1",
            "notes": [],
        },
        "samples": samples,
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    artifact_path = tmp_path / "classifier.pt"
    metrics = train_classifier_baseline(
        ClassifierTrainConfig(
            manifest_path=str(manifest_path),
            artifact_path=str(artifact_path),
            image_size=96,
            batch_size=8,
            epochs=1,
            patience=1,
        )
    )

    assert artifact_path.exists()
    assert metrics["sample_count"] == len(samples)
    assert len(metrics["labels"]) == 15
    assert metrics["artifact_path"].endswith("classifier.pt")
