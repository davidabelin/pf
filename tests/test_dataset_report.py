from __future__ import annotations

from pathlib import Path

from canonical_dataset import CanonicalBuildConfig, build_canonical_dataset
from dataset_report import build_dataset_report


def test_dataset_report_builds_summary_and_contact_sheet(tmp_path: Path):
    dataset_dir = tmp_path / "canonical_core"
    build_canonical_dataset(
        CanonicalBuildConfig(
            out_dir=str(dataset_dir),
            solids=("hexa",),
            test_mode=True,
            max_attempts_per_solid=5000,
        )
    )

    contact_sheet = tmp_path / "hexa_sheet.png"
    markdown_path = tmp_path / "hexa_report.md"
    report = build_dataset_report(
        dataset_dir,
        contact_sheet_path=contact_sheet,
        markdown_path=markdown_path,
        per_label=1,
        image_size=96,
    )

    assert report["sample_count"] == 9
    assert report["coverage_by_solid"]["hexa"]["valid_topology_count"] == 3
    assert report["asset_qa"]["vector_json_missing"] == []
    assert report["asset_qa"]["canonical_svg_missing"] == []
    assert report["asset_qa"]["repair_svg_missing"] == []
    assert report["asset_qa"]["completion_faces_missing"] == []
    assert report["render_profile"]["palette"]["face_fill"] == "#dbe0e7"
    assert contact_sheet.exists()
    assert markdown_path.exists()
