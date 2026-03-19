"""Dataset reporting and exemplar generation for Polyfolds manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from polyfolds_ml.manifest import build_manifest, load_manifest_rows
from vector_render import NEUTRAL_RENDER_PROFILE_ID, neutral_render_palette, render_faces_image


EXPECTED_STATES = ("valid", "incomplete", "invalid")
SOLID_ORDER = ("tetra", "hexa", "octa", "dodeca", "icosa")


def _load_manifest_like(input_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(input_path).resolve()
    if path.is_dir():
        return path, build_manifest([path], dataset_name=path.name)

    payload = load_manifest_rows(path)
    if "samples" in payload:
        return path, payload

    if (path.parent / "samples.jsonl").exists() or (path.parent / "labels.jsonl").exists():
        return path.parent, build_manifest([path.parent], dataset_name=path.parent.name)

    raise RuntimeError("Expected a manifest JSON with samples or a dataset root containing samples.jsonl or labels.jsonl.")


def _ordered_solids(rows: list[dict[str, Any]]) -> list[str]:
    present = {str(row.get("solid", "unknown")) for row in rows}
    ordered = [solid for solid in SOLID_ORDER if solid in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            SOLID_ORDER.index(str(row.get("solid"))) if str(row.get("solid")) in SOLID_ORDER else len(SOLID_ORDER),
            EXPECTED_STATES.index(str(row.get("state") or row.get("class_label")))
            if str(row.get("state") or row.get("class_label")) in EXPECTED_STATES
            else len(EXPECTED_STATES),
            str(row.get("topology_hash") or row.get("sample_id") or ""),
            str(row.get("sample_id") or ""),
        ),
    )


def _path_exists(value: str | None) -> bool:
    return bool(value) and Path(str(value)).exists()


def _balance_repeat_factors(counter: Counter[str]) -> dict[str, float]:
    if not counter:
        return {}
    target = max(counter.values())
    return {key: round(float(target) / float(value), 3) for key, value in sorted(counter.items())}


def _family_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    states_by_family: dict[str, set[str]] = defaultdict(set)
    split_by_family: dict[str, set[str]] = defaultdict(set)
    duplicates: Counter[str] = Counter()

    for row in rows:
        solid = str(row.get("solid", "unknown"))
        state = str(row.get("state") or row.get("class_label") or "unknown")
        topology_hash = str(row.get("topology_hash") or row.get("sample_id") or "missing")
        family_key = f"{solid}:{topology_hash}"
        states_by_family[family_key].add(state)
        split_by_family[family_key].add(str(row.get("split", "train")))
        duplicates.update([f"{solid}:{state}:{topology_hash}"])

    missing_triples = {
        family_key: sorted(set(EXPECTED_STATES).difference(states))
        for family_key, states in sorted(states_by_family.items())
        if set(EXPECTED_STATES).difference(states)
    }
    split_leaks = {family_key: sorted(splits) for family_key, splits in sorted(split_by_family.items()) if len(splits) > 1}
    duplicate_state_topologies = {key: int(value) for key, value in sorted(duplicates.items()) if value > 1}
    return {
        "family_count": int(len(states_by_family)),
        "families_missing_states": missing_triples,
        "families_spanning_multiple_splits": split_leaks,
        "duplicate_state_topologies": duplicate_state_topologies,
    }


def _asset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vector_json_missing: list[str] = []
    canonical_svg_missing: list[str] = []
    repair_svg_missing: list[str] = []
    completion_faces_missing: list[str] = []

    for row in rows:
        if str(row.get("source_kind", "legacy")) != "canonical":
            continue
        sample_id = str(row.get("sample_id", "unknown"))
        if not _path_exists(row.get("vector_json_path")):
            vector_json_missing.append(sample_id)
        if not _path_exists(row.get("canonical_svg_path")):
            canonical_svg_missing.append(sample_id)

        state = str(row.get("state") or row.get("class_label") or "unknown")
        repair_target = row.get("repair_target") or {}
        if state in {"incomplete", "invalid"} and not _path_exists(repair_target.get("target_svg_path")):
            repair_svg_missing.append(sample_id)
        if state == "incomplete" and not (repair_target.get("completion_face_indices") or []):
            completion_faces_missing.append(sample_id)

    return {
        "vector_json_missing": vector_json_missing,
        "canonical_svg_missing": canonical_svg_missing,
        "repair_svg_missing": repair_svg_missing,
        "completion_faces_missing": completion_faces_missing,
    }


def _coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("solid", "unknown"))].append(row)

    for solid, solid_rows in sorted(grouped.items()):
        valid_rows = [row for row in solid_rows if str(row.get("state") or row.get("class_label")) == "valid"]
        modes = Counter(str((row.get("metadata") or {}).get("generation_mode", "unknown")) for row in valid_rows)
        full_counts = {int((row.get("metadata") or {}).get("full_valid_count")) for row in valid_rows if (row.get("metadata") or {}).get("full_valid_count")}
        summary[solid] = {
            "valid_topology_count": int(len(valid_rows)),
            "full_valid_count": int(next(iter(full_counts))) if len(full_counts) == 1 else None,
            "coverage_kind": next(iter(modes.keys())) if len(modes) == 1 and modes else "mixed",
            "coverage_ratio": round(float(len(valid_rows)) / float(next(iter(full_counts))), 4) if len(full_counts) == 1 and next(iter(full_counts)) else None,
        }
    return summary


def _dataset_contract() -> dict[str, Any]:
    return {
        "canonical_root_files": (
            "samples.jsonl",
            "dataset_manifest.json",
            "vector_json/<solid>/<sample_id>.json",
            "svg/<solid>/<sample_id>.svg",
            "preview/<solid>/<sample_id>.png (optional)",
        ),
        "required_sample_fields": (
            "sample_id",
            "split",
            "solid",
            "state",
            "joint_label",
            "topology_hash",
            "vector_json_path",
            "canonical_svg_path",
            "render_profile_id",
            "source_kind",
            "vector_faces",
            "vector_edges",
        ),
        "state_space": EXPECTED_STATES,
        "repair_contract": (
            "repair_target.target_svg_path",
            "repair_target.completion_face_indices for incomplete samples",
            "metadata.edit_recipe for derived incomplete and invalid samples",
        ),
    }


def _select_exemplars(rows: list[dict[str, Any]], *, per_label: int) -> list[dict[str, Any]]:
    ordered_rows = _ordered_rows(rows)
    selected: list[dict[str, Any]] = []
    for solid in _ordered_solids(ordered_rows):
        for state in EXPECTED_STATES:
            matches = [
                row
                for row in ordered_rows
                if str(row.get("solid")) == solid and str(row.get("state") or row.get("class_label")) == state
            ]
            selected.extend(matches[: int(per_label)])
    return selected


def write_exemplar_contact_sheet(
    rows: list[dict[str, Any]],
    *,
    out_path: str | Path,
    per_label: int = 1,
    image_size: int = 192,
) -> str:
    """Write a contact sheet of deterministic exemplar renders."""

    exemplars = _select_exemplars(rows, per_label=max(1, int(per_label)))
    if not exemplars:
        raise RuntimeError("Need at least one sample to build an exemplar contact sheet.")

    columns = max(1, len(EXPECTED_STATES) * max(1, int(per_label)))
    rows_count = (len(exemplars) + columns - 1) // columns
    label_height = 34
    pad = 12
    width = columns * image_size + (columns + 1) * pad
    height = rows_count * (image_size + label_height) + (rows_count + 1) * pad
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for index, row in enumerate(exemplars):
        col = index % columns
        row_index = index // columns
        x = pad + col * (image_size + pad)
        y = pad + row_index * (image_size + label_height + pad)
        tile = render_faces_image(
            row.get("vector_faces", []),
            vector_edges=row.get("vector_edges", []),
            image_size=image_size,
        )
        canvas.paste(tile, (x, y))
        label = f'{row.get("solid")}:{row.get("state") or row.get("class_label")}'
        draw.text((x, y + image_size + 4), label, fill=(24, 32, 40))
        draw.text((x, y + image_size + 18), str(row.get("topology_hash") or row.get("sample_id")), fill=(110, 118, 128))

    path = Path(out_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def _markdown_report(input_path: Path, report: dict[str, Any], *, contact_sheet_path: str | None) -> str:
    lines = [
        f"# Polyfolds Dataset Report: `{input_path}`",
        "",
        "## Summary",
        f"- sample count: {report['sample_count']}",
        f"- dataset kind: {report['dataset_kind']}",
        f"- coverage kind: {report['coverage_kind']}",
        f"- label space: {report['label_space_version']}",
        f"- render profile: {report['render_profile']['render_profile_id']}",
        "",
        "## Counts",
    ]
    for solid, count in report["counts"]["by_solid"].items():
        lines.append(f"- {solid}: {count} samples")
    lines.append("")
    lines.append("## Joint Labels")
    for label, count in report["counts"]["by_joint_label"].items():
        lines.append(f"- {label}: {count}")

    lines.extend(["", "## Coverage"])
    for solid, payload in report["coverage_by_solid"].items():
        lines.append(
            f"- {solid}: {payload['valid_topology_count']} valid topologies, coverage `{payload['coverage_kind']}`, full count `{payload['full_valid_count']}`"
        )

    lines.extend(
        [
            "",
            "## Asset QA",
            f"- missing vector JSON: {len(report['asset_qa']['vector_json_missing'])}",
            f"- missing canonical SVG: {len(report['asset_qa']['canonical_svg_missing'])}",
            f"- missing repair SVG: {len(report['asset_qa']['repair_svg_missing'])}",
            f"- incomplete samples missing completion faces: {len(report['asset_qa']['completion_faces_missing'])}",
        ]
    )

    if contact_sheet_path:
        lines.extend(["", "## Exemplars", f"- contact sheet: `{contact_sheet_path}`"])

    lines.extend(["", "## Render Palette"])
    for key, value in report["render_profile"]["palette"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Training Balance"])
    for key, value in report["balance_plan"]["joint_label_repeat_factor"].items():
        lines.append(f"- {key}: repeat factor {value}")

    return "\n".join(lines) + "\n"


def build_dataset_report(
    input_path: str | Path,
    *,
    contact_sheet_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    json_path: str | Path | None = None,
    per_label: int = 1,
    image_size: int = 192,
) -> dict[str, Any]:
    """Build a JSON-ready report for a canonical or legacy Polyfolds dataset."""

    resolved_input, payload = _load_manifest_like(input_path)
    manifest = payload.get("manifest", {})
    rows = list(payload.get("samples", []))
    state_counter = Counter(str(row.get("state") or row.get("class_label") or "unknown") for row in rows)
    solid_counter = Counter(str(row.get("solid", "unknown")) for row in rows)
    joint_counter = Counter(str(row.get("joint_label") or row.get("class_label") or "unknown") for row in rows)
    split_counter = Counter(str(row.get("split", "train")) for row in rows)
    source_counter = Counter(str(row.get("source_kind", "legacy")) for row in rows)
    generation_counter = Counter(str((row.get("metadata") or {}).get("generation_mode", "unknown")) for row in rows)

    report = {
        "input_path": str(resolved_input),
        "dataset_name": str(manifest.get("dataset_name", resolved_input.name)),
        "sample_count": int(len(rows)),
        "dataset_kind": str(manifest.get("dataset_kind", "unknown")),
        "coverage_kind": str(manifest.get("coverage_kind", "unknown")),
        "label_space_version": str(manifest.get("label_space_version", "unknown")),
        "counts": {
            "by_state": dict(sorted(state_counter.items())),
            "by_solid": dict(sorted(solid_counter.items())),
            "by_joint_label": dict(sorted(joint_counter.items())),
            "by_split": dict(sorted(split_counter.items())),
            "by_source_kind": dict(sorted(source_counter.items())),
            "by_generation_mode": dict(sorted(generation_counter.items())),
        },
        "coverage_by_solid": _coverage_summary(rows),
        "family_qa": _family_summary(rows),
        "asset_qa": _asset_summary(rows),
        "balance_plan": {
            "joint_label_repeat_factor": _balance_repeat_factors(joint_counter),
            "solid_repeat_factor": _balance_repeat_factors(solid_counter),
        },
        "render_profile": {
            "render_profile_id": str(next((row.get("render_profile_id") for row in rows if row.get("render_profile_id")), NEUTRAL_RENDER_PROFILE_ID)),
            "palette": neutral_render_palette(),
        },
        "dataset_contract": _dataset_contract(),
    }

    sheet_value = None
    if contact_sheet_path is not None:
        sheet_value = write_exemplar_contact_sheet(rows, out_path=contact_sheet_path, per_label=per_label, image_size=image_size)
        report["contact_sheet_path"] = sheet_value

    if markdown_path is not None:
        markdown_text = _markdown_report(resolved_input, report, contact_sheet_path=sheet_value)
        path = Path(markdown_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_text, encoding="utf-8")

    if json_path is not None:
        path = Path(json_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a Polyfolds dataset root or manifest and optionally emit exemplar previews.")
    parser.add_argument("input", help="Dataset root, dataset_manifest.json path, or manifest.json path.")
    parser.add_argument("--output-json", help="Optional JSON report path.")
    parser.add_argument("--output-md", help="Optional Markdown report path.")
    parser.add_argument("--contact-sheet", help="Optional PNG contact sheet path.")
    parser.add_argument("--per-label", type=int, default=1, help="Number of exemplars to include per solid-state label in the contact sheet.")
    parser.add_argument("--image-size", type=int, default=192, help="Per-exemplar render size for the contact sheet.")
    args = parser.parse_args(argv)

    report = build_dataset_report(
        args.input,
        contact_sheet_path=args.contact_sheet,
        markdown_path=args.output_md,
        json_path=args.output_json,
        per_label=int(args.per_label),
        image_size=int(args.image_size),
    )
    print(json.dumps(report, indent=2))
    return 0
