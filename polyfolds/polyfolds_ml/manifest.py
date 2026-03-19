"""Manifest builder for legacy and canonical Polyfolds datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polyfolds_ml.schema import DatasetManifest, PolyfoldSample, RasterAsset, RepairTarget, SCHEMA_VERSION, VectorEdge, VectorFace


def _round_point(point: tuple[float, float], decimals: int = 6) -> tuple[float, float]:
    return (round(float(point[0]), decimals), round(float(point[1]), decimals))


def _faces_from_cells(cells: list[list[int]], *, present: bool = True, offset: int = 0) -> list[VectorFace]:
    faces: list[VectorFace] = []
    for idx, cell in enumerate(cells):
        x = float(cell[0])
        y = float(cell[1])
        faces.append(
            VectorFace(
                face_index=offset + idx,
                polygon=((x, y), (x + 1.0, y), (x + 1.0, y + 1.0), (x, y + 1.0)),
                present=present,
            )
        )
    return faces


def _faces_from_polygons(net_payload: dict[str, Any]) -> list[VectorFace]:
    out: list[VectorFace] = []
    for item in net_payload.get("faces", []) or []:
        xy = tuple(_round_point((pair[0], pair[1])) for pair in item.get("xy", []))
        if len(xy) < 3:
            continue
        out.append(
            VectorFace(
                face_index=int(item.get("face_index", len(out))),
                polygon=xy,
                present=True,
            )
        )
    return out


def _edges_from_faces(faces: list[VectorFace]) -> list[VectorEdge]:
    edge_map: dict[tuple[tuple[float, float], tuple[float, float]], dict[str, Any]] = {}
    for face in faces:
        polygon = list(face.polygon)
        for index in range(len(polygon)):
            start = polygon[index]
            end = polygon[(index + 1) % len(polygon)]
            key = tuple(sorted((_round_point(start), _round_point(end))))
            record = edge_map.setdefault(key, {"start": start, "end": end, "face_indices": []})
            record["face_indices"].append(int(face.face_index))

    edges: list[VectorEdge] = []
    for index, (key, record) in enumerate(sorted(edge_map.items())):
        edges.append(
            VectorEdge(
                edge_id=f"e{index:05d}",
                start=key[0],
                end=key[1],
                face_indices=tuple(sorted(int(value) for value in record["face_indices"])),
                edge_group="edge_shared" if len(record["face_indices"]) >= 2 else "edge_cut",
            )
        )
    return edges


def _topology_hash_from_tree_edges(solid: str, tree_edges: list[Any]) -> str:
    normalized: list[tuple[int, int]] = []
    for item in tree_edges or []:
        if len(item) >= 2:
            normalized.append(tuple(sorted((int(item[0]), int(item[1])))))
    payload = json.dumps({"solid": solid, "tree_edges": sorted(normalized)}, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _repair_target_from_row(row: dict[str, Any]) -> RepairTarget | None:
    completion_cells = row.get("completion_cells") or []
    completion_faces = row.get("completion_faces") or []
    if not completion_cells and not completion_faces:
        return None
    face_indices: list[int] = []
    for item in completion_faces:
        if isinstance(item, dict) and "face_index" in item:
            face_indices.append(int(item["face_index"]))
    return RepairTarget(
        target_raster=None,
        target_svg_path=None,
        completion_face_indices=tuple(sorted(face_indices)),
        edit_budget=len(completion_cells) + len(completion_faces),
    )


def _sample_from_legacy_row(row: dict[str, Any], *, source_dir: Path) -> dict[str, Any]:
    raster_path = source_dir / str(row["file"])
    solid = str(row.get("solid", source_dir.name.replace("dataset_", "")))
    class_label = str(row.get("class", "unknown"))

    vector_faces = []
    if row.get("cells"):
        vector_faces.extend(_faces_from_cells(row.get("cells") or [], present=True, offset=0))
    elif isinstance(row.get("net"), dict):
        vector_faces.extend(_faces_from_polygons(row["net"]))

    completion_cells = row.get("completion_cells") or []
    if completion_cells:
        vector_faces.extend(_faces_from_cells(completion_cells, present=False, offset=len(vector_faces)))

    vector_edges = _edges_from_faces(vector_faces)
    topology_hash = None
    if isinstance(row.get("net"), dict):
        topology_hash = _topology_hash_from_tree_edges(solid, row["net"].get("tree_edges") or [])

    sample = PolyfoldSample(
        sample_id=raster_path.stem,
        split=str(row.get("split", "train")),
        class_label=class_label,
        solid=solid,
        source_dataset=source_dir.name,
        raster_input=RasterAsset(path=str(raster_path.resolve())),
        state=class_label,
        joint_label=f"{solid}:{class_label}",
        topology_hash=topology_hash,
        vector_json_path=None,
        canonical_svg_path=None,
        render_profile_id=None,
        source_kind="legacy",
        vector_faces=tuple(vector_faces),
        vector_edges=tuple(vector_edges),
        repair_target=_repair_target_from_row(row),
        metadata={
            "invalid_reason": row.get("invalid_reason"),
            "faces_total": row.get("faces_total"),
            "faces_present": row.get("faces_present"),
            "seed": row.get("seed"),
            "generation_mode": "imported",
        },
        schema_version=SCHEMA_VERSION,
    )
    return sample.to_dict()


def _resolve_relative_path(value: str | None, *, source_dir: Path) -> str | None:
    if not value:
        return value
    path = Path(str(value))
    if path.is_absolute():
        return str(path.resolve())
    return str((source_dir / path).resolve())


def _normalize_canonical_row(row: dict[str, Any], *, source_dir: Path) -> dict[str, Any]:
    out = dict(row)
    out["source_kind"] = str(out.get("source_kind", "canonical"))
    out["state"] = str(out.get("state") or out.get("class_label") or "unknown")
    out["class_label"] = str(out.get("class_label") or out["state"])
    out["solid"] = str(out.get("solid") or source_dir.name)
    out["joint_label"] = str(out.get("joint_label") or f'{out["solid"]}:{out["state"]}')
    out["source_dataset"] = str(out.get("source_dataset") or source_dir.name)
    out["schema_version"] = int(out.get("schema_version", SCHEMA_VERSION))
    out["vector_json_path"] = _resolve_relative_path(out.get("vector_json_path"), source_dir=source_dir)
    out["canonical_svg_path"] = _resolve_relative_path(out.get("canonical_svg_path"), source_dir=source_dir)

    raster_input = out.get("raster_input")
    if isinstance(raster_input, dict):
        raster_copy = dict(raster_input)
        raster_copy["path"] = _resolve_relative_path(raster_copy.get("path"), source_dir=source_dir)
        out["raster_input"] = raster_copy

    repair_target = out.get("repair_target")
    if isinstance(repair_target, dict):
        repair_copy = dict(repair_target)
        repair_copy["target_svg_path"] = _resolve_relative_path(repair_copy.get("target_svg_path"), source_dir=source_dir)
        target_raster = repair_copy.get("target_raster")
        if isinstance(target_raster, dict):
            target_raster_copy = dict(target_raster)
            target_raster_copy["path"] = _resolve_relative_path(target_raster_copy.get("path"), source_dir=source_dir)
            repair_copy["target_raster"] = target_raster_copy
        out["repair_target"] = repair_copy

    return out


def _iter_legacy_rows(dataset_dir: Path):
    labels_path = dataset_dir / "labels.jsonl"
    if not labels_path.exists():
        return
    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield _sample_from_legacy_row(json.loads(text), source_dir=dataset_dir)


def _iter_canonical_rows(dataset_dir: Path):
    samples_path = dataset_dir / "samples.jsonl"
    if not samples_path.exists():
        return
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield _normalize_canonical_row(json.loads(text), source_dir=dataset_dir)


def build_manifest(dataset_roots: list[Path], *, output_path: Path | None = None, dataset_name: str = "polyfolds_v1") -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    solid_counter: Counter[str] = Counter()
    dataset_kinds: set[str] = set()
    coverage_modes: set[str] = set()
    label_space_versions: set[str] = set()

    for root in dataset_roots:
        dataset_dir = Path(root).resolve()
        if (dataset_dir / "samples.jsonl").exists():
            rows = list(_iter_canonical_rows(dataset_dir) or [])
            dataset_kinds.add("canonical")
            coverage_modes.update(
                str((row.get("metadata") or {}).get("generation_mode", "sampled"))
                for row in rows
                if isinstance(row, dict)
            )
            label_space_versions.add("solid_state_v1")
        else:
            rows = list(_iter_legacy_rows(dataset_dir) or [])
            dataset_kinds.add("legacy")
            coverage_modes.add("imported")
            label_space_versions.add("state_only_v1")

        for row in rows:
            samples.append(row)
            label = str(row.get("joint_label") or row.get("class_label") or "unknown")
            class_counter.update([label])
            solid_counter.update([str(row.get("solid", "unknown"))])

    manifest = DatasetManifest(
        dataset_name=dataset_name,
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        source_roots=tuple(str(Path(root).resolve()) for root in dataset_roots),
        classes=tuple(sorted(class_counter.keys())),
        solids=tuple(sorted(solid_counter.keys())),
        sample_count=len(samples),
        dataset_kind=next(iter(dataset_kinds)) if len(dataset_kinds) == 1 else "mixed",
        coverage_kind=next(iter(coverage_modes)) if len(coverage_modes) == 1 else "mixed",
        label_space_version=next(iter(label_space_versions)) if len(label_space_versions) == 1 else "mixed",
        notes=(
            "Legacy roots are imported but remain non-canonical.",
            "Canonical roots provide vector JSON plus SVG assets for the primary workflow.",
        ),
    )

    payload = {"manifest": manifest.to_dict(), "samples": samples}
    if output_path is not None:
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_manifest_rows(manifest_path: Path) -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def summarize_manifest(manifest_payload: dict[str, Any]) -> dict[str, Any]:
    rows = manifest_payload.get("samples", [])
    classes = Counter(str(row.get("joint_label") or row.get("class_label") or "unknown") for row in rows)
    solids = Counter(str(row.get("solid", "unknown")) for row in rows)
    return {
        "sample_count": int(len(rows)),
        "classes": dict(sorted(classes.items())),
        "solids": dict(sorted(solids.items())),
        "dataset_kind": str((manifest_payload.get("manifest") or {}).get("dataset_kind", "unknown")),
        "coverage_kind": str((manifest_payload.get("manifest") or {}).get("coverage_kind", "unknown")),
    }
