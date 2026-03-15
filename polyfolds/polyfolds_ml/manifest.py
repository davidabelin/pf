"""Manifest builder for legacy Polyfolds datasets.

Role
----
Translate older Polyfolds dataset folders into the normalized raster+vector
manifest schema used by the next phase of offline model training.

Cross-Repo Context
------------------
This is the bridge from the historical data-generation scripts into the newer
``polyfolds_ml`` workflow. The resulting manifest is meant to be consumed by
offline training now and by ``pf_web`` inference workflows later.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polyfolds_ml.schema import DatasetManifest, PolyfoldSample, RasterAsset, RepairTarget, VectorEdge, VectorFace


def _round_point(point: tuple[float, float], decimals: int = 6) -> tuple[float, float]:
    """Round one 2D point into a stable manifest-friendly coordinate key."""

    return (round(float(point[0]), decimals), round(float(point[1]), decimals))


def _faces_from_cells(cells: list[list[int]], *, present: bool = True, offset: int = 0) -> list[VectorFace]:
    """Convert axis-aligned cell coordinates into square ``VectorFace`` records."""

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
    """Convert legacy polygon payloads into normalized ``VectorFace`` records."""

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
    """Derive explicit edge records from a face list.

    Role
    ----
    Make adjacency and shared-boundary structure first-class in the manifest so
    later repair models do not have to rediscover it from raster images alone.
    """

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
            )
        )
    return edges


def _repair_target_from_row(row: dict[str, Any], *, source_dir: Path) -> RepairTarget | None:
    """Build a repair-target record from one legacy label row when available.

    Notes
    -----
    Legacy datasets rarely contain full repair payloads, so this function keeps
    the contract permissive and emits a partial target when only completion-face
    hints are available.
    """

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


def _sample_from_legacy_row(row: dict[str, Any], *, source_dir: Path) -> PolyfoldSample:
    """Convert one legacy label row into the normalized sample schema.

    Role
    ----
    This is the main compatibility shim from historical Polyfolds outputs into
    the newer raster-plus-vector sample contract.
    """

    raster_path = source_dir / str(row["file"])
    vector_faces = []
    if row.get("cells"):
        vector_faces.extend(_faces_from_cells(row.get("cells") or [], present=True, offset=0))
    elif isinstance(row.get("net"), dict):
        vector_faces.extend(_faces_from_polygons(row["net"]))

    completion_cells = row.get("completion_cells") or []
    if completion_cells:
        vector_faces.extend(_faces_from_cells(completion_cells, present=False, offset=len(vector_faces)))

    vector_edges = _edges_from_faces(vector_faces)
    return PolyfoldSample(
        sample_id=raster_path.stem,
        split=str(row.get("split", "train")),
        class_label=str(row.get("class", "unknown")),
        solid=str(row.get("solid", source_dir.name.replace("dataset_", ""))),
        source_dataset=source_dir.name,
        raster_input=RasterAsset(path=str(raster_path.resolve())),
        vector_faces=tuple(vector_faces),
        vector_edges=tuple(vector_edges),
        repair_target=_repair_target_from_row(row, source_dir=source_dir),
        metadata={
            "invalid_reason": row.get("invalid_reason"),
            "faces_total": row.get("faces_total"),
            "faces_present": row.get("faces_present"),
            "seed": row.get("seed"),
        },
    )


def _iter_legacy_label_rows(dataset_dir: Path):
    """Yield parsed rows from one legacy ``labels.jsonl`` file if present."""

    labels_path = dataset_dir / "labels.jsonl"
    if not labels_path.exists():
        return
    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield json.loads(text)


def build_manifest(dataset_roots: list[Path], *, output_path: Path | None = None, dataset_name: str = "polyfolds_v1") -> dict[str, Any]:
    """Build a normalized manifest payload from one or more legacy datasets.

    Returns
    -------
    dict[str, Any]
        JSON-ready payload containing a manifest header and normalized sample
        rows.

    Role
    ----
    This is the canonical offline handoff from historical dataset folders to
    the normalized Polyfolds ML workflow.
    """

    samples: list[PolyfoldSample] = []
    class_counter: Counter[str] = Counter()
    solid_counter: Counter[str] = Counter()

    for root in dataset_roots:
        dataset_dir = Path(root).resolve()
        for row in _iter_legacy_label_rows(dataset_dir) or []:
            sample = _sample_from_legacy_row(row, source_dir=dataset_dir)
            samples.append(sample)
            class_counter.update([sample.class_label])
            solid_counter.update([sample.solid])

    manifest = DatasetManifest(
        dataset_name=dataset_name,
        schema_version=1,
        created_at=datetime.now(UTC).isoformat(),
        source_roots=tuple(str(Path(root).resolve()) for root in dataset_roots),
        classes=tuple(sorted(class_counter.keys())),
        solids=tuple(sorted(solid_counter.keys())),
        sample_count=len(samples),
        notes=(
            "Legacy labels were normalized into a raster+vector schema.",
            "Edge color groups are reserved for future generators and may be null in legacy rows.",
        ),
    )

    payload = {
        "manifest": manifest.to_dict(),
        "samples": [sample.to_dict() for sample in samples],
    }
    if output_path is not None:
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_manifest_rows(manifest_path: Path) -> dict[str, Any]:
    """Load one previously written manifest JSON payload from disk."""

    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def summarize_manifest(manifest_payload: dict[str, Any]) -> dict[str, Any]:
    """Return a compact class/solid summary for one manifest payload.

    Used By
    -------
    CLI diagnostics and quick sanity checks before longer training runs.
    """

    rows = manifest_payload.get("samples", [])
    classes = Counter(str(row.get("class_label", "unknown")) for row in rows)
    solids = Counter(str(row.get("solid", "unknown")) for row in rows)
    return {
        "sample_count": int(len(rows)),
        "classes": dict(sorted(classes.items())),
        "solids": dict(sorted(solids.items())),
    }
