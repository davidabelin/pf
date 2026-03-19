"""Build a canonical vector-first Polyfolds dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bootstrap_paths import ensure_polyfolds_paths

ensure_polyfolds_paths()

from platonic_nets import Net2D, NetFace2D, net_to_json, unfold_random_net
from solid_dodeca import SPEC as DODECA_SPEC
from solid_hexa import _NETS_SPEC as HEXA_SPEC
from solid_icosa import SPEC as ICOSA_SPEC
from solid_octa import SPEC as OCTA_SPEC
from solid_polyface import (
    _derive_seed,
    _flip_subtree_about_shared_edge,
    _invalidate_by_detaching_subtree,
    _make_incomplete,
    _net_has_duplicate_polygons,
    _net_has_overlap,
)
from solid_tetra import SPEC as TETRA_SPEC
from vector_render import NEUTRAL_RENDER_PROFILE_ID, neutral_render_palette, save_faces_png, write_faces_svg

from polyfolds_ml.schema import DatasetManifest, PolyfoldSample, RasterAsset, RepairTarget, VectorEdge, VectorFace


FULL_VALID_COUNTS = {"tetra": 2, "hexa": 11, "octa": 11, "dodeca": 43380, "icosa": 43380}
TEST_VALID_COUNTS = {"tetra": 2, "hexa": 3, "octa": 3, "dodeca": 4, "icosa": 4}
SOLID_SPECS = {
    "tetra": TETRA_SPEC,
    "hexa": HEXA_SPEC,
    "octa": OCTA_SPEC,
    "dodeca": DODECA_SPEC,
    "icosa": ICOSA_SPEC,
}


@dataclass(slots=True)
class CanonicalBuildConfig:
    out_dir: str
    solids: tuple[str, ...] = ("tetra", "hexa", "octa", "dodeca", "icosa")
    hard_valid_limit: int = 2048
    seed: int = 20260318
    preview_png: bool = False
    preview_size: int = 256
    max_tries_per_net: int = 500
    max_attempts_per_solid: int = 120000
    test_mode: bool = False


def _round_point(point: tuple[float, float], decimals: int = 6) -> tuple[float, float]:
    return (round(float(point[0]), decimals), round(float(point[1]), decimals))


def _topology_hash(net: Net2D, *, solid_key: str) -> str:
    edges = sorted(tuple(sorted((int(a), int(b)))) for a, b, _edge in net.tree_edges)
    payload = json.dumps({"solid": solid_key, "tree_edges": edges}, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _split_from_topology_hash(topology_hash: str) -> str:
    bucket = int(str(topology_hash)[:8], 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"


def _target_valid_count(solid_key: str, config: CanonicalBuildConfig) -> tuple[int, str]:
    if config.test_mode:
        return int(TEST_VALID_COUNTS[solid_key]), "sampled"
    if solid_key in {"tetra", "hexa", "octa"}:
        return int(FULL_VALID_COUNTS[solid_key]), "exhaustive"
    return int(config.hard_valid_limit), "sampled"


def _vector_faces_from_net(net: Net2D, *, completion_faces: list[NetFace2D] | None = None) -> list[VectorFace]:
    by_index: dict[int, VectorFace] = {}
    for face in net.faces:
        by_index[int(face.face_index)] = VectorFace(
            face_index=int(face.face_index),
            polygon=tuple(_round_point(point) for point in face.xy),
            present=True,
        )
    for face in completion_faces or []:
        by_index[int(face.face_index)] = VectorFace(
            face_index=int(face.face_index),
            polygon=tuple(_round_point(point) for point in face.xy),
            present=False,
        )
    return [by_index[index] for index in sorted(by_index.keys())]


def _vector_edges_from_faces(faces: list[VectorFace]) -> list[VectorEdge]:
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
                start=tuple(key[0]),
                end=tuple(key[1]),
                face_indices=tuple(sorted(int(v) for v in record["face_indices"])),
                edge_group="edge_shared" if len(record["face_indices"]) >= 2 else "edge_cut",
            )
        )
    return edges


def _asset_paths(base_dir: Path, solid_key: str, sample_id: str) -> tuple[Path, Path, Path]:
    vector_path = base_dir / "vector_json" / solid_key / f"{sample_id}.json"
    svg_path = base_dir / "svg" / solid_key / f"{sample_id}.svg"
    png_path = base_dir / "preview" / solid_key / f"{sample_id}.png"
    return vector_path, svg_path, png_path


def _render_roles() -> dict[str, str]:
    return {
        "face_present": "neutral_fill",
        "completion_target": "neutral_outline",
        "edge_shared": "shared_edge",
        "edge_cut": "cut_edge",
    }


def _write_vector_payload(
    *,
    out_path: Path,
    solid_key: str,
    state: str,
    topology_hash: str,
    net: Net2D,
    completion_faces: list[NetFace2D] | None,
    edit_recipe: dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "solid": solid_key,
        "state": state,
        "topology_hash": topology_hash,
        "render_profile_id": NEUTRAL_RENDER_PROFILE_ID,
        "render_roles": _render_roles(),
        "render_palette": neutral_render_palette(),
        "net": net_to_json(net),
        "completion_faces": [
            {"face_index": int(face.face_index), "vertex_ids": list(face.vertex_ids), "xy": [list(point) for point in face.xy]}
            for face in completion_faces or []
        ],
        "edit_recipe": edit_recipe,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sample_from_net(
    *,
    dataset_name: str,
    solid_key: str,
    state: str,
    topology_hash: str,
    split: str,
    net: Net2D,
    faces_total: int,
    generation_mode: str,
    full_valid_count: int,
    valid_target_sample_id: str,
    completion_faces: list[NetFace2D] | None,
    edit_recipe: dict[str, Any],
    valid_target_svg_path: str | None,
    vector_json_path: Path,
    canonical_svg_path: Path,
    preview_png_path: Path | None,
    preview_size: int,
    seed_value: int,
) -> dict[str, Any]:
    sample_id = f"{solid_key}_{topology_hash}_{state}"
    vector_faces = _vector_faces_from_net(net, completion_faces=completion_faces)
    vector_edges = _vector_edges_from_faces(vector_faces)

    _write_vector_payload(
        out_path=vector_json_path,
        solid_key=solid_key,
        state=state,
        topology_hash=topology_hash,
        net=net,
        completion_faces=completion_faces,
        edit_recipe=edit_recipe,
    )
    write_faces_svg(
        vector_faces,
        out_path=canonical_svg_path,
        vector_edges=vector_edges,
        metadata={"sample_id": sample_id, "state": state, "solid": solid_key, "topology_hash": topology_hash},
    )

    raster_input = None
    if preview_png_path is not None:
        save_faces_png(vector_faces, out_path=preview_png_path, vector_edges=vector_edges, image_size=preview_size)
        raster_input = RasterAsset(
            path=str(preview_png_path.resolve()),
            width=int(preview_size),
            height=int(preview_size),
            color_mode="rgb",
        )

    repair_target = None
    if state in {"incomplete", "invalid"} and valid_target_svg_path:
        repair_target = RepairTarget(
            target_raster=None,
            target_svg_path=str(valid_target_svg_path),
            completion_face_indices=tuple(sorted(int(face.face_index) for face in completion_faces or [])),
            edit_budget=len(completion_faces or []) if state == "incomplete" else 1,
        )

    sample = PolyfoldSample(
        sample_id=sample_id,
        split=split,
        class_label=state,
        solid=solid_key,
        source_dataset=dataset_name,
        raster_input=raster_input,
        state=state,
        joint_label=f"{solid_key}:{state}",
        topology_hash=topology_hash,
        vector_json_path=str(vector_json_path.resolve()),
        canonical_svg_path=str(canonical_svg_path.resolve()),
        render_profile_id=NEUTRAL_RENDER_PROFILE_ID,
        source_kind="canonical",
        vector_faces=tuple(vector_faces),
        vector_edges=tuple(vector_edges),
        repair_target=repair_target,
        metadata={
            "generation_mode": generation_mode,
            "faces_total": int(faces_total),
            "faces_present": int(len(net.faces)),
            "render_roles": _render_roles(),
            "render_palette": neutral_render_palette(),
            "edit_recipe": edit_recipe,
            "full_valid_count": int(full_valid_count),
            "valid_target_sample_id": valid_target_sample_id,
            "seed": int(seed_value),
        },
    )
    return sample.to_dict()


def _derive_invalid(net: Net2D, *, topology_hash: str) -> tuple[Net2D, dict[str, Any]]:
    seed_value = int(topology_hash[:12], 16)
    for attempt in range(12):
        rng = random.Random(seed_value + attempt)
        flipped = _flip_subtree_about_shared_edge(net, rng=rng)
        if flipped is None or _net_has_duplicate_polygons(flipped):
            continue
        if _net_has_overlap(flipped):
            return flipped, {"operation": "overlap_flip", "attempt": attempt}
    detached = _invalidate_by_detaching_subtree(net, rng=random.Random(seed_value), sidelength=float(net.sidelength))
    return detached, {"operation": "detach_subtree", "attempt": 0}


def _derive_incomplete(net: Net2D, *, solid_key: str, topology_hash: str) -> tuple[Net2D, list[NetFace2D], dict[str, Any]]:
    spec = SOLID_SPECS[solid_key]
    rng = random.Random(int(topology_hash[4:16], 16))
    partial, completion = _make_incomplete(net, rng=rng, max_missing=int(spec.default_missing), faces_total=int(spec.faces_total))
    return partial, completion, {"operation": "remove_leaf_faces", "removed_face_indices": [int(face.face_index) for face in completion]}


def _collect_valid_families(spec, *, target_count: int, config: CanonicalBuildConfig) -> list[tuple[str, Net2D, int]]:
    unique: dict[str, tuple[Net2D, int]] = {}
    attempt = 0
    while len(unique) < int(target_count) and attempt < int(config.max_attempts_per_solid):
        seed_value = _derive_seed(int(config.seed), int(spec.polyhedra_id), attempt)
        net = unfold_random_net(
            solid_id=int(spec.polyhedra_id),
            sidelength=1.0,
            seed=int(seed_value),
            max_tries=int(config.max_tries_per_net),
        )
        topology_hash = _topology_hash(net, solid_key=str(spec.key))
        unique.setdefault(topology_hash, (net, int(seed_value)))
        attempt += 1

    if len(unique) < int(target_count):
        raise RuntimeError(f"Collected {len(unique)} unique {spec.key} valid topologies, need {target_count}.")

    return [(hash_value, unique[hash_value][0], unique[hash_value][1]) for hash_value in sorted(unique.keys())[: int(target_count)]]


def build_canonical_dataset(config: CanonicalBuildConfig) -> dict[str, Any]:
    """Generate the canonical vector-first Polyfolds dataset."""

    out_dir = Path(config.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = out_dir.name

    samples: list[dict[str, Any]] = []
    joint_labels: Counter[str] = Counter()
    solids: Counter[str] = Counter()
    generation_modes: set[str] = set()
    per_solid_summary: dict[str, Any] = {}

    for solid_key in config.solids:
        spec = SOLID_SPECS[solid_key]
        target_valid, generation_mode = _target_valid_count(solid_key, config)
        generation_modes.add(generation_mode)
        valid_families = _collect_valid_families(spec, target_count=target_valid, config=config)
        per_solid_summary[solid_key] = {"valid_topologies": len(valid_families), "generation_mode": generation_mode}

        for topology_hash, valid_net, seed_value in valid_families:
            split = _split_from_topology_hash(topology_hash)
            valid_sample_id = f"{solid_key}_{topology_hash}_valid"

            vector_json_path, svg_path, png_path = _asset_paths(out_dir, solid_key, valid_sample_id)
            valid_sample = _sample_from_net(
                dataset_name=dataset_name,
                solid_key=solid_key,
                state="valid",
                topology_hash=topology_hash,
                split=split,
                net=valid_net,
                faces_total=int(spec.faces_total),
                generation_mode=generation_mode,
                full_valid_count=int(FULL_VALID_COUNTS[solid_key]),
                valid_target_sample_id=valid_sample_id,
                completion_faces=None,
                edit_recipe={"operation": "identity"},
                valid_target_svg_path=str(svg_path.resolve()),
                vector_json_path=vector_json_path,
                canonical_svg_path=svg_path,
                preview_png_path=png_path if config.preview_png else None,
                preview_size=int(config.preview_size),
                seed_value=int(seed_value),
            )
            samples.append(valid_sample)
            joint_labels.update([str(valid_sample["joint_label"])])
            solids.update([str(valid_sample["solid"])])

            incomplete_net, completion_faces, incomplete_recipe = _derive_incomplete(valid_net, solid_key=solid_key, topology_hash=topology_hash)
            vector_json_path, incomplete_svg_path, incomplete_png_path = _asset_paths(out_dir, solid_key, f"{solid_key}_{topology_hash}_incomplete")
            incomplete_sample = _sample_from_net(
                dataset_name=dataset_name,
                solid_key=solid_key,
                state="incomplete",
                topology_hash=topology_hash,
                split=split,
                net=incomplete_net,
                faces_total=int(spec.faces_total),
                generation_mode=generation_mode,
                full_valid_count=int(FULL_VALID_COUNTS[solid_key]),
                valid_target_sample_id=valid_sample_id,
                completion_faces=completion_faces,
                edit_recipe=incomplete_recipe,
                valid_target_svg_path=str(svg_path.resolve()),
                vector_json_path=vector_json_path,
                canonical_svg_path=incomplete_svg_path,
                preview_png_path=incomplete_png_path if config.preview_png else None,
                preview_size=int(config.preview_size),
                seed_value=int(seed_value),
            )
            samples.append(incomplete_sample)
            joint_labels.update([str(incomplete_sample["joint_label"])])
            solids.update([str(incomplete_sample["solid"])])

            invalid_net, invalid_recipe = _derive_invalid(valid_net, topology_hash=topology_hash)
            vector_json_path, invalid_svg_path, invalid_png_path = _asset_paths(out_dir, solid_key, f"{solid_key}_{topology_hash}_invalid")
            invalid_sample = _sample_from_net(
                dataset_name=dataset_name,
                solid_key=solid_key,
                state="invalid",
                topology_hash=topology_hash,
                split=split,
                net=invalid_net,
                faces_total=int(spec.faces_total),
                generation_mode=generation_mode,
                full_valid_count=int(FULL_VALID_COUNTS[solid_key]),
                valid_target_sample_id=valid_sample_id,
                completion_faces=None,
                edit_recipe=invalid_recipe,
                valid_target_svg_path=str(svg_path.resolve()),
                vector_json_path=vector_json_path,
                canonical_svg_path=invalid_svg_path,
                preview_png_path=invalid_png_path if config.preview_png else None,
                preview_size=int(config.preview_size),
                seed_value=int(seed_value),
            )
            samples.append(invalid_sample)
            joint_labels.update([str(invalid_sample["joint_label"])])
            solids.update([str(invalid_sample["solid"])])

    coverage_kind = "mixed" if len(generation_modes) > 1 else next(iter(generation_modes), "sampled")
    manifest = DatasetManifest(
        dataset_name=dataset_name,
        schema_version=2,
        created_at=datetime.now(UTC).isoformat(),
        source_roots=(str(out_dir),),
        classes=tuple(sorted(joint_labels.keys())),
        solids=tuple(sorted(solids.keys())),
        sample_count=len(samples),
        dataset_kind="canonical",
        coverage_kind=coverage_kind,
        label_space_version="solid_state_v1",
        notes=(
            "Canonical vector-first Polyfolds dataset.",
            "Neutral render roles are stored now; final semantic colors remain deferred.",
            "Legacy raster datasets under polyfolds/dataset_* are reference material only.",
        ),
    )

    samples_path = out_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row) + "\n")

    payload = {"manifest": manifest.to_dict(), "per_solid": per_solid_summary}
    (out_dir / "dataset_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the canonical vector-first Polyfolds dataset.")
    parser.add_argument("--out-dir", default="data\\canonical_core", help="Output dataset root.")
    parser.add_argument(
        "--solid",
        action="append",
        dest="solids",
        choices=sorted(SOLID_SPECS.keys()),
        help="Limit generation to one or more solids. Repeat the flag to choose multiple.",
    )
    parser.add_argument("--hard-valid-limit", type=int, default=2048, help="Unique valid topology target for dodeca and icosa.")
    parser.add_argument("--seed", type=int, default=20260318, help="Base deterministic seed.")
    parser.add_argument("--preview-png", action="store_true", help="Write optional neutral preview PNGs.")
    parser.add_argument("--preview-size", type=int, default=256, help="Preview PNG size.")
    parser.add_argument("--max-tries-per-net", type=int, default=500, help="Max unfold attempts per net sample.")
    parser.add_argument("--max-attempts-per-solid", type=int, default=120000, help="Unique topology search budget per solid.")
    parser.add_argument("--test", action="store_true", help="Generate a tiny sampled dataset for smoke testing.")
    args = parser.parse_args(argv)

    payload = build_canonical_dataset(
        CanonicalBuildConfig(
            out_dir=str(args.out_dir),
            solids=tuple(args.solids or tuple(SOLID_SPECS.keys())),
            hard_valid_limit=int(args.hard_valid_limit),
            seed=int(args.seed),
            preview_png=bool(args.preview_png),
            preview_size=int(args.preview_size),
            max_tries_per_net=int(args.max_tries_per_net),
            max_attempts_per_solid=int(args.max_attempts_per_solid),
            test_mode=bool(args.test),
        )
    )
    print(json.dumps(payload, indent=2))
    return 0
