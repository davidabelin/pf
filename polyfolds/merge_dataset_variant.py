from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


def _sanitize_token(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "x"


def _make_new_basename(*, old_basename: str, solid: str, variant: str) -> str:
    solid_t = _sanitize_token(solid)
    variant_t = _sanitize_token(variant)

    name = Path(old_basename).name
    m = re.match(r"^(valid|incomplete|invalid)_(.+?)_(\d+)\.png$", name, flags=re.IGNORECASE)
    if m:
        cls, solid0, num = m.groups()
        solid0_t = _sanitize_token(solid0)
        if solid0_t == solid_t:
            return f"{cls.lower()}_{solid_t}_{variant_t}_{int(num):07d}.png"

    stem = Path(name).stem
    m2 = re.match(r"^(valid|incomplete|invalid)_(.+)$", stem, flags=re.IGNORECASE)
    if m2:
        cls, rest = m2.groups()
        rest_t = str(rest)
        rest_low = rest_t.lower()
        solid_prefix = solid_t + "_"
        if rest_low.startswith(solid_prefix):
            rest_t = rest_t[len(solid_prefix) :]
        rest_t = rest_t.strip("_") or "0"
        return f"{cls.lower()}_{solid_t}_{variant_t}_{rest_t}.png"

    return f"misc_{solid_t}_{variant_t}_{stem}.png"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line_num, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON on {path}:{line_num}") from e
        if not isinstance(obj, dict):
            raise RuntimeError(f"Expected object JSON on {path}:{line_num}")
        out.append(obj)
    return out


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Merge a dataset variant into a destination dataset by renaming images and appending labels.jsonl.",
    )
    p.add_argument("--src", required=True, help="source dataset dir (e.g. dataset_icosa_pale)")
    p.add_argument("--dst", required=True, help="destination dataset dir (e.g. dataset_icosa)")
    p.add_argument("--solid", required=True, help="solid token (e.g. icosa)")
    p.add_argument("--variant", required=True, help="variant token inserted into filenames (e.g. pale)")
    p.add_argument("--delete-src", action="store_true", help="delete the entire --src directory after merge")
    p.add_argument("--dry-run", action="store_true", help="print actions but do not modify files")
    args = p.parse_args(argv)

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    src_images = src_dir / "images"
    dst_images = dst_dir / "images"

    if not src_dir.exists():
        raise SystemExit(f"Missing --src: {src_dir}")
    if not src_images.exists():
        raise SystemExit(f"Missing images dir: {src_images}")
    if not dst_dir.exists():
        raise SystemExit(f"Missing --dst: {dst_dir}")

    solid = str(args.solid)
    variant = str(args.variant)

    dst_images.mkdir(parents=True, exist_ok=True)

    pngs = sorted(src_images.glob("*.png"))
    if not pngs:
        raise SystemExit(f"No PNGs found in: {src_images}")

    rename_map: dict[str, str] = {}
    for src_path in pngs:
        old_basename = src_path.name
        new_basename = _make_new_basename(old_basename=old_basename, solid=solid, variant=variant)

        candidate = dst_images / new_basename
        if candidate.exists():
            stem = candidate.stem
            suffix = candidate.suffix
            k = 1
            while True:
                cand2 = dst_images / f"{stem}_dup{k}{suffix}"
                if not cand2.exists():
                    candidate = cand2
                    new_basename = cand2.name
                    break
                k += 1

        rename_map[old_basename] = new_basename

        print(f"move: {src_path} -> {candidate}")
        if not args.dry_run:
            shutil.move(str(src_path), str(candidate))

    # Merge labels
    dst_labels_path = dst_dir / "labels.jsonl"
    src_labels_path = src_dir / "labels.jsonl"
    if not src_labels_path.exists():
        raise SystemExit(f"Missing labels.jsonl in --src: {src_labels_path}")

    dst_records = _read_jsonl(dst_labels_path)
    dst_files = {str(rec.get("file", "")).strip() for rec in dst_records}

    src_records = _read_jsonl(src_labels_path)
    appended = 0
    for rec in src_records:
        old_file = str(rec.get("file", "")).strip()
        if not old_file:
            continue
        old_basename = Path(old_file).name
        new_basename = rename_map.get(old_basename)
        if not new_basename:
            # If an image didn't move (or labels refer to a non-image), skip it.
            continue

        rec2 = dict(rec)
        rec2["file"] = f"images\\{new_basename}"
        if rec2["file"] in dst_files:
            continue
        dst_records.append(rec2)
        dst_files.add(rec2["file"])
        appended += 1

    print(f"labels: append {appended} records -> {dst_labels_path}")
    if not args.dry_run:
        _write_jsonl(dst_labels_path, dst_records)

    if bool(args.delete_src):
        print(f"delete: {src_dir}")
        if not args.dry_run:
            shutil.rmtree(src_dir)


if __name__ == "__main__":
    main()

