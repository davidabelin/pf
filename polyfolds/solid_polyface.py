"""CLI workflow helpers for generating Polyfolds nets and labeled datasets.

Role
----
This module wraps the lower-level geometric generators in `platonic_nets` with
task-oriented commands for one-time offline use: sample valid nets, synthesize
incomplete and invalid variants, render PNG datasets, and emit `labels.jsonl`
records for later manifest/training stages.

Cross-Repo Context
------------------
These commands are intentionally offline and developer-facing. The deployed
`pf_web` app should eventually serve trained models, not run these expensive
generation steps on demand.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

from bootstrap_paths import ensure_polyfolds_paths

ensure_polyfolds_paths()

from platonic_nets import Net2D, NetFace2D, net_to_json, render_net_png, unfold_random_net, unfold_random_net_with_collision
from solid_common import DatasetDefaults, NetsDefaults, add_dataset_args, add_nets_args, apply_test_scale


@dataclass(frozen=True)
class SolidSpec:
    """Static configuration describing one platonic-solid family."""

    key: str  # short name: tetra|octa|dodeca|icosa
    polyhedra_id: int  # polyhedra.PlatonicSolid id
    faces_total: int
    nets_max: int | None
    defaults_dataset: DatasetDefaults
    defaults_nets: NetsDefaults
    default_missing: int


def build_parser(spec: SolidSpec) -> argparse.ArgumentParser:
    """Build the CLI parser for one solid-specific tool entrypoint."""

    p = argparse.ArgumentParser(description=f"{spec.key} tools")
    sub = p.add_subparsers(dest="cmd")

    nets = sub.add_parser("nets", help=f"Generate unfolded 2D {spec.key} nets (random non-overlapping)")
    add_nets_args(nets, defaults=spec.defaults_nets)

    ds = sub.add_parser("dataset", help=f"Generate {spec.key} net datasets (PNG + labels.jsonl)")
    add_dataset_args(ds, defaults=spec.defaults_dataset)
    ds.add_argument("--sidelength", type=float, default=1.0, help="edge length (default: 1)")
    ds.add_argument("--max-tries", type=int, default=spec.defaults_nets.max_tries, help=f"max attempts per sample before failing (default: {spec.defaults_nets.max_tries})")
    ds.add_argument("--missing", type=int, default=spec.default_missing, help=f"max missing faces for incomplete class (default: {spec.default_missing})")

    return p


def _derive_seed(base_seed: int, cls_code: int, idx: int) -> int:
    x = (base_seed & 0xFFFFFFFFFFFFFFFF) ^ ((cls_code & 0xFF) << 48) ^ (idx & 0xFFFFFFFFFFFF)
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return int(z & 0xFFFFFFFF)


def _split_of(i: int, total: int) -> str:
    r = i / max(1, total)
    return "train" if r < 0.8 else ("val" if r < 0.9 else "test")


def _tree_adj(face_count: int, edges: Sequence[tuple[int, int, tuple[int, int]]]) -> dict[int, set[int]]:
    adj = {i: set() for i in range(face_count)}
    for a, b, _e in edges:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


def _pick_leaf_removals(
    *,
    face_count: int,
    tree_edges: Sequence[tuple[int, int, tuple[int, int]]],
    root: int,
    k: int,
    rng: random.Random,
) -> set[int]:
    """Choose removable leaf faces for an incomplete-net variant.

    Notes
    -----
    Removing leaves preserves a connected remainder more often than removing
    arbitrary faces, which makes incomplete samples closer to plausible partial
    folding work rather than arbitrary graph damage.
    """

    adj = _tree_adj(face_count, tree_edges)
    removed: set[int] = set()
    root = int(root)

    for _ in range(max(0, int(k))):
        leaves = [f for f, ns in adj.items() if f != root and f not in removed and len(ns) == 1]
        if not leaves:
            break
        leaf = int(rng.choice(leaves))
        (nb,) = tuple(adj[leaf])
        adj[nb].remove(leaf)
        adj[leaf].clear()
        removed.add(leaf)

    return removed


def _subset_net(net: Net2D, *, keep_faces: set[int]) -> Net2D:
    faces = tuple(f for f in net.faces if int(f.face_index) in keep_faces)
    edges = tuple((a, b, e) for (a, b, e) in net.tree_edges if int(a) in keep_faces and int(b) in keep_faces)
    return Net2D(
        solid_name=net.solid_name,
        solid_id=net.solid_id,
        sidelength=net.sidelength,
        root_face=net.root_face,
        tree_edges=edges,
        faces=faces,
    )


def _faces_by_index(faces: Iterable[NetFace2D]) -> dict[int, NetFace2D]:
    return {int(f.face_index): f for f in faces}


def _make_incomplete(net: Net2D, *, rng: random.Random, max_missing: int, faces_total: int) -> tuple[Net2D, list[NetFace2D]]:
    """Create an incomplete variant and return the held-out completion faces."""

    max_missing = max(1, int(max_missing))
    choices = [1] * 3 + [2]
    if max_missing >= 3:
        choices.append(max_missing)
    k = int(rng.choice(choices))
    k = max(1, min(k, max_missing))

    removed = _pick_leaf_removals(face_count=faces_total, tree_edges=net.tree_edges, root=net.root_face, k=k, rng=rng)
    if not removed:
        candidates = [int(f.face_index) for f in net.faces if int(f.face_index) != int(net.root_face)]
        removed = {int(rng.choice(candidates))}

    keep = {int(f.face_index) for f in net.faces} - removed
    partial = _subset_net(net, keep_faces=keep)

    by_idx = _faces_by_index(net.faces)
    completion = [by_idx[i] for i in sorted(list(removed)) if i in by_idx]
    return partial, completion


def _centroid(poly: Sequence[tuple[float, float]]) -> tuple[float, float]:
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    return (float(cx), float(cy))


def _tree_parent_children(net: Net2D) -> tuple[dict[int, int], dict[int, list[int]], dict[tuple[int, int], tuple[int, int]]]:
    parent: dict[int, int] = {int(net.root_face): -1}
    children: dict[int, list[int]] = {}
    shared_edge: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b, e in net.tree_edges:
        pa = int(a)
        ch = int(b)
        parent[ch] = pa
        children.setdefault(pa, []).append(ch)
        shared_edge[(pa, ch)] = (int(e[0]), int(e[1]))
    return parent, children, shared_edge


def _subtree_nodes(children: dict[int, list[int]], root: int) -> set[int]:
    out: set[int] = set()
    stack = [int(root)]
    while stack:
        u = int(stack.pop())
        if u in out:
            continue
        out.add(u)
        stack.extend(children.get(u, []))
    return out


def _translate_face(face: NetFace2D, *, dx: float, dy: float) -> NetFace2D:
    return NetFace2D(
        face_index=face.face_index,
        vertex_ids=face.vertex_ids,
        xy=tuple((float(x + dx), float(y + dy)) for x, y in face.xy),
    )


def _reflect_point_about_line(
    p: tuple[float, float],
    *,
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-18:
        return (float(px), float(py))
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    qx = ax + t * dx
    qy = ay + t * dy
    rx = 2.0 * qx - px
    ry = 2.0 * qy - py
    return (float(rx), float(ry))


def _reflect_face_about_line(face: NetFace2D, *, a: tuple[float, float], b: tuple[float, float]) -> NetFace2D:
    return NetFace2D(
        face_index=face.face_index,
        vertex_ids=face.vertex_ids,
        xy=tuple(_reflect_point_about_line((float(x), float(y)), a=a, b=b) for x, y in face.xy),
    )


def _flip_subtree_about_shared_edge(net: Net2D, *, rng: random.Random) -> Net2D | None:
    """Reflect one subtree through a hinge edge to induce an invalid overlap."""

    parent, children, shared_edge = _tree_parent_children(net)
    root = int(net.root_face)
    by_idx = _faces_by_index(net.faces)

    candidates = [int(f.face_index) for f in net.faces if int(f.face_index) != root and int(f.face_index) in parent]
    if not candidates:
        return None

    cut_root = int(rng.choice(candidates))
    cut_parent = int(parent.get(cut_root, -1))
    if cut_parent < 0:
        return None

    va, vb = shared_edge.get((cut_parent, cut_root), (None, None))
    if va is None or vb is None:
        return None

    face_child = by_idx.get(cut_root)
    if face_child is None:
        return None

    mp = {int(v): (float(x), float(y)) for v, (x, y) in zip(face_child.vertex_ids, face_child.xy)}
    if int(va) not in mp or int(vb) not in mp:
        return None

    a = mp[int(va)]
    b = mp[int(vb)]
    subtree = _subtree_nodes(children, cut_root)

    new_faces: list[NetFace2D] = []
    for f in net.faces:
        fi = int(f.face_index)
        if fi in subtree:
            new_faces.append(_reflect_face_about_line(f, a=a, b=b))
        else:
            new_faces.append(f)

    return Net2D(
        solid_name=net.solid_name,
        solid_id=net.solid_id,
        sidelength=net.sidelength,
        root_face=net.root_face,
        tree_edges=net.tree_edges,
        faces=tuple(new_faces),
    )


def _project_poly(poly: Sequence[tuple[float, float]], axis: tuple[float, float]) -> tuple[float, float]:
    ax, ay = axis
    vals = [(x * ax + y * ay) for x, y in poly]
    return (min(vals), max(vals))


def _sat_convex_overlap(poly_a: Sequence[tuple[float, float]], poly_b: Sequence[tuple[float, float]], eps: float) -> bool:
    def axes(poly: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for i in range(len(poly)):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % len(poly)]
            ex, ey = (x1 - x0), (y1 - y0)
            nx, ny = (-ey, ex)
            ln = (nx * nx + ny * ny) ** 0.5
            if ln <= 1e-18:
                continue
            out.append((nx / ln, ny / ln))
        return out

    for axis in (axes(poly_a) + axes(poly_b)):
        a0, a1 = _project_poly(poly_a, axis)
        b0, b1 = _project_poly(poly_b, axis)
        if a1 <= b0 + eps or b1 <= a0 + eps:
            return False
    return True


def _net_has_overlap(net: Net2D, *, eps: float = 1e-6) -> bool:
    faces = list(net.faces)
    for i in range(len(faces)):
        poly_a = [(float(x), float(y)) for x, y in faces[i].xy]
        for j in range(i + 1, len(faces)):
            poly_b = [(float(x), float(y)) for x, y in faces[j].xy]
            if _sat_convex_overlap(poly_a, poly_b, eps):
                return True
    return False


def _net_has_duplicate_polygons(net: Net2D, *, decimals: int = 3) -> bool:
    def sig(face: NetFace2D) -> tuple[tuple[float, float], ...]:
        pts = [(round(float(x), decimals), round(float(y), decimals)) for x, y in face.xy]
        # Polygon vertex order can differ; compare as a multiset signature.
        return tuple(sorted(pts))

    seen: set[tuple[tuple[float, float], ...]] = set()
    for f in net.faces:
        s = sig(f)
        if s in seen:
            return True
        seen.add(s)
    return False


def _invalidate_by_detaching_subtree(net: Net2D, *, rng: random.Random, sidelength: float) -> Net2D:
    """Create a disconnected invalid net by translating one attached subtree."""

    parent, children, shared_edge = _tree_parent_children(net)
    root = int(net.root_face)
    candidates = [int(f.face_index) for f in net.faces if int(f.face_index) != root and int(f.face_index) in parent]
    if not candidates:
        return net

    cut_root = int(rng.choice(candidates))
    cut_parent = int(parent[cut_root])
    if cut_parent < 0:
        return net

    subtree = _subtree_nodes(children, cut_root)
    by_idx = _faces_by_index(net.faces)
    face_child = by_idx.get(cut_root)
    face_parent = by_idx.get(cut_parent)
    if face_child is None or face_parent is None:
        return net

    va, vb = shared_edge.get((cut_parent, cut_root), (None, None))
    if va is None or vb is None:
        return net

    mp_child = {int(v): (float(x), float(y)) for v, (x, y) in zip(face_child.vertex_ids, face_child.xy)}
    if int(va) not in mp_child or int(vb) not in mp_child:
        return net
    ax, ay = mp_child[int(va)]
    bx, by = mp_child[int(vb)]
    ex, ey = (bx - ax), (by - ay)
    ln = (ex * ex + ey * ey) ** 0.5
    if ln <= 1e-12:
        return net

    # Unit perpendicular (either sign).
    px, py = (-ey / ln, ex / ln)

    cpx, cpy = _centroid(face_parent.xy)
    ccx, ccy = _centroid(face_child.xy)
    vx, vy = (ccx - cpx), (ccy - cpy)
    if (px * vx + py * vy) < 0.0:
        px, py = (-px, -py)

    gap = rng.uniform(0.10, 0.22) * float(sidelength)
    dx, dy = (px * gap, py * gap)

    new_faces: list[NetFace2D] = []
    for f in net.faces:
        fi = int(f.face_index)
        if fi in subtree:
            new_faces.append(_translate_face(f, dx=dx, dy=dy))
        else:
            new_faces.append(f)

    return Net2D(
        solid_name=net.solid_name,
        solid_id=net.solid_id,
        sidelength=net.sidelength,
        root_face=net.root_face,
        tree_edges=net.tree_edges,
        faces=tuple(new_faces),
    )


def _nets_worker_task(task: dict[str, Any]) -> tuple[int, str, str | None]:
    """Generate and optionally render one valid net task for the `nets` command."""

    i = int(task["i"])
    seed = int(task["seed"])
    net = unfold_random_net(
        solid_id=int(task["polyhedra_id"]),
        sidelength=float(task["sidelength"]),
        seed=seed,
        max_tries=int(task["max_tries"]),
    )
    json_line = json.dumps(net_to_json(net))

    png_path = task.get("png_path")
    if png_path:
        render_net_png(
            net,
            out_path=str(png_path),
            img_size=int(task["img_size"]),
            palette=str(task["palette"]),
            line_width=float(task["line_width"]),
            labels=bool(task["labels"]),
            seed=seed,
        )
        return i, json_line, str(png_path)
    return i, json_line, None


def run_nets(spec: SolidSpec, args: argparse.Namespace) -> None:
    """Generate a batch of valid nets plus optional PNG previews.

    Role
    ----
    This is the lighter-weight developer tool used to inspect raw unfoldings
    before stepping up to full dataset generation.
    """

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "nets.jsonl")

    workers = 0 if bool(getattr(args, "fast", False)) else int(args.workers)
    if workers <= 0:
        workers = int(os.cpu_count() or 1)
    chunksize = max(1, int(args.chunksize))

    tasks: list[dict[str, Any]] = []
    for i in range(int(args.count)):
        seed = int(args.seed) + i
        png_path = None if bool(args.no_png) else os.path.join(out_dir, f"net_{i:04d}.png")
        tasks.append(
            {
                "i": i,
                "seed": seed,
                "polyhedra_id": int(spec.polyhedra_id),
                "sidelength": float(args.sidelength),
                "max_tries": int(args.max_tries),
                "img_size": int(args.img_size),
                "palette": str(args.palette),
                "line_width": float(args.line_width),
                "labels": bool(args.labels),
                "png_path": png_path,
            }
        )

    results: list[tuple[int, str, str | None]] = []
    if workers == 1:
        for t in tasks:
            results.append(_nets_worker_task(t))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for r in pool.imap(_nets_worker_task, tasks, chunksize=chunksize):
                results.append(r)

    results.sort(key=lambda x: x[0])
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _i, json_line, _png in results:
            f.write(json_line + "\n")

    print(f"Wrote: {jsonl_path}")
    if not bool(args.no_png) and int(args.count) > 0:
        print(f"Wrote: {os.path.join(out_dir, 'net_0000.png')}")


def _dataset_worker_task(task: dict[str, Any]) -> tuple[int, str]:
    """Generate and label one dataset sample for valid/incomplete/invalid data.

    Notes
    -----
    Invalid samples are attempted in order of realism: natural overlap,
    hinge-flip overlap, then disconnected corruption as a fallback.
    """

    order = int(task["order"])
    idx = int(task["idx"])
    cls = str(task["class"])
    total = int(task["total"])
    base_seed = int(task["seed"])
    rng = random.Random(base_seed)

    spec: SolidSpec = task["spec"]
    max_tries = int(task["max_tries"])
    sidelength = float(task["sidelength"])
    img_size = int(task["img_size"])
    palette = str(task["palette"])
    line_width = float(task["line_width"])
    out_path = str(task["out_path"])

    overlaps = False
    invalid_reason = None

    if cls == "valid":
        net = unfold_random_net(solid_id=spec.polyhedra_id, sidelength=sidelength, seed=base_seed, max_tries=max_tries)
        completion_faces: list[NetFace2D] = []
    elif cls == "incomplete":
        full = unfold_random_net(solid_id=spec.polyhedra_id, sidelength=sidelength, seed=base_seed, max_tries=max_tries)
        net, completion_faces = _make_incomplete(full, rng=rng, max_missing=int(task["missing"]), faces_total=spec.faces_total)
    elif cls == "invalid":
        # Prefer realistic "bad nets":
        # 1) An unfolding that overlaps (rare for some solids).
        # 2) A connected net where a subtree is flipped through a shared edge (can induce overlap).
        # 3) A disconnected net (a subtree detached by a small gap).
        net = None
        for j in range(80):
            net_j, collision = unfold_random_net_with_collision(
                solid_id=spec.polyhedra_id,
                sidelength=sidelength,
                seed=base_seed + j,
                max_tries=max_tries,
            )
            net = net_j
            if collision:
                overlaps = True
                invalid_reason = "overlap"
                break

        if net is None:
            raise RuntimeError("failed to generate invalid sample")

        if not overlaps:
            # Try forcing an overlap while keeping edge attachments intact.
            base = unfold_random_net(solid_id=spec.polyhedra_id, sidelength=sidelength, seed=base_seed, max_tries=max_tries)
            for _k in range(40):
                flipped = _flip_subtree_about_shared_edge(base, rng=rng)
                if flipped is None:
                    continue
                if _net_has_duplicate_polygons(flipped):
                    continue
                if _net_has_overlap(flipped):
                    net = flipped
                    overlaps = True
                    invalid_reason = "overlap_flip"
                    break

        if not overlaps:
            net = _invalidate_by_detaching_subtree(net, rng=rng, sidelength=sidelength)
            invalid_reason = "disconnected"
        completion_faces = []
    else:
        raise ValueError(f"unknown class: {cls}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    render_net_png(
        net,
        out_path=out_path,
        img_size=img_size,
        palette=palette,
        line_width=line_width,
        labels=False,
        seed=base_seed,
        dedupe_edges=(cls != "invalid"),
    )

    rec = {
        "file": os.path.join("images", os.path.basename(out_path)),
        "split": _split_of(idx, total),
        "class": cls,
        "solid": spec.key,
        "faces_total": spec.faces_total,
        "faces_present": len(net.faces),
        "net": net_to_json(net),
        "completion_faces": [
            {"face_index": f.face_index, "vertex_ids": list(f.vertex_ids), "xy": [list(p) for p in f.xy]}
            for f in completion_faces
        ],
        "overlaps": bool(overlaps),
        "invalid_reason": invalid_reason,
        "seed": int(base_seed),
    }
    return order, json.dumps(rec)


def _write_preview(out_dir: str, solid_key: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = ["valid", "incomplete", "invalid"]
    files = [
        os.path.join(out_dir, "images", f"valid_{solid_key}_0000000.png"),
        os.path.join(out_dir, "images", f"incomplete_{solid_key}_0000000.png"),
        os.path.join(out_dir, "images", f"invalid_{solid_key}_0000000.png"),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, title, path in zip(axs, titles, files):
        ax.set_title(title)
        ax.axis("off")
        if os.path.exists(path):
            img = plt.imread(path)
            ax.imshow(img)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "preview.png"), dpi=200)
    plt.close(fig)


def run_dataset(spec: SolidSpec, args: argparse.Namespace) -> None:
    """Generate a labeled PNG dataset and `labels.jsonl` for one solid family.

    Role
    ----
    This is the main one-time offline dataset command that later feeds the
    manifest builder and baseline classifier training flow.
    """

    out_dir = str(args.out_dir)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    n_valid = int(args.n_valid)
    n_incomplete = int(args.n_incomplete)
    n_invalid = int(args.n_invalid)
    if bool(args.test):
        n_valid = apply_test_scale(n_valid)
        n_incomplete = apply_test_scale(n_incomplete)
        n_invalid = apply_test_scale(n_invalid)

    max_tries = int(args.max_tries)
    sidelength = float(args.sidelength)

    workers = 0 if bool(getattr(args, "fast", False)) else int(args.workers)
    if workers <= 0:
        workers = int(os.cpu_count() or 1)
    chunksize = max(1, int(args.chunksize))

    order = 0

    def add_tasks(cls: str, cls_code: int, total: int) -> list[dict[str, Any]]:
        nonlocal order
        tasks: list[dict[str, Any]] = []
        for i in range(total):
            seed = _derive_seed(int(args.seed), cls_code, i)
            fname = f"{cls}_{spec.key}_{i:07d}.png"
            tasks.append(
                {
                    "order": order,
                    "idx": i,
                    "total": total,
                    "class": cls,
                    "seed": seed,
                    "out_path": os.path.join(img_dir, fname),
                    "img_size": int(args.img_size),
                    "palette": str(args.palette),
                    "line_width": float(args.line_width),
                    "max_tries": max_tries,
                    "sidelength": sidelength,
                    "missing": int(args.missing),
                    "spec": spec,
                }
            )
            order += 1
        return tasks

    tasks_all = [
        *add_tasks("valid", 1, n_valid),
        *add_tasks("incomplete", 2, n_incomplete),
        *add_tasks("invalid", 3, n_invalid),
    ]

    labels_path = os.path.join(out_dir, "labels.jsonl")
    with open(labels_path, "w", encoding="utf-8") as f:
        if workers == 1:
            for t in tasks_all:
                _order, rec_json = _dataset_worker_task(t)
                f.write(rec_json + "\n")
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                for _order, rec_json in pool.imap(_dataset_worker_task, tasks_all, chunksize=chunksize):
                    f.write(rec_json + "\n")

    print(f"Wrote: {labels_path}")
    if bool(args.plot):
        _write_preview(out_dir, spec.key)


def main(spec: SolidSpec, argv: List[str] | None = None) -> None:
    """Dispatch one solid-specific CLI command.

    Used By
    -------
    The `solid_*.py` entrypoints that bind a concrete `SolidSpec` and expose
    it as a runnable developer tool.
    """

    args = build_parser(spec).parse_args(argv)

    if args.cmd == "nets":
        run_nets(spec, args)
        return

    if args.cmd == "dataset":
        run_dataset(spec, args)
        return

    build_parser(spec).print_help()
