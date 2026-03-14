from __future__ import annotations

import argparse
import colorsys
import json
import math
import multiprocessing as mp
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, List, Set, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection, PolyCollection

from cube_nets_discrete import ALL_NORMALS, DIRS, Cell, enumerate_free_polyominoes, foldable_cube_net_discrete, is_connected

try:
    # `polyhedra` is optional for this file, but preferred when installed.
    from polyhedra import rotate_about_line
except Exception:  # pragma: no cover
    rotate_about_line = None


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-length vector")
    return v / n


@dataclass
class Frame3D:
    origin: np.ndarray  # (3,)
    ex: np.ndarray  # (3,) local +x edge direction
    ey: np.ndarray  # (3,) local +y edge direction

    def normal(self) -> np.ndarray:
        return _unit(np.cross(self.ex, self.ey))

    def corners(self) -> list[np.ndarray]:
        o = self.origin
        return [o, o + self.ex, o + self.ey, o + self.ex + self.ey]


def _orthonormalize(ex: np.ndarray, ey: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ex = _unit(ex)
    ey = ey - ex * float(np.dot(ex, ey))
    ey = _unit(ey)
    return ex, ey


def _quantize_axis(v: np.ndarray, *, tol: float = 1e-6) -> tuple[int, int, int] | None:
    r = np.rint(v).astype(int)
    t = (int(r[0]), int(r[1]), int(r[2]))
    if t not in ALL_NORMALS:
        return None
    if not np.allclose(v, np.asarray(r, dtype=float), atol=tol):
        return None
    return t


def _rotate_frame_about_line(fr: Frame3D, *, base_pt: np.ndarray, axis: np.ndarray, theta: float) -> Frame3D:
    if rotate_about_line is None:
        raise RuntimeError("polyhedra is not available (rotate_about_line missing)")

    def _rotate_about_line_safe(p: np.ndarray) -> np.ndarray:
        pv = np.asarray(p, dtype=float)
        bpv = np.asarray(base_pt, dtype=float)
        av = np.asarray(axis, dtype=float)
        an = float(np.linalg.norm(av))
        if an == 0.0:
            raise ValueError("zero-length rotation axis")

        # polyhedra.tools.rotate_about_line() divides by ||cross(axis, rv1)||,
        # which is zero when the point lies on the axis.
        dist_times_an = float(np.linalg.norm(np.cross(pv - bpv, av)))
        if dist_times_an <= 1e-12 * an:
            return pv.copy()

        return rotate_about_line(pv, bpv, av, theta)

    o = _rotate_about_line_safe(fr.origin)
    ex_end = _rotate_about_line_safe(fr.origin + fr.ex)
    ey_end = _rotate_about_line_safe(fr.origin + fr.ey)
    ex, ey = _orthonormalize(ex_end - o, ey_end - o)
    return Frame3D(origin=o, ex=ex, ey=ey)


def _step_frame_3d(parent: Frame3D, dir2: str) -> Frame3D:
    if dir2 == "E":
        pre = Frame3D(origin=parent.origin + parent.ex, ex=parent.ex, ey=parent.ey)
        return _rotate_frame_about_line(
            pre,
            base_pt=parent.origin + parent.ex,
            axis=parent.ey,
            theta=+math.pi / 2,
        )
    if dir2 == "W":
        pre = Frame3D(origin=parent.origin - parent.ex, ex=parent.ex, ey=parent.ey)
        return _rotate_frame_about_line(
            pre,
            base_pt=parent.origin,
            axis=parent.ey,
            theta=-math.pi / 2,
        )
    if dir2 == "N":
        pre = Frame3D(origin=parent.origin + parent.ey, ex=parent.ex, ey=parent.ey)
        return _rotate_frame_about_line(
            pre,
            base_pt=parent.origin + parent.ey,
            axis=parent.ex,
            theta=-math.pi / 2,
        )
    if dir2 == "S":
        pre = Frame3D(origin=parent.origin - parent.ey, ex=parent.ex, ey=parent.ey)
        return _rotate_frame_about_line(
            pre,
            base_pt=parent.origin,
            axis=parent.ex,
            theta=+math.pi / 2,
        )
    raise ValueError(dir2)


def _key3(p: np.ndarray, *, decimals: int = 6) -> tuple[float, float, float]:
    r = np.round(p.astype(float), decimals=decimals)
    return (float(r[0]), float(r[1]), float(r[2]))


def _foldable_cube_net_polyhedra(poly: Iterable[Cell]) -> bool:
    cells = set(poly)
    if len(cells) != 6 or not is_connected(cells):
        return False

    adj: dict[Cell, list[tuple[Cell, str]]] = defaultdict(list)
    for x, y in cells:
        for d, (dx, dy) in DIRS.items():
            nb = (x + dx, y + dy)
            if nb in cells:
                adj[(x, y)].append((nb, d))

    root = next(iter(cells))
    root_fr = Frame3D(
        origin=np.asarray([0.0, 0.0, 0.0]),
        ex=np.asarray([1.0, 0.0, 0.0]),
        ey=np.asarray([0.0, 1.0, 0.0]),
    )
    frame: dict[Cell, Frame3D] = {root: root_fr}
    used_norms: Set[tuple[int, int, int]] = set()

    q = deque([root])
    while q:
        c = q.popleft()
        fr = frame[c]
        qn = _quantize_axis(fr.normal())
        if qn is None:
            return False
        used_norms.add(qn)

        for nb, d in adj[c]:
            fr_nb = _step_frame_3d(fr, d)
            if nb in frame:
                prev = frame[nb]
                if not (
                    np.allclose(prev.origin, fr_nb.origin, atol=1e-6)
                    and np.allclose(prev.ex, fr_nb.ex, atol=1e-6)
                    and np.allclose(prev.ey, fr_nb.ey, atol=1e-6)
                ):
                    return False
                continue

            qn_nb = _quantize_axis(fr_nb.normal())
            if qn_nb is None or qn_nb in used_norms:
                return False

            frame[nb] = fr_nb
            used_norms.add(qn_nb)
            q.append(nb)

    if used_norms != ALL_NORMALS:
        return False

    verts: Set[tuple[float, float, float]] = set()
    for fr in frame.values():
        for v in fr.corners():
            verts.add(_key3(v))
    return len(verts) == 8


def foldable_cube_net(poly: Iterable[Cell]) -> bool:
    """
    Returns True iff the 6-cell polyomino can fold to a cube (i.e., is one of the 11 nets).

    Prefers a polyhedra-based 3D hinge fold simulation when available; otherwise falls back
    to a discrete orientation propagation method.
    """

    if rotate_about_line is not None:
        return _foldable_cube_net_polyhedra(poly)
    return foldable_cube_net_discrete(poly)


def remove_cells_for_incomplete(net: Tuple[Cell, ...], k: int) -> Tuple[Tuple[Cell, ...], List[Cell]]:
    return remove_cells_for_incomplete_rng(net, k, rng=random)


def remove_cells_for_incomplete_rng(
    net: Tuple[Cell, ...],
    k: int,
    *,
    rng: random.Random,
) -> Tuple[Tuple[Cell, ...], List[Cell]]:
    cells = list(net)
    for _ in range(200):
        removed = rng.sample(cells, k)
        remain = set(cells) - set(removed)
        if len(remain) >= 1 and is_connected(remain):
            return tuple(sorted(remain)), sorted(removed)
    removed = rng.sample(cells, k)
    remain = set(cells) - set(removed)
    return tuple(sorted(remain)), sorted(removed)


def poly_to_polygons(cells: Iterable[Cell], jitter: float = 0.0) -> List[np.ndarray]:
    polys: List[np.ndarray] = []
    vertex_offsets: dict[tuple[int, int], np.ndarray] = {}

    def v(ix: int, iy: int) -> np.ndarray:
        key = (int(ix), int(iy))
        off = vertex_offsets.get(key)
        if off is None:
            if jitter > 0:
                off = np.random.uniform(-jitter, jitter, size=(2,))
            else:
                off = np.zeros((2,), dtype=float)
            vertex_offsets[key] = off
        return np.asarray([float(ix), float(iy)], dtype=float) + off

    for x, y in cells:
        pts = np.vstack([v(x, y), v(x + 1, y), v(x + 1, y + 1), v(x, y + 1)])
        polys.append(pts)
    return polys


def _pastel_rgba(*, alpha: float = 1.0) -> tuple[float, float, float, float]:
    h = random.random()
    l = random.uniform(0.78, 0.92)
    s = random.uniform(0.10, 0.30)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (float(r), float(g), float(b), float(alpha))


def _derive_seed(base_seed: int, cls_code: int, idx: int) -> int:
    x = (base_seed & 0xFFFFFFFFFFFFFFFF) ^ ((cls_code & 0xFF) << 48) ^ (idx & 0xFFFFFFFFFFFF)
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return int(z & 0xFFFFFFFF)


def _render_task(task: dict[str, Any]) -> str:
    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)

    render_sample(
        task["out_path"],
        task["cells"],
        img_size=int(task["img_size"]),
        palette=str(task["palette"]),
        line_width=float(task["line_width"]),
        bg=tuple(task["bg"]),
    )
    return json.dumps(task["record"])


def render_sample(
    out_path: str,
    cells: Iterable[Cell],
    *,
    img_size: int = 512,
    bg: Tuple[float, float, float] = (1, 1, 1),
    palette: str = "pastel",
    line_width: float = 4.0,
) -> None:
    polys = poly_to_polygons(cells, jitter=0.0)

    all_pts = np.vstack(polys)
    centroid = all_pts.mean(axis=0)

    angle = random.choice([0, 90, 180, 270]) + random.uniform(-7, 7)
    theta = math.radians(angle)
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    scale = random.uniform(60, 110)

    tpolys = []
    for pts in polys:
        q = (pts - centroid) @ R.T
        q *= (scale / 1.0)
        tpolys.append(q)

    allq = np.vstack(tpolys)
    minxy = allq.min(axis=0)
    maxxy = allq.max(axis=0)
    span = maxxy - minxy
    pad = random.uniform(40, 90)

    target_span = (img_size - 2 * pad)
    s2 = min(target_span / span[0], target_span / span[1])
    tpolys2 = []
    for q in tpolys:
        qq = (q - minxy) * s2
        tpolys2.append(qq)

    allq2 = np.vstack(tpolys2)
    maxxy2 = allq2.max(axis=0)
    ox = random.uniform(pad * 0.5, img_size - maxxy2[0] - pad * 0.5)
    oy = random.uniform(pad * 0.5, img_size - maxxy2[1] - pad * 0.5)

    palette_l = palette.lower().strip()
    if palette_l in {"gray", "grey"}:
        base = random.uniform(0.70, 0.90)
        face_colors = [(base, base, base, 1.0)] * len(tpolys2)
    elif palette_l in {"random"}:
        face_colors = [(random.uniform(0.1, 0.95), random.uniform(0.1, 0.95), random.uniform(0.1, 0.95), 1.0) for _ in tpolys2]
    else:  # pastel (default)
        base = _pastel_rgba(alpha=1.0)
        face_colors = [base] * len(tpolys2)

    fig = plt.figure(figsize=(img_size / 100, img_size / 100), dpi=100)
    ax = plt.gca()
    ax.set_facecolor(bg)

    polys_img = [q + np.array([ox, oy]) for q in tpolys2]
    ax.add_collection(PolyCollection(polys_img, facecolors=face_colors, edgecolors="none", antialiased=False))

    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for q in polys_img:
        for i in range(len(q)):
            p0 = (float(q[i][0]), float(q[i][1]))
            p1 = (float(q[(i + 1) % len(q)][0]), float(q[(i + 1) % len(q)][1]))
            a = (round(p0[0], 4), round(p0[1], 4))
            b = (round(p1[0], 4), round(p1[1], 4))
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            segs.append((p0, p1))

    ax.add_collection(LineCollection(segs, colors=(0, 0, 0, 1), linewidths=float(line_width), antialiased=True))

    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def make_dataset(
    out_dir: str = "dataset",
    n_valid: int = 2000,
    n_incomplete: int = 2000,
    n_invalid: int = 2000,
    solid_name: str = "cube",
    seed: int = 2025,
    img_size: int = 512,
    palette: str = "pastel",
    line_width: float = 4.0,
    workers: int = 0,
    chunksize: int = 20,
) -> None:
    solid_key = "".join((c if (c.isalnum() or c in {"_", "-"}) else "_") for c in str(solid_name).strip().lower()) or "cube"

    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    labels_path = os.path.join(out_dir, "labels.jsonl")
    hexes = enumerate_free_polyominoes(6)
    assert len(hexes) == 35, f"Expected 35 free hexominoes, got {len(hexes)}"

    valid_nets = [h for h in hexes if foldable_cube_net(h)]
    invalid_hexes = [h for h in hexes if not foldable_cube_net(h)]

    print("Free hexominoes:", len(hexes))
    print(f"Valid {solid_key} nets:", len(valid_nets))
    print("Invalid hexominoes:", len(invalid_hexes))

    def split_of(i: int, total: int) -> str:
        r = i / max(1, total)
        return "train" if r < 0.8 else ("val" if r < 0.9 else "test")

    bg = (1.0, 1.0, 1.0)
    tasks: list[dict[str, Any]] = []

    def add_task(idx: int, total: int, cls: str, cls_code: int, cells: Tuple[Cell, ...], completion: List[Cell]) -> None:
        fname = f"{cls}_{solid_key}_{idx:07d}.png"
        out_path = os.path.join(img_dir, fname)
        rec = {
            "file": os.path.join("images", fname),
            "split": split_of(idx, total),
            "class": cls,
            "solid": solid_key,
            "cells": list(map(list, cells)),
            "completion_cells": list(map(list, completion)),
        }
        tasks.append(
            {
                "seed": _derive_seed(int(seed), cls_code, int(idx)),
                "out_path": out_path,
                "cells": cells,
                "img_size": int(img_size),
                "palette": str(palette),
                "line_width": float(line_width),
                "bg": bg,
                "record": rec,
            }
        )

    for i in range(n_valid):
        sample_seed = _derive_seed(int(seed), 1, int(i))
        rng = random.Random(sample_seed)
        net = rng.choice(valid_nets)
        add_task(i, n_valid, "valid", 1, net, [])

    for i in range(n_incomplete):
        sample_seed = _derive_seed(int(seed), 2, int(i))
        rng = random.Random(sample_seed)
        net = rng.choice(valid_nets)
        k = rng.choice([1, 1, 1, 2])
        remain, completion = remove_cells_for_incomplete_rng(net, k, rng=rng)
        add_task(i, n_incomplete, "incomplete", 2, remain, completion)

    for i in range(n_invalid):
        sample_seed = _derive_seed(int(seed), 3, int(i))
        rng = random.Random(sample_seed)
        shape = rng.choice(invalid_hexes)
        add_task(i, n_invalid, "invalid", 3, shape, [])

    workers_i = int(workers)
    if workers_i <= 0:
        workers_i = int(os.cpu_count() or 1)
    chunksize_i = max(1, int(chunksize))

    with open(labels_path, "w", encoding="utf-8") as f:
        if workers_i == 1:
            for t in tasks:
                f.write(_render_task(t) + "\n")
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers_i) as pool:
                for rec_json in pool.imap(_render_task, tasks, chunksize=chunksize_i):
                    f.write(rec_json + "\n")

    print("Wrote:", labels_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate cube-net square datasets (PNG + labels.jsonl).")
    p.add_argument("--out-dir", default="dataset", help="output directory (default: dataset)")
    p.add_argument("--n-valid", type=int, default=2000, help="number of valid samples (default: 2000)")
    p.add_argument("--n-incomplete", type=int, default=2000, help="number of incomplete samples (default: 2000)")
    p.add_argument("--n-invalid", type=int, default=2000, help="number of invalid samples (default: 2000)")
    p.add_argument("--solid-name", default="cube", help="solid name used for labels/filenames (default: cube)")
    p.add_argument("--seed", type=int, default=2025, help="random seed (default: 2025)")
    p.add_argument("--img-size", type=int, default=512, help="output PNG size (default: 512)")
    p.add_argument("--palette", default="pastel", help="face color palette: pastel|random|gray (default: pastel)")
    p.add_argument("--line-width", type=float, default=4.0, help="outline line width in pixels (default: 4)")
    p.add_argument("--workers", type=int, default=0, help="worker processes (0=all cores, default: 0)")
    p.add_argument("--chunksize", type=int, default=20, help="work chunk size per process (default: 20)")
    p.add_argument("--plot", action="store_true", help="write a quick preview image to <out-dir>/preview.png")
    p.add_argument("--normals", action="store_true", help="(deprecated) normals are always shown in --plot")
    p.add_argument("--plot-normals", action="store_true", help="(deprecated) normals are always shown in --plot")
    p.add_argument(
        "--test",
        action="store_true",
        help="run ~1/10th-sized dataset for quick iteration (keeps splits/labels)",
    )
    return p


def _plot_preview(
    out_path: str,
    *,
    seed: int,
    palette: str,
    line_width: float,
) -> None:
    rng = random.Random(int(seed))
    hexes = enumerate_free_polyominoes(6)
    valid_nets = [h for h in hexes if foldable_cube_net(h)]
    invalid_hexes = [h for h in hexes if not foldable_cube_net(h)]

    valid = rng.choice(valid_nets)
    base = rng.choice(valid_nets)
    k = rng.choice([1, 1, 1, 2])
    remain, completion = remove_cells_for_incomplete_rng(base, k, rng=rng)
    invalid = rng.choice(invalid_hexes)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("valid", valid, []),
        ("incomplete", remain, completion),
        ("invalid", invalid, []),
    ]

    for ax, (title, cells, completion_cells) in zip(axs, panels):
        cells_set = set(cells)
        xs = [x for x, _ in cells_set]
        ys = [y for _, y in cells_set]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        ax.set_aspect("equal")
        ax.set_title(title)
        ax.axis("off")

        palette_l = palette.lower().strip()
        if palette_l in {"gray", "grey"}:
            base_c = (0.80, 0.80, 0.80, 1.0)
        elif palette_l == "random":
            base_c = (rng.uniform(0.1, 0.95), rng.uniform(0.1, 0.95), rng.uniform(0.1, 0.95), 1.0)
        else:
            base_c = _pastel_rgba(alpha=1.0)

        for (x, y) in sorted(cells_set):
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=base_c, edgecolor="none"))

        if completion_cells:
            for (x, y) in completion_cells:
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        1,
                        1,
                        facecolor=(0, 0, 0, 0),
                        edgecolor=(0.8, 0.1, 0.1, 1),
                        linewidth=max(1.0, float(line_width) * 0.6),
                        linestyle="--",
                    )
                )

        # Unique edges
        segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for (x, y) in cells_set:
            corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
            edges = list(zip(corners, corners[1:] + corners[:1]))
            for a, b in edges:
                key = (a, b) if a <= b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                segs.append(((a[0], a[1]), (b[0], b[1])))

        ax.add_collection(LineCollection(segs, colors=(0, 0, 0, 1), linewidths=float(line_width)))

        ax.set_xlim(minx - 0.5, maxx + 1.5)
        ax.set_ylim(miny - 0.5, maxy + 1.5)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv: List[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    n_valid = int(args.n_valid)
    n_incomplete = int(args.n_incomplete)
    n_invalid = int(args.n_invalid)
    if args.test:
        n_valid = max(1, n_valid // 10)
        n_incomplete = max(1, n_incomplete // 10)
        n_invalid = max(1, n_invalid // 10)

    make_dataset(
        out_dir=str(args.out_dir),
        n_valid=n_valid,
        n_incomplete=n_incomplete,
        n_invalid=n_invalid,
        solid_name=str(args.solid_name),
        seed=int(args.seed),
        img_size=int(args.img_size),
        palette=str(args.palette),
        line_width=float(args.line_width),
        workers=int(args.workers),
        chunksize=int(args.chunksize),
    )

    if args.plot:
        _plot_preview(
            os.path.join(str(args.out_dir), "preview.png"),
            seed=int(args.seed),
            palette=str(args.palette),
            line_width=float(args.line_width),
        )


if __name__ == "__main__":
    main()
