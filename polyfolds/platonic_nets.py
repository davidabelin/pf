from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class NetFace2D:
    face_index: int
    vertex_ids: Tuple[int, ...]
    xy: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class Net2D:
    solid_name: str
    solid_id: int
    sidelength: float
    root_face: int
    tree_edges: Tuple[Tuple[int, int, Tuple[int, int]], ...]  # (parent_face, child_face, (va,vb))
    faces: Tuple[NetFace2D, ...]


def _require_polyhedra() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        from polyhedra import PlatonicSolid
        from polyhedra import rotate_about_line

        return np, PlatonicSolid, rotate_about_line
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: install `polyhedra` in this environment.") from e


def _unit(np: Any, v: Any) -> Any:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-length vector")
    return v / n


def _face_points(np: Any, solid: Any, face_index: int) -> list[Any]:
    f = solid.faces[face_index]
    return [np.asarray(solid.get_vertex(int(i)), dtype=float) for i in f.vertex_ids]


def _face_normal(np: Any, pts: Sequence[Any]) -> Any:
    a, b, c = pts[0], pts[1], pts[2]
    n = np.cross(b - a, c - b)
    return _unit(np, n)


def _basis_from_three(np: Any, a: Any, b: Any, c: Any) -> Any:
    ex = _unit(np, b - a)
    ey_raw = c - a
    ey_raw = ey_raw - ex * float(np.dot(ex, ey_raw))
    ey = _unit(np, ey_raw)
    ez = _unit(np, np.cross(ex, ey))
    return np.column_stack([ex, ey, ez])


def _rigid_transform_from_3pts(np: Any, a: Any, b: Any, c: Any, ap: Any, bp: Any, cp: Any) -> tuple[Any, Any]:
    B = _basis_from_three(np, a, b, c)
    Bp = _basis_from_three(np, ap, bp, cp)
    R = Bp @ B.T
    t = ap - (R @ a)
    return R, t


def _solid_face_adjacency(solid: Any) -> tuple[list[list[int]], dict[tuple[int, int], tuple[int, int]]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    edge_to_oriented: dict[tuple[int, int], tuple[int, int]] = {}

    for fi, face in enumerate(solid.faces):
        ids = [int(i) for i in face.vertex_ids]
        for i in range(len(ids)):
            a = ids[i]
            b = ids[(i + 1) % len(ids)]
            key = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(key, []).append(fi)
            edge_to_oriented.setdefault((fi, key[0], key[1]), (a, b))

    adj: list[set[int]] = [set() for _ in solid.faces]
    shared_edge: dict[tuple[int, int], tuple[int, int]] = {}
    for ekey, faces in edge_to_faces.items():
        if len(faces) == 2:
            f0, f1 = faces
            adj[f0].add(f1)
            adj[f1].add(f0)
            shared_edge[(min(f0, f1), max(f0, f1))] = ekey

    return [sorted(list(s)) for s in adj], shared_edge


def _shared_edge_vertices(solid: Any, f_parent: int, f_child: int) -> tuple[int, int]:
    face_p = solid.faces[f_parent]
    set_p = set(int(i) for i in face_p.vertex_ids)
    face_c = solid.faces[f_child]
    set_c = set(int(i) for i in face_c.vertex_ids)
    shared = sorted(list(set_p.intersection(set_c)))
    if len(shared) != 2:
        raise ValueError("faces are not adjacent")

    ids = [int(i) for i in face_p.vertex_ids]
    for i in range(len(ids)):
        a = ids[i]
        b = ids[(i + 1) % len(ids)]
        if {a, b} == set(shared):
            return (a, b)
    return (shared[0], shared[1])


def _random_spanning_tree(rng: Any, adjacency: Sequence[Sequence[int]], root: int) -> list[tuple[int, int]]:
    visited = {root}
    stack = [root]
    edges: list[tuple[int, int]] = []

    while stack:
        u = stack[-1]
        nbs = list(adjacency[u])
        rng.shuffle(nbs)
        nxt = None
        for v in nbs:
            if v not in visited:
                nxt = v
                break
        if nxt is None:
            stack.pop()
            continue
        visited.add(nxt)
        edges.append((u, nxt))
        stack.append(nxt)

    if len(visited) != len(adjacency):
        raise RuntimeError("failed to build spanning tree")
    return edges


def _signed_angle_about_axis(np: Any, axis: Any, v_from: Any, v_to: Any) -> float:
    axis_u = _unit(np, axis)
    cross = np.cross(v_from, v_to)
    s = float(np.dot(axis_u, cross))
    c = float(np.dot(v_from, v_to))
    return float(np.arctan2(s, c))


def _rotate_points_about_edge(np: Any, rotate_about_line: Any, pts: Sequence[Any], a: Any, b: Any, theta: float) -> list[Any]:
    axis = np.asarray(b - a, dtype=float)
    base = np.asarray(a, dtype=float)

    def rot(p: Any) -> Any:
        pv = np.asarray(p, dtype=float)
        an = float(np.linalg.norm(axis))
        if an == 0.0:
            raise ValueError("zero-length rotation axis")

        # polyhedra.tools.rotate_about_line() divides by ||cross(axis, rv1)||,
        # which is 0 when the point lies on the axis; handle that case explicitly.
        dist_times_an = float(np.linalg.norm(np.cross(pv - base, axis)))
        if dist_times_an <= 1e-12 * an:
            return pv.copy()

        return np.asarray(rotate_about_line(pv, base, axis, theta), dtype=float)

    return [rot(p) for p in pts]


def _polygon_edges(poly: Sequence[tuple[int, tuple[float, float]]]) -> list[tuple[tuple[int, int], tuple[tuple[float, float], tuple[float, float]]]]:
    out = []
    n = len(poly)
    for i in range(n):
        ida, pa = poly[i]
        idb, pb = poly[(i + 1) % n]
        key = (ida, idb) if ida < idb else (idb, ida)
        out.append((key, (pa, pb)))
    return out


def _orient2d(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float], eps: float) -> bool:
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersection_kind(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
    eps: float,
) -> str:
    o1 = _orient2d(a1, a2, b1)
    o2 = _orient2d(a1, a2, b2)
    o3 = _orient2d(b1, b2, a1)
    o4 = _orient2d(b1, b2, a2)

    def sgn(x: float) -> int:
        if x > eps:
            return 1
        if x < -eps:
            return -1
        return 0

    s1, s2, s3, s4 = sgn(o1), sgn(o2), sgn(o3), sgn(o4)

    # Colinear case
    if s1 == 0 and s2 == 0 and s3 == 0 and s4 == 0:
        # Project onto dominant axis and check overlap length
        if abs(a1[0] - a2[0]) >= abs(a1[1] - a2[1]):
            a_min, a_max = sorted([a1[0], a2[0]])
            b_min, b_max = sorted([b1[0], b2[0]])
        else:
            a_min, a_max = sorted([a1[1], a2[1]])
            b_min, b_max = sorted([b1[1], b2[1]])
        overlap = min(a_max, b_max) - max(a_min, b_min)
        if overlap > eps:
            return "colinear"
        # single-point touch (or disjoint)
        if _on_segment(a1, a2, b1, eps) or _on_segment(a1, a2, b2, eps) or _on_segment(b1, b2, a1, eps) or _on_segment(b1, b2, a2, eps):
            return "touch"
        return "none"

    # Proper crossing
    if s1 * s2 < 0 and s3 * s4 < 0:
        return "proper"

    # Touching (endpoint or T-junction)
    if (s1 == 0 and _on_segment(a1, a2, b1, eps)) or (s2 == 0 and _on_segment(a1, a2, b2, eps)) or (s3 == 0 and _on_segment(b1, b2, a1, eps)) or (s4 == 0 and _on_segment(b1, b2, a2, eps)):
        return "touch"

    return "none"


def _point_in_convex(poly: Sequence[tuple[float, float]], p: tuple[float, float], eps: float) -> bool:
    sign = 0
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        o = _orient2d(a, b, p)
        if abs(o) <= eps:
            continue
        s = 1 if o > 0 else -1
        if sign == 0:
            sign = s
        elif sign != s:
            return False
    return sign != 0


def _sat_convex_overlap(poly_a: Sequence[tuple[float, float]], poly_b: Sequence[tuple[float, float]], eps: float) -> bool:
    def axes(poly: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        out = []
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx = x2 - x1
            dy = y2 - y1
            nx, ny = -dy, dx
            ln = (nx * nx + ny * ny) ** 0.5
            if ln <= eps:
                continue
            out.append((nx / ln, ny / ln))
        return out

    def proj(poly: Sequence[tuple[float, float]], ax: tuple[float, float]) -> tuple[float, float]:
        axx, axy = ax
        vals = [p[0] * axx + p[1] * axy for p in poly]
        return (min(vals), max(vals))

    for ax in [*axes(poly_a), *axes(poly_b)]:
        a0, a1 = proj(poly_a, ax)
        b0, b1 = proj(poly_b, ax)
        # If separated or only touching (within eps), then no area overlap.
        if a1 <= b0 + eps or b1 <= a0 + eps:
            return False
    return True


def _polygons_collide(
    poly_a: Sequence[tuple[int, tuple[float, float]]],
    poly_b: Sequence[tuple[int, tuple[float, float]]],
    *,
    eps: float = 1e-6,
) -> bool:
    pts_a = [p for _vid, p in poly_a]
    pts_b = [p for _vid, p in poly_b]
    # Convex SAT overlap detects positive-area intersection; edge/vertex touching is allowed.
    return _sat_convex_overlap(pts_a, pts_b, eps)


def _unfold_random_net_internal(
    *,
    solid_id: int,
    sidelength: float = 1.0,
    seed: int = 0,
    root_face: int | None = None,
    max_tries: int = 500,
    require_non_overlapping: bool = True,
) -> tuple[Net2D, bool]:
    np, PlatonicSolid, rotate_about_line = _require_polyhedra()

    solid = PlatonicSolid("solid", int(solid_id), float(sidelength))
    adjacency, _shared = _solid_face_adjacency(solid)

    rng = __import__("random").Random(int(seed))
    root = int(root_face) if root_face is not None else rng.randrange(len(solid.faces))

    for _attempt in range(max_tries):
        try:
            tree = _random_spanning_tree(rng, adjacency, root)
            parent_of: dict[int, int] = {root: -1}
            order = [root]
            for u, v in tree:
                parent_of[v] = u
                order.append(v)

            face0_pts = _face_points(np, solid, root)
            ids0 = [int(i) for i in solid.faces[root].vertex_ids]
            a0, b0, c0 = face0_pts[0], face0_pts[1], face0_pts[2]
            B0 = _basis_from_three(np, a0, b0, c0)
            Bp = np.eye(3, dtype=float)
            R0 = Bp @ B0.T
            t0 = -R0 @ a0

            poses: dict[int, tuple[Any, Any]] = {root: (R0, t0)}

            def apply_pose(pose: tuple[Any, Any], p: Any) -> Any:
                R, t = pose
                return (R @ p) + t

            unfolded_faces: dict[int, list[tuple[int, tuple[float, float]]]] = {}
            tree_edges_out: list[tuple[int, int, tuple[int, int]]] = []

            poly0 = []
            for vid in ids0:
                p = apply_pose(poses[root], np.asarray(solid.get_vertex(vid), dtype=float))
                poly0.append((vid, (float(p[0]), float(p[1]))))
            unfolded_faces[root] = poly0

            ok = True
            for f in order[1:]:
                parent = parent_of[f]
                if parent < 0 or parent not in poses:
                    ok = False
                    break

                shared_va, shared_vb = _shared_edge_vertices(solid, parent, f)
                tree_edges_out.append((parent, f, (shared_va, shared_vb)))

                pose_p = poses[parent]
                face_ids = [int(i) for i in solid.faces[f].vertex_ids]
                pts_folded = [apply_pose(pose_p, np.asarray(solid.get_vertex(vid), dtype=float)) for vid in face_ids]

                parent_pts = [
                    apply_pose(pose_p, np.asarray(solid.get_vertex(int(i)), dtype=float))
                    for i in solid.faces[parent].vertex_ids[:3]
                ]
                n_p = _face_normal(np, parent_pts)
                n_c = _face_normal(np, pts_folded[:3])

                pa = apply_pose(pose_p, np.asarray(solid.get_vertex(shared_va), dtype=float))
                pb = apply_pose(pose_p, np.asarray(solid.get_vertex(shared_vb), dtype=float))
                axis = pb - pa
                theta = _signed_angle_about_axis(np, axis, n_c, n_p)

                pts_unfolded = _rotate_points_about_edge(np, rotate_about_line, pts_folded, pa, pb, theta)

                vid_a, vid_b, vid_c = face_ids[0], face_ids[1], face_ids[2]
                A = np.asarray(solid.get_vertex(vid_a), dtype=float)
                B = np.asarray(solid.get_vertex(vid_b), dtype=float)
                C = np.asarray(solid.get_vertex(vid_c), dtype=float)
                Ap = pts_unfolded[0]
                Bp2 = pts_unfolded[1]
                Cp2 = pts_unfolded[2]
                Rf, tf = _rigid_transform_from_3pts(np, A, B, C, Ap, Bp2, Cp2)
                poses[f] = (Rf, tf)

                poly_f = []
                for vid in face_ids:
                    p = apply_pose(poses[f], np.asarray(solid.get_vertex(vid), dtype=float))
                    poly_f.append((vid, (float(p[0]), float(p[1]))))
                unfolded_faces[f] = poly_f

            if not ok or len(unfolded_faces) != len(solid.faces):
                continue
        except Exception:
            continue

        face_indices = sorted(unfolded_faces.keys())
        collision = False
        for i in range(len(face_indices)):
            for j in range(i + 1, len(face_indices)):
                fi = face_indices[i]
                fj = face_indices[j]
                if _polygons_collide(unfolded_faces[fi], unfolded_faces[fj]):
                    collision = True
                    break
            if collision:
                break

        if require_non_overlapping and collision:
            continue

        faces_out: list[NetFace2D] = []
        for fi in sorted(unfolded_faces.keys()):
            poly = unfolded_faces[fi]
            faces_out.append(
                NetFace2D(
                    face_index=int(fi),
                    vertex_ids=tuple(int(v) for v, _p in poly),
                    xy=tuple(tuple(map(float, p)) for _v, p in poly),
                )
            )

        net = Net2D(
            solid_name=str(getattr(solid, "name", "solid")),
            solid_id=int(solid_id),
            sidelength=float(sidelength),
            root_face=int(root),
            tree_edges=tuple((int(a), int(b), (int(e[0]), int(e[1]))) for a, b, e in tree_edges_out),
            faces=tuple(faces_out),
        )
        return net, collision

    raise RuntimeError(f"Failed to generate a net after {max_tries} tries.")


def unfold_random_net(
    *,
    solid_id: int,
    sidelength: float = 1.0,
    seed: int = 0,
    root_face: int | None = None,
    max_tries: int = 500,
) -> Net2D:
    net, _collision = _unfold_random_net_internal(
        solid_id=solid_id,
        sidelength=sidelength,
        seed=seed,
        root_face=root_face,
        max_tries=max_tries,
        require_non_overlapping=True,
    )
    return net


def unfold_random_net_with_collision(
    *,
    solid_id: int,
    sidelength: float = 1.0,
    seed: int = 0,
    root_face: int | None = None,
    max_tries: int = 500,
) -> tuple[Net2D, bool]:
    return _unfold_random_net_internal(
        solid_id=solid_id,
        sidelength=sidelength,
        seed=seed,
        root_face=root_face,
        max_tries=max_tries,
        require_non_overlapping=False,
    )


def net_to_json(net: Net2D) -> dict[str, Any]:
    return {
        "solid_id": net.solid_id,
        "sidelength": net.sidelength,
        "root_face": net.root_face,
        "tree_edges": [[a, b, [e0, e1]] for a, b, (e0, e1) in net.tree_edges],
        "faces": [
            {"face_index": f.face_index, "vertex_ids": list(f.vertex_ids), "xy": [list(p) for p in f.xy]}
            for f in net.faces
        ],
    }


def net_bbox(net: Net2D) -> tuple[float, float, float, float]:
    xs = [p[0] for f in net.faces for p in f.xy]
    ys = [p[1] for f in net.faces for p in f.xy]
    return (min(xs), min(ys), max(xs), max(ys))


def render_net_png(
    net: Net2D,
    *,
    out_path: str,
    img_size: int = 512,
    palette: str = "pastel",
    line_width: float = 1.0,
    labels: bool = False,
    seed: int = 0,
    dedupe_edges: bool = True,
) -> None:
    import colorsys
    import random

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    rng = random.Random(int(seed))

    xmin, ymin, xmax, ymax = net_bbox(net)
    spanx = float(xmax - xmin)
    spany = float(ymax - ymin)
    span = max(spanx, spany) or 1.0

    # Preserve aspect ratio: use a single uniform scale and center in the square.
    pad_px = float(img_size) * 0.08
    usable = max(1.0, float(img_size) - 2.0 * pad_px)
    scale = usable / span
    cx = (float(xmin) + float(xmax)) / 2.0
    cy = (float(ymin) + float(ymax)) / 2.0

    def to_img(p: tuple[float, float]) -> tuple[float, float]:
        x = (float(p[0]) - cx) * scale + float(img_size) / 2.0
        y = (float(p[1]) - cy) * scale + float(img_size) / 2.0
        return (x, y)

    face_polys = [[to_img(p) for p in f.xy] for f in net.faces]

    pal = palette.lower().strip()
    if pal in {"gray", "grey"}:
        base = rng.uniform(0.75, 0.9)
        alpha = 0.90
        face_colors = [(base, base, base, alpha)] * len(face_polys)
    elif pal == "random":
        face_colors = [
            (rng.uniform(0.1, 0.95), rng.uniform(0.1, 0.95), rng.uniform(0.1, 0.95), 1.0) for _ in face_polys
        ]
    else:
        # Pastel palette: use per-face variation (still washed out) so overlaps remain visible.
        n = max(1, len(face_polys))
        h0 = rng.random()
        alpha = 0.90
        face_colors = []
        for i in range(n):
            # Spread hues around the wheel with a bit of jitter.
            h = (h0 + (i / n) * rng.uniform(0.65, 0.95) + rng.uniform(-0.03, 0.03)) % 1.0
            l = rng.uniform(0.78, 0.92)
            s = rng.uniform(0.10, 0.28)
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            face_colors.append((r, g, b, alpha))

    fig = plt.figure(figsize=(img_size / 100, img_size / 100), dpi=100)
    ax = plt.gca()
    ax.set_facecolor((1, 1, 1))

    ax.add_collection(PolyCollection(face_polys, facecolors=face_colors, edgecolors="none", antialiased=False))

    segs = []
    seen = set()
    for poly in face_polys:
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i + 1) % len(poly)]
            if dedupe_edges:
                key = (round(a[0], 3), round(a[1], 3), round(b[0], 3), round(b[1], 3))
                key2 = (round(b[0], 3), round(b[1], 3), round(a[0], 3), round(a[1], 3))
                if key in seen or key2 in seen:
                    continue
                seen.add(key)
            segs.append((a, b))

    ax.add_collection(LineCollection(segs, colors=(0, 0, 0, 1), linewidths=float(line_width), antialiased=True))

    if labels:
        for f, poly in zip(net.faces, face_polys):
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            ax.text(cx, cy, str(f.face_index), ha="center", va="center", fontsize=14, color=(0.1, 0.1, 0.1))

    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)

    import os
    from pathlib import Path

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
