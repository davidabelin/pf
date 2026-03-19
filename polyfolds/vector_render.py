"""Shared neutral vector rendering helpers for canonical Polyfolds assets."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


NEUTRAL_RENDER_PROFILE_ID = "neutral_v1"

_BACKGROUND = (255, 255, 255)
_FACE_FILL = (219, 224, 231)
_FACE_OUTLINE = (24, 32, 40)
_MISSING_OUTLINE = (112, 124, 138)
_SHARED_EDGE = (110, 118, 128)
_CUT_EDGE = (24, 32, 40)


def _get_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _points_from_face(face: Any) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in _get_value(face, "polygon", ()) or ())


def _projector(vector_faces: Iterable[Any], image_size: int, pad_ratio: float = 0.08):
    polys = [_points_from_face(face) for face in vector_faces if _points_from_face(face)]
    if not polys:
        raise ValueError("Need at least one polygon to render.")

    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    span = max(float(xmax - xmin), float(ymax - ymin), 1.0)
    pad_px = float(image_size) * float(pad_ratio)
    usable = max(1.0, float(image_size) - 2.0 * pad_px)
    scale = usable / span
    cx = (float(xmin) + float(xmax)) / 2.0
    cy = (float(ymin) + float(ymax)) / 2.0

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x = (float(point[0]) - cx) * scale + float(image_size) / 2.0
        y = (cy - float(point[1])) * scale + float(image_size) / 2.0
        return (x, y)

    return project


def render_faces_image(
    vector_faces: Iterable[Any],
    *,
    vector_edges: Iterable[Any] | None = None,
    image_size: int = 192,
) -> Image.Image:
    """Render vector faces into a neutral RGB image."""

    faces = list(vector_faces)
    project = _projector(faces, image_size=image_size)
    image = Image.new("RGB", (int(image_size), int(image_size)), _BACKGROUND)
    draw = ImageDraw.Draw(image)

    for face in sorted(faces, key=lambda item: (_get_value(item, "present", True) is False, _get_value(item, "face_index", 0))):
        pts = [project(point) for point in _points_from_face(face)]
        if len(pts) < 3:
            continue
        present = bool(_get_value(face, "present", True))
        if present:
            draw.polygon(pts, fill=_FACE_FILL, outline=_FACE_OUTLINE)
        else:
            draw.polygon(pts, fill=None, outline=_MISSING_OUTLINE)

    if vector_edges:
        for edge in vector_edges:
            start = project(tuple(_get_value(edge, "start", (0.0, 0.0))))
            end = project(tuple(_get_value(edge, "end", (0.0, 0.0))))
            face_indices = tuple(int(v) for v in (_get_value(edge, "face_indices", ()) or ()))
            color = _SHARED_EDGE if len(face_indices) >= 2 else _CUT_EDGE
            width = 1 if len(face_indices) >= 2 else 2
            draw.line([start, end], fill=color, width=width)

    return image


def render_faces_array(
    vector_faces: Iterable[Any],
    *,
    vector_edges: Iterable[Any] | None = None,
    image_size: int = 192,
) -> np.ndarray:
    """Render faces into a normalized float32 CHW tensor-friendly array."""

    image = render_faces_image(vector_faces, vector_edges=vector_edges, image_size=image_size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def save_faces_png(
    vector_faces: Iterable[Any],
    *,
    out_path: str | Path,
    vector_edges: Iterable[Any] | None = None,
    image_size: int = 256,
) -> None:
    """Write one neutral preview PNG to disk."""

    image = render_faces_image(vector_faces, vector_edges=vector_edges, image_size=image_size)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_faces_svg(
    vector_faces: Iterable[Any],
    *,
    out_path: str | Path,
    vector_edges: Iterable[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a canonical neutral SVG with semantic face and edge role markers."""

    faces = list(vector_faces)
    polys = [_points_from_face(face) for face in faces if _points_from_face(face)]
    if not polys:
        raise ValueError("Need at least one polygon to write SVG.")

    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    pad = max(0.25, max(xmax - xmin, ymax - ymin) * 0.05)
    width = (xmax - xmin) + 2.0 * pad
    height = (ymax - ymin) + 2.0 * pad

    def svg_point(point: tuple[float, float]) -> str:
        x = float(point[0])
        y = float(point[1])
        return f"{x:.6f},{-y:.6f}"

    view_box = f"{xmin - pad:.6f} {-ymax - pad:.6f} {width:.6f} {height:.6f}"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" fill="none" stroke-linejoin="round" stroke-linecap="round">',
        f'  <rect x="{xmin - pad:.6f}" y="{-ymax - pad:.6f}" width="{width:.6f}" height="{height:.6f}" fill="#ffffff" />',
    ]

    if metadata:
        payload = html.escape(json.dumps(metadata, sort_keys=True))
        lines.append(f"  <metadata>{payload}</metadata>")

    for face in sorted(faces, key=lambda item: int(_get_value(item, "face_index", 0))):
        points = " ".join(svg_point(point) for point in _points_from_face(face))
        present = bool(_get_value(face, "present", True))
        face_role = "face_present" if present else "completion_target"
        fill = "#dbe0e7" if present else "none"
        stroke = "#182028" if present else "#707c8a"
        dash = "" if present else ' stroke-dasharray="0.20 0.12"'
        lines.append(
            "  "
            + f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="0.03"{dash} '
            + f'data-face-index="{int(_get_value(face, "face_index", 0))}" data-face-role="{face_role}" />'
        )

    for edge in vector_edges or ():
        start = tuple(float(v) for v in _get_value(edge, "start", (0.0, 0.0)))
        end = tuple(float(v) for v in _get_value(edge, "end", (0.0, 0.0)))
        face_indices = tuple(int(v) for v in (_get_value(edge, "face_indices", ()) or ()))
        role = "edge_shared" if len(face_indices) >= 2 else "edge_cut"
        stroke = "#6e7680" if len(face_indices) >= 2 else "#182028"
        width_value = "0.02" if len(face_indices) >= 2 else "0.03"
        lines.append(
            "  "
            + f'<line x1="{start[0]:.6f}" y1="{-start[1]:.6f}" x2="{end[0]:.6f}" y2="{-end[1]:.6f}" '
            + f'stroke="{stroke}" stroke-width="{width_value}" data-edge-role="{role}" />'
        )

    lines.append("</svg>")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
