"""Structured schema for Polyfolds raster+vector datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = 1


@dataclass(slots=True)
class RasterAsset:
    path: str
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"


@dataclass(slots=True)
class VectorFace:
    face_index: int
    polygon: tuple[tuple[float, float], ...]
    present: bool = True
    edge_group: str | None = None


@dataclass(slots=True)
class VectorEdge:
    edge_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    face_indices: tuple[int, ...] = ()
    edge_group: str | None = None


@dataclass(slots=True)
class RepairTarget:
    target_raster: RasterAsset | None = None
    target_svg_path: str | None = None
    completion_face_indices: tuple[int, ...] = ()
    edit_budget: int | None = None


@dataclass(slots=True)
class PolyfoldSample:
    sample_id: str
    split: str
    class_label: str
    solid: str
    source_dataset: str
    raster_input: RasterAsset
    vector_faces: tuple[VectorFace, ...] = ()
    vector_edges: tuple[VectorEdge, ...] = ()
    repair_target: RepairTarget | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetManifest:
    dataset_name: str
    schema_version: int
    created_at: str
    source_roots: tuple[str, ...]
    classes: tuple[str, ...]
    solids: tuple[str, ...]
    sample_count: int
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
