"""Structured schema for Polyfolds raster+vector datasets.

Role
----
Define the normalized dataset contract shared by the Polyfolds manifest builder,
offline training scripts, and the future deployed inference surface.

Cross-Repo Context
------------------
This schema is the hand-off point between the offline ``pf/polyfolds``
workspace and the eventual ``pf_web`` interactive app. It is intentionally
designed to preserve both raster and vector structure so later repair models do
not get trapped in raster-only outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = 1


@dataclass(slots=True)
class RasterAsset:
    """Reference to one raster asset used as model input or target output."""

    path: str
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"


@dataclass(slots=True)
class VectorFace:
    """One polygonal face in the 2D vector representation of a net."""

    face_index: int
    polygon: tuple[tuple[float, float], ...]
    present: bool = True
    edge_group: str | None = None


@dataclass(slots=True)
class VectorEdge:
    """One explicit edge record derived from vector faces."""

    edge_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    face_indices: tuple[int, ...] = ()
    edge_group: str | None = None


@dataclass(slots=True)
class RepairTarget:
    """Optional repair/completion target attached to one training sample."""

    target_raster: RasterAsset | None = None
    target_svg_path: str | None = None
    completion_face_indices: tuple[int, ...] = ()
    edit_budget: int | None = None


@dataclass(slots=True)
class PolyfoldSample:
    """One normalized Polyfolds sample spanning raster and vector views."""

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
        """Serialize the dataclass into a plain JSON-ready dictionary."""

        return asdict(self)


@dataclass(slots=True)
class DatasetManifest:
    """Top-level metadata block for one normalized Polyfolds dataset export."""

    dataset_name: str
    schema_version: int
    created_at: str
    source_roots: tuple[str, ...]
    classes: tuple[str, ...]
    solids: tuple[str, ...]
    sample_count: int
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest header into a plain JSON-ready dictionary."""

        return asdict(self)
