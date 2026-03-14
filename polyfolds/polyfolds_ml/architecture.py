"""Architecture descriptors for Polyfolds classification and repair models."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ConvStage:
    channels: int
    kernel_size: int
    stride: int = 1
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    input_size: int
    stages: tuple[ConvStage, ...]
    projection_dim: int


@dataclass(frozen=True, slots=True)
class HeadSpec:
    hidden_dims: tuple[int, ...]
    output_dim: int
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class PolyfoldsModelSpec:
    name: str
    objective: str
    raster_encoder: EncoderSpec
    vector_encoder_dim: int
    classifier_head: HeadSpec | None = None
    repair_head: HeadSpec | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def default_classifier_spec() -> PolyfoldsModelSpec:
    return PolyfoldsModelSpec(
        name="polyfolds_cnn_classifier_v0",
        objective="classify valid vs invalid vs incomplete",
        raster_encoder=EncoderSpec(
            input_size=192,
            stages=(
                ConvStage(channels=24, kernel_size=7, dropout=0.05),
                ConvStage(channels=48, kernel_size=5, dropout=0.08),
                ConvStage(channels=96, kernel_size=3, dropout=0.1),
            ),
            projection_dim=192,
        ),
        vector_encoder_dim=64,
        classifier_head=HeadSpec(hidden_dims=(192, 96), output_dim=3, dropout=0.15),
        notes=(
            "Start with raster input, but preserve vector hooks from day one.",
            "Edge-group colors should stay aligned between raster and vector representations.",
            "This spec is meant for later PyTorch implementation, not current execution.",
        ),
    )


def default_repair_spec() -> PolyfoldsModelSpec:
    return PolyfoldsModelSpec(
        name="polyfolds_repair_hybrid_v0",
        objective="predict least-change repair targets in vector/raster space",
        raster_encoder=EncoderSpec(
            input_size=192,
            stages=(
                ConvStage(channels=32, kernel_size=7, dropout=0.05),
                ConvStage(channels=64, kernel_size=5, dropout=0.1),
                ConvStage(channels=128, kernel_size=3, dropout=0.12),
            ),
            projection_dim=256,
        ),
        vector_encoder_dim=128,
        classifier_head=HeadSpec(hidden_dims=(192,), output_dim=3, dropout=0.1),
        repair_head=HeadSpec(hidden_dims=(256, 128), output_dim=64, dropout=0.15),
        notes=(
            "Repair should target structured vector edits or SVG-like output, not only raster pixels.",
            "A critic can score rendered repairs later, but the generator should stay geometry-aware.",
        ),
    )
