"""Architecture descriptors for Polyfolds classification and repair models.

Role
----
Define the intended model-design vocabulary for Polyfolds before the heavier
training implementation is finalized. These specs document the future shape of
the classifier and repair models without coupling the current codebase to one
runtime framework yet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ConvStage:
    """One convolutional encoder stage in a planned Polyfolds raster stack."""

    channels: int
    kernel_size: int
    stride: int = 1
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    """High-level description of one raster encoder stack."""

    input_size: int
    stages: tuple[ConvStage, ...]
    projection_dim: int


@dataclass(frozen=True, slots=True)
class HeadSpec:
    """High-level description of one prediction head on top of the encoder."""

    hidden_dims: tuple[int, ...]
    output_dim: int
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class PolyfoldsModelSpec:
    """Complete planned model specification for one Polyfolds objective."""

    name: str
    objective: str
    raster_encoder: EncoderSpec
    vector_encoder_dim: int
    classifier_head: HeadSpec | None = None
    repair_head: HeadSpec | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Serialize the declarative model spec into a JSON-ready dictionary."""

        return asdict(self)


def default_classifier_spec() -> PolyfoldsModelSpec:
    """Return the default planned classifier architecture spec."""

    return PolyfoldsModelSpec(
        name="polyfolds_cnn_classifier_v1",
        objective="joint solid-plus-state classification over canonical vector-derived renders",
        raster_encoder=EncoderSpec(
            input_size=192,
            stages=(
                ConvStage(channels=32, kernel_size=3, dropout=0.0),
                ConvStage(channels=64, kernel_size=3, dropout=0.0),
                ConvStage(channels=128, kernel_size=3, dropout=0.0),
                ConvStage(channels=256, kernel_size=3, dropout=0.0),
            ),
            projection_dim=256,
        ),
        vector_encoder_dim=64,
        classifier_head=HeadSpec(hidden_dims=(256, 128), output_dim=15, dropout=0.15),
        notes=(
            "Raster input is rendered on demand from canonical vector geometry.",
            "The first label space is solid plus state, not state-only classification.",
            "Semantic color mapping remains deferred; the canonical render profile is neutral for now.",
        ),
    )


def default_repair_spec() -> PolyfoldsModelSpec:
    """Return the default planned repair-model architecture spec."""

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
            "A critic can score rendered repairs later, but only after supervised vector reconstruction is stable.",
        ),
    )
