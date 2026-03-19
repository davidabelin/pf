"""Architecture and training descriptors for Polyfolds models.

Role
----
Document the agreed classifier and repair-model shapes alongside the default
training regimen used by the canonical Polyfolds workflow.
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
class AffineAugmentationSpec:
    """Declarative description of the raster affine augmentation budget."""

    rotation_degrees: float
    scale_min: float
    scale_max: float
    translate_fraction: float


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    """Default training regimen for one Polyfolds model objective."""

    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    optimizer: str
    balanced_sampling: str
    loss: str
    split_policy: str
    augmentation: AffineAugmentationSpec
    num_workers: int = 0
    seed: int = 42
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Serialize the declarative training spec into a JSON-ready dictionary."""

        return asdict(self)


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


def default_classifier_training_spec() -> TrainingSpec:
    """Return the default training regimen for the shared classifier."""

    return TrainingSpec(
        batch_size=16,
        epochs=12,
        learning_rate=3e-4,
        weight_decay=1e-4,
        patience=4,
        optimizer="AdamW",
        balanced_sampling="joint_label_weighted_random",
        loss="cross_entropy",
        split_policy="topology_hash family split with leakage check",
        augmentation=AffineAugmentationSpec(
            rotation_degrees=12.0,
            scale_min=0.92,
            scale_max=1.08,
            translate_fraction=0.08,
        ),
        notes=(
            "Balanced sampling equalizes the 15 solid-plus-state labels during training epochs.",
            "If balanced sampling is disabled, inverse-frequency class weighting is used instead.",
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
