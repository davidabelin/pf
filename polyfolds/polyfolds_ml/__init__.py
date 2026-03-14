"""Polyfolds ML scaffolding for dataset manifests and baseline training."""

from polyfolds_ml.architecture import PolyfoldsModelSpec, default_classifier_spec, default_repair_spec
from polyfolds_ml.manifest import build_manifest, load_manifest_rows, summarize_manifest
from polyfolds_ml.schema import DatasetManifest, PolyfoldSample
from polyfolds_ml.training import ClassifierTrainConfig, train_classifier_baseline

__all__ = [
    "ClassifierTrainConfig",
    "DatasetManifest",
    "PolyfoldSample",
    "PolyfoldsModelSpec",
    "build_manifest",
    "default_classifier_spec",
    "default_repair_spec",
    "load_manifest_rows",
    "summarize_manifest",
    "train_classifier_baseline",
]
