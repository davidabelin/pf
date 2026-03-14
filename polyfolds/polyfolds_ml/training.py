"""Baseline classifier training for Polyfolds manifests."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
except Exception as exc:  # pragma: no cover
    LogisticRegression = None
    MLPClassifier = None
    train_test_split = None
    accuracy_score = None
    f1_score = None
    SKLEARN_IMPORT_ERROR = str(exc)
else:
    SKLEARN_IMPORT_ERROR = None


@dataclass(slots=True)
class ClassifierTrainConfig:
    manifest_path: str
    artifact_path: str
    model_type: str = "logistic"
    image_size: int = 64
    hidden_layer_sizes: tuple[int, ...] = (128, 64)
    max_iter: int = 300
    test_size: float = 0.2
    random_state: int = 42


def _read_raster(path: str, image_size: int) -> np.ndarray:
    image = mpimg.imread(path)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    image = np.asarray(image, dtype=float)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    target = int(image_size)
    row_idx = np.linspace(0, max(0, image.shape[0] - 1), target).astype(int)
    col_idx = np.linspace(0, max(0, image.shape[1] - 1), target).astype(int)
    resized = image[np.ix_(row_idx, col_idx)]
    return resized.reshape(-1).astype(np.float32)


def _load_xy(manifest_path: str, image_size: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = payload.get("samples", [])
    labels = sorted({str(row["class_label"]) for row in rows})
    label_to_index = {label: index for index, label in enumerate(labels)}
    X = np.asarray([_read_raster(str(row["raster_input"]["path"]), image_size=image_size) for row in rows], dtype=np.float32)
    y = np.asarray([label_to_index[str(row["class_label"])] for row in rows], dtype=np.int64)
    return X, y, labels


def train_classifier_baseline(config: ClassifierTrainConfig) -> dict[str, Any]:
    if LogisticRegression is None or MLPClassifier is None or train_test_split is None:
        raise RuntimeError(
            "scikit-learn is required for polyfolds baseline training"
            + (f". Import error: {SKLEARN_IMPORT_ERROR}" if SKLEARN_IMPORT_ERROR else "")
        )

    X, y, labels = _load_xy(config.manifest_path, image_size=config.image_size)
    if len(X) < 12:
        raise RuntimeError("Need at least 12 samples for the baseline classifier scaffold.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(config.test_size),
        random_state=int(config.random_state),
        stratify=y if len(set(y.tolist())) > 1 else None,
    )
    if config.model_type == "logistic":
        model = LogisticRegression(max_iter=int(config.max_iter), random_state=int(config.random_state))
    elif config.model_type == "nn":
        model = MLPClassifier(
            hidden_layer_sizes=tuple(int(v) for v in config.hidden_layer_sizes),
            max_iter=int(config.max_iter),
            random_state=int(config.random_state),
        )
    else:
        raise ValueError("model_type must be 'logistic' or 'nn'.")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, pred)) if accuracy_score else float(np.mean(pred == y_test))
    macro_f1 = float(f1_score(y_test, pred, average="macro")) if f1_score else accuracy

    artifact = {
        "config": asdict(config),
        "labels": labels,
        "model": model,
    }
    artifact_path = Path(config.artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(pickle.dumps(artifact))

    return {
        "sample_count": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": labels,
        "artifact_path": str(artifact_path.resolve()),
    }
