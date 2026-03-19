"""PyTorch classifier training for canonical Polyfolds manifests."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from vector_render import render_faces_array


@dataclass(slots=True)
class ClassifierTrainConfig:
    manifest_path: str
    artifact_path: str
    image_size: int = 192
    batch_size: int = 16
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 4
    num_workers: int = 0
    seed: int = 42


def _label_for_row(row: dict[str, Any]) -> str:
    solid = str(row.get("solid", "unknown"))
    state = str(row.get("state") or row.get("class_label") or "unknown")
    return str(row.get("joint_label") or f"{solid}:{state}")


def _read_manifest(manifest_path: str) -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def _rows_for_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("split", "train")) == split]


def _ensure_non_empty_splits(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = _rows_for_split(rows, "train")
    val_rows = _rows_for_split(rows, "val")
    test_rows = _rows_for_split(rows, "test")

    if not train_rows:
        raise RuntimeError("Need at least one training sample in the manifest.")

    if not val_rows or not test_rows:
        ordered = sorted(rows, key=lambda row: str(row.get("topology_hash") or row.get("sample_id") or ""))
        if not val_rows:
            val_rows = ordered[: max(1, len(ordered) // 10)]
        if not test_rows:
            test_rows = ordered[-max(1, len(ordered) // 10) :]
        held_out_ids = {str(row.get("sample_id")) for row in val_rows + test_rows}
        train_rows = [row for row in train_rows if str(row.get("sample_id")) not in held_out_ids]
        if not train_rows:
            train_rows = ordered[max(1, len(ordered) // 5) :]
    return train_rows, val_rows, test_rows


def _confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true, y_pred):
        matrix[int(true_idx), int(pred_idx)] += 1
    return matrix


def _macro_f1(matrix: np.ndarray) -> float:
    scores: list[float] = []
    for index in range(matrix.shape[0]):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - tp)
        fn = float(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall))
    return float(sum(scores) / len(scores)) if scores else 0.0


class VectorManifestDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], *, label_to_index: dict[str, int], image_size: int, augment: bool) -> None:
        self.rows = list(rows)
        self.label_to_index = dict(label_to_index)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        array = render_faces_array(
            row.get("vector_faces", []),
            vector_edges=row.get("vector_edges", []),
            image_size=self.image_size,
        )
        tensor = torch.from_numpy(array).float()
        if self.augment:
            tensor = _apply_affine_augmentation(tensor)
        label = self.label_to_index[_label_for_row(row)]
        return tensor, torch.tensor(label, dtype=torch.long)


def _apply_affine_augmentation(image: torch.Tensor) -> torch.Tensor:
    theta_deg = float(torch.empty(1).uniform_(-12.0, 12.0).item())
    scale = float(torch.empty(1).uniform_(0.92, 1.08).item())
    tx = float(torch.empty(1).uniform_(-0.08, 0.08).item())
    ty = float(torch.empty(1).uniform_(-0.08, 0.08).item())
    radians = np.deg2rad(theta_deg)
    cos_v = float(np.cos(radians) * scale)
    sin_v = float(np.sin(radians) * scale)
    theta = torch.tensor([[cos_v, -sin_v, tx], [sin_v, cos_v, ty]], dtype=image.dtype, device=image.device).unsqueeze(0)
    grid = F.affine_grid(theta, size=(1, image.shape[0], image.shape[1], image.shape[2]), align_corners=False)
    out = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out[0]


class PolyfoldsCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        channels = [3, 32, 64, 128, 256]
        blocks: list[nn.Module] = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(128, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def _evaluate(model: nn.Module, loader: DataLoader, *, device: torch.device, labels: list[str]) -> dict[str, Any]:
    model.eval()
    true_all: list[int] = []
    pred_all: list[int] = []
    loss_values: list[float] = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            pred = torch.argmax(logits, dim=1)
            true_all.extend(int(v) for v in y.cpu().tolist())
            pred_all.extend(int(v) for v in pred.cpu().tolist())
            loss_values.append(float(loss.item()))

    matrix = _confusion_matrix(true_all, pred_all, num_classes=len(labels))
    accuracy = float(sum(int(a == b) for a, b in zip(true_all, pred_all)) / max(1, len(true_all)))
    return {
        "loss": float(sum(loss_values) / len(loss_values)) if loss_values else 0.0,
        "accuracy": accuracy,
        "macro_f1": _macro_f1(matrix),
        "confusion_matrix": matrix.tolist(),
        "samples": int(len(true_all)),
    }


def train_classifier_baseline(config: ClassifierTrainConfig) -> dict[str, Any]:
    """Train the shared Polyfolds CNN classifier."""

    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))

    payload = _read_manifest(config.manifest_path)
    rows = list(payload.get("samples", []))
    if len(rows) < 12:
        raise RuntimeError("Need at least 12 samples for the classifier training path.")

    labels = sorted({_label_for_row(row) for row in rows})
    label_to_index = {label: index for index, label in enumerate(labels)}
    train_rows, val_rows, test_rows = _ensure_non_empty_splits(rows)

    train_ds = VectorManifestDataset(train_rows, label_to_index=label_to_index, image_size=int(config.image_size), augment=True)
    val_ds = VectorManifestDataset(val_rows, label_to_index=label_to_index, image_size=int(config.image_size), augment=False)
    test_ds = VectorManifestDataset(test_rows, label_to_index=label_to_index, image_size=int(config.image_size), augment=False)

    train_loader = DataLoader(train_ds, batch_size=int(config.batch_size), shuffle=True, num_workers=int(config.num_workers))
    val_loader = DataLoader(val_ds, batch_size=int(config.batch_size), shuffle=False, num_workers=int(config.num_workers))
    test_loader = DataLoader(test_ds, batch_size=int(config.batch_size), shuffle=False, num_workers=int(config.num_workers))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolyfoldsCNN(num_classes=len(labels)).to(device)
    artifact_path = Path(config.artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_path.with_suffix(artifact_path.suffix + ".best.tmp")

    train_counter = Counter(_label_for_row(row) for row in train_rows)
    class_weights = torch.tensor(
        [len(train_rows) / max(1, train_counter[label]) for label in labels],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))

    history: list[dict[str, Any]] = []
    best_metric = -1.0
    epochs_without_improvement = 0

    for epoch in range(int(config.epochs)):
        model.train()
        batch_losses: list[float] = []
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        val_metrics = _evaluate(model, val_loader, device=device, labels=labels)
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(sum(batch_losses) / len(batch_losses)) if batch_losses else 0.0,
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
        )

        if float(val_metrics["macro_f1"]) > best_metric:
            best_metric = float(val_metrics["macro_f1"])
            torch.save(model.state_dict(), checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config.patience):
                break

    if checkpoint_path.exists():
        best_state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(best_state)
    val_metrics = _evaluate(model, val_loader, device=device, labels=labels)
    test_metrics = _evaluate(model, test_loader, device=device, labels=labels)

    artifact = {
        "config": asdict(config),
        "labels": labels,
        "model_state_dict": model.state_dict(),
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_name": "polyfolds_cnn_classifier_v1",
    }
    torch.save(artifact, artifact_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return {
        "sample_count": int(len(rows)),
        "train_samples": int(len(train_rows)),
        "val_samples": int(len(val_rows)),
        "test_samples": int(len(test_rows)),
        "best_val_macro_f1": float(val_metrics["macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "labels": labels,
        "artifact_path": str(artifact_path.resolve()),
        "device": str(device),
        "history": history,
        "test_confusion_matrix": test_metrics["confusion_matrix"],
    }
