"""
Experiment 1 — Baseline: ResNet18 (frozen) + LSTM.

3-class jump type head (failed excluded).
Rotation head is auxiliary and not the primary metric.
"""
import json
import os
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import FigureSkatingDataset
from model import FigureSkatingModel
from reporting_utils import save_history, save_training_curves, save_confusion_matrix

BATCH_SIZE = 8
EPOCHS     = 30
LR         = 1e-4
PATIENCE   = 7
ROTATION_LOSS_WEIGHT = 0.5

FAILED_LABEL = 3
LABEL_NAMES  = ["salchow", "axel", "toe_loop"]

ROOT       = Path(__file__).resolve().parent.parent
TRAIN_META = ROOT / "data" / "frames" / "dataset_metadata_train.json"
VAL_META   = ROOT / "data" / "frames" / "dataset_metadata_val.json"
CKPT_DIR   = ROOT / "ai_training" / "checkpoints"
REPORT_DIR = ROOT / "reports" / "figures" / "exp1_baseline"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("GPU not found, training on CPU")

# ── Class weights (3-class, exclude failed) ────────────────────────────────────
with open(TRAIN_META, encoding="utf-8") as f:
    train_meta_raw = json.load(f)

counts  = Counter(
    v["jump_type_label"] for v in train_meta_raw.values()
    if v["jump_type_label"] != FAILED_LABEL
)
n_total = sum(counts.values())
weights = [n_total / (3 * counts[i]) for i in range(3)]
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: salchow={weights[0]:.2f}  axel={weights[1]:.2f}  toe_loop={weights[2]:.2f}")


# ── Data ───────────────────────────────────────────────────────────────────────
def make_subset(meta_path, is_train: bool) -> Subset:
    ds   = FigureSkatingDataset(str(meta_path), is_train=is_train)
    keep = [i for i, e in enumerate(ds.data) if e["jump_type_label"] != FAILED_LABEL]
    return Subset(ds, keep)


train_dataset = make_subset(TRAIN_META, is_train=True)
val_dataset   = make_subset(VAL_META,   is_train=False)
print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    drop_last=True, num_workers=0, pin_memory=(DEVICE.type == "cuda"),
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(DEVICE.type == "cuda"),
)

# ── Model ──────────────────────────────────────────────────────────────────────
model      = FigureSkatingModel().to(DEVICE)
criterion  = nn.CrossEntropyLoss(weight=class_weights)
optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_rot_acc": []}
best_val_acc      = 0.0
epochs_no_improve = 0

# ── Training loop ──────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    model.train()
    running_loss = correct = total = 0

    for videos, type_labels, rot_labels in train_loader:
        videos, type_labels, rot_labels = (
            videos.to(DEVICE), type_labels.to(DEVICE), rot_labels.to(DEVICE)
        )
        optimizer.zero_grad()
        type_logits, rot_logits = model(videos)
        loss = criterion(type_logits, type_labels) + \
               ROTATION_LOSS_WEIGHT * nn.functional.cross_entropy(rot_logits, rot_labels - 1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * type_labels.size(0)
        _, pred = torch.max(type_logits, 1)
        correct += (pred == type_labels).sum().item()
        total   += type_labels.size(0)

    scheduler.step()
    train_loss, train_acc = running_loss / total, correct / total
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")

    model.eval()
    val_loss = val_correct = val_total = rot_correct = rot_total = 0

    with torch.no_grad():
        for videos, type_labels, rot_labels in val_loader:
            videos, type_labels, rot_labels = (
                videos.to(DEVICE), type_labels.to(DEVICE), rot_labels.to(DEVICE)
            )
            type_logits, rot_logits = model(videos)
            loss = criterion(type_logits, type_labels) + \
                   ROTATION_LOSS_WEIGHT * nn.functional.cross_entropy(rot_logits, rot_labels - 1)
            val_loss    += loss.item() * type_labels.size(0)
            _, pred      = torch.max(type_logits, 1)
            val_correct += (pred == type_labels).sum().item()
            val_total   += type_labels.size(0)
            rot_mask = rot_labels > 0
            if rot_mask.any():
                _, rot_pred  = torch.max(rot_logits[rot_mask], 1)
                rot_correct += (rot_pred == rot_labels[rot_mask] - 1).sum().item()
                rot_total   += rot_mask.sum().item()

    val_loss /= val_total
    val_acc      = val_correct / val_total
    val_rot_acc  = rot_correct / rot_total if rot_total > 0 else 0.0
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_rot_acc"].append(val_rot_acc)
    save_training_curves(history, REPORT_DIR / "training_curves.png", "Baseline ResNet18+LSTM")
    print(f"Epoch [{epoch+1}/{EPOCHS}] Val Loss: {val_loss:.4f}  "
          f"Type Acc: {val_acc:.4f}  Rot Acc: {val_rot_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), CKPT_DIR / "best_model.pth")
        print(f"  Best model saved (val_acc={val_acc:.4f})")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print("Early stopping triggered")
        break

save_history(history, CKPT_DIR / "history_baseline.json")
print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")

# ── Confusion matrix on val set ────────────────────────────────────────────────
model.load_state_dict(torch.load(CKPT_DIR / "best_model.pth", map_location=DEVICE, weights_only=True))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for videos, type_labels, rot_labels in val_loader:
        videos = videos.to(DEVICE)
        type_logits, _ = model(videos)
        _, pred = torch.max(type_logits, 1)
        y_true.extend(type_labels.tolist())
        y_pred.extend(pred.cpu().tolist())

save_confusion_matrix(y_true, y_pred, LABEL_NAMES, REPORT_DIR / "confusion_matrix.png",
                      "Exp 1: Baseline ResNet18+LSTM")
print("Type confusion matrix saved.")

r_true, r_pred = [], []
with torch.no_grad():
    for videos, type_labels, rot_labels in val_loader:
        videos, rot_labels = videos.to(DEVICE), rot_labels.to(DEVICE)
        _, rot_logits = model(videos)
        rot_mask = rot_labels > 0
        if rot_mask.any():
            _, pred = torch.max(rot_logits[rot_mask], 1)
            r_true.extend((rot_labels[rot_mask] - 1).tolist())  # {1,2,3} → {0,1,2}
            r_pred.extend(pred.cpu().tolist())
save_confusion_matrix(r_true, r_pred, ["single", "double", "triple"],
                      REPORT_DIR / "confusion_matrix_rotation.png",
                      "Exp 1: Rotation (1x/2x/3x)", labels=[0, 1, 2])
print("Rotation confusion matrix saved.")