"""
Experiment 3 — Pose LSTM: MediaPipe landmarks (99-d/frame) → BiLSTM classifier.

3-class model (failed excluded). Rotation head is auxiliary.
"""
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from pose_dataset import PoseDataset
from pose_model import PoseModel
from reporting_utils import save_history, save_training_curves, save_confusion_matrix

BATCH_SIZE  = 16
EPOCHS      = 50
LR          = 3e-4
PATIENCE    = 10
ROTATION_LOSS_WEIGHT = 0.5

FAILED_LABEL = 3
LABEL_NAMES  = ["salchow", "axel", "toe_loop"]

ROOT       = Path(__file__).resolve().parent.parent
TRAIN_META = ROOT / "data" / "frames" / "dataset_metadata_train.json"
VAL_META   = ROOT / "data" / "frames" / "dataset_metadata_val.json"
CKPT_DIR   = ROOT / "ai_training" / "checkpoints"
REPORT_DIR = ROOT / "reports" / "figures" / "exp3_pose_lstm"

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
def make_subset(meta_path, failed_label: int) -> Subset:
    ds   = PoseDataset(meta_path)
    keep = [i for i, e in enumerate(ds.data) if e["jump_type_label"] != failed_label]
    return Subset(ds, keep)


train_dataset = make_subset(TRAIN_META, FAILED_LABEL)
val_dataset   = make_subset(VAL_META,   FAILED_LABEL)
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
model     = PoseModel().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc      = 0.0
epochs_no_improve = 0

# ── Training loop ──────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    model.train()
    running_loss = correct = total = 0

    for poses, type_labels, rot_labels in train_loader:
        poses, type_labels, rot_labels = (
            poses.to(DEVICE), type_labels.to(DEVICE), rot_labels.to(DEVICE)
        )
        optimizer.zero_grad()
        type_logits, rot_logits = model(poses)
        loss = criterion(type_logits, type_labels) + \
               ROTATION_LOSS_WEIGHT * nn.functional.cross_entropy(rot_logits, rot_labels)
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
    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")

    model.eval()
    val_loss = val_correct = val_total = 0

    with torch.no_grad():
        for poses, type_labels, rot_labels in val_loader:
            poses, type_labels, rot_labels = (
                poses.to(DEVICE), type_labels.to(DEVICE), rot_labels.to(DEVICE)
            )
            type_logits, rot_logits = model(poses)
            loss = criterion(type_logits, type_labels) + \
                   ROTATION_LOSS_WEIGHT * nn.functional.cross_entropy(rot_logits, rot_labels)
            val_loss    += loss.item() * type_labels.size(0)
            _, pred      = torch.max(type_logits, 1)
            val_correct += (pred == type_labels).sum().item()
            val_total   += type_labels.size(0)

    val_loss /= val_total
    val_acc   = val_correct / val_total
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    save_training_curves(history, REPORT_DIR / "training_curves.png", "Pose LSTM")
    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), CKPT_DIR / "best_pose_model.pth")
        print(f"  Best model saved (val_acc={val_acc:.4f})")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print("Early stopping triggered")
        break

save_history(history, CKPT_DIR / "history_pose.json")
print(f"\nTraining Pose LSTM complete. Best val acc: {best_val_acc:.4f}")

# ── Confusion matrix on val set ────────────────────────────────────────────────
model.load_state_dict(torch.load(CKPT_DIR / "best_pose_model.pth", map_location=DEVICE, weights_only=True))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for poses, type_labels, rot_labels in val_loader:
        poses = poses.to(DEVICE)
        type_logits, _ = model(poses)
        _, pred = torch.max(type_logits, 1)
        y_true.extend(type_labels.tolist())
        y_pred.extend(pred.cpu().tolist())

save_confusion_matrix(y_true, y_pred, LABEL_NAMES, REPORT_DIR / "confusion_matrix.png",
                      "Exp 3: Pose LSTM")
print("Confusion matrix saved.")