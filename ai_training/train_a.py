"""
Experiment 2 — Option A: ResNet18 + LSTM with unfrozen layer4.

Differential learning rates: backbone LR = 1e-5, heads LR = 1e-4.
4-class model (includes failed), class-weighted loss.
"""
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FigureSkatingDataset
from model_a import FigureSkatingModelA
from reporting_utils import save_history, save_training_curves, save_confusion_matrix

BATCH_SIZE  = 8
EPOCHS      = 30
LR          = 1e-4
LR_BACKBONE = 1e-5
PATIENCE    = 7
ROTATION_LOSS_WEIGHT = 0.5

LABEL_NAMES = ["salchow", "axel", "toe_loop", "failed"]

ROOT       = Path(__file__).resolve().parent.parent
TRAIN_META = ROOT / "data" / "frames" / "dataset_metadata_train.json"
VAL_META   = ROOT / "data" / "frames" / "dataset_metadata_val.json"
CKPT_DIR   = ROOT / "ai_training" / "checkpoints"
REPORT_DIR = ROOT / "reports" / "figures" / "exp2_resnet_finetune"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("GPU not found, training on CPU")

# ── Class weights (4-class including failed) ───────────────────────────────────
with open(TRAIN_META, encoding="utf-8") as f:
    train_meta_raw = json.load(f)

counts  = Counter(v["jump_type_label"] for v in train_meta_raw.values())
n_total = sum(counts.values())
weights = [n_total / (4 * counts[i]) for i in range(4)]
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: {[f'{w:.2f}' for w in weights]}")

# ── Data ───────────────────────────────────────────────────────────────────────
train_dataset = FigureSkatingDataset(str(TRAIN_META), is_train=True)
val_dataset   = FigureSkatingDataset(str(VAL_META),   is_train=False)
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
model     = FigureSkatingModelA().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam([
    {"params": model.cnn_finetune.parameters(), "lr": LR_BACKBONE},
    {"params": model.lstm.parameters(),         "lr": LR},
    {"params": model.batchnorm.parameters(),    "lr": LR},
    {"params": model.head_type.parameters(),    "lr": LR},
    {"params": model.head_rotation.parameters(),"lr": LR},
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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
        loss_type = criterion(type_logits, type_labels)
        rot_mask  = rot_labels > 0
        loss_rot  = nn.functional.cross_entropy(rot_logits[rot_mask], rot_labels[rot_mask] - 1) \
                    if rot_mask.any() else torch.tensor(0.0, device=DEVICE)
        loss = loss_type + ROTATION_LOSS_WEIGHT * loss_rot
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
            loss_type = criterion(type_logits, type_labels)
            rot_mask  = rot_labels > 0
            loss_rot  = nn.functional.cross_entropy(rot_logits[rot_mask], rot_labels[rot_mask] - 1) \
                        if rot_mask.any() else torch.tensor(0.0, device=DEVICE)
            val_loss    += (loss_type + ROTATION_LOSS_WEIGHT * loss_rot).item() * type_labels.size(0)
            _, pred      = torch.max(type_logits, 1)
            val_correct += (pred == type_labels).sum().item()
            val_total   += type_labels.size(0)
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
    save_training_curves(history, REPORT_DIR / "training_curves.png", "ResNet18+LSTM (layer4 unfrozen)")
    print(f"Epoch [{epoch+1}/{EPOCHS}] Val Loss: {val_loss:.4f}  "
          f"Type Acc: {val_acc:.4f}  Rot Acc: {val_rot_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), CKPT_DIR / "best_model_a.pth")
        print(f"  Best model saved (val_acc={val_acc:.4f})")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print("Early stopping triggered")
        break

save_history(history, CKPT_DIR / "history_a.json")
print(f"\nTraining A complete. Best val acc: {best_val_acc:.4f}")

# ── Confusion matrix on val set ────────────────────────────────────────────────
model.load_state_dict(torch.load(CKPT_DIR / "best_model_a.pth", map_location=DEVICE, weights_only=True))
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
                      "Exp 2: ResNet18+LSTM (layer4 unfrozen)")
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
                      "Exp 2: Rotation (1x/2x/3x)", labels=[0, 1, 2])
print("Rotation confusion matrix saved.")