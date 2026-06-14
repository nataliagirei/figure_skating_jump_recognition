"""Shared plotting utilities for training scripts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_training_curves(history: dict, out_path: Path, title: str = "") -> None:
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4))

    ax_l.plot(ep, history["train_loss"], label="train")
    ax_l.plot(ep, history["val_loss"],   label="val")
    ax_l.set_title(f"Loss - {title}" if title else "Loss")
    ax_l.set_xlabel("Epoch")
    ax_l.legend(); ax_l.grid(True)

    ax_a.plot(ep, history["train_acc"], label="train type")
    ax_a.plot(ep, history["val_acc"],   label="val type")
    if history.get("val_rot_acc"):
        ax_a.plot(ep, history["val_rot_acc"], label="val rotation", linestyle="--")
    ax_a.set_title(f"Accuracy - {title}" if title else "Accuracy")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylim(0, 1)
    ax_a.legend(); ax_a.grid(True)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def save_confusion_matrix(
    y_true: list, y_pred: list, label_names: list[str], out_path: Path, title: str = "",
    labels: list[int] | None = None,
) -> None:
    if labels is None:
        labels = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def save_history(history: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(history, indent=2), encoding="utf-8")