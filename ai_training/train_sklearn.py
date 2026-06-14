"""
Experiment 6 — Sklearn on combined features.

Feature vector per video (base 1366-d):
  - 512-d : R3D-18 frozen backbone              (data/features/{stem}.npy)
  - 297-d : pose mean/std/velocity              (data/poses/{stem}.npy)
  - 45-d  : optical flow mean_u/mean_v/mag      (data/features_flow/{stem}.npy)
  - 512-d : ResNet18 on lower-body takeoff crop (data/features_local/{stem}.npy)

Optional YOLO features (512-d, data/features_yolo/{stem}.npy):
  - ResNet18 on YOLO-guided blade crop (bottom 30% of person bbox)
  - Compared against the heuristic local crop in a separate ablation section
  - Run extract_yolo_features.py first to generate these features

5-fold stratified CV on all 305 videos (train + val + test).
Saves ablation bar chart and confusion matrix to reports/figures/exp6_sklearn/.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

ROOT            = Path(__file__).resolve().parent.parent
CKPT_DIR        = Path(__file__).resolve().parent / "checkpoints"
VID_DIR         = ROOT / "data" / "features"
POSE_DIR        = ROOT / "data" / "poses"
FLOW_DIR        = ROOT / "data" / "features_flow"
LOCAL_DIR       = ROOT / "data" / "features_local"
YOLO_DIR        = ROOT / "data" / "features_yolo"
LOCAL_BOT_DIR   = ROOT / "data" / "features_local_bottom"
YOLO_BOT_DIR    = ROOT / "data" / "features_yolo_bottom"
METAS           = [
    ROOT / "data" / "frames" / "dataset_metadata_train.json",
    ROOT / "data" / "frames" / "dataset_metadata_val.json",
    ROOT / "data" / "frames" / "dataset_metadata_test.json",
]
REPORT_DIR = ROOT / "reports" / "figures" / "exp6_sklearn"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES  = ["salchow", "axel", "toe_loop"]
FAILED_LABEL = 3

VID_DIM   = 512
POSE_DIM  = 297
FLOW_DIM  = 45
LOCAL_DIM = 512
YOLO_DIM  = 512


# ── Metadata ──────────────────────────────────────────────────────────────────
def load_metadata() -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for p in METAS:
        entries.update(json.loads(p.read_text(encoding="utf-8")))
    return entries


# ── Pose feature: (T, 99) → (297,) ───────────────────────────────────────────
def pose_features(poses: np.ndarray) -> np.ndarray:
    mean_pos = poses.mean(axis=0)
    std_pos  = poses.std(axis=0)
    velocity = np.abs(np.diff(poses, axis=0)).mean(axis=0)
    return np.concatenate([mean_pos, std_pos, velocity])


# ── Dataset assembly ──────────────────────────────────────────────────────────
def build_dataset():
    metadata = load_metadata()

    X_vid, X_pose, X_flow, X_local, y = [], [], [], [], []
    yolo_raw:      list[np.ndarray | None] = []
    local_bot_raw: list[np.ndarray | None] = []
    yolo_bot_raw:  list[np.ndarray | None] = []
    skipped = {"no_video": 0, "no_pose": 0, "no_flow": 0, "no_local": 0}

    for stem, entry in metadata.items():
        label = entry["jump_type_label"]
        if label == FAILED_LABEL:
            continue

        vid_path   = VID_DIR   / f"{stem}.npy"
        pose_path  = POSE_DIR  / f"{stem}.npy"
        flow_path  = FLOW_DIR  / f"{stem}.npy"
        local_path = LOCAL_DIR / f"{stem}.npy"

        if not vid_path.exists():   skipped["no_video"] += 1;  continue
        if not pose_path.exists():  skipped["no_pose"] += 1;   continue
        if not flow_path.exists():  skipped["no_flow"] += 1;   continue
        if not local_path.exists(): skipped["no_local"] += 1;  continue

        X_vid.append(np.load(vid_path))
        X_pose.append(pose_features(np.load(pose_path)))
        X_flow.append(np.load(flow_path))
        X_local.append(np.load(local_path))
        y.append(label)

        yolo_path     = YOLO_DIR     / f"{stem}.npy"
        local_bot_path = LOCAL_BOT_DIR / f"{stem}.npy"
        yolo_bot_path  = YOLO_BOT_DIR  / f"{stem}.npy"

        yolo_raw.append(np.load(yolo_path)      if yolo_path.exists()     else None)
        local_bot_raw.append(np.load(local_bot_path) if local_bot_path.exists() else None)
        yolo_bot_raw.append(np.load(yolo_bot_path)   if yolo_bot_path.exists()  else None)

    X_vid   = np.array(X_vid,   dtype=np.float32)
    X_pose  = np.array(X_pose,  dtype=np.float32)
    X_flow  = np.array(X_flow,  dtype=np.float32)
    X_local = np.array(X_local, dtype=np.float32)
    y       = np.array(y,       dtype=np.int64)

    def _optional_array(raw: list) -> np.ndarray | None:
        n_missing = sum(1 for x in raw if x is None)
        return np.array(raw, dtype=np.float32) if n_missing == 0 else None

    X_yolo     = _optional_array(yolo_raw)
    X_local_bot = _optional_array(local_bot_raw)
    X_yolo_bot  = _optional_array(yolo_bot_raw)

    print(f"Dataset: {len(y)} videos")
    for k, v in skipped.items():
        if v: print(f"  skipped ({k}): {v}")
    for lbl, name in enumerate(LABEL_NAMES):
        print(f"  {name}: {(y == lbl).sum()}")

    def _feat_status(name: str, arr: np.ndarray | None, raw: list) -> None:
        if arr is not None:
            print(f"  {name}: available ({len(y)} videos)")
        else:
            n = sum(1 for x in raw if x is None)
            print(f"  {name}: missing for {n}/{len(y)} - run extractor first")

    _feat_status("YOLO features (center)",      X_yolo,     yolo_raw)
    _feat_status("Local bottom crop",           X_local_bot, local_bot_raw)
    _feat_status("YOLO bottom crop",            X_yolo_bot,  yolo_bot_raw)

    return X_vid, X_pose, X_flow, X_local, X_yolo, X_local_bot, X_yolo_bot, y


# ── CV helper ─────────────────────────────────────────────────────────────────
def run_cv(name: str, clf, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> float:
    pipe   = make_pipeline(StandardScaler(), clf)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"  {name}")
    print(f"    per-fold: {[f'{s:.3f}' for s in scores]}")
    print(f"    mean +/- std: {scores.mean():.3f} +/- {scores.std():.3f}")
    return float(scores.mean())


# ── Ablation ──────────────────────────────────────────────────────────────────
def ablation(X_vid, X_pose, X_flow, X_local, y, cv):
    print("\n-- Ablation (LogReg C=0.05) -----------------------------------------")
    clf = LogisticRegression(C=0.05, max_iter=1000, random_state=42)

    results = {
        "Video only (512-d)":       run_cv("Video only        (512-d)", clf, X_vid,   y, cv),
        "Pose only (297-d)":        run_cv("Pose only         (297-d)", clf, X_pose,  y, cv),
        "Flow only (45-d)":         run_cv("Flow only          (45-d)", clf, X_flow,  y, cv),
        "Local crop only (512-d)":  run_cv("Local crop only   (512-d)", clf, X_local, y, cv),
        "Video+Local (1024-d)":     run_cv("Video + Local    (1024-d)", clf, np.hstack([X_vid, X_local]),  y, cv),
        "Video+Pose (809-d)":       run_cv("Video + Pose     ( 809-d)", clf, np.hstack([X_vid, X_pose]),   y, cv),
        "All four (1366-d)":        run_cv("All four        (1366-d)",  clf, np.hstack([X_vid, X_pose, X_flow, X_local]), y, cv),
    }

    # ── Ablation bar chart ────────────────────────────────────────────────────
    _save_bar_chart(
        results,
        title="Exp 6 — Feature Ablation (LogReg C=0.05)",
        out_path=REPORT_DIR / "ablation_bar_chart.png",
    )
    return results


def ablation_yolo(X_vid, X_pose, X_flow, X_local, X_yolo,
                  X_local_bot, X_yolo_bot, y, cv):
    """
    Compare all crop variants:
      - Heuristic center crop (current)
      - YOLO center crop
      - Heuristic bottom crop  (new)
      - YOLO bottom crop       (new)
    """
    print("\n-- Crop variant ablation (LogReg C=0.05) ----------------------------")
    clf = LogisticRegression(C=0.05, max_iter=1000, random_state=42)

    # ── single-stream comparisons ─────────────────────────────────────────────
    results: dict[str, float] = {}

    def add(label: str, log_label: str, X: np.ndarray | None) -> None:
        if X is not None:
            results[label] = run_cv(log_label, clf, X, y, cv)

    add("Heuristic center (512-d)",  "Heuristic center  (512-d)", X_local)
    add("YOLO center (512-d)",       "YOLO center       (512-d)", X_yolo)
    add("Heuristic bottom (512-d)",  "Heuristic bottom  (512-d)", X_local_bot)
    add("YOLO bottom (512-d)",       "YOLO bottom       (512-d)", X_yolo_bot)

    # ── combined with video ───────────────────────────────────────────────────
    def add_combo(label: str, log_label: str, X_crop: np.ndarray | None) -> None:
        if X_crop is not None:
            add(label, log_label, np.hstack([X_vid, X_crop]))

    add_combo("Video + Heuristic center (1024-d)", "Video+Heuristic ctr (1024-d)", X_local)
    add_combo("Video + YOLO center (1024-d)",      "Video+YOLO center   (1024-d)", X_yolo)
    add_combo("Video + Heuristic bottom (1024-d)", "Video+Heuristic bot (1024-d)", X_local_bot)
    add_combo("Video + YOLO bottom (1024-d)",      "Video+YOLO bottom   (1024-d)", X_yolo_bot)

    # ── best prior + each crop ────────────────────────────────────────────────
    def add_vp_combo(label: str, log_label: str, X_crop: np.ndarray | None) -> None:
        if X_crop is not None:
            add(label, log_label, np.hstack([X_vid, X_pose, X_crop]))

    add_vp_combo("Video+Pose+Heuristic ctr (1321-d)", "V+P+Heur ctr (1321-d)", X_local)
    add_vp_combo("Video+Pose+YOLO ctr (1321-d)",      "V+P+YOLO ctr (1321-d)", X_yolo)
    add_vp_combo("Video+Pose+Heuristic bot (1321-d)", "V+P+Heur bot (1321-d)", X_local_bot)
    add_vp_combo("Video+Pose+YOLO bot (1321-d)",      "V+P+YOLO bot (1321-d)", X_yolo_bot)

    if results:
        _save_bar_chart(
            results,
            title="Exp 6 — Crop variant comparison (LogReg C=0.05)",
            out_path=REPORT_DIR / "ablation_yolo_bar_chart.png",
            figsize=(12, max(6, len(results) * 0.55)),
        )
    return results


def _save_bar_chart(
    results: dict,
    title: str,
    out_path: Path,
    figsize: tuple = (10, 5),
) -> None:
    labels = list(results.keys())
    values = list(results.values())
    colors = ["#4878CF" if v < max(values) else "#E34A33" for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("5-fold CV Accuracy")
    ax.set_title(title)
    ax.axvline(1/3, color="gray", linestyle="--", linewidth=0.8, label="chance (33%)")
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Bar chart -> {out_path}")


# ── Full grid on best combo ───────────────────────────────────────────────────
def full_eval(X: np.ndarray, y: np.ndarray, cv: StratifiedKFold, label: str = "1366-d"):
    print(f"\n-- Grid search on combined {label} -----------------------------------")
    grid_results = {}
    for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
        name = f"LogReg C={C}"
        grid_results[C] = run_cv(name,
                                 LogisticRegression(C=C, max_iter=1000, random_state=42),
                                 X, y, cv)

    run_cv("LinearSVC C=0.1", LinearSVC(C=0.1, max_iter=2000, random_state=42), X, y, cv)
    run_cv("SVM RBF C=1",     SVC(C=1.0,  kernel="rbf", random_state=42), X, y, cv)
    run_cv("SVM RBF C=10",    SVC(C=10.0, kernel="rbf", random_state=42), X, y, cv)

    # ── Grid search bar chart ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    c_labels = [f"LogReg C={c}" for c in grid_results]
    c_vals   = list(grid_results.values())
    ax.bar(c_labels, c_vals, color="#4878CF")
    ax.set_ylim(0, 1)
    ax.set_ylabel("5-fold CV Accuracy")
    ax.set_title(f"Exp 6 — LogReg C Grid Search ({label})")
    ax.axhline(1/3, color="gray", linestyle="--", linewidth=0.8, label="chance (33%)")
    ax.legend()
    for i, v in enumerate(c_vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "grid_search_bar_chart.png", dpi=120)
    plt.close(fig)

    best_C = max(grid_results, key=lambda k: grid_results[k])
    print(f"\n-- Confusion matrix (LogReg C={best_C}, out-of-fold) ----------------")
    pipe   = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=best_C, max_iter=1000, random_state=42),
    )
    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    cm     = confusion_matrix(y, y_pred, labels=[0, 1, 2])
    print(cm)
    print(classification_report(y, y_pred, target_names=LABEL_NAMES))

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Exp 6 — All Features, LogReg C={best_C} (5-fold OOF)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    print(f"  Plots saved to {REPORT_DIR}")


def save_final_logreg(X_vid: np.ndarray, X_pose: np.ndarray, y: np.ndarray) -> None:
    """Fit Video+Pose LogReg on all available data and save for inference."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CKPT_DIR / "final_logreg_type.pkl"
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=1000, random_state=42),
    )
    pipe.fit(np.hstack([X_vid, X_pose]), y)
    joblib.dump(pipe, out_path)
    print(f"\n  Final LogReg (Video+Pose, C=0.05) fitted on {len(y)} videos -> {out_path}")


def main():
    X_vid, X_pose, X_flow, X_local, X_yolo, X_local_bot, X_yolo_bot, y = build_dataset()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    ablation(X_vid, X_pose, X_flow, X_local, y, cv)

    have_any_crop_variant = any(
        x is not None for x in [X_yolo, X_local_bot, X_yolo_bot]
    )
    if have_any_crop_variant:
        ablation_yolo(X_vid, X_pose, X_flow, X_local, X_yolo,
                      X_local_bot, X_yolo_bot, y, cv)

    # Grid search on the richest available feature set
    extras = [x for x in [X_yolo, X_local_bot, X_yolo_bot] if x is not None]
    X_all  = np.hstack([X_vid, X_pose, X_flow, X_local] + extras)
    label  = f"{X_all.shape[1]}-d (all available)"
    full_eval(X_all, y, cv, label=label)

    # Fit and save final model for inference (used by final_model.py)
    save_final_logreg(X_vid, X_pose, y)


if __name__ == "__main__":
    main()