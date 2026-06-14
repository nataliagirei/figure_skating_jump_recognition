"""
Generate per-video debug MP4s for training and validation splits.

Each video shows all 16 frames side-by-side with a panel:
  Left  — frame image with:
            • coloured border showing jump phase (gray/green/orange/blue)
            • phase label top-left;  ★ on the takeoff frame
  Right — info panel:
            • CORRECT / WRONG verdict
            • Ground-truth class + rotation
            • Type confidence bars (all 3 classes)
            • Rotation confidence bars 1×/2×/3× (PyTorch models only)
            • Predicted class + rotation + confidence

Output:
    videos/<exp>/train/<stem>.mp4
    videos/<exp>/val/<stem>.mp4

Usage:
    python ai_training/make_debug_videos.py --exp exp1
    python ai_training/make_debug_videos.py --exp exp3   # needs mediapipe
    python ai_training/make_debug_videos.py --exp exp6   # needs pre-extracted features
    python ai_training/make_debug_videos.py --exp exp7_yolo_bot --split val
    python ai_training/make_debug_videos.py --exp exp4 --split val --fps 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai_training"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_NAMES = ["Salchow", "Axel", "Toe Loop"]
LABEL_BGR   = [
    ( 60,  76, 231),   # Salchow  → #e74c3c (red)
    (219, 152,  52),   # Axel     → #3498db (blue)
    (113, 204,  46),   # Toe Loop → #2ecc71 (green)
]

ROT_NAMES = ["?", "Single", "Double", "Triple"]
ROT_SHORT = ["?", "1×", "2×", "3×"]

PHASE_BGR = {
    "pre-takeoff": (110, 110, 110),
    "takeoff":     ( 40, 220, 100),
    "flight":      ( 30, 180, 255),
    "landing":     (255, 140,   0),
}

CORRECT_BGR = ( 50, 210,  50)
WRONG_BGR   = ( 50,  50, 220)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

N_FRAMES = 16
FRAME_H  = 224
FRAME_W  = 224
PANEL_W  = 216
OUT_W    = FRAME_W + PANEL_W   # 440
OUT_H    = FRAME_H             # 224


# ── Image helpers ──────────────────────────────────────────────────────────────

def unnormalize(arr: np.ndarray) -> np.ndarray:
    """(3, H, W) float32 ImageNet-normed → (H, W, 3) uint8 BGR."""
    rgb = arr.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def _text(img, txt, xy, scale, color, thickness=1):
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


# ── Phase detection ────────────────────────────────────────────────────────────

def compute_phases(frames: list[np.ndarray]) -> tuple[list[str], int]:
    """(3,H,W) npy list → (phase_names, takeoff_idx) via optical flow."""
    mean_vs: list[float] = []
    for f1, f2 in list(zip(frames, frames[1:]))[:15]:
        g1 = cv2.cvtColor(unnormalize(f1), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(unnormalize(f2), cv2.COLOR_BGR2GRAY).astype(np.float32)
        fl = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mean_vs.append(float(fl[..., 1].mean()))

    takeoff_idx = min(int(np.argmin(mean_vs)) + 1 if mean_vs else 0, len(frames) - 1)
    n          = len(frames)
    flight_end = takeoff_idx + max(2, n // 4)

    phases = []
    for i in range(n):
        if i < takeoff_idx:
            phases.append("pre-takeoff")
        elif i <= takeoff_idx + 1:
            phases.append("takeoff")
        elif i <= flight_end:
            phases.append("flight")
        else:
            phases.append("landing")
    return phases, takeoff_idx


def annotate_frame(frame_bgr: np.ndarray, phase: str) -> np.ndarray:
    """Draw coloured border + phase label on the frame (in-place copy)."""
    out   = frame_bgr.copy()
    color = PHASE_BGR[phase]

    cv2.rectangle(out, (0, 0), (FRAME_W - 1, FRAME_H - 1), color, 3)

    label = ("★ " if phase == "takeoff" else "") + phase.upper()
    tw    = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
    cv2.rectangle(out, (0, 0), (tw + 8, 18), (0, 0, 0), -1)
    _text(out, label, (4, 13), 0.38, color)
    return out


# ── Info panel ─────────────────────────────────────────────────────────────────

def draw_panel(
    probs:     np.ndarray | None,
    rot_probs: np.ndarray | None,
    gt:        int,
    gt_rot:    int,
    frame_idx: int,
    n_frames:  int,
    stem:      str,
    phase:     str,
    na_reason: str = "",
) -> np.ndarray:
    panel = np.full((OUT_H, PANEL_W, 3), 28, dtype=np.uint8)

    # ── N/A state ─────────────────────────────────────────────────────────────
    if probs is None:
        _text(panel, "N/A", (6, 22), 0.65, (150, 150, 150), 2)
        y = 46
        for chunk in (na_reason[i:i + 20] for i in range(0, len(na_reason), 20)):
            _text(panel, chunk, (6, y), 0.33, (110, 110, 110))
            y += 15
        _text(panel, stem[:24], (4, OUT_H - 6), 0.28, (110, 110, 110))
        return panel

    pred     = int(np.argmax(probs))
    pred_rot = int(np.argmax(rot_probs[1:])) + 1 if rot_probs is not None else None
    is_correct = pred == gt

    # ── CORRECT / WRONG ───────────────────────────────────────────────────────
    _text(panel, "CORRECT" if is_correct else "WRONG",
          (6, 19), 0.52, CORRECT_BGR if is_correct else WRONG_BGR, 2)

    # ── Frame counter + phase ─────────────────────────────────────────────────
    phase_clr = PHASE_BGR[phase]
    _text(panel, f"Frame {frame_idx+1}/{n_frames}", (6, 33), 0.36, (170, 170, 170))
    _text(panel, phase.upper(), (PANEL_W - cv2.getTextSize(
        phase.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0][0] - 4, 33),
        0.33, phase_clr)

    # ── Ground truth ──────────────────────────────────────────────────────────
    gt_name = LABEL_NAMES[gt] if gt < 3 else "Failed"
    gt_clr  = LABEL_BGR[gt]   if gt < 3 else (150, 150, 150)
    gt_rot_str = f" {ROT_SHORT[gt_rot]}" if gt_rot > 0 else ""
    _text(panel, "GT:", (6, 47), 0.37, (180, 180, 180))
    _text(panel, f"{gt_name}{gt_rot_str}", (32, 47), 0.40, gt_clr)

    cv2.line(panel, (4, 53), (PANEL_W - 4, 53), (70, 70, 70), 1)

    # ── Type confidence bars ───────────────────────────────────────────────────
    _text(panel, "TYPE", (6, 63), 0.33, (140, 140, 140))
    bx, max_b, bh, gap = 6, PANEL_W - 14, 14, 5
    y0 = 68
    for i, (name, p) in enumerate(zip(LABEL_NAMES, probs)):
        y   = y0 + i * (bh + gap)
        bw  = max(1, int(p * max_b))
        clr = LABEL_BGR[i] if i == pred else (60, 60, 60)
        cv2.rectangle(panel, (bx, y), (bx + max_b, y + bh), (45, 45, 45), -1)
        cv2.rectangle(panel, (bx, y), (bx + bw,    y + bh), clr, -1)
        cv2.rectangle(panel, (bx, y), (bx + max_b, y + bh), (85, 85, 85), 1)
        _text(panel, name[:3], (bx + 2, y + bh - 3), 0.30, (220, 220, 220))
        pct = f"{p:.0%}"
        tw  = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)[0][0]
        _text(panel, pct, (bx + max_b - tw - 2, y + bh - 3), 0.30, (220, 220, 220))

    # ── Rotation bars (PyTorch models only, skip "unknown" class 0) ────────────
    rot_y0 = y0 + 3 * (bh + gap) + 8
    if rot_probs is not None:
        _text(panel, "ROTATION", (6, rot_y0), 0.33, (140, 140, 140))
        rbh, rgap = 12, 4
        for i in range(1, 4):                 # 1=single, 2=double, 3=triple
            p   = float(rot_probs[i])
            y   = rot_y0 + 5 + (i - 1) * (rbh + rgap)
            bw  = max(1, int(p * max_b))
            is_pred_rot = (i == pred_rot)
            clr = (180, 140, 50) if is_pred_rot else (55, 55, 55)
            cv2.rectangle(panel, (bx, y), (bx + max_b, y + rbh), (40, 40, 40), -1)
            cv2.rectangle(panel, (bx, y), (bx + bw,    y + rbh), clr, -1)
            cv2.rectangle(panel, (bx, y), (bx + max_b, y + rbh), (80, 80, 80), 1)
            _text(panel, ROT_SHORT[i], (bx + 2, y + rbh - 3), 0.29, (210, 210, 210))
            pct = f"{p:.0%}"
            tw  = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.29, 1)[0][0]
            _text(panel, pct, (bx + max_b - tw - 2, y + rbh - 3), 0.29, (210, 210, 210))
        div_y = rot_y0 + 5 + 3 * (rbh + rgap)
    else:
        div_y = rot_y0 + 4

    # ── Prediction summary ─────────────────────────────────────────────────────
    cv2.line(panel, (4, div_y), (PANEL_W - 4, div_y), (70, 70, 70), 1)
    pred_name = LABEL_NAMES[pred] if pred < 3 else "Failed"
    pred_clr  = LABEL_BGR[pred]   if pred < 3 else (150, 150, 150)
    _text(panel, "PRED:", (6, div_y + 13), 0.36, (180, 180, 180))
    rot_str = f" {ROT_SHORT[pred_rot]}" if pred_rot is not None else ""
    _text(panel, f"{pred_name}{rot_str}", (6, div_y + 26), 0.46, pred_clr, 1)
    conf_str = f"{probs[pred]:.0%}"
    if rot_probs is not None and pred_rot is not None:
        conf_str += f"  rot {rot_probs[pred_rot]:.0%}"
    _text(panel, conf_str, (6, div_y + 39), 0.38, pred_clr)

    _text(panel, stem[:26], (4, OUT_H - 5), 0.27, (100, 100, 100))
    return panel


# ── Video writer ───────────────────────────────────────────────────────────────

def write_debug_video(
    frames:    list[np.ndarray],
    probs:     np.ndarray | None,
    rot_probs: np.ndarray | None,
    gt:        int,
    gt_rot:    int,
    out_path:  Path,
    stem:      str,
    fps:       int,
    na_reason: str = "",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer  = cv2.VideoWriter(str(out_path),
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (OUT_W, OUT_H))

    phases, _ = compute_phases(frames) if probs is not None else (
        ["pre-takeoff"] * len(frames), 0)

    for i, arr in enumerate(frames):
        phase  = phases[i]
        canvas = np.empty((OUT_H, OUT_W, 3), dtype=np.uint8)
        canvas[:, :FRAME_W] = annotate_frame(unnormalize(arr), phase)
        canvas[:, FRAME_W:] = draw_panel(
            probs, rot_probs, gt, gt_rot, i, len(frames), stem, phase, na_reason)
        writer.write(canvas)

    writer.release()


# ── Experiment configs ─────────────────────────────────────────────────────────

EXP_CONFIGS: dict[str, dict] = {
    "exp1": {
        "type":   "video",
        "ckpt":   "best_model.pth",
        "module": "model",
        "cls":    "FigureSkatingModel",
        "kwargs": {},
        "n_type": 3,
    },
    "exp2": {
        "type":   "video",
        "ckpt":   "best_model_a.pth",
        "module": "model_a",
        "cls":    "FigureSkatingModelA",
        "kwargs": {},
        "n_type": 4,
    },
    "exp3": {
        "type":   "pose",
        "ckpt":   "best_pose_model.pth",
        "module": "pose_model",
        "cls":    "PoseModel",
        "kwargs": {},
        "n_type": 3,
    },
    "exp4": {
        "type":   "video",
        "ckpt":   "best_model_b_finetune.pth",
        "module": "model_b",
        "cls":    "FigureSkatingModelB",
        "kwargs": {"freeze_backbone": False},
        "n_type": 3,
    },
    "exp5": {
        "type":   "video",
        "ckpt":   "best_model_b.pth",
        "module": "model_b",
        "cls":    "FigureSkatingModelB",
        "kwargs": {"freeze_backbone": True},
        "n_type": 3,
    },
    "exp6": {
        "type": "sklearn",
    },
    # Exp 7 — crop variant ablation (4 separate LogReg pipelines)
    "exp7_heur_ctr": {
        "type":        "sklearn_crop",
        "feature_dir": "features_local",
        "label":       "Heuristic center crop",
    },
    "exp7_yolo_ctr": {
        "type":        "sklearn_crop",
        "feature_dir": "features_yolo",
        "label":       "YOLO center crop",
    },
    "exp7_heur_bot": {
        "type":        "sklearn_crop",
        "feature_dir": "features_local_bottom",
        "label":       "Heuristic bottom crop",
    },
    "exp7_yolo_bot": {
        "type":        "sklearn_crop",
        "feature_dir": "features_yolo_bottom",
        "label":       "YOLO bottom crop",
    },
}


# ── Model loading ──────────────────────────────────────────────────────────────

def load_pytorch_model(cfg: dict):
    mod   = __import__(cfg["module"])
    cls   = getattr(mod, cfg["cls"])
    model = cls(**cfg["kwargs"]).to(DEVICE)
    ckpt  = ROOT / "ai_training" / "checkpoints" / cfg["ckpt"]
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def load_sklearn_pipeline():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    VID   = ROOT / "data" / "features"
    POSE  = ROOT / "data" / "poses"
    FLOW  = ROOT / "data" / "features_flow"
    LOCAL = ROOT / "data" / "features_local"
    METAS = [ROOT / "data" / "frames" / f"dataset_metadata_{s}.json"
             for s in ("train", "val", "test")]

    meta: dict = {}
    for p in METAS:
        if p.exists():
            meta.update(json.loads(p.read_text(encoding="utf-8")))

    X_rows, y_rows = [], []
    for stem, entry in meta.items():
        label = entry.get("jump_type_label", 3)
        if label == 3:
            continue
        paths = [VID / f"{stem}.npy", POSE / f"{stem}.npy",
                 FLOW / f"{stem}.npy", LOCAL / f"{stem}.npy"]
        if not all(p.exists() for p in paths):
            continue
        pose_raw  = np.load(paths[1])
        pose_feat = np.concatenate([pose_raw.mean(0), pose_raw.std(0),
                                    np.abs(np.diff(pose_raw, axis=0)).mean(0)])
        X_rows.append(np.concatenate([np.load(paths[0]), pose_feat,
                                      np.load(paths[2]), np.load(paths[3])]))
        y_rows.append(label)

    if not X_rows:
        print("  Sklearn: no pre-extracted features found - skipping exp6")
        return None

    pipe = make_pipeline(
        StandardScaler(),
        SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
    )
    pipe.fit(np.array(X_rows, dtype=np.float32), np.array(y_rows))
    print(f"  Sklearn SVM fitted on {len(y_rows)} videos.")
    return pipe


def load_sklearn_crop(feature_dir_name: str):
    """Fit LogReg (C=0.05) on a single 512-d crop feature using train split only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feat_dir   = ROOT / "data" / feature_dir_name
    meta_train = ROOT / "data" / "frames" / "dataset_metadata_train.json"

    if not feat_dir.exists():
        print(f"  Sklearn crop: {feature_dir_name}/ missing - skipping")
        return None

    meta = json.loads(meta_train.read_text(encoding="utf-8"))
    X, y = [], []
    for stem, entry in meta.items():
        label = entry.get("jump_type_label", 3)
        if label == 3:
            continue
        p = feat_dir / f"{stem}.npy"
        if not p.exists():
            continue
        X.append(np.load(p))
        y.append(label)

    if not X:
        print(f"  Sklearn crop: no features in {feature_dir_name}/ - skipping")
        return None

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=1000, random_state=42),
    )
    pipe.fit(np.array(X, dtype=np.float32), np.array(y))
    print(f"  Sklearn crop ({feature_dir_name}) fitted on {len(y)} train videos.")
    return pipe


# ── Pose extraction (MediaPipe) ────────────────────────────────────────────────

_MEDIAPIPE_OK: bool | None = None


def _check_mediapipe() -> bool:
    global _MEDIAPIPE_OK
    if _MEDIAPIPE_OK is None:
        try:
            from mediapipe.tasks import python as _
            _MEDIAPIPE_OK = True
        except ImportError:
            _MEDIAPIPE_OK = False
            print("  mediapipe not installed - exp3 will show N/A")
    return _MEDIAPIPE_OK


def _ensure_pose_task() -> str | None:
    model_path = ROOT / "pose_extraction" / "pose_landmarker_full.task"
    if model_path.exists():
        return str(model_path)
    import urllib.request
    url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
           "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
    try:
        print("  Downloading pose landmarker model...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, model_path)
        return str(model_path)
    except Exception as e:
        print(f"  Could not download pose model: {e}")
        return None


def extract_poses(bgr_frames: list[np.ndarray]) -> np.ndarray | None:
    """list of BGR uint8 (H,W,3) → (T, 99) float32, or None."""
    if not _check_mediapipe():
        return None
    task_path = _ensure_pose_task()
    if task_path is None:
        return None

    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=task_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    pose_seq: list[np.ndarray] = []
    last_valid = np.zeros(99, dtype=np.float32)

    with mp_vision.PoseLandmarker.create_from_options(opts) as lm:
        for frame_bgr in bgr_frames:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            det = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if det.pose_landmarks:
                kpts = []
                for p in det.pose_landmarks[0]:
                    kpts.extend([p.x, p.y, p.visibility])
                last_valid = np.array(kpts, dtype=np.float32)
            pose_seq.append(last_valid.copy())

    poses = np.stack(pose_seq)
    LH, RH, LS, RS = 23, 24, 11, 12
    out = poses.copy()
    for t in range(len(poses)):
        frame  = poses[t].reshape(33, 3)
        xy     = frame[:, :2]
        hip_c  = (xy[LH] + xy[RH]) / 2
        sw     = np.linalg.norm(xy[LS] - xy[RS])
        nf     = frame.copy()
        nf[:, :2] = (xy - hip_c) / (sw if sw > 1e-6 else 1.0)
        out[t] = nf.reshape(-1)
    return out.astype(np.float32)


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_video(model, frames: list[np.ndarray],
                n_type: int) -> tuple[np.ndarray, np.ndarray]:
    """→ (type_probs (3,), rot_probs (4,))"""
    batch               = torch.from_numpy(np.stack(frames)).unsqueeze(0).float().to(DEVICE)
    type_logits, rot_logits = model(batch)
    type_probs = F.softmax(type_logits[:, :n_type], dim=-1).cpu().numpy()[0, :3]
    s = type_probs.sum()
    type_probs = type_probs / s if s > 1e-9 else type_probs
    rot_probs  = F.softmax(rot_logits, dim=-1).cpu().numpy()[0]   # (4,)
    return type_probs, rot_probs


@torch.no_grad()
def infer_pose(model, frames: list[np.ndarray]) -> tuple[np.ndarray | None,
                                                          np.ndarray | None]:
    """→ (type_probs (3,), rot_probs (4,)) or (None, None) if MediaPipe unavailable."""
    bgr_frames = [unnormalize(f) for f in frames]
    poses = extract_poses(bgr_frames)
    if poses is None:
        return None, None
    x = torch.from_numpy(poses).unsqueeze(0).float().to(DEVICE)
    type_logits, rot_logits = model(x)
    return (F.softmax(type_logits, dim=-1).cpu().numpy()[0],
            F.softmax(rot_logits,  dim=-1).cpu().numpy()[0])


def infer_sklearn(pipe, stem: str) -> tuple[np.ndarray | None, str]:
    VID   = ROOT / "data" / "features"
    POSE  = ROOT / "data" / "poses"
    FLOW  = ROOT / "data" / "features_flow"
    LOCAL = ROOT / "data" / "features_local"

    paths   = [VID / f"{stem}.npy", POSE / f"{stem}.npy",
                FLOW / f"{stem}.npy", LOCAL / f"{stem}.npy"]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        return None, f"Missing: {', '.join(missing)}"

    pose_raw  = np.load(paths[1])
    pose_feat = np.concatenate([pose_raw.mean(0), pose_raw.std(0),
                                np.abs(np.diff(pose_raw, axis=0)).mean(0)])
    x = np.concatenate([np.load(paths[0]), pose_feat,
                        np.load(paths[2]), np.load(paths[3])]).reshape(1, -1)

    probs   = pipe.predict_proba(x.astype(np.float32))[0]
    classes = list(pipe.classes_)
    result  = np.zeros(3, dtype=np.float32)
    for c, p in zip(classes, probs):
        if 0 <= c < 3:
            result[c] = p
    s = result.sum()
    return (result / s if s > 1e-9 else result), ""


def infer_sklearn_crop(pipe, stem: str,
                       feature_dir_name: str) -> tuple[np.ndarray | None, str]:
    feat_path = ROOT / "data" / feature_dir_name / f"{stem}.npy"
    if not feat_path.exists():
        return None, f"Missing: {feat_path.name}"

    x       = np.load(feat_path).reshape(1, -1).astype(np.float32)
    probs   = pipe.predict_proba(x)[0]
    classes = list(pipe.classes_)
    result  = np.zeros(3, dtype=np.float32)
    for c, p in zip(classes, probs):
        if 0 <= c < 3:
            result[c] = p
    s = result.sum()
    return (result / s if s > 1e-9 else result), ""


# ── Split processing ───────────────────────────────────────────────────────────

def process_split(exp: str, state: dict, meta_path: Path,
                  split_name: str, out_dir: Path, fps: int) -> None:
    cfg   = EXP_CONFIGS[exp]
    meta  = json.loads(meta_path.read_text(encoding="utf-8"))
    items = [(stem, e) for stem, e in meta.items()
             if e.get("jump_type_label", 3) != 3]

    print(f"  {split_name}: {len(items)} videos")

    wrong = na = 0
    for idx, (stem, entry) in enumerate(items, 1):
        gt      = entry["jump_type_label"]
        gt_rot  = entry.get("rotation", 0)
        paths   = entry.get("frames", [])
        if not paths:
            continue

        frames    = [np.load(p) for p in paths]
        probs:     np.ndarray | None = None
        rot_probs: np.ndarray | None = None
        na_reason = ""

        if cfg["type"] == "video":
            probs, rot_probs = infer_video(state["model"], frames, cfg["n_type"])

        elif cfg["type"] == "pose":
            probs, rot_probs = infer_pose(state["model"], frames)
            if probs is None:
                na_reason = "MediaPipe unavailable"
                na += 1

        elif cfg["type"] == "sklearn":
            if state.get("sklearn") is None:
                probs, na_reason = None, "No pre-extracted features"
                na += 1
            else:
                probs, na_reason = infer_sklearn(state["sklearn"], stem)
                if probs is None:
                    na += 1

        elif cfg["type"] == "sklearn_crop":
            if state.get("sklearn_crop") is None:
                probs, na_reason = None, f"No features: {cfg['feature_dir']}"
                na += 1
            else:
                probs, na_reason = infer_sklearn_crop(
                    state["sklearn_crop"], stem, cfg["feature_dir"])
                if probs is None:
                    na += 1

        if probs is not None:
            wrong += int(int(np.argmax(probs)) != gt)

        out_path = out_dir / split_name / f"{stem}.mp4"
        write_debug_video(frames, probs, rot_probs, gt, gt_rot,
                          out_path, stem, fps, na_reason)

        if idx % 20 == 0 or idx == len(items):
            valid   = idx - na
            acc_str = f"acc={1-wrong/valid:.1%}" if valid > 0 else "acc=N/A"
            print(f"    {idx}/{len(items)}  {acc_str}  wrong={wrong}  n/a={na}")

    if items:
        valid = len(items) - na
        if valid > 0:
            print(f"  {split_name} done: acc={1-wrong/valid:.1%}  "
                  f"wrong={wrong}/{valid}  n/a={na}")
        else:
            print(f"  {split_name} done: all {na} videos were N/A")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-video debug videos for a trained experiment."
    )
    parser.add_argument("--exp",   default="exp1", choices=list(EXP_CONFIGS))
    parser.add_argument("--split", default="both", choices=["train", "val", "both"])
    parser.add_argument("--fps",   type=int, default=4)
    args = parser.parse_args()

    cfg = EXP_CONFIGS[args.exp]
    print(f"Device : {DEVICE}")
    print(f"Exp    : {args.exp}  ({cfg['type']})")

    state: dict = {}

    if cfg["type"] in ("video", "pose"):
        print(f"Loading checkpoint: {cfg['ckpt']} ...")
        state["model"] = load_pytorch_model(cfg)

    elif cfg["type"] == "sklearn":
        print("Fitting sklearn SVM from pre-extracted features ...")
        state["sklearn"] = load_sklearn_pipeline()

    elif cfg["type"] == "sklearn_crop":
        print(f"Fitting sklearn LogReg on {cfg['feature_dir']} (train split) ...")
        state["sklearn_crop"] = load_sklearn_crop(cfg["feature_dir"])

    frames_dir = ROOT / "data" / "frames"
    out_dir    = ROOT / "videos" / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": frames_dir / "dataset_metadata_train.json",
        "val":   frames_dir / "dataset_metadata_val.json",
    }
    targets = list(split_map.items()) if args.split == "both" \
              else [(args.split, split_map[args.split])]

    for name, path in targets:
        if not path.exists():
            print(f"  {name}: metadata not found, skipping")
            continue
        process_split(args.exp, state, path, name, out_dir, args.fps)

    print(f"\nDone. Videos saved to: {out_dir}")


if __name__ == "__main__":
    main()