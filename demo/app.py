"""
Gradio demo — Figure Skating Jump Recognition.

Layout:
  Row 1 : Upload + crop mode + Run (left) | Prediction text + confidence bars (right)
  Row 2 : Key frames panel (4 phases, skeleton overlay)
  Row 3 : Joint velocity — interactive Plotly chart (what BiLSTM reads)
  Accordion : Exp 1–6 type comparison

Usage:
    pip install gradio ultralytics plotly
    pip install mediapipe
    python demo/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai_training"))

# ── Constants ─────────────────────────────────────────────────────────────────

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FRAMES   = 16
CROP_SIZE  = 224
SHORT_SIDE = 256

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TYPE_NAMES  = ["Salchow", "Axel", "Toe Loop"]
ROT_NAMES   = ["Single",  "Double", "Triple"]
_TYPE_KEYS  = ["salchow", "axel", "toe_loop"]
_ROT_KEYS   = ["single",  "double", "triple"]
TYPE_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]
ROT_COLORS  = ["#9b59b6", "#f39c12", "#1abc9c"]

PHASE_COLOR = {
    "pre-takeoff": "#888888",
    "takeoff":     "#2ecc71",
    "flight":      "#f39c12",
    "landing":     "#3498db",
}
# Semi-transparent fill for Plotly phase shading
PHASE_FILL = {
    "pre-takeoff": "rgba(136,136,136,0.07)",
    "takeoff":     "rgba(46,204,113,0.12)",
    "flight":      "rgba(243,156,18,0.12)",
    "landing":     "rgba(52,152,219,0.10)",
}

# MediaPipe pose connections (torso + limbs, no face clutter)
_SKEL_CONN = [
    (11,12),(11,13),(12,14),(13,15),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(24,26),(25,27),(26,28),
    (27,29),(27,31),(28,30),(28,32),
]

# Joint groups for velocity plot
_VEL_JOINTS = {
    "Wrists (L+R)":   [15, 16],
    "Shoulders (L+R)": [11, 12],
    "Knees (L+R)":    [25, 26],
}
_VEL_COLORS = ["#e74c3c", "#3498db", "#2ecc71"]

CKPT = ROOT / "ai_training" / "checkpoints"


# ── Video loading ─────────────────────────────────────────────────────────────

def _resize_short(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    s = SHORT_SIDE / min(h, w)
    return cv2.resize(frame, (int(w * s), int(h * s)))


def _center_crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x = w // 2 - CROP_SIZE // 2
    y = h // 2 - CROP_SIZE // 2
    return frame[y:y + CROP_SIZE, x:x + CROP_SIZE]


def _bottom_crop(frame: np.ndarray) -> np.ndarray:
    """Center 224×224 at 75% height — captures blade/ice contact for tall portrait video."""
    h, w = frame.shape[:2]
    x     = w // 2 - CROP_SIZE // 2
    cy    = int(h * 0.75)
    y     = min(max(cy - CROP_SIZE // 2, 0), h - CROP_SIZE)
    return frame[y:y + CROP_SIZE, x:x + CROP_SIZE]


def _to_tensor(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def load_video(path: str, use_bottom: bool = False) -> tuple[list[np.ndarray], list[np.ndarray]]:
    crop_fn = _bottom_crop if use_bottom else _center_crop
    cap = cv2.VideoCapture(path)
    all_raw: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_raw.append(crop_fn(_resize_short(frame)))
    cap.release()
    if not all_raw:
        return [], []
    idxs = list(np.linspace(0, len(all_raw) - 1, N_FRAMES, dtype=int))
    raw  = [all_raw[i] for i in idxs]
    return raw, [_to_tensor(f) for f in raw]


# ── Optical flow ──────────────────────────────────────────────────────────────

def compute_flow(raw_frames: list[np.ndarray]) -> tuple[np.ndarray, int]:
    mean_vs: list[float] = []
    feats:   list[float] = []
    for f1, f2 in list(zip(raw_frames, raw_frames[1:]))[:15]:
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        fl = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        u, v = fl[..., 0], fl[..., 1]
        feats.extend([float(u.mean()), float(v.mean()), float(np.sqrt(u**2+v**2).mean())])
        mean_vs.append(float(v.mean()))
    while len(feats) < 45:
        feats.append(0.0)
    pair_idx = int(np.argmin(mean_vs)) if mean_vs else 0
    return np.array(feats[:45], dtype=np.float32), min(pair_idx + 1, N_FRAMES - 1)


def assign_phases(n: int, takeoff: int) -> list[str]:
    flight_end = takeoff + max(2, n // 4)
    return [
        "pre-takeoff" if i < takeoff
        else "takeoff" if i <= takeoff + 1
        else "flight"  if i <= flight_end
        else "landing"
        for i in range(n)
    ]


def pick_phase_frames(phases: list[str]) -> dict[str, int]:
    """One representative frame per phase — middle of flight, first of others."""
    buckets: dict[str, list[int]] = {}
    for i, ph in enumerate(phases):
        buckets.setdefault(ph, []).append(i)
    result = {}
    for ph in ["pre-takeoff", "takeoff", "flight", "landing"]:
        idxs = buckets.get(ph, [])
        if idxs:
            result[ph] = idxs[len(idxs) // 2] if ph == "flight" else idxs[0]
    return result


# ── Pose extraction ───────────────────────────────────────────────────────────

_MEDIAPIPE_OK: bool | None = None


def _check_mediapipe() -> bool:
    global _MEDIAPIPE_OK
    if _MEDIAPIPE_OK is None:
        try:
            from mediapipe.tasks import python as _
            _MEDIAPIPE_OK = True
        except ImportError:
            _MEDIAPIPE_OK = False
    return _MEDIAPIPE_OK


def _ensure_pose_model() -> str | None:
    p = ROOT / "pose_extraction" / "pose_landmarker_full.task"
    if p.exists():
        return str(p)
    import urllib.request
    url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
           "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
    try:
        urllib.request.urlretrieve(url, p)
        return str(p)
    except Exception as e:
        print(f"Pose model download failed: {e}")
        return None


def _draw_skeleton(frame_bgr: np.ndarray, landmarks) -> np.ndarray:
    h, w  = frame_bgr.shape[:2]
    out   = frame_bgr.copy()
    for i, j in _SKEL_CONN:
        li, lj = landmarks[i], landmarks[j]
        if li.visibility > 0.25 and lj.visibility > 0.25:
            cv2.line(out,
                     (int(li.x * w), int(li.y * h)),
                     (int(lj.x * w), int(lj.y * h)),
                     (0, 230, 230), 1)
    for lm in landmarks:
        if lm.visibility > 0.25:
            cv2.circle(out, (int(lm.x * w), int(lm.y * h)), 2, (255, 230, 0), -1)
    return out


def extract_poses_full(
    raw_frames: list[np.ndarray],
) -> tuple[np.ndarray | None, list[np.ndarray], int]:
    """
    Returns (normalised_pose_seq, skeleton_frames, n_detected).
    normalised_pose_seq: (T, 99) hip-centred shoulder-width-normalised — BiLSTM input.
    skeleton_frames: raw_frames with pose skeleton drawn on each detected frame.
    n_detected: number of frames where pose was successfully detected.
    """
    skeleton_frames = [f.copy() for f in raw_frames]

    if not _check_mediapipe():
        return None, skeleton_frames, 0
    model_path = _ensure_pose_model()
    if model_path is None:
        return None, skeleton_frames, 0

    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    pose_seq:   list[np.ndarray] = []
    last_valid  = np.zeros(99, dtype=np.float32)
    n_detected  = 0

    with mp_vision.PoseLandmarker.create_from_options(opts) as det:
        for t, frame_bgr in enumerate(raw_frames):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.pose_landmarks:
                landmarks = res.pose_landmarks[0]
                skeleton_frames[t] = _draw_skeleton(frame_bgr, landmarks)
                kpts = []
                for p in landmarks:
                    kpts.extend([p.x, p.y, p.visibility])
                last_valid = np.array(kpts, dtype=np.float32)
                n_detected += 1
            pose_seq.append(last_valid.copy())

    poses = np.stack(pose_seq)
    LH, RH, LS, RS = 23, 24, 11, 12
    out = poses.copy()
    for t in range(len(poses)):
        frm = poses[t].reshape(33, 3)
        xy  = frm[:, :2]
        hc  = (xy[LH] + xy[RH]) / 2
        sw  = np.linalg.norm(xy[LS] - xy[RS])
        nf        = frm.copy()
        nf[:, :2] = (xy - hc) / (sw if sw > 1e-6 else 1.0)
        out[t]    = nf.reshape(-1)

    return out.astype(np.float32), skeleton_frames, n_detected


# ── Model loading ─────────────────────────────────────────────────────────────

_STATE: dict = {}


def _ensure_loaded():
    if _STATE:
        return
    print(f"Loading models on {DEVICE} ...")

    from model      import FigureSkatingModel
    from model_a    import FigureSkatingModelA
    from model_b    import FigureSkatingModelB
    from pose_model import PoseModel

    def _load(cls, ckpt_name, **kw):
        ck = CKPT / ckpt_name
        if not ck.exists():
            print(f"  [skip] {ckpt_name}")
            return None
        m = cls(**kw).to(DEVICE)
        m.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True))
        m.eval()
        return m

    _STATE["models"] = {
        "exp1": _load(FigureSkatingModel,  "best_model.pth"),
        "exp2": _load(FigureSkatingModelA, "best_model_a.pth"),
        "exp3": _load(PoseModel,           "best_pose_model_rotation.pth"),
        "exp4": _load(FigureSkatingModelB, "best_model_b_finetune.pth", freeze_backbone=False),
        "exp5": _load(FigureSkatingModelB, "best_model_b.pth",          freeze_backbone=True),
    }
    print("  PyTorch models loaded.")

    try:
        from final_model import FinalModel
        lok, bok = FinalModel.checkpoints_ready()
        _STATE["final_model"] = FinalModel() if (lok and bok) else None
        print("  FinalModel loaded." if _STATE["final_model"] else "  FinalModel: checkpoints missing.")
    except Exception as e:
        _STATE["final_model"] = None
        print(f"  FinalModel failed: {e}")

    try:
        from ultralytics import YOLO
        yp = ROOT / "ai_training" / "yolo11n.pt"
        _STATE["yolo"] = YOLO(str(yp if yp.exists() else "yolo11n.pt"))
        print("  YOLO loaded.")
    except ImportError:
        _STATE["yolo"] = None


# ── Inference ─────────────────────────────────────────────────────────────────

def _batch(tensor_frames: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(tensor_frames)).unsqueeze(0).float().to(DEVICE)


@torch.no_grad()
def _softmax3(model, tensor_frames) -> np.ndarray:
    logits, _ = model(_batch(tensor_frames))
    return F.softmax(logits, dim=-1).cpu().numpy()[0, :3]


@torch.no_grad()
def _infer_pose_type(model, pose_seq: np.ndarray | None) -> np.ndarray | None:
    if model is None or pose_seq is None:
        return None
    x = torch.from_numpy(pose_seq).unsqueeze(0).float().to(DEVICE)
    logits, _ = model(x)
    return F.softmax(logits, dim=-1).cpu().numpy()[0]


@torch.no_grad()
def _extract_r3d(models: dict, tensor_frames) -> np.ndarray | None:
    m = models.get("exp5")
    if m is None:
        return None
    x = _batch(tensor_frames).permute(0, 2, 1, 3, 4)
    return m.backbone(x).flatten().cpu().numpy()


def _infer_final(final_model, models, tensor_frames, pose_seq) -> dict | None:
    if final_model is None or pose_seq is None:
        return None
    vf = _extract_r3d(models, tensor_frames)
    if vf is None:
        return None
    try:
        return final_model.predict(vf, pose_seq)
    except Exception as e:
        print(f"FinalModel error: {e}")
        return None


# ── Output builders ───────────────────────────────────────────────────────────

def make_result_md(fm: dict | None, n_detected: int) -> str:
    """Prediction headline + diagnostic warning + pose quality indicator."""
    pose_line = (
        f"Pose detected in **{n_detected}/{N_FRAMES}** frames"
        if n_detected > 0
        else "*Pose extraction unavailable (install mediapipe)*"
    )
    if n_detected > 0 and n_detected < N_FRAMES * 0.5:
        pose_line += " ⚠️ low — skater may be partially out of frame"

    if fm is None:
        return (
            "### Final Model unavailable\n"
            "> Run `python run_all.py --exp 3` and `--exp 6` first.\n\n"
            f"{pose_line}"
        )

    rot  = fm["rotation"].capitalize()
    jump = fm["type"].capitalize().replace("_", " ")
    tc   = fm["type_proba"][fm["type"]]
    rc   = fm["rotation_proba"][fm["rotation"]]

    result = (
        f"### Predicted: **{rot} {jump}**\n"
        f"Type confidence: **{tc:.0%}** &nbsp;|&nbsp; "
        f"Rotation confidence: **{rc:.0%}**\n\n"
        f"{pose_line}"
    )

    # Diagnostic warning: Salchow ↔ Toe Loop confusion
    sp = fm["type_proba"]["salchow"]
    tp = fm["type_proba"]["toe_loop"]
    if sp > 0.22 and tp > 0.22:
        result += (
            "\n\n> ⚠️ **Diagnostic:** Salchow and Toe Loop both score high "
            f"({sp:.0%} vs {tp:.0%}). "
            "These jumps differ only at the blade contact point — "
            "a sub-10-pixel detail at 224×224 that global features cannot reliably separate. "
            "This uncertainty is expected; see Exp 6 analysis."
        )

    return result


def make_confidence_figure(fm: dict | None) -> plt.Figure:
    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(8, 2.8))
    fig.suptitle("Final Model — confidence", fontsize=11)

    for ax, names, keys, colors, label, fm_key in [
        (ax_t, TYPE_NAMES, _TYPE_KEYS, TYPE_COLORS, "Jump Type",  "type_proba"),
        (ax_r, ROT_NAMES,  _ROT_KEYS,  ROT_COLORS,  "Rotation",   "rotation_proba"),
    ]:
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 1)
        ax.axhline(1 / 3, color="#ccc", linewidth=0.8, linestyle="--")

        if fm is None:
            ax.set_facecolor("#f5f5f5")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    fontsize=12, transform=ax.transAxes, color="#aaa")
            ax.set_xticks([])
            continue

        proba  = [fm[fm_key][k] for k in keys]
        pred   = int(np.argmax(proba))
        barcl  = [colors[i] if i == pred else "#ddd" for i in range(len(names))]
        bars   = ax.bar(names, proba, color=barcl, width=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(axis="y", labelsize=8)

        for bar, p in zip(bars, proba):
            if p > 0.06:
                ax.text(bar.get_x() + bar.get_width() / 2, p - 0.03,
                        f"{p:.0%}", ha="center", va="top",
                        fontsize=9, color="white", fontweight="bold")
        ax.set_xlabel(
            f"-> {names[pred]}  {proba[pred]:.0%}",
            fontsize=9, fontweight="bold", color=colors[pred],
        )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def make_frames_figure(
    skeleton_frames: list[np.ndarray],
    phases: list[str],
    takeoff_idx: int,
) -> plt.Figure:
    """4 key frames (one per phase) with skeleton overlay."""
    phase_frames = pick_phase_frames(phases)
    ordered = [ph for ph in ["pre-takeoff", "takeoff", "flight", "landing"] if ph in phase_frames]
    n = len(ordered)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.text(0.5, 0.5, "No frames", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]

    phase_labels = {
        "pre-takeoff": "Pre-takeoff",
        "takeoff":     "Takeoff ★",
        "flight":      "Mid-flight",
        "landing":     "Landing",
    }

    for ax, ph in zip(axes, ordered):
        idx = phase_frames[ph]
        rgb = cv2.cvtColor(skeleton_frames[idx], cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.axis("off")
        color = PHASE_COLOR[ph]
        ax.set_title(f"{phase_labels[ph]}\nframe {idx}", fontsize=9,
                     color=color, fontweight="bold", pad=4)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    fig.suptitle(
        "Key frames — MediaPipe skeleton (yellow dots = joints, cyan lines = bones)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def make_velocity_plotly(
    pose_seq: np.ndarray | None,
    takeoff_idx: int,
    phases: list[str],
):
    """Interactive Plotly velocity chart — hover shows frame index + phase."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        # Fallback: return a plain matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "pip install plotly for interactive chart",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="#888")
        return fig

    fig = go.Figure()

    if pose_seq is None or len(pose_seq) < 2:
        fig.add_annotation(
            text="Pose extraction unavailable — install mediapipe",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#aaa"),
        )
        fig.update_layout(height=300)
        return fig

    velocity = np.abs(np.diff(pose_seq, axis=0))  # (T-1, 99)
    t_axis   = list(range(1, len(pose_seq)))

    # Phase background shading
    boundaries: list[tuple[int, int, str]] = []
    start, cur_ph = 0, phases[0]
    for i in range(1, len(phases)):
        if phases[i] != cur_ph:
            boundaries.append((start, i - 1, cur_ph))
            start, cur_ph = i, phases[i]
    boundaries.append((start, len(phases) - 1, cur_ph))

    for s, e, ph in boundaries:
        fig.add_vrect(
            x0=s - 0.5, x1=e + 0.5,
            fillcolor=PHASE_FILL[ph],
            line_width=0,
        )
        mid = (s + e) / 2
        if e - s >= 1:
            fig.add_annotation(
                x=mid, y=1.0, yref="paper",
                text=ph, showarrow=False,
                font=dict(size=8, color=PHASE_COLOR[ph]),
                yanchor="top",
            )

    # Joint velocity traces
    for (group, joint_ids), color in zip(_VEL_JOINTS.items(), _VEL_COLORS):
        xy_cols = [j * 3 for j in joint_ids] + [j * 3 + 1 for j in joint_ids]
        vel     = velocity[:, xy_cols].mean(axis=1).tolist()
        custom  = [[phases[t], round(vel[t - 1], 4)] for t in range(1, len(pose_seq))]

        fig.add_trace(go.Scatter(
            x=t_axis,
            y=vel,
            name=group,
            line=dict(color=color, width=2),
            customdata=custom,
            hovertemplate=(
                f"<b>{group}</b><br>"
                "Frame: <b>%{x}</b><br>"
                "Phase: %{customdata[0]}<br>"
                "Velocity: %{customdata[1]:.4f}"
                "<extra></extra>"
            ),
        ))

    # Takeoff vertical line
    fig.add_vline(
        x=takeoff_idx,
        line_dash="dash", line_color=PHASE_COLOR["takeoff"], line_width=2,
        annotation_text=f"<b>Takeoff</b> (frame {takeoff_idx})",
        annotation_position="top right",
        annotation_font=dict(size=10, color=PHASE_COLOR["takeoff"]),
    )

    fig.update_layout(
        title=dict(
            text="Joint velocity — what Pose BiLSTM reads for rotation counting"
                 "  <i>(hover to inspect frame index and phase)</i>",
            font=dict(size=11),
        ),
        xaxis=dict(
            title="Frame index",
            tickmode="linear", dtick=1,
            range=[-0.5, len(pose_seq) - 0.5],
        ),
        yaxis=dict(title="Mean joint velocity<br>(normalised coords)", title_font_size=10),
        legend=dict(
            orientation="h",
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.75)", borderwidth=0,
        ),
        hovermode="x unified",
        height=320,
        margin=dict(t=55, b=45, l=65, r=15),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.07)")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.07)")
    return fig


EXP_META = [
    ("exp1", "Exp 1\nResNet18+LSTM\nfrozen"),
    ("exp2", "Exp 2\nResNet18+LSTM\nlayer4 FT"),
    ("exp3", "Exp 3\nPose BiLSTM\ntype head"),
    ("exp4", "Exp 4\nR3D-18\nfull FT"),
    ("exp5", "Exp 5\nR3D-18\nfrozen probe"),
    ("exp6", "Exp 6\nLogReg\nVideo+Pose"),
]


def make_exp_figure(results: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    fig.suptitle("Experiment Comparison — Jump Type", fontsize=11)
    for ax, (key, label) in zip(axes.flatten(), EXP_META):
        probs = results.get(key)
        ax.set_title(label, fontsize=8.5, pad=4)
        ax.set_ylim(0, 1)
        ax.axhline(1 / 3, color="#ccc", linewidth=0.7, linestyle="--")
        if probs is None:
            ax.set_facecolor("#f5f5f5")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    fontsize=11, transform=ax.transAxes, color="#aaa")
            ax.set_xticks([])
            continue
        p3    = np.array(probs[:3], dtype=float)
        pred  = int(np.argmax(p3))
        barcl = [TYPE_COLORS[i] if i == pred else "#ddd" for i in range(3)]
        bars  = ax.bar(TYPE_NAMES, p3, color=barcl, width=0.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(TYPE_NAMES, fontsize=8, rotation=20, ha="right")
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(axis="y", labelsize=7)
        for bar, p in zip(bars, p3):
            if p > 0.07:
                ax.text(bar.get_x() + bar.get_width() / 2, p - 0.03,
                        f"{p:.0%}", ha="center", va="top",
                        fontsize=7.5, color="white", fontweight="bold")
        ax.set_xlabel(
            f"-> {TYPE_NAMES[pred]}  {p3[pred]:.0%}",
            fontsize=8, fontweight="bold", color=TYPE_COLORS[pred],
        )
    for ax in axes[:, 0]:
        ax.set_ylabel("Confidence", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ── Main pipeline ─────────────────────────────────────────────────────────────

_EMPTY = ("*Upload a video and click Run.*", None, None, None, None)


def run(video_path: str | None, crop_mode: str = "Center crop"):
    if video_path is None:
        return _EMPTY

    _ensure_loaded()
    models      = _STATE["models"]
    final_model = _STATE.get("final_model")

    use_bottom                   = "Bottom" in crop_mode
    raw_frames, tensor_frames    = load_video(video_path, use_bottom=use_bottom)
    if not raw_frames:
        return ("*Could not read video.*", None, None, None, None)

    _, takeoff_idx = compute_flow(raw_frames)
    phases         = assign_phases(len(raw_frames), takeoff_idx)

    # Pose extraction (once) — shared by Exp 3, Final Model, and visualizations
    pose_seq, skeleton_frames, n_detected = extract_poses_full(raw_frames)

    # Experiment type predictions
    results: dict[str, np.ndarray | None] = {}

    m1 = models.get("exp1")
    results["exp1"] = _softmax3(m1, tensor_frames) if m1 else None

    m2 = models.get("exp2")
    if m2 is not None:
        p2 = _softmax3(m2, tensor_frames)
        s  = p2[:3].sum()
        results["exp2"] = p2[:3] / s if s > 0 else p2[:3]
    else:
        results["exp2"] = None

    results["exp3"] = _infer_pose_type(models.get("exp3"), pose_seq)

    m4 = models.get("exp4")
    results["exp4"] = _softmax3(m4, tensor_frames) if m4 else None

    m5 = models.get("exp5")
    results["exp5"] = _softmax3(m5, tensor_frames) if m5 else None

    fm_result = _infer_final(final_model, models, tensor_frames, pose_seq)

    results["exp6"] = (
        np.array([fm_result["type_proba"][k] for k in _TYPE_KEYS])
        if fm_result is not None else None
    )

    return (
        make_result_md(fm_result, n_detected),
        make_confidence_figure(fm_result),
        make_frames_figure(skeleton_frames, phases, takeoff_idx),
        make_velocity_plotly(pose_seq, takeoff_idx, phases),
        make_exp_figure(results),
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_app():
    import gradio as gr

    with gr.Blocks(title="Figure Skating Jump Recognition", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## Figure Skating Jump Recognition\n"
            "Upload a short jump clip (~3–8 sec) and click **Run**."
        )

        with gr.Tabs():

            # ── Tab 1: Prediction ──────────────────────────────────────────
            with gr.TabItem("Prediction"):
                with gr.Row(equal_height=True):
                    # Left column: upload + controls (narrower)
                    with gr.Column(scale=2, min_width=220):
                        vid_in   = gr.Video(label="Upload video clip", height=220)
                        crop_sel = gr.Radio(
                            choices=[
                                "Center crop",
                                "Bottom-biased crop (captures blade/ice better)",
                            ],
                            value="Center crop",
                            label="Crop mode",
                        )
                        run_btn = gr.Button("Run", variant="primary", size="lg")

                    # Right column: result headline (wider)
                    with gr.Column(scale=3):
                        result_md = gr.Markdown("*Upload a video and click Run.*")

                # Confidence bars below, full width
                conf_plot = gr.Plot(label="Confidence — type & rotation")

            # ── Tab 2: What the model sees ─────────────────────────────────
            with gr.TabItem("What the model sees"):
                gr.Markdown(
                    "**Key frames** with MediaPipe pose skeleton (cyan bones, yellow joints).  \n"
                    "**Velocity chart** shows the signal the Pose BiLSTM reads to count rotations — "
                    "hover over any point to inspect frame index, phase, and velocity."
                )
                frames_plot = gr.Plot()
                vel_plot    = gr.Plot()

            # ── Tab 3: Experiment comparison ───────────────────────────────
            with gr.TabItem("Experiment comparison"):
                gr.Markdown(
                    "Jump type confidence for all 6 models on this clip.  \n"
                    "**Exp 6 = LogReg (Video+Pose)** — same model as Final Model type head.  \n"
                    "*Rotation prediction is only available in Exp 3 (Pose BiLSTM) and the Final Model.*"
                )
                exp_plot = gr.Plot()

        # Wire button (inside Tab 1) to all outputs across all tabs
        run_btn.click(
            fn=run,
            inputs=[vid_in, crop_sel],
            outputs=[result_md, conf_plot, frames_plot, vel_plot, exp_plot],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()