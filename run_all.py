"""
Master pipeline — runs everything from raw videos to debug output.

Usage:
    python run_all.py                        # full pipeline
    python run_all.py --skip-prep            # skip registry + frame extraction + poses
    python run_all.py --skip-features        # skip feature extraction
    python run_all.py --skip-training        # skip all PyTorch / sklearn training
    python run_all.py --skip-debug           # skip debug video generation
    python run_all.py --exp 5                # only experiment 5 (no prep/debug)
    python run_all.py --yolo-test            # YOLO zero-shot diagnostic only

Pipeline order:
  ── Preparation ──────────────────────────────────────────────────────────────
  P1  build_registry.py            → registry.json
  P2  training_preparation.py      → data/frames/ + dataset_metadata_*.json
  P3  pose_extraction/extract_poses.py → data/poses/   (needed for exp3)

  ── Feature extraction ───────────────────────────────────────────────────────
  F1  extract_video_features.py    → data/features/
  F2  extract_flow_features.py     → data/features_flow/
  F3  extract_local_features.py    → data/features_local/
  F4  extract_yolo_features.py     → data/features_yolo/
  F5  extract_bottom_frames.py     → data/frames_bottom/
  F6  extract_local_features.py --bottom → data/features_local_bottom/
  F7  extract_yolo_features.py  --bottom → data/features_yolo_bottom/

  ── Experiments ──────────────────────────────────────────────────────────────
  1   Baseline: ResNet18 (frozen) + LSTM
  2   ResNet18 + LSTM, layer4 unfrozen
  3   Pose LSTM (MediaPipe landmarks)   → checkpoints/best_pose_model_rotation.pth
  4   R3D-18 full fine-tuning (overfitting demo)
  5   R3D-18 frozen — linear probe
  6   Sklearn: combined features ablation + grid search
                                         → checkpoints/final_logreg_type.pkl

  ── Final model (built by Exp 3 + Exp 6) ─────────────────────────────────────
  FM  final_model.py --test            type: LogReg(Video+Pose)  rotation: Pose BiLSTM

  ── Debug videos ─────────────────────────────────────────────────────────────
  D   make_debug_videos.py         → videos/<exp>/train|val/*.mp4
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
AI_DIR   = ROOT / "ai_training"
POSE_DIR = ROOT / "pose_extraction"
CKPT_DIR = AI_DIR / "checkpoints"
REPORT   = ROOT / "reporting"

PYTHON = sys.executable


# ── Helpers ────────────────────────────────────────────────────────────────────

def run(script: Path, cwd: Path, label: str, extra_args: list[str] | None = None) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    cmd = [PYTHON, str(script)] + (extra_args or [])
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        print(f"[FAILED] {label} - exit code {result.returncode}")
        return False
    return True


def skip(reason: str) -> None:
    print(f"  [skip] {reason}")


# ── Preparation ────────────────────────────────────────────────────────────────

def run_preparation(force: bool = False) -> None:
    print(f"\n{'='*60}")
    print("  Preparation")
    print(f"{'='*60}")

    # P1 — build registry (fast, always re-run to stay up to date)
    run(ROOT / "build_registry.py", ROOT, "P1 -- build_registry.py -> registry.json")

    # P2 — extract frames + create splits
    meta_exists = all(
        (AI_DIR.parent / "data" / "frames" / f"dataset_metadata_{s}.json").exists()
        for s in ("train", "val", "test")
    )
    if meta_exists and not force:
        skip("P2 -- training_preparation.py (data/frames/dataset_metadata_*.json already exist)")
    else:
        run(AI_DIR / "training_preparation.py", AI_DIR,
            "P2 -- training_preparation.py -> data/frames/")

    # P3 — pose extraction (mediapipe) — optional, only needed for exp3
    poses_dir = ROOT / "data" / "poses"
    pose_files = list(poses_dir.glob("*.npy")) if poses_dir.exists() else []
    if pose_files and not force:
        skip(f"P3 -- extract_poses.py ({len(pose_files)} pose files already in data/poses/)")
    else:
        pose_script = POSE_DIR / "extract_poses.py"
        if pose_script.exists():
            run(pose_script, ROOT, "P3 -- extract_poses.py -> data/poses/")
        else:
            skip("P3 -- extract_poses.py not found (exp3 will skip pose-missing videos)")


# ── Feature extraction ─────────────────────────────────────────────────────────

def run_feature_extraction() -> None:
    print(f"\n{'='*60}")
    print("  Feature Extraction")
    print(f"{'='*60}")

    for script_name, extra_args, label in [
        ("extract_video_features.py",  [],           "F1 -- R3D-18 video features      -> data/features/"),
        ("extract_flow_features.py",   [],           "F2 -- Optical flow features      -> data/features_flow/"),
        ("extract_local_features.py",  [],           "F3 -- Local crop (center)        -> data/features_local/"),
        ("extract_yolo_features.py",   [],           "F4 -- YOLO crop (center)         -> data/features_yolo/"),
        ("extract_bottom_frames.py",   [],           "F5 -- Bottom frames              -> data/frames_bottom/"),
        ("extract_local_features.py",  ["--bottom"], "F6 -- Local crop (bottom)        -> data/features_local_bottom/"),
        ("extract_yolo_features.py",   ["--bottom"], "F7 -- YOLO crop (bottom)         -> data/features_yolo_bottom/"),
    ]:
        script = AI_DIR / script_name
        print(f"\n  Running: {script_name} {' '.join(extra_args)}")
        result = subprocess.run([PYTHON, str(script)] + extra_args, cwd=str(AI_DIR))
        if result.returncode != 0:
            print(f"  [WARN] {script_name} failed - sklearn exp may be incomplete")


# ── Experiments ────────────────────────────────────────────────────────────────

EXP_CFG = {
    1: dict(script="train.py",            ckpt="best_model.pth",
            label="Exp 1 — Baseline: ResNet18 (frozen) + LSTM"),
    2: dict(script="train_a.py",          ckpt="best_model_a.pth",
            label="Exp 2 — ResNet18 + LSTM (layer4 unfrozen)"),
    3: dict(script="train_pose.py",       ckpt="best_pose_model.pth",
            label="Exp 3 — Pose LSTM (MediaPipe landmarks)"),
    4: dict(script="train_b_finetune.py", ckpt="best_model_b_finetune.pth",
            label="Exp 4 — R3D-18 full fine-tuning"),
    5: dict(script="train_b.py",          ckpt="best_model_b.pth",
            label="Exp 5 — R3D-18 frozen (linear probe)"),
    6: dict(script="train_sklearn.py",    ckpt=None,
            label="Exp 6 — Sklearn combined features"),
}


def run_exp(n: int, skip_training: bool) -> None:
    cfg   = EXP_CFG[n]
    ckpt  = CKPT_DIR / cfg["ckpt"] if cfg["ckpt"] else None
    label = cfg["label"]

    if skip_training:
        skip(f"{label} (--skip-training)")
        return
    if ckpt and ckpt.exists():
        skip(f"{label} — checkpoint exists ({cfg['ckpt']})")
        return

    run(AI_DIR / cfg["script"], AI_DIR, label)


# ── Debug videos ───────────────────────────────────────────────────────────────

DEBUG_EXPS = {
    "exp1": "best_model.pth",
    "exp2": "best_model_a.pth",
    "exp3": "best_pose_model.pth",
    "exp4": "best_model_b_finetune.pth",
    "exp5": "best_model_b.pth",
    "exp6": None,   # sklearn — no checkpoint, check features
}

# Exp 7: 4 crop variants (sklearn LogReg). Key = exp name, value = required feature dir.
EXP7_CROP_DIRS = {
    "exp7_heur_ctr": "features_local",
    "exp7_yolo_ctr": "features_yolo",
    "exp7_heur_bot": "features_local_bottom",
    "exp7_yolo_bot": "features_yolo_bottom",
}


def run_debug_videos() -> None:
    print(f"\n{'='*60}")
    print("  Debug Videos")
    print(f"{'='*60}")

    script = AI_DIR / "make_debug_videos.py"
    if not script.exists():
        skip("make_debug_videos.py not found")
        return

    features_ok = (ROOT / "data" / "features").exists()

    # Exp 1–6
    for exp, ckpt_name in DEBUG_EXPS.items():
        if ckpt_name is not None:
            if not (CKPT_DIR / ckpt_name).exists():
                skip(f"{exp} debug videos — checkpoint {ckpt_name} missing")
                continue
        else:
            if not features_ok:
                skip(f"{exp} debug videos — data/features/ missing")
                continue
        run(script, AI_DIR, f"Debug videos — {exp}",
            extra_args=["--exp", exp, "--split", "both"])

    # Exp 7 — crop variant ablation (each needs its own feature dir)
    for exp, feat_dir_name in EXP7_CROP_DIRS.items():
        feat_dir = ROOT / "data" / feat_dir_name
        if not feat_dir.exists():
            skip(f"{exp} debug videos — data/{feat_dir_name}/ missing")
            continue
        run(script, AI_DIR, f"Debug videos — {exp}",
            extra_args=["--exp", exp, "--split", "both"])


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full figure skating pipeline")
    parser.add_argument("--skip-prep",      action="store_true",
                        help="Skip preparation (registry, frame extraction, poses)")
    parser.add_argument("--skip-features",  action="store_true",
                        help="Skip feature extraction")
    parser.add_argument("--skip-training",  action="store_true",
                        help="Skip all training steps")
    parser.add_argument("--skip-debug",     action="store_true",
                        help="Skip debug video generation")
    parser.add_argument("--force-prep",     action="store_true",
                        help="Re-run preparation even if outputs exist")
    parser.add_argument("--exp", type=int, default=None,
                        help="Run only this experiment number (1–6), skips prep and debug")
    parser.add_argument("--yolo-test",      action="store_true",
                        help="Run YOLO zero-shot diagnostic (requires: pip install ultralytics)")
    args = parser.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    single_exp = args.exp is not None

    # ── YOLO diagnostic ────────────────────────────────────────────────────────
    if args.yolo_test:
        print(f"\n{'='*60}")
        print("  YOLO Zero-Shot Diagnostic")
        print(f"{'='*60}")
        run(AI_DIR / "yolo_zeroshot_test.py", AI_DIR, "YOLO zero-shot test")
        if single_exp is False:
            return

    # ── Preparation ────────────────────────────────────────────────────────────
    if not args.skip_prep and not single_exp:
        run_preparation(force=args.force_prep)

    # ── Feature extraction ─────────────────────────────────────────────────────
    if not args.skip_features and not single_exp:
        run_feature_extraction()

    # ── Experiments ────────────────────────────────────────────────────────────
    experiments = [args.exp] if single_exp else list(range(1, 7))
    for n in experiments:
        run_exp(n, skip_training=args.skip_training)

    # ── Sample frames report ───────────────────────────────────────────────────
    if not single_exp:
        frames_dir = ROOT / "reports" / "figures" / "sample_frames"
        if list(frames_dir.glob("*.png")):
            skip("sample frames already exist")
        else:
            print(f"\n{'='*60}")
            print("  Reporting - Sample Frames")
            print(f"{'='*60}")
            run(REPORT / "save_sample_frames.py", REPORT, "Save 3 frames per class")

    # ── Final model sanity check ───────────────────────────────────────────────
    if not single_exp and not args.skip_training:
        final_model_script = AI_DIR / "final_model.py"
        logreg_ckpt  = CKPT_DIR / "final_logreg_type.pkl"
        bilstm_ckpt  = CKPT_DIR / "best_pose_model_rotation.pth"
        bilstm_fallback = CKPT_DIR / "best_pose_model.pth"
        if logreg_ckpt.exists() and (bilstm_ckpt.exists() or bilstm_fallback.exists()):
            run(final_model_script, AI_DIR, "FM -- Final model quick eval on val set")
        else:
            missing = []
            if not logreg_ckpt.exists():  missing.append("final_logreg_type.pkl (run Exp 6)")
            if not bilstm_ckpt.exists() and not bilstm_fallback.exists():
                missing.append("best_pose_model*.pth (run Exp 3)")
            skip(f"Final model — missing: {', '.join(missing)}")

    # ── Debug videos ───────────────────────────────────────────────────────────
    if not args.skip_debug and not single_exp:
        run_debug_videos()

    print(f"\n{'='*60}")
    print("  All done.  Reports -> reports/figures/")
    print(f"  Debug  ->  videos/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()