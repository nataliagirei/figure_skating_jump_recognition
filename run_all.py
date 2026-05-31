"""
Master pipeline — runs all 6 experiments in order.

Usage:
    python run_all.py                    # run everything
    python run_all.py --skip-training    # only run feature extraction + reporting
    python run_all.py --exp 5            # run only experiment 5

Skip logic: if a checkpoint already exists the training step is skipped.
Feature extraction: skipped per-video (handled inside each extractor).

Experiment order:
  1  Baseline: ResNet18 (frozen) + LSTM
  2  ResNet18 + LSTM, layer4 unfrozen
  3  Pose LSTM (MediaPipe landmarks)
  4  R3D-18 full fine-tuning (overfitting demo)
  5  R3D-18 frozen — linear probe  ← best result
  6  Sklearn: combined features ablation + grid search

Reporting:
  - Training curves saved during each run (reports/figures/{exp}/)
  - Confusion matrices saved at end of each training script
  - Sklearn bar charts saved by train_sklearn.py
  - Sample frames: reports/figures/sample_frames/
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT        = Path(__file__).resolve().parent
AI_DIR      = ROOT / "ai_training"
CKPT_DIR    = AI_DIR / "checkpoints"
REPORTING   = ROOT / "reporting"

PYTHON = sys.executable


def run(script: Path, cwd: Path, label: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run([PYTHON, str(script)], cwd=str(cwd))
    if result.returncode != 0:
        print(f"[FAILED] {label} exited with code {result.returncode}")
        return False
    return True


def skip(reason: str) -> None:
    print(f"  [skip] {reason}")


def extract_features() -> None:
    """Run all three feature extractors (each skips already-processed videos)."""
    print(f"\n{'='*60}")
    print("  Feature Extraction")
    print(f"{'='*60}")

    for script_name, label in [
        ("extract_video_features.py",  "R3D-18 video features  → data/features/"),
        ("extract_flow_features.py",   "Optical flow features  → data/features_flow/"),
        ("extract_local_features.py",  "Local crop features    → data/features_local/"),
    ]:
        script = AI_DIR / script_name
        print(f"\n  Running: {script_name}")
        result = subprocess.run([PYTHON, str(script)], cwd=str(AI_DIR))
        if result.returncode != 0:
            print(f"  [WARN] {script_name} failed — sklearn exp may be incomplete")


def run_exp(n: int, skip_training: bool) -> None:
    exp_cfg = {
        1: dict(
            script="train.py",
            ckpt="best_model.pth",
            label="Exp 1 — Baseline: ResNet18 (frozen) + LSTM",
        ),
        2: dict(
            script="train_a.py",
            ckpt="best_model_a.pth",
            label="Exp 2 — ResNet18 + LSTM (layer4 unfrozen)",
        ),
        3: dict(
            script="train_pose.py",
            ckpt="best_pose_model.pth",
            label="Exp 3 — Pose LSTM (MediaPipe landmarks)",
        ),
        4: dict(
            script="train_b_finetune.py",
            ckpt="best_model_b_finetune.pth",
            label="Exp 4 — R3D-18 full fine-tuning",
        ),
        5: dict(
            script="train_b.py",
            ckpt="best_model_b.pth",
            label="Exp 5 — R3D-18 frozen (linear probe)",
        ),
        6: dict(
            script="train_sklearn.py",
            ckpt=None,          # sklearn has no checkpoint; always run
            label="Exp 6 — Sklearn combined features (ablation + grid search)",
        ),
    }

    if n not in exp_cfg:
        print(f"[ERROR] Unknown experiment number: {n}")
        return

    cfg    = exp_cfg[n]
    ckpt   = CKPT_DIR / cfg["ckpt"] if cfg["ckpt"] else None
    label  = cfg["label"]
    script = AI_DIR / cfg["script"]

    if skip_training:
        skip(f"{label} (--skip-training)")
        return

    if ckpt and ckpt.exists():
        skip(f"{label} — checkpoint exists at {ckpt.name}")
        return

    run(script, AI_DIR, label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all figure skating experiments")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip all PyTorch training steps")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature extraction")
    parser.add_argument("--exp", type=int, default=None,
                        help="Run only this experiment number (1-6)")
    args = parser.parse_args()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Feature extraction ─────────────────────────────────────────────────────
    if not args.skip_features and args.exp is None:
        extract_features()

    # ── Experiments ────────────────────────────────────────────────────────────
    experiments = [args.exp] if args.exp else list(range(1, 7))
    for n in experiments:
        run_exp(n, skip_training=args.skip_training)

    # ── Sample frames ──────────────────────────────────────────────────────────
    if args.exp is None:
        print(f"\n{'='*60}")
        print("  Reporting — Sample Frames")
        print(f"{'='*60}")
        frames_dir = ROOT / "reports" / "figures" / "sample_frames"
        if list(frames_dir.glob("*.png")):
            skip("sample frames already exist")
        else:
            run(REPORTING / "save_sample_frames.py", REPORTING, "Save 3 frames per class")

    print(f"\n{'='*60}")
    print("  All done.  Reports -> reports/figures/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()