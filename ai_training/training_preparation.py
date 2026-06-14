"""
Prepares data/frames/ from raw videos in data/.

1. Read registry.json for labels and video paths
2. Stratified 70/15/15 train/val/test split (failed excluded, seed=42)
3. For each video: resize short-side→256, center crop 224×224, ImageNet-normalise,
   sample NUM_FRAMES uniform frames → save as (3,224,224) float32 .npy
4. Write data/frames/dataset_metadata_{split}.json

Run after build_registry.py, before any training or feature-extraction scripts.
Idempotent: already-extracted videos (full set of .npy present) are skipped.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parent.parent
REGISTRY   = ROOT / "registry.json"
FRAMES_DIR = ROOT / "data" / "frames"

NUM_FRAMES        = 16
TARGET_SHORT_SIDE = 256
CROP_SIZE         = 224
TRAIN_FRAC        = 0.70
VAL_FRAC          = 0.15
SEED              = 42

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Frame helpers ──────────────────────────────────────────────────────────────

def _resize_short(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if h < w:
        nh, nw = TARGET_SHORT_SIDE, int(w * TARGET_SHORT_SIDE / h)
    else:
        nh, nw = int(h * TARGET_SHORT_SIDE / w), TARGET_SHORT_SIDE
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)


def _center_crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x = w // 2 - CROP_SIZE // 2
    y = h // 2 - CROP_SIZE // 2
    return frame[y:y + CROP_SIZE, x:x + CROP_SIZE]


def _to_tensor(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (H,W,3) → float32 (3,H,W) ImageNet-normed."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)


def extract_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    raw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw.append(_center_crop(_resize_short(frame)))
    cap.release()
    if not raw:
        return []
    indices = np.linspace(0, len(raw) - 1, NUM_FRAMES, dtype=int)
    return [_to_tensor(raw[i]) for i in indices]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    registry: dict = json.loads(REGISTRY.read_text(encoding="utf-8"))

    # Exclude failed jump (label 3)
    videos = {stem: e for stem, e in registry.items() if e["jump_type"] != "failed"}
    print(f"Videos (excluding failed): {len(videos)}")
    counts = Counter(e["jump_type"] for e in videos.values())
    for t, c in sorted(counts.items()):
        print(f"  {t}: {c}")

    # Stratified split
    rng = random.Random(SEED)
    groups: dict[str, list] = defaultdict(list)
    for stem, e in videos.items():
        groups[e["jump_type"]].append(stem)

    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for jump_type, stems in sorted(groups.items()):
        rng.shuffle(stems)
        n, t = len(stems), int(len(stems) * TRAIN_FRAC)
        v = t + int(n * VAL_FRAC)
        splits["train"] += stems[:t]
        splits["val"]   += stems[t:v]
        splits["test"]  += stems[v:]

    for name, stems in splits.items():
        c = Counter(videos[s]["jump_type"] for s in stems)
        print(f"  {name:5s}: {len(stems)}  {dict(sorted(c.items()))}")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    total_ok = total_skip = total_fail = 0

    for split_name, stems in splits.items():
        split_dir = FRAMES_DIR / split_name
        split_dir.mkdir(exist_ok=True)
        meta_out: dict = {}

        for stem in tqdm(stems, desc=split_name):
            entry     = videos[stem]
            frame_dir = split_dir / stem

            # Already done?
            existing = list(frame_dir.glob("*.npy")) if frame_dir.exists() else []
            if len(existing) == NUM_FRAMES:
                frame_paths = [str(frame_dir / f"{i:03d}.npy") for i in range(NUM_FRAMES)]
                meta_out[stem] = {**entry, "frames": frame_paths}
                total_skip += 1
                continue

            video_path = ROOT / entry["filepath"]
            if not video_path.exists():
                print(f"\n  [WARN] not found: {video_path}")
                total_fail += 1
                continue

            frames = extract_frames(video_path)
            if len(frames) < NUM_FRAMES:
                print(f"\n  [WARN] too few frames: {stem} ({len(frames)})")
                total_fail += 1
                continue

            frame_dir.mkdir(exist_ok=True)
            frame_paths = []
            for i, f in enumerate(frames):
                p = frame_dir / f"{i:03d}.npy"
                np.save(p, f)
                frame_paths.append(str(p))

            meta_out[stem] = {**entry, "frames": frame_paths}
            total_ok += 1

        out = FRAMES_DIR / f"dataset_metadata_{split_name}.json"
        out.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {split_name}: {len(meta_out)} videos -> {out.name}")

    print(f"\nDone.  new={total_ok}  skipped={total_skip}  failed={total_fail}")
    print(f"Output: {FRAMES_DIR}")


if __name__ == "__main__":
    main()