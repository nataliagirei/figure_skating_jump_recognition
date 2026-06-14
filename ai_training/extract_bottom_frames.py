"""
Extracts frames from raw videos using a bottom-biased crop.

Current preprocessing (preprocessing.py) uses a center crop, which for
portrait TikTok videos (576x1024) discards the bottom ~25% of the original
frame — exactly where the blade-ice contact happens.

This script reads raw videos directly and applies a bottom-biased crop:
  - Resize short side to 256 (same as current pipeline)
  - Crop 224x224 centred at 75% of frame height (instead of 50%)
  - For 576x1024 → covers original y=515..1020 (bottom half)

Output: data/frames_bottom/{split}/{stem}/000.npy ... 015.npy
        data/frames_bottom/dataset_metadata_{split}.json

The metadata format is identical to data/frames/dataset_metadata_{split}.json
so all downstream extractors can switch to this variant with a path swap.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT         = Path(__file__).resolve().parent.parent
REGISTRY     = ROOT / "registry.json"
FRAMES_DIR   = ROOT / "data" / "frames"
OUT_DIR      = ROOT / "data" / "frames_bottom"
METAS_IN     = [
    (FRAMES_DIR / "dataset_metadata_train.json", "train"),
    (FRAMES_DIR / "dataset_metadata_val.json",   "val"),
    (FRAMES_DIR / "dataset_metadata_test.json",  "test"),
]

NUM_FRAMES        = 16
TARGET_SHORT_SIDE = 256
CROP_SIZE         = 224
Y_FRAC            = 0.75   # centre crop at 75% height — captures feet/blade

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── crop helpers ───────────────────────────────────────────────────────────────

def resize_short_side(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if h < w:
        new_h = TARGET_SHORT_SIDE
        new_w = int(w * TARGET_SHORT_SIDE / h)
    else:
        new_w = TARGET_SHORT_SIDE
        new_h = int(h * TARGET_SHORT_SIDE / w)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def bottom_biased_crop(frame: np.ndarray) -> np.ndarray:
    """
    Crop CROP_SIZE×CROP_SIZE centred at Y_FRAC of frame height.
    For portrait 256x455: centres at y=341, crop y=229..453 →
    covers the bottom ~50% of the original video where blade contact is.
    Horizontal: same centre crop as the original pipeline.
    """
    h, w  = frame.shape[:2]
    x     = max(0, w // 2 - CROP_SIZE // 2)
    y_ctr = int(h * Y_FRAC)
    y     = max(0, min(h - CROP_SIZE, y_ctr - CROP_SIZE // 2))
    return frame[y:y + CROP_SIZE, x:x + CROP_SIZE]


def normalise(frame_rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (H,W,3)  →  float32 (3,224,224) ImageNet-normalised."""
    f = frame_rgb.astype(np.float32) / 255.0
    f = (f - IMAGENET_MEAN) / IMAGENET_STD
    return f.transpose(2, 0, 1).astype(np.float32)


# ── video extraction ───────────────────────────────────────────────────────────

def extract_frames(video_path: Path) -> list[np.ndarray]:
    """
    Read all frames, apply resize + bottom crop, return NUM_FRAMES uniform
    samples as normalised (3,224,224) float32 arrays.
    Returns empty list if video can't be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    raw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_short_side(frame)
        frame = bottom_biased_crop(frame)
        raw.append(frame)
    cap.release()

    if not raw:
        return []

    indices = np.linspace(0, len(raw) - 1, NUM_FRAMES, dtype=int)
    return [normalise(raw[i]) for i in indices]


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    stem_to_video: dict[str, Path] = {
        stem: ROOT / entry["filepath"]
        for stem, entry in registry.items()
    }

    total_ok = total_skip = total_fail = 0

    for meta_path, split in METAS_IN:
        meta_in = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_out: dict = {}

        split_dir = OUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for stem, entry in tqdm(meta_in.items(), desc=split):
            video_path = stem_to_video.get(stem)
            if video_path is None or not video_path.exists():
                total_skip += 1
                continue

            frame_dir = split_dir / stem
            # Check if already done
            if frame_dir.exists() and len(list(frame_dir.glob("*.npy"))) == NUM_FRAMES:
                frame_paths = [str(frame_dir / f"{i:03d}.npy") for i in range(NUM_FRAMES)]
                meta_out[stem] = {**entry, "frames": frame_paths}
                total_ok += 1
                continue

            frames = extract_frames(video_path)
            if len(frames) < NUM_FRAMES:
                total_fail += 1
                continue

            frame_dir.mkdir(exist_ok=True)
            frame_paths = []
            for i, f in enumerate(frames):
                out_path = frame_dir / f"{i:03d}.npy"
                np.save(out_path, f)
                frame_paths.append(str(out_path))

            meta_out[stem] = {**entry, "frames": frame_paths}
            total_ok += 1

        out_meta = OUT_DIR / f"dataset_metadata_{split}.json"
        out_meta.write_text(
            json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  {split}: {len(meta_out)} videos -> {out_meta.name}")

    print(f"\nDone.  ok={total_ok}  skipped={total_skip}  failed={total_fail}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()