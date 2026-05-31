"""
Extracts dense optical flow features from pre-extracted frames.

For each video: 15 consecutive frame pairs → [mean_u, mean_v, mean_magnitude]
per pair → 45-d feature vector.

mean_u captures horizontal motion (axel approaches forward → distinctive sign),
mean_v captures vertical motion (jump height profile),
mean_magnitude captures overall speed (spin acceleration).

Output: data/features_flow/{stem}.npy  shape (45,)

Run after training_preparation.py (frames must exist).
"""
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parent.parent
FLOW_DIR = ROOT / "data" / "features_flow"
METAS    = [
    ROOT / "data" / "frames" / "dataset_metadata_train.json",
    ROOT / "data" / "frames" / "dataset_metadata_val.json",
    ROOT / "data" / "frames" / "dataset_metadata_test.json",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Farneback parameters
FB_PYR_SCALE  = 0.5
FB_LEVELS     = 3
FB_WINSIZE    = 15
FB_ITERATIONS = 3
FB_POLY_N     = 5
FB_POLY_SIGMA = 1.2


def _to_gray(npy_path: str) -> np.ndarray:
    """(3,224,224) float32 ImageNet-normalised → (224,224) uint8 grayscale."""
    t   = np.load(npy_path)                            # (3, H, W)
    img = t.transpose(1, 2, 0)                         # (H, W, 3)
    img = (img * IMAGENET_STD + IMAGENET_MEAN) * 255   # [0, 255]
    img = img.clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def flow_features(frame_paths: list[str]) -> np.ndarray:
    """Returns (45,) vector: 15 pairs × [mean_u, mean_v, mean_magnitude]."""
    grays = [_to_gray(p) for p in frame_paths]

    features: list[float] = []
    for prev, curr in zip(grays[:-1], grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            FB_PYR_SCALE, FB_LEVELS, FB_WINSIZE,
            FB_ITERATIONS, FB_POLY_N, FB_POLY_SIGMA, 0,
        )
        u, v   = flow[..., 0], flow[..., 1]
        mag    = np.hypot(u, v)
        features.extend([float(u.mean()), float(v.mean()), float(mag.mean())])

    return np.array(features, dtype=np.float32)   # (45,)


def main() -> None:
    FLOW_DIR.mkdir(parents=True, exist_ok=True)

    all_entries: dict[str, list[str]] = {}
    for meta_path in METAS:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        for stem, entry in data.items():
            all_entries[stem] = entry["frames"]

    print(f"Videos to process: {len(all_entries)}")

    for stem, frame_paths in tqdm(all_entries.items()):
        out_path = FLOW_DIR / f"{stem}.npy"
        if out_path.exists():
            continue
        if not all(Path(p).exists() for p in frame_paths):
            continue
        feat = flow_features(frame_paths)
        np.save(out_path, feat)

    print(f"Done. Saved to {FLOW_DIR}")


if __name__ == "__main__":
    main()