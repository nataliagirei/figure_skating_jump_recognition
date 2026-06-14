"""
Extracts lower-body crop features focused on the takeoff window.

Takeoff detection:
  - Uses pre-computed optical flow (data/features_flow/{stem}.npy).
  - The mean_v (vertical flow) values are at indices 1, 4, 7, ... of the 45-d
    flow vector (one per consecutive frame pair).
  - The pair with the most negative mean_v has the strongest upward motion →
    that is the approximate takeoff moment.
  - Falls back to frames 0-4 if flow features are missing.

Crop:
  - 4 frames centred around the takeoff frame index.
  - Bottom 50% of each frame cropped (hips → feet → ice surface).
  - Crop resized back to 224×224 and passed through frozen ResNet18.
  - Features averaged across the 4 frames → 512-d output.

Output: data/features_local/{stem}.npy  shape (512,)

Flags:
  --bottom   Read from data/frames_bottom/ → write to data/features_local_bottom/
             Run extract_bottom_frames.py first to generate bottom-cropped frames.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bottom", action="store_true",
                   help="Use bottom-cropped frames (data/frames_bottom/) as input")
    return p.parse_args()

_args     = _parse_args()
_variant  = "frames_bottom" if _args.bottom else "frames"
_out_name = "features_local_bottom" if _args.bottom else "features_local"

LOCAL_DIR = ROOT / "data" / _out_name
FLOW_DIR  = ROOT / "data" / "features_flow"
METAS     = [
    ROOT / "data" / _variant / "dataset_metadata_train.json",
    ROOT / "data" / _variant / "dataset_metadata_val.json",
    ROOT / "data" / _variant / "dataset_metadata_test.json",
]

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CROP_START_REL = 0.50   # crop bottom 50% (hips → feet)
N_WINDOW       = 4      # frames around takeoff to use


def build_backbone() -> nn.Module:
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # drop FC
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone.to(DEVICE)


def find_takeoff_idx(flow_path: Path, n_frames: int) -> int:
    """
    Returns the sampled frame index where takeoff occurs.
    Uses minimum mean_v (most upward motion) from flow features.
    """
    flow    = np.load(flow_path)      # (45,)
    mean_v  = flow[1::3]              # (15,) — vertical flow per pair
    pair_idx = int(np.argmin(mean_v)) # pair with max upward motion
    # pair_idx i → between frame i and i+1; takeoff is frame i+1
    return min(pair_idx + 1, n_frames - 1)


def takeoff_frame_indices(takeoff: int, n_frames: int) -> list[int]:
    """N_WINDOW frames centred on takeoff, clamped to [0, n_frames)."""
    half  = N_WINDOW // 2
    start = max(0, takeoff - half)
    end   = min(n_frames, start + N_WINDOW)
    start = max(0, end - N_WINDOW)
    return list(range(start, end))


@torch.no_grad()
def extract_local(
    backbone:    nn.Module,
    frame_paths: list[str],
    frame_idxs:  list[int],
) -> np.ndarray:
    feats = []
    for i in frame_idxs:
        frame = torch.from_numpy(np.load(frame_paths[i]))  # (3, H, W)
        h = frame.shape[1]
        crop_start = int(h * CROP_START_REL)
        crop = frame[:, crop_start:, :]                    # (3, H/2, W)
        crop = F.interpolate(
            crop.unsqueeze(0).float(),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).to(DEVICE)
        feat = backbone(crop).flatten()                    # (512,)
        feats.append(feat.cpu().numpy())

    return np.mean(feats, axis=0).astype(np.float32)      # (512,)


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    all_entries: dict[str, list[str]] = {}
    for meta_path in METAS:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        for stem, entry in data.items():
            all_entries[stem] = entry["frames"]

    print(f"Videos to process: {len(all_entries)}")
    backbone = build_backbone()

    no_flow = 0
    for stem, frame_paths in tqdm(all_entries.items()):
        out_path = LOCAL_DIR / f"{stem}.npy"
        if out_path.exists():
            continue
        if not all(Path(p).exists() for p in frame_paths):
            continue

        flow_path = FLOW_DIR / f"{stem}.npy"
        if flow_path.exists():
            takeoff = find_takeoff_idx(flow_path, len(frame_paths))
        else:
            takeoff = 2   # fallback: third frame
            no_flow += 1

        idxs = takeoff_frame_indices(takeoff, len(frame_paths))
        feat = extract_local(backbone, frame_paths, idxs)
        np.save(out_path, feat)

    print(f"Done. Saved to {LOCAL_DIR}")
    if no_flow:
        print(f"  {no_flow} videos used fallback (no flow features found)")


if __name__ == "__main__":
    main()