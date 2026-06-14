"""
Extracts YOLO-guided blade-region crop features.

Replaces the bottom-50% heuristic in extract_local_features.py with a
tight crop from a YOLOv11n person detection:
  - Find takeoff frame window (same flow-based logic as extract_local_features.py)
  - For each of 4 takeoff frames: run YOLO, crop bottom FOOT_FRAC of person bbox
  - Per-frame fallback to bottom-50% heuristic if no person detected
  - Pass crop through frozen ResNet18 avgpool → 512-d
  - Average across 4 frames → final (512,) feature vector

Output: data/features_yolo/{stem}.npy  shape (512,)

Flags:
  --bottom   Read from data/frames_bottom/ → write to data/features_yolo_bottom/
             Run extract_bottom_frames.py first to generate bottom-cropped frames.

Requires: pip install ultralytics
"""
import argparse
import json
import sys
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
_out_name = "features_yolo_bottom" if _args.bottom else "features_yolo"

YOLO_DIR  = ROOT / "data" / _out_name
FLOW_DIR  = ROOT / "data" / "features_flow"
METAS     = [
    ROOT / "data" / _variant / "dataset_metadata_train.json",
    ROOT / "data" / _variant / "dataset_metadata_val.json",
    ROOT / "data" / _variant / "dataset_metadata_test.json",
]

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOOT_FRAC = 0.30   # bottom 30% of person bbox = blade region
N_WINDOW  = 4      # frames around takeoff to average
CONF_THRESH = 0.25
COCO_PERSON = 0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_resnet_backbone() -> nn.Module:
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone.to(DEVICE)


def find_takeoff_idx(stem: str, n_frames: int) -> int:
    flow_path = FLOW_DIR / f"{stem}.npy"
    if not flow_path.exists():
        return 2
    flow    = np.load(flow_path)
    mean_v  = flow[1::3]
    pair_idx = int(np.argmin(mean_v))
    return min(pair_idx + 1, n_frames - 1)


def takeoff_frame_indices(takeoff: int, n_frames: int) -> list[int]:
    half  = N_WINDOW // 2
    start = max(0, takeoff - half)
    end   = min(n_frames, start + N_WINDOW)
    start = max(0, end - N_WINDOW)
    return list(range(start, end))


def denorm(frame_npy: np.ndarray) -> np.ndarray:
    """(3,H,W) float32 ImageNet-normalised  ->  (H,W,3) uint8 RGB"""
    img = frame_npy.transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def yolo_blade_crop(
    frame_npy: np.ndarray, yolo_model, H: int, W: int
) -> tuple[np.ndarray, bool]:
    """
    Detect person with YOLO, return (blade_crop, detected) where:
      blade_crop — (3,224,224) float32 ImageNet-normalised crop
      detected   — True if person found, False if fallback was used

    Falls back to bottom-50% heuristic if no person detected.
    YOLO receives BGR (its native format); the RGB crop is used for ResNet.
    """
    orig     = denorm(frame_npy)                    # (H,W,3) uint8 RGB
    orig_bgr = orig[:, :, ::-1]                     # BGR for YOLO
    results  = yolo_model(orig_bgr, verbose=False)
    result   = results[0]

    best_conf = -1.0
    best_box  = None
    for box in result.boxes:
        if int(box.cls.item()) != COCO_PERSON:
            continue
        conf = float(box.conf.item())
        if conf > CONF_THRESH and conf > best_conf:
            best_conf = conf
            best_box  = box.xyxy[0].cpu().numpy()

    if best_box is not None:
        x1, y1, x2, y2 = (
            max(0, int(best_box[0])),
            max(0, int(best_box[1])),
            min(W, int(best_box[2])),
            min(H, int(best_box[3])),
        )
        bbox_h  = y2 - y1
        foot_y1 = max(y1, int(y2 - bbox_h * FOOT_FRAC))
        blade   = orig[foot_y1:y2, x1:x2]          # (h', w', 3) RGB uint8
        if blade.size > 0:
            blade_f = blade.astype(np.float32) / 255.0
            blade_f = (blade_f - IMAGENET_MEAN) / IMAGENET_STD
            blade_t = torch.from_numpy(blade_f.transpose(2, 0, 1)).unsqueeze(0)
            blade_t = F.interpolate(blade_t, size=(224, 224),
                                    mode="bilinear", align_corners=False)
            return blade_t.squeeze(0).numpy(), True  # (3,224,224) float32, detected

    # Fallback: bottom-50% of frame (same as extract_local_features.py)
    crop_start = H // 2
    crop       = frame_npy[:, crop_start:, :]
    crop_t     = torch.from_numpy(crop).unsqueeze(0).float()
    crop_t     = F.interpolate(crop_t, size=(224, 224),
                               mode="bilinear", align_corners=False)
    return crop_t.squeeze(0).numpy(), False          # (3,224,224) float32, fallback


@torch.no_grad()
def extract_yolo(
    backbone:    nn.Module,
    yolo_model,
    frame_paths: list[str],
    frame_idxs:  list[int],
) -> tuple[np.ndarray, int]:
    """Returns (feature_512d, n_fallback_frames)."""
    feats      = []
    n_fallback = 0
    for i in frame_idxs:
        frame_npy = np.load(frame_paths[i])
        H, W = frame_npy.shape[1], frame_npy.shape[2]
        crop, detected = yolo_blade_crop(frame_npy, yolo_model, H, W)
        if not detected:
            n_fallback += 1
        crop_t = torch.from_numpy(crop).unsqueeze(0).to(DEVICE)
        feat   = backbone(crop_t).flatten()
        feats.append(feat.cpu().numpy())
    return np.mean(feats, axis=0).astype(np.float32), n_fallback


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not found.  Install with:  pip install ultralytics")
        sys.exit(1)

    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    all_entries: dict[str, list[str]] = {}
    for meta_path in METAS:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        for stem, entry in data.items():
            all_entries[stem] = entry["frames"]

    print(f"Videos to process: {len(all_entries)}")

    print("Loading models ...")
    backbone   = build_resnet_backbone()
    yolo_model = YOLO("yolo11n.pt")

    n_fallback = 0
    for stem, frame_paths in tqdm(all_entries.items()):
        out_path = YOLO_DIR / f"{stem}.npy"
        if out_path.exists():
            continue
        if not all(Path(p).exists() for p in frame_paths):
            continue

        takeoff = find_takeoff_idx(stem, len(frame_paths))
        idxs    = takeoff_frame_indices(takeoff, len(frame_paths))

        feat, n_fb = extract_yolo(backbone, yolo_model, frame_paths, idxs)
        n_fallback += n_fb
        np.save(out_path, feat)

    print(f"Done. Saved to {YOLO_DIR}")
    if n_fallback:
        frames_total = len(all_entries) * N_WINDOW
        print(f"  Fallback to heuristic crop: {n_fallback}/{frames_total} frames "
              f"({n_fallback / max(frames_total, 1) * 100:.1f}%)")


if __name__ == "__main__":
    main()