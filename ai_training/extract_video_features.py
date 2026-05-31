"""
Extracts 512-d R3D-18 (Kinetics-400) feature vectors from pre-extracted frames.
Saves one .npy per video: shape (512,).
Output: data/features/{stem}.npy

Run once before train_sklearn.py.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.models.video import r3d_18, R3D_18_Weights
from tqdm import tqdm

ROOT         = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"
METAS        = [
    ROOT / "data" / "frames" / "dataset_metadata_train.json",
    ROOT / "data" / "frames" / "dataset_metadata_val.json",
    ROOT / "data" / "frames" / "dataset_metadata_test.json",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def build_backbone() -> torch.nn.Module:
    backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    # Drop final FC; after avgpool output is (B, 512, 1, 1, 1)
    model = torch.nn.Sequential(*list(backbone.children())[:-1])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(DEVICE)


@torch.no_grad()
def extract(backbone: torch.nn.Module, frame_paths: list[str]) -> np.ndarray:
    frames = [torch.from_numpy(np.load(p)) for p in frame_paths]  # each (3,224,224)
    clip   = torch.stack(frames)          # (T, 3, 224, 224)
    clip   = clip.unsqueeze(0)            # (1, T, 3, 224, 224)
    clip   = clip.permute(0, 2, 1, 3, 4) # (1, 3, T, 224, 224)
    clip   = clip.to(DEVICE)
    feat   = backbone(clip)               # (1, 512, 1, 1, 1)
    return feat.squeeze().cpu().numpy()   # (512,)


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all entries across splits (stem → frame_paths)
    all_entries: dict[str, list[str]] = {}
    for meta_path in METAS:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        for stem, entry in data.items():
            all_entries[stem] = entry["frames"]

    print(f"Total videos to process: {len(all_entries)}")

    backbone = build_backbone()

    skipped = 0
    for stem, frame_paths in tqdm(all_entries.items()):
        out_path = FEATURES_DIR / f"{stem}.npy"
        if out_path.exists():
            continue
        if not all(Path(p).exists() for p in frame_paths):
            skipped += 1
            continue
        feat = extract(backbone, frame_paths)
        np.save(out_path, feat)

    print(f"Done. Features saved to {FEATURES_DIR}")
    if skipped:
        print(f"Skipped {skipped} videos with missing frame files.")


if __name__ == "__main__":
    main()