"""
Saves 3 representative frames per jump class for documentation.

For each class (salchow, axel, toe_loop):
  - takeoff:  frames[0]          — first sampled frame (approach)
  - mid-air:  frames[len//2]     — middle frame (peak of jump)
  - landing:  frames[-1]         — last sampled frame (landing)

Output: reports/figures/sample_frames/{class}_{phase}.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
METAS      = [
    ROOT / "data" / "frames" / "dataset_metadata_train.json",
    ROOT / "data" / "frames" / "dataset_metadata_val.json",
]
OUT_DIR    = ROOT / "reports" / "figures" / "sample_frames"

LABEL_NAMES  = {0: "salchow", 1: "axel", 2: "toe_loop"}
FAILED_LABEL = 3

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def npy_to_rgb(path: str) -> np.ndarray:
    """(3, H, W) ImageNet-normalised float32 → (H, W, 3) uint8."""
    t   = np.load(path)                           # (3, H, W)
    img = t.transpose(1, 2, 0)                    # (H, W, 3)
    img = img * IMAGENET_STD + IMAGENET_MEAN      # un-normalise
    return img.clip(0, 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_entries: dict[str, dict] = {}
    for meta in METAS:
        all_entries.update(json.loads(meta.read_text(encoding="utf-8")))

    # Collect one example per class
    examples: dict[int, dict] = {}
    for stem, entry in all_entries.items():
        label = entry["jump_type_label"]
        if label == FAILED_LABEL or label in examples:
            continue
        if not all(Path(p).exists() for p in entry["frames"]):
            continue
        examples[label] = entry
        if len(examples) == len(LABEL_NAMES):
            break

    for label, entry in examples.items():
        class_name   = LABEL_NAMES[label]
        frame_paths  = entry["frames"]
        n            = len(frame_paths)
        phase_idxs   = {"takeoff": 0, "mid-air": n // 2, "landing": n - 1}

        for phase, idx in phase_idxs.items():
            img      = npy_to_rgb(frame_paths[idx])
            out_path = OUT_DIR / f"{class_name}_{phase}.png"

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img)
            ax.set_title(f"{class_name} — {phase}", fontsize=11)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"  Saved: {out_path.name}")

    print(f"\nDone. {len(examples) * 3} frames saved to {OUT_DIR}")


if __name__ == "__main__":
    main()