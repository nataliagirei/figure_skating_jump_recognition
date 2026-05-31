import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT      = Path(__file__).resolve().parent.parent
POSES_DIR = ROOT / "data" / "poses"


class PoseDataset(Dataset):
    """
    Loads pre-extracted MediaPipe pose sequences (.npy) from data/poses/.

    Each sample: (pose_sequence, jump_type_label, rotation_label)
      pose_sequence:   (T, 99) float32 — 33 landmarks × (x, y, visibility)
      jump_type_label: 0=salchow, 1=axel, 2=toe_loop
      rotation_label:  0=single, 1=double, 2=triple
    """

    def __init__(self, split_path: str | Path) -> None:
        with open(split_path, encoding="utf-8") as f:
            raw = json.load(f)

        # Keep only videos that have a pose file
        self.data  : list[dict] = []
        self.stems : list[str]  = []
        missing = 0

        for stem, entry in raw.items():
            pose_path = POSES_DIR / f"{stem}.npy"
            if not pose_path.exists():
                missing += 1
                continue
            self.data.append(entry)
            self.stems.append(stem)

        if missing:
            print(f"  [PoseDataset] Skipped {missing} videos with no pose file")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        stem  = self.stems[idx]
        entry = self.data[idx]

        poses = np.load(POSES_DIR / f"{stem}.npy").astype(np.float32)  # (T, 99)
        poses = torch.from_numpy(poses)

        return (
            poses,
            torch.tensor(entry["jump_type_label"], dtype=torch.long),
            torch.tensor(entry["rotation"],        dtype=torch.long),
        )