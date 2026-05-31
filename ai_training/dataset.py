import json
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

IMAGE_SIZE = 224


class FigureSkatingDataset(Dataset):
    """
    Loads pre-extracted frame .npy files (3, 224, 224) float32, already
    ImageNet-normalised. Applies lightweight tensor-level augmentation in
    train mode; no-op in eval mode.

    jump_type_label: 0=salchow, 1=axel, 2=toe_loop, 3=failed
    rotation:        0=unknown, 1=single, 2=double,  3=triple
    """

    def __init__(
        self,
        split_path: str,
        is_train:   bool = False,
        num_frames: int  = 16,
    ) -> None:
        with open(split_path, encoding="utf-8") as f:
            raw = json.load(f)

        self.data       = list(raw.values())
        self.is_train   = is_train
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        entry       = self.data[idx]
        frame_paths = entry["frames"]

        tensors = self._load_frames(frame_paths)   # list of (3,224,224) tensors

        if self.is_train:
            tensors = self._augment(tensors)

        frames = torch.stack(tensors)   # (T, 3, 224, 224)

        return (
            frames,
            torch.tensor(entry["jump_type_label"], dtype=torch.long),
            torch.tensor(entry["rotation"],        dtype=torch.long),
        )

    def _load_frames(self, frame_paths: list[str]) -> list[torch.Tensor]:
        total = len(frame_paths)

        if self.is_train:
            seg_size = total / self.num_frames
            indices  = [
                min(int(i * seg_size + random.random() * seg_size), total - 1)
                for i in range(self.num_frames)
            ]
        else:
            indices = list(np.linspace(0, total - 1, self.num_frames, dtype=int))

        tensors: list[torch.Tensor] = []
        for i in indices:
            arr = np.load(frame_paths[i])          # (3, 224, 224) float32
            tensors.append(torch.from_numpy(arr))

        if tensors:
            while len(tensors) < self.num_frames:
                tensors.append(tensors[-1])
        else:
            tensors = [torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)] * self.num_frames

        return tensors[:self.num_frames]

    def _augment(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        # Same random params for all frames in the clip
        do_flip   = random.random() < 0.5
        angle     = random.uniform(-10, 10)

        out = []
        for t in tensors:
            if do_flip:
                t = TF.hflip(t)
            t = TF.rotate(t, angle)
            out.append(t)
        return out