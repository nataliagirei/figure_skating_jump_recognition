import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

NUM_ROTATIONS = 3   # single / double / triple (raw labels 1,2,3 → remapped 0,1,2 at train time)


class FigureSkatingModelB(nn.Module):
    """
    R3D-18 pretrained on Kinetics-400, frozen backbone, linear heads only.

    Input:  (B, T, C, H, W)
    Output: (type_logits, rotation_logits)
    """

    def __init__(
        self, num_jump_types: int = 3, dropout_prob: float = 0.3, freeze_backbone: bool = True
    ) -> None:
        super().__init__()

        backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        # Drop the final 400-class FC; after avgpool shape is (B, 512, 1, 1, 1)
        self.backbone    = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 512

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.batchnorm = nn.BatchNorm1d(self.feature_dim)
        self.dropout   = nn.Dropout(dropout_prob)

        self.head_type     = nn.Linear(self.feature_dim, num_jump_types)
        self.head_rotation = nn.Linear(self.feature_dim, NUM_ROTATIONS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1, 3, 4)          # (B,T,C,H,W) → (B,C,T,H,W)

        features = self.backbone(x).flatten(1) # (B, 512)
        features = self.batchnorm(features)
        features = self.dropout(features)

        return self.head_type(features), self.head_rotation(features)