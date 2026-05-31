import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# jump_type:  0=salchow, 1=axel, 2=toe_loop  (failed excluded)
# rotation:   0=unknown, 1=single, 2=double, 3=triple
NUM_JUMP_TYPES = 3
NUM_ROTATIONS  = 4


class FigureSkatingModel(nn.Module):
    """
    ResNet18 (frozen) + LSTM backbone with two classification heads.

    CNN is fully frozen — ImageNet features generalise well and
    prevent overfitting on the small dataset (~300 videos).
    Fine-tuning is not beneficial below ~1000 labelled examples.
    """

    def __init__(self, dropout_prob: float = 0.3) -> None:
        super().__init__()

        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn         = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        for param in self.cnn.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        self.batchnorm = nn.BatchNorm1d(128)
        self.dropout   = nn.Dropout(dropout_prob)

        self.head_type     = nn.Linear(128, NUM_JUMP_TYPES)
        self.head_rotation = nn.Linear(128, NUM_ROTATIONS)  # 4-class: unknown/single/double/triple

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, frames, C, H, W = x.shape

        x        = x.view(batch_size * frames, C, H, W)
        features = self.cnn(x).view(batch_size * frames, self.feature_dim)
        features = features.view(batch_size, frames, self.feature_dim)

        lstm_out, _ = self.lstm(features)
        feat = lstm_out[:, -1, :] + lstm_out.mean(dim=1)

        feat = self.batchnorm(feat)
        feat = self.dropout(feat)

        return self.head_type(feat), self.head_rotation(feat)