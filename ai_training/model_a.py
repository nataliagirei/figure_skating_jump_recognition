import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# jump_type:  0=salchow, 1=axel, 2=toe_loop, 3=failed
# rotation:   0=unknown, 1=single, 2=double, 3=triple
NUM_JUMP_TYPES = 4
NUM_ROTATIONS  = 3   # single / double / triple (raw labels 1,2,3 → remapped 0,1,2 at train time)


class FigureSkatingModelA(nn.Module):
    """
    Option A: ResNet18 + LSTM with partially unfrozen backbone.

    Changes vs baseline:
    - layer4 of ResNet is trainable (fine-tuned with lower LR)
    - layers 0-6 remain frozen
    - LSTM hidden size increased to 256
    - BN → Dropout(0.3) order preserved
    """

    def __init__(self, dropout_prob: float = 0.3) -> None:
        super().__init__()

        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        children = list(resnet.children())

        # children indices: 0=conv1, 1=bn1, 2=relu, 3=maxpool,
        #                   4=layer1, 5=layer2, 6=layer3,
        #                   7=layer4, 8=avgpool, 9=fc
        self.cnn_frozen   = nn.Sequential(*children[:7])   # conv1…layer3, frozen
        self.cnn_finetune = nn.Sequential(*children[7:9])  # layer4 + avgpool, trainable
        self.feature_dim  = 512

        for param in self.cnn_frozen.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
        )

        self.batchnorm = nn.BatchNorm1d(256)
        self.dropout   = nn.Dropout(dropout_prob)

        self.head_type     = nn.Linear(256, NUM_JUMP_TYPES)
        self.head_rotation = nn.Linear(256, NUM_ROTATIONS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, frames, C, H, W = x.shape

        x = x.view(batch_size * frames, C, H, W)
        features = self.cnn_frozen(x)
        features = self.cnn_finetune(features)
        features = features.view(batch_size * frames, self.feature_dim)
        features = features.view(batch_size, frames, self.feature_dim)

        lstm_out, _ = self.lstm(features)
        feat = lstm_out[:, -1, :]

        feat = self.batchnorm(feat)
        feat = self.dropout(feat)

        return self.head_type(feat), self.head_rotation(feat)