import torch
import torch.nn as nn

# jump_type:  0=salchow, 1=axel, 2=toe_loop
# rotation:   0=unknown, 1=single, 2=double, 3=triple
NUM_JUMP_TYPES = 3
NUM_ROTATIONS  = 4

POSE_DIM = 99   # 33 MediaPipe landmarks × (x, y, visibility)


class PoseModel(nn.Module):
    """
    LSTM classifier on body pose sequences.

    Much fewer parameters than CNN+LSTM → less overfitting on small datasets.
    Pose features are already position/scale normalised → model focuses on
    movement patterns (takeoff edge, arm positions, rotation speed).

    Input:  (B, T, 99)   — batch of pose sequences
    Output: (type_logits, rotation_logits)  each (B, 3)
    """

    def __init__(self, hidden_size: int = 256, num_layers: int = 2,
                 dropout_prob: float = 0.3) -> None:
        super().__init__()

        # Project raw keypoints to a richer representation
        self.input_proj = nn.Sequential(
            nn.Linear(POSE_DIM, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0,
            bidirectional=True,   # both past and future context
        )

        lstm_out_dim = hidden_size * 2   # bidirectional

        self.norm    = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(dropout_prob)

        self.head_type     = nn.Linear(lstm_out_dim, NUM_JUMP_TYPES)
        self.head_rotation = nn.Linear(lstm_out_dim, NUM_ROTATIONS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, 99)
        x = self.input_proj(x)          # (B, T, 128)

        lstm_out, _ = self.lstm(x)      # (B, T, 512)

        # Use both last timestep and mean pooling for richer representation
        last = lstm_out[:, -1, :]
        mean = lstm_out.mean(dim=1)
        feat = last + mean              # (B, 512)

        feat = self.norm(feat)
        feat = self.dropout(feat)

        return self.head_type(feat), self.head_rotation(feat)