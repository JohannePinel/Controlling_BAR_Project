import torch
import torch.nn as nn



# ── Modèle ─────────────────────────────────────────────────────────────────────

class ObstacleNet(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 5 * 2 * 721 * 2  # 14420

        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 64),        nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
        )
        self.danger_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.gains_head  = nn.Linear(32, 2)

    def forward(self, x):
        features = self.backbone(x)
        danger   = self.danger_head(features).squeeze(-1)  # (B,)
        gains    = self.gains_head(features)               # (B, 2)
        return danger, gains
    
