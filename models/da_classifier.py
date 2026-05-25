"""Standalone image-level domain classifier head.

Used externally to the YOLO model (dumb-model principle): the model returns
backbone_feat; train_GRL.py runs DAImgHead on the feature with gradient_scalar
applied for the GRL-attached pass.
"""
import torch
from torch import nn


class DAImgHead(nn.Module):
    """Conv -> ReLU -> Conv producing [B, 1, H, W] logits.

    Initialized with small weights (std=0.001) to match DA-Detect's DAImgHead.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        for layer in (self.conv1, self.conv2):
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        return self.conv2(x)
