"""U-Net with a ResNet-18 encoder for monocular depth estimation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.conv = _conv(c_in + c_skip, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DepthUNet(nn.Module):
    """Input: normalized RGB (B, 3, H, W). Output: depth in meters (B, 1, H, W)."""

    def __init__(self, pretrained: bool = True, max_depth: float = 10.0):
        super().__init__()
        enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # 1/2, 64
        self.pool = enc.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.head = nn.Conv2d(32, 1, 3, padding=1)
        self.max_depth = max_depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        d = self.up4(s4, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        d = F.interpolate(self.head(d), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(d) * self.max_depth
