from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18RGB(nn.Module):
    """标准 ResNet18（RGB），使用预训练权重，输出 7 类。"""

    def __init__(self, num_classes: int = 7, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


