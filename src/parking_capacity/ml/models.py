"""Modèles de régression sur image."""

from __future__ import annotations

import torch
import torch.nn as nn

ARCHITECTURES = ("tiny", "resnet18", "resnet50", "efficientnet_b0")


class TinyChipRegressor(nn.Module):
    """
    Régresseur adapté aux images quasi-uniformes (tests synthétiques) :
    moyenne globale RGB → MLP → capacité.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.pool(x).flatten(1)
        return self.fc(v).squeeze(-1)


def build_model(
    name: str,
    *,
    pretrained: bool = True,
) -> nn.Module:
    if name == "tiny":
        return TinyChipRegressor()
    if name in ARCHITECTURES and name != "tiny":
        try:
            import torchvision.models as tvm
        except ImportError as e:
            raise ImportError(
                f"torchvision requis pour {name} : pip install -e '.[train]'"
            ) from e
        if name == "resnet18":
            try:
                w = tvm.ResNet18_Weights.DEFAULT if pretrained else None
                m = tvm.resnet18(weights=w)
            except AttributeError:
                m = tvm.resnet18(pretrained=pretrained)
            m.fc = nn.Linear(m.fc.in_features, 1)
            return m
        if name == "resnet50":
            try:
                w = tvm.ResNet50_Weights.DEFAULT if pretrained else None
                m = tvm.resnet50(weights=w)
            except AttributeError:
                m = tvm.resnet50(pretrained=pretrained)
            m.fc = nn.Linear(m.fc.in_features, 1)
            return m
        if name == "efficientnet_b0":
            try:
                w = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                m = tvm.efficientnet_b0(weights=w)
            except AttributeError:
                m = tvm.efficientnet_b0(pretrained=pretrained)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
            return m
    raise ValueError(f"Architecture inconnue : {name}")
