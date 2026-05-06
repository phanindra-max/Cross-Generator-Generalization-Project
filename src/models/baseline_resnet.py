"""
Baseline ResNet model for deepfake detection.

Standard ImageNet-pretrained ResNet fine-tuned for binary classification.
Used as a comparison point against the constrained CNN approach.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BaselineResNet(nn.Module):
    """ResNet-based binary classifier for deepfake detection."""

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            num_classes: Number of output classes.
            backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50').
            pretrained: Whether to use ImageNet-pretrained weights.
            freeze_backbone: If True, freeze all backbone parameters.
        """
        super().__init__()

        weights = "IMAGENET1K_V1" if pretrained else None
        if backbone == "resnet18":
            self.model = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            self.model = models.resnet34(weights=weights)
        elif backbone == "resnet50":
            self.model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate features (before classification head)."""
        modules = list(self.model.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        return feature_extractor(x).flatten(1)
