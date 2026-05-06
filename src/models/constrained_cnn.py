"""
Constrained CNN for image manipulation detection.

Implements the constrained convolutional layer from:
    Bayar, B. & Stamm, M.C. (2018). "Constrained Convolutional Neural Networks:
    A New Approach Towards General Purpose Image Manipulation Detection." IEEE TIFS.

The key idea: the first convolutional layer is constrained so its filters
sum to zero (suppressing image content and emphasizing residual noise patterns).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedConvLayer(nn.Module):
    """
    Constrained convolutional layer where filter weights are normalized so that
    the center weight is -1 and all other weights sum to 1. This forces the
    layer to act as a high-pass / prediction-error filter.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 5, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        constrained_weights = self._apply_constraint()
        return F.conv2d(x, constrained_weights, padding=self.kernel_size // 2)

    def _apply_constraint(self) -> torch.Tensor:
        """
        Apply the Bayar constraint: center pixel weight = -1,
        remaining weights normalized to sum to 1 per filter per input channel.
        """
        weights = self.weight.clone()
        center = self.kernel_size // 2

        # Zero out center, normalize the rest to sum to 1, then set center to -1
        with torch.no_grad():
            weights[:, :, center, center] = 0

        # Normalize non-center weights to sum to 1
        weight_sum = weights.sum(dim=(2, 3), keepdim=True)
        weight_sum[weight_sum == 0] = 1.0
        weights = weights / weight_sum

        # Set center to -1
        weights[:, :, center, center] = -1.0

        return weights


class ConstrainedCNN(nn.Module):
    """
    Full constrained CNN architecture for binary deepfake detection.

    Architecture:
        1. Constrained conv layer (high-pass filtering)
        2. Several standard conv blocks with batch norm and ReLU
        3. Global average pooling
        4. Fully connected classifier
    """

    def __init__(
        self,
        num_classes: int = 2,
        constrained_filters: int = 5,
        base_channels: int = 32,
    ):
        super().__init__()

        self.constrained_conv = ConstrainedConvLayer(
            in_channels=3, out_channels=constrained_filters, kernel_size=5
        )

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(constrained_filters, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(base_channels * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.constrained_conv(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the constrained-conv residual for visualization."""
        return self.constrained_conv(x)
