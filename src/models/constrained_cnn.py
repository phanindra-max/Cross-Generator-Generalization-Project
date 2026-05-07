"""
Constrained CNN for image manipulation detection.

Implements the constrained convolutional layer from:
    Bayar, B. & Stamm, M.C. (2018). "Constrained Convolutional Neural Networks:
    A New Approach Towards General Purpose Image Manipulation Detection." IEEE TIFS.

The key idea: the first convolutional layer is constrained so its filters
sum to zero (suppressing image content and emphasizing residual noise patterns).

Deviations from Bayar 2018 (documented for the writeup, not correctness bugs):
  - Bayar uses grayscale input with 3 filters; this implementation uses RGB
    input with 5 filters per output channel. Treat as "Bayar-style" rather
    than a literal reproduction.
  - Bayar's reference implementation uses padding=0 on the constrained layer.
    This implementation uses same-padding for downstream shape convenience.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedConvLayer(nn.Module):
    """
    Constrained convolutional layer (Bayar Algorithm 1).

    The parameter `self.weight` is constrained so that, for every (out_ch, in_ch)
    filter slice:
        w[c, c]                = -1
        sum(w[h, w] for (h,w) != (c,c)) = +1
    Equivalent to a high-pass / prediction-error filter (filter sums to zero).

    Constraint is enforced by HARD projection (`project()`), not by a
    differentiable normalization in forward. The training loop must call
    `project()` after every `optimizer.step()` so the parameter itself stays
    on the constraint manifold between updates. This matches Bayar's Algorithm 1.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 5, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.xavier_normal_(self.weight)
        self.project()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=self.kernel_size // 2)

    @torch.no_grad()
    def project(self) -> None:
        """
        Hard-project `self.weight` onto the Bayar constraint manifold.
        Call once at init and after every optimizer.step().
        """
        c = self.kernel_size // 2
        w = self.weight.data
        w[:, :, c, c] = 0.0

        # Sum is over off-center positions only (center is now zero).
        denom = w.sum(dim=(2, 3), keepdim=True)
        # Guard against degenerate filters whose off-center sum is ~0.
        # Falling back to 1 leaves them un-rescaled this step; the next
        # gradient + projection iterates them out of the degenerate region.
        safe = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
        w.div_(safe)

        w[:, :, c, c] = -1.0


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

    def project_constraints(self) -> None:
        """Re-project the constrained front end. Call after optimizer.step()."""
        self.constrained_conv.project()

    def get_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the constrained-conv residual for visualization."""
        return self.constrained_conv(x)
