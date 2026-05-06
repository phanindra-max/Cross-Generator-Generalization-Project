"""
FaceForensics++ dataset loader.

Supports loading original and manipulated face images from the FaceForensics++ dataset
with configurable compression quality and manipulation method selection.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


MANIPULATION_METHODS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
]

COMPRESSION_LEVELS = ["raw", "c23", "c40"]


class FaceForensicsDataset(Dataset):
    """Dataset class for FaceForensics++ face forgery detection."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        manipulation: str = "Deepfakes",
        compression: str = "c23",
        transform: Optional[A.Compose] = None,
        max_frames_per_video: int = 10,
    ):
        """
        Args:
            root_dir: Path to FaceForensics++ dataset root.
            split: One of 'train', 'val', 'test'.
            manipulation: Manipulation method to load.
            compression: Compression quality level.
            transform: Albumentations transform pipeline.
            max_frames_per_video: Max frames to sample per video sequence.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.manipulation = manipulation
        self.compression = compression
        self.transform = transform or self._default_transform()
        self.max_frames_per_video = max_frames_per_video

        assert manipulation in MANIPULATION_METHODS, (
            f"Unknown manipulation: {manipulation}. Choose from {MANIPULATION_METHODS}"
        )
        assert compression in COMPRESSION_LEVELS, (
            f"Unknown compression: {compression}. Choose from {COMPRESSION_LEVELS}"
        )

        self.samples = self._load_split()

    def _default_transform(self) -> A.Compose:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def _load_split(self) -> List[Tuple[str, int]]:
        """Load file paths and labels for the given split."""
        split_file = self.root_dir / "splits" / f"{self.split}.json"

        real_dir = self.root_dir / "original_sequences" / "youtube" / self.compression / "images"
        fake_dir = (
            self.root_dir / "manipulated_sequences" / self.manipulation
            / self.compression / "images"
        )

        samples = []

        if real_dir.exists():
            for video_dir in sorted(real_dir.iterdir()):
                if video_dir.is_dir():
                    frames = sorted(video_dir.glob("*.png"))[:self.max_frames_per_video]
                    samples.extend([(str(f), 0) for f in frames])

        if fake_dir.exists():
            for video_dir in sorted(fake_dir.iterdir()):
                if video_dir.is_dir():
                    frames = sorted(video_dir.glob("*.png"))[:self.max_frames_per_video]
                    samples.extend([(str(f), 1) for f in frames])

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image)
        image = transformed["image"]

        return image, label


def get_dataloader(
    root_dir: str,
    split: str = "train",
    manipulation: str = "Deepfakes",
    compression: str = "c23",
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[A.Compose] = None,
) -> DataLoader:
    """Create a DataLoader for FaceForensics++ data."""
    dataset = FaceForensicsDataset(
        root_dir=root_dir,
        split=split,
        manipulation=manipulation,
        compression=compression,
        transform=transform,
    )
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
