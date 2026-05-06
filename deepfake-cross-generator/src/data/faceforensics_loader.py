"""
FaceForensics++ dataset loader.

Supports loading from CSV manifests (produced by create_splits.py) or
from the raw FaceForensics++ directory structure.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd
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
        root_dir: str = "",
        split: str = "train",
        manipulation: str = "FaceSwap",
        compression: str = "c23",
        transform: Optional[A.Compose] = None,
        max_frames_per_video: int = 10,
        csv_path: Optional[str] = None,
    ):
        """
        Args:
            root_dir: Path to FaceForensics++ dataset root (used for directory mode).
            split: One of 'train', 'val', 'test'.
            manipulation: Manipulation method to load (directory mode only).
            compression: Compression quality level (directory mode only).
            transform: Albumentations transform pipeline.
            max_frames_per_video: Max frames to sample per video (directory mode only).
            csv_path: Path to a CSV manifest file. If provided, ignores directory-based loading.
        """
        self.root_dir = Path(root_dir) if root_dir else None
        self.split = split
        self.manipulation = manipulation
        self.compression = compression
        self.transform = transform or self._default_transform()
        self.max_frames_per_video = max_frames_per_video
        self.csv_path = csv_path

        if csv_path:
            self.samples = self._load_from_csv(csv_path)
        else:
            assert manipulation in MANIPULATION_METHODS, (
                f"Unknown manipulation: {manipulation}. Choose from {MANIPULATION_METHODS}"
            )
            assert compression in COMPRESSION_LEVELS, (
                f"Unknown compression: {compression}. Choose from {COMPRESSION_LEVELS}"
            )
            self.samples = self._load_from_directory()

    def _default_transform(self) -> A.Compose:
        return A.Compose([
            A.Resize(128, 128),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def _load_from_csv(self, csv_path: str) -> List[Tuple[str, int]]:
        """Load samples from a CSV manifest file."""
        df = pd.read_csv(csv_path)
        samples = [(row["path"], int(row["label"])) for _, row in df.iterrows()]
        return samples

    def _load_from_directory(self) -> List[Tuple[str, int]]:
        """Load file paths and labels by walking the FF++ directory structure."""
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
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image)
        image = transformed["image"]

        return image, label


def get_dataloader(
    root_dir: str = "",
    split: str = "train",
    manipulation: str = "FaceSwap",
    compression: str = "c23",
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[A.Compose] = None,
    csv_path: Optional[str] = None,
) -> DataLoader:
    """
    Create a DataLoader for FaceForensics++ data.

    Can load from a CSV manifest (preferred) or from the raw directory structure.
    """
    dataset = FaceForensicsDataset(
        root_dir=root_dir,
        split=split,
        manipulation=manipulation,
        compression=compression,
        transform=transform,
        csv_path=csv_path,
    )
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
