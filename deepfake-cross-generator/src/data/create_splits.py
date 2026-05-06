"""
Create reproducible train/test split CSV manifests.

Splits:
  - train.csv: 1600 real + 1600 FaceSwap
  - test_indist.csv: 400 real + 400 FaceSwap
  - test_ood.csv: 400 real (held-out) + 1000 diffusion fakes
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def collect_images(directory: str) -> List[str]:
    """Collect all image file paths from a directory."""
    extensions = {".png", ".jpg", ".jpeg"}
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    images = sorted([
        str(f.relative_to(dir_path.parent.parent))
        for f in dir_path.iterdir()
        if f.suffix.lower() in extensions
    ])
    return images


def create_splits(
    real_dir: str = "data/real",
    faceswap_dir: str = "data/faceswap",
    diffusion_dir: str = "data/diffusion_sd15",
    output_dir: str = "data/splits",
    seed: int = 42,
):
    """
    Create train/test split CSVs with deterministic randomization.

    Split allocation:
      - Real images: 1600 train + 400 in-dist test + 400 OOD test = 2400 total
        (if only 2000 available, use 1600 train + 400 test, share for OOD)
      - FaceSwap: 1600 train + 400 in-dist test = 2000 total
      - Diffusion: 1000 all go to OOD test
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    # Collect all image paths
    real_images = collect_images(real_dir)
    faceswap_images = collect_images(faceswap_dir)
    diffusion_images = collect_images(diffusion_dir)

    print(f"Found {len(real_images)} real images")
    print(f"Found {len(faceswap_images)} FaceSwap images")
    print(f"Found {len(diffusion_images)} diffusion images")

    # Shuffle with fixed seed
    random.shuffle(real_images)
    random.shuffle(faceswap_images)

    # Allocate real images
    real_train = real_images[:1600]
    real_test_indist = real_images[1600:2000]
    # For OOD test, reuse a held-out portion or extend if we have enough
    if len(real_images) >= 2400:
        real_test_ood = real_images[2000:2400]
    else:
        # Reuse the in-dist test reals for OOD test (same real baseline)
        real_test_ood = real_test_indist

    # Allocate FaceSwap
    faceswap_train = faceswap_images[:1600]
    faceswap_test = faceswap_images[1600:2000]

    # Build DataFrames
    train_records = []
    for path in real_train:
        train_records.append({"path": path, "label": 0, "split": "train", "source": "ffhq"})
    for path in faceswap_train:
        train_records.append({"path": path, "label": 1, "split": "train", "source": "faceswap_c23"})

    test_indist_records = []
    for path in real_test_indist:
        test_indist_records.append({"path": path, "label": 0, "split": "test_indist", "source": "ffhq"})
    for path in faceswap_test:
        test_indist_records.append({"path": path, "label": 1, "split": "test_indist", "source": "faceswap_c23"})

    test_ood_records = []
    for path in real_test_ood:
        test_ood_records.append({"path": path, "label": 0, "split": "test_ood", "source": "ffhq"})
    for path in diffusion_images:
        test_ood_records.append({"path": path, "label": 1, "split": "test_ood", "source": "stable_diffusion_v1_5"})

    # Save CSVs
    train_df = pd.DataFrame(train_records)
    test_indist_df = pd.DataFrame(test_indist_records)
    test_ood_df = pd.DataFrame(test_ood_records)

    train_df.to_csv(output_path / "train.csv", index=False)
    test_indist_df.to_csv(output_path / "test_indist.csv", index=False)
    test_ood_df.to_csv(output_path / "test_ood.csv", index=False)

    print(f"\nSplit summary:")
    print(f"  train.csv:       {len(train_df)} samples ({train_df['label'].value_counts().to_dict()})")
    print(f"  test_indist.csv: {len(test_indist_df)} samples ({test_indist_df['label'].value_counts().to_dict()})")
    print(f"  test_ood.csv:    {len(test_ood_df)} samples ({test_ood_df['label'].value_counts().to_dict()})")

    return train_df, test_indist_df, test_ood_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create data split manifests")
    parser.add_argument("--real-dir", type=str, default="data/real")
    parser.add_argument("--faceswap-dir", type=str, default="data/faceswap")
    parser.add_argument("--diffusion-dir", type=str, default="data/diffusion_sd15")
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_splits(
        real_dir=args.real_dir,
        faceswap_dir=args.faceswap_dir,
        diffusion_dir=args.diffusion_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
