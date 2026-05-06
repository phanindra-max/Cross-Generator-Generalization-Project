"""
Download and prepare FFHQ thumbnail images (128x128) from HuggingFace.

Downloads a subset of the FFHQ dataset to serve as real face images
for the deepfake detection pipeline.
"""

import os
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm


def download_ffhq(
    output_dir: str,
    num_images: int = 2000,
    resolution: int = 128,
):
    """
    Download FFHQ thumbnails from HuggingFace and save as 128x128 PNGs.

    Args:
        output_dir: Directory to save the images.
        num_images: Number of images to download.
        resolution: Target resolution (images are already 128x128 from this dataset).
    """
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading FFHQ thumbnails dataset from HuggingFace...")
    ds = load_dataset("nuwandaa/ffhq128", split="train")

    num_images = min(num_images, len(ds))
    print(f"Saving {num_images} images to {output_path}...")

    for i in tqdm(range(num_images), desc="Saving FFHQ images"):
        img = ds[i]["image"]

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution), Image.LANCZOS)

        img = img.convert("RGB")
        img.save(output_path / f"{i:05d}.png")

    print(f"Done. Saved {num_images} images to {output_path}")
    return num_images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FFHQ thumbnails")
    parser.add_argument("--output-dir", type=str, default="data/real")
    parser.add_argument("--num-images", type=int, default=2000)
    args = parser.parse_args()

    download_ffhq(output_dir=args.output_dir, num_images=args.num_images)
