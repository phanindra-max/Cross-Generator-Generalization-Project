"""
Download and prepare FFHQ thumbnail images (128x128) from HuggingFace.

Downloads a subset of the FFHQ dataset to serve as real face images
for the deepfake detection pipeline.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def _save_image(args):
    """Save a single image to disk (used by thread pool)."""
    img, filepath, resolution = args
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)
    img = img.convert("RGB")
    img.save(filepath, compress_level=1)


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

    existing = set(output_path.glob("*.png"))
    if len(existing) >= num_images:
        print(f"Already have {len(existing)} images in {output_path}, skipping download.")
        return len(existing)

    print("Loading FFHQ thumbnails dataset from HuggingFace...")
    ds = load_dataset("nuwandaa/ffhq128", split="train", streaming=False)

    num_images = min(num_images, len(ds))
    subset = ds.select(range(num_images))

    print(f"Saving {num_images} images to {output_path}...")

    batch_size = 64
    num_workers = min(8, os.cpu_count() or 4)
    saved = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch_start in tqdm(range(0, num_images, batch_size), desc="Saving FFHQ images"):
            batch_end = min(batch_start + batch_size, num_images)
            batch = subset[batch_start:batch_end]
            images = batch["image"]

            tasks = [
                (img, output_path / f"{batch_start + j:05d}.png", resolution)
                for j, img in enumerate(images)
            ]
            list(executor.map(_save_image, tasks))
            saved += len(tasks)

    print(f"Done. Saved {saved} images to {output_path}")
    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FFHQ thumbnails")
    parser.add_argument("--output-dir", type=str, default="data/real")
    parser.add_argument("--num-images", type=int, default=2000)
    args = parser.parse_args()

    download_ffhq(output_dir=args.output_dir, num_images=args.num_images)
