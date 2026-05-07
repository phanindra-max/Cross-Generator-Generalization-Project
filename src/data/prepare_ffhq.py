"""
Download and prepare FFHQ thumbnail images (128x128) from HuggingFace.

Downloads a subset of the FFHQ dataset to serve as real face images
for the deepfake detection pipeline.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def download_ffhq(
    output_dir: str,
    num_images: int = 2000,
    resolution: int = 128,
    max_workers: int = 8,
):
    """
    Download FFHQ thumbnails from HuggingFace and save as PNGs.

    Uses non-streaming load with a row slice so the underlying parquet
    shards download in parallel via huggingface_hub. PNG encode + write
    are dispatched to a thread pool.

    Args:
        output_dir: Directory to save the images.
        num_images: Number of images to download.
        resolution: Target resolution (images are already 128x128 from this dataset).
        max_workers: Number of threads for parallel image saves.
    """
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing = list(output_path.glob("*.png"))
    if len(existing) >= num_images:
        print(f"Already have {len(existing)} images in {output_path}, skipping download.")
        return len(existing)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(
        f"Downloading FFHQ thumbnails (first {num_images}) from HuggingFace "
        f"(auth={'yes' if token else 'no'})..."
    )

    ds = load_dataset(
        "nuwandaa/ffhq128",
        split=f"train[:{num_images}]",
        token=token,
    )

    print(f"Saving {num_images} images to {output_path} ({max_workers} workers)...")

    def _save_one(idx: int) -> int:
        img_path = output_path / f"{idx:05d}.png"
        if img_path.exists():
            return idx
        img = _prepare_image(ds[idx], resolution)
        img.save(img_path, compress_level=1)
        return idx

    saved = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_save_one, i) for i in range(num_images)]
        for fut in tqdm(as_completed(futures), total=num_images, desc="Saving FFHQ images"):
            fut.result()
            saved += 1

    print(f"Done. Saved {saved} images to {output_path}")
    return saved


def _prepare_image(example, resolution: int):
    """Convert a HuggingFace image example to the requested RGB resolution."""
    from PIL import Image as PILImage

    img = example["image"]
    if not isinstance(img, PILImage.Image):
        img = PILImage.fromarray(img)
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), PILImage.LANCZOS)
    return img.convert("RGB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FFHQ thumbnails")
    parser.add_argument("--output-dir", type=str, default="data/real")
    parser.add_argument("--num-images", type=int, default=2000)
    args = parser.parse_args()

    download_ffhq(output_dir=args.output_dir, num_images=args.num_images)
