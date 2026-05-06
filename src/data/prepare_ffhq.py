"""
Download and prepare FFHQ thumbnail images (128x128) from HuggingFace.

Downloads a subset of the FFHQ dataset to serve as real face images
for the deepfake detection pipeline.
"""

from pathlib import Path

from tqdm import tqdm


def download_ffhq(
    output_dir: str,
    num_images: int = 2000,
    resolution: int = 128,
):
    """
    Download FFHQ thumbnails from HuggingFace and save as PNGs.

    Args:
        output_dir: Directory to save the images.
        num_images: Number of images to download.
        resolution: Target resolution (images are already 128x128 from this dataset).
    """
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing = list(output_path.glob("*.png"))
    if len(existing) >= num_images:
        print(f"Already have {len(existing)} images in {output_path}, skipping download.")
        return len(existing)

    print("Streaming FFHQ thumbnails dataset from HuggingFace...")
    ds = load_dataset("nuwandaa/ffhq128", split="train", streaming=True)

    print(f"Saving {num_images} images to {output_path}...")

    saved = 0
    with tqdm(total=num_images, desc="Saving FFHQ images") as progress:
        for idx, example in enumerate(ds):
            if idx >= num_images:
                break

            img_path = output_path / f"{idx:05d}.png"
            if img_path.exists():
                saved += 1
                progress.update(1)
                continue

            img = _prepare_image(example, resolution)
            img.save(img_path, compress_level=1)
            saved += 1
            progress.update(1)

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
