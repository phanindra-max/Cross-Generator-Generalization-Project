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

    print("Loading FFHQ thumbnails dataset from HuggingFace...")
    ds = load_dataset("nuwandaa/ffhq128", split="train", keep_in_memory=True)

    num_images = min(num_images, len(ds))
    subset = ds.select(range(num_images))

    print(f"Saving {num_images} images to {output_path}...")

    out_dir_str = str(output_path)
    res = resolution

    def save_example(example, idx):
        from PIL import Image as PILImage

        img = example["image"]
        if not isinstance(img, PILImage.Image):
            img = PILImage.fromarray(img)
        if img.size != (res, res):
            img = img.resize((res, res), PILImage.LANCZOS)
        img.convert("RGB").save(f"{out_dir_str}/{idx:05d}.png", compress_level=1)
        return example

    subset.map(save_example, with_indices=True, num_proc=4, desc="Saving FFHQ images")

    saved = len(list(output_path.glob("*.png")))
    print(f"Done. Saved {saved} images to {output_path}")
    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FFHQ thumbnails")
    parser.add_argument("--output-dir", type=str, default="data/real")
    parser.add_argument("--num-images", type=int, default=2000)
    args = parser.parse_args()

    download_ffhq(output_dir=args.output_dir, num_images=args.num_images)
