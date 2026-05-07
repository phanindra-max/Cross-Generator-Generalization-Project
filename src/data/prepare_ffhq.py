"""
Download and prepare FFHQ thumbnail images (128x128) from HuggingFace.

The `nuwandaa/ffhq128` dataset is published as a single zip archive
(`thumbnails128x128.zip`) containing ~70k PNG thumbnails. Going through
the `datasets` library forces a full extraction + Arrow cache build over
all 70k files even when we only want a small subset, which is what made
the previous implementation take ~30+ minutes on Colab. This module
downloads the zip directly with `huggingface_hub` and extracts only the
first N entries.
"""

import io
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

REPO_ID = "nuwandaa/ffhq128"
ARCHIVE_FILENAME = "thumbnails128x128.zip"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def download_ffhq(
    output_dir: str,
    num_images: int = 2000,
    resolution: int = 128,
    max_workers: int = 8,
):
    """
    Download FFHQ thumbnails from HuggingFace and save as PNGs.

    Strategy: pull the source zip in one shot via `huggingface_hub`, then
    extract only the first `num_images` entries. Decoding and saving are
    dispatched to a thread pool. This avoids the `datasets` library's
    full-archive extraction + Arrow cache build, which dominated runtime.

    Args:
        output_dir: Directory to save the images.
        num_images: Number of images to download.
        resolution: Target resolution. FFHQ128 is already 128x128, so this
            is a no-op for the default; a different value triggers a resize.
        max_workers: Number of threads for parallel decode + save.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing = list(output_path.glob("*.png"))
    if len(existing) >= num_images:
        print(f"Already have {len(existing)} images in {output_path}, skipping download.")
        return len(existing)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    zip_path = _download_archive(token)
    entries = _select_entries(zip_path, num_images)

    print(f"Extracting {len(entries)} images to {output_path} ({max_workers} workers)...")

    # ZipFile is not thread-safe for concurrent reads, so each worker thread
    # opens its own handle. The zip is a local file at this point, so opening
    # a handle is cheap (just reads the central directory).
    thread_local = threading.local()

    def _zf() -> zipfile.ZipFile:
        zf = getattr(thread_local, "zf", None)
        if zf is None:
            zf = zipfile.ZipFile(zip_path)
            thread_local.zf = zf
        return zf

    def _save_one(idx: int, member_name: str) -> int:
        out_file = output_path / f"{idx:05d}.png"
        if out_file.exists():
            return idx
        with _zf().open(member_name) as src:
            data = src.read()
        img = _prepare_image(data, resolution)
        img.save(out_file, compress_level=1)
        return idx

    saved = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_save_one, i, name) for i, name in enumerate(entries)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Saving FFHQ images"):
            fut.result()
            saved += 1

    print(f"Done. Saved {saved} images to {output_path}")
    return saved


def _download_archive(token):
    """Fetch the FFHQ128 zip via huggingface_hub (cached on subsequent runs)."""
    from huggingface_hub import hf_hub_download

    print(
        f"Downloading {ARCHIVE_FILENAME} from {REPO_ID} "
        f"(auth={'yes' if token else 'no'})..."
    )
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=ARCHIVE_FILENAME,
        repo_type="dataset",
        token=token,
    )
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Archive ready at {zip_path} ({size_mb:.1f} MB)")
    return zip_path


def _select_entries(zip_path: str, num_images: int):
    """Return the first `num_images` image entries from the archive."""
    with zipfile.ZipFile(zip_path) as zf:
        image_entries = sorted(
            n for n in zf.namelist()
            if n.lower().endswith(IMAGE_SUFFIXES) and not n.endswith("/")
        )
    if not image_entries:
        raise RuntimeError(f"No image entries found in {zip_path}")
    if len(image_entries) < num_images:
        print(
            f"Warning: archive only contains {len(image_entries)} images, "
            f"requested {num_images}. Using all available."
        )
        num_images = len(image_entries)
    return image_entries[:num_images]


def _prepare_image(data: bytes, resolution: int):
    """Decode raw image bytes and conform to the requested RGB resolution."""
    from PIL import Image as PILImage

    img = PILImage.open(io.BytesIO(data))
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
