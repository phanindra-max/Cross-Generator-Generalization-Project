"""
Prepare FaceForensics++ FaceSwap frames from Kaggle dataset.

Downloads the FF++ c23 dataset via kagglehub, extracts frames from
the FaceSwap manipulation subset, and resizes to 128x128.
"""

import os
import random
from pathlib import Path
from typing import Optional, List

import cv2
from tqdm import tqdm


def download_faceforensics_kaggle() -> str:
    """
    Download FaceForensics++ c23 from Kaggle using kagglehub.

    Returns:
        Path to the downloaded dataset root.
    """
    import kagglehub

    print("Downloading FaceForensics++ c23 from Kaggle...")
    path = kagglehub.dataset_download("xdxd003/ff-c23")
    print(f"Dataset downloaded to: {path}")
    return path


def find_faceswap_dir(kaggle_root: str) -> Path:
    """
    Locate the FaceSwap images directory within the Kaggle download.
    The structure varies, so we search for it.
    """
    kaggle_path = Path(kaggle_root)

    candidates = [
        kaggle_path / "manipulated_sequences" / "FaceSwap" / "c23" / "images",
        kaggle_path / "FaceSwap" / "c23" / "images",
        kaggle_path / "c23" / "FaceSwap" / "images",
        kaggle_path / "FaceSwap" / "images",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: search recursively for a directory named "FaceSwap"
    for root, dirs, _ in os.walk(str(kaggle_path)):
        if "FaceSwap" in dirs:
            potential = Path(root) / "FaceSwap"
            # Look for images subdirectory
            for sub in potential.rglob("images"):
                if sub.is_dir() and any(sub.iterdir()):
                    return sub

    raise FileNotFoundError(
        f"Could not find FaceSwap images directory in {kaggle_root}. "
        f"Please check the dataset structure."
    )


def extract_frames(
    faceswap_dir: str,
    output_dir: str,
    num_videos: int = 500,
    frames_per_video: int = 4,
    resolution: int = 128,
    seed: int = 42,
):
    """
    Extract evenly-spaced frames from FaceSwap video directories.

    Args:
        faceswap_dir: Path to FaceSwap images directory (contains subdirs per video).
        output_dir: Directory to save extracted frames.
        num_videos: Number of video subdirectories to sample from.
        frames_per_video: Number of frames to extract per video.
        resolution: Target resolution for output images.
        seed: Random seed for video selection.
    """
    faceswap_path = Path(faceswap_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted([d for d in faceswap_path.iterdir() if d.is_dir()])
    print(f"Found {len(video_dirs)} video directories")

    random.seed(seed)
    if len(video_dirs) > num_videos:
        video_dirs = random.sample(video_dirs, num_videos)
    else:
        num_videos = len(video_dirs)
        print(f"Only {num_videos} videos available, using all of them")

    frame_count = 0
    for video_dir in tqdm(video_dirs, desc="Extracting frames"):
        frames = sorted([
            f for f in video_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])

        if len(frames) == 0:
            continue

        # Select evenly-spaced frames
        if len(frames) >= frames_per_video:
            indices = [int(i * (len(frames) - 1) / (frames_per_video - 1))
                       for i in range(frames_per_video)]
        else:
            indices = list(range(len(frames)))

        for idx in indices:
            img = cv2.imread(str(frames[idx]))
            if img is None:
                continue

            img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
            output_file = output_path / f"{frame_count:05d}.png"
            cv2.imwrite(str(output_file), img)
            frame_count += 1

    print(f"Done. Extracted {frame_count} frames to {output_path}")
    return frame_count


def prepare_faceforensics(
    output_dir: str = "data/faceswap",
    num_videos: int = 500,
    frames_per_video: int = 4,
    resolution: int = 128,
):
    """Full pipeline: download from Kaggle and extract frames."""
    kaggle_root = download_faceforensics_kaggle()
    faceswap_dir = find_faceswap_dir(kaggle_root)
    print(f"Found FaceSwap directory: {faceswap_dir}")

    num_extracted = extract_frames(
        faceswap_dir=str(faceswap_dir),
        output_dir=output_dir,
        num_videos=num_videos,
        frames_per_video=frames_per_video,
        resolution=resolution,
    )
    return num_extracted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare FaceForensics++ FaceSwap frames")
    parser.add_argument("--output-dir", type=str, default="data/faceswap")
    parser.add_argument("--num-videos", type=int, default=500)
    parser.add_argument("--frames-per-video", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()

    prepare_faceforensics(
        output_dir=args.output_dir,
        num_videos=args.num_videos,
        frames_per_video=args.frames_per_video,
        resolution=args.resolution,
    )
