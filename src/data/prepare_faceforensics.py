"""
Prepare FaceForensics++ FaceSwap frames from Kaggle dataset.

Downloads the FF++ c23 dataset via kagglehub, extracts frames from
the FaceSwap manipulation subset, and resizes to 128x128.
"""

import os
import random
from pathlib import Path

import cv2
from tqdm import tqdm

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


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


def _list_image_frames(path: Path):
    return sorted([f for f in path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES])


def _list_video_files(path: Path):
    return sorted([f for f in path.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_SUFFIXES])


def _has_extractable_media(path: Path) -> bool:
    if not path.is_dir():
        return False
    if _list_image_frames(path) or _list_video_files(path):
        return True
    return any(child.is_dir() and _list_image_frames(child) for child in path.iterdir())


def find_faceswap_dir(kaggle_root: str) -> Path:
    """
    Locate the FaceSwap media directory within the Kaggle download.
    The structure varies, so we search for it.
    """
    kaggle_path = Path(kaggle_root)

    candidates = [
        kaggle_path / "manipulated_sequences" / "FaceSwap" / "c23" / "images",
        kaggle_path / "manipulated_sequences" / "FaceSwap" / "c23" / "videos",
        kaggle_path / "manipulated_sequences" / "FaceSwap" / "images",
        kaggle_path / "manipulated_sequences" / "FaceSwap" / "videos",
        kaggle_path / "FaceSwap" / "c23" / "images",
        kaggle_path / "FaceSwap" / "c23" / "videos",
        kaggle_path / "c23" / "FaceSwap" / "images",
        kaggle_path / "c23" / "FaceSwap" / "videos",
        kaggle_path / "FaceSwap" / "images",
        kaggle_path / "FaceSwap" / "videos",
    ]

    for candidate in candidates:
        if _has_extractable_media(candidate):
            return candidate

    # Fallback: search case-insensitively under any FaceSwap folder.
    faceswap_roots = [
        path for path in kaggle_path.rglob("*")
        if path.is_dir() and path.name.lower() == "faceswap"
    ]
    for faceswap_root in faceswap_roots:
        if _has_extractable_media(faceswap_root):
            return faceswap_root

        for sub in faceswap_root.rglob("*"):
            if sub.is_dir() and _has_extractable_media(sub):
                return sub

    available = sorted(str(path.relative_to(kaggle_path)) for path in faceswap_roots[:10])
    hint = f" FaceSwap-like directories found: {available}." if available else ""

    raise FileNotFoundError(
        f"Could not find FaceSwap images or videos in {kaggle_root}."
        f"{hint} Please inspect the dataset structure with: "
        f"!find {kaggle_root} -maxdepth 4 -type d | head -50"
    )


def _evenly_spaced_indices(total_frames: int, frames_per_video: int):
    if total_frames <= 0:
        return []
    if frames_per_video <= 1 or total_frames == 1:
        return [0]
    if total_frames >= frames_per_video:
        return [
            int(i * (total_frames - 1) / (frames_per_video - 1))
            for i in range(frames_per_video)
        ]
    return list(range(total_frames))


def _write_resized_frame(img, output_path: Path, frame_count: int, resolution: int) -> int:
    if img is None:
        return frame_count

    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    output_file = output_path / f"{frame_count:05d}.png"
    cv2.imwrite(str(output_file), img)
    return frame_count + 1


def _extract_from_image_dir(video_dir: Path, output_path: Path, frame_count: int, frames_per_video: int, resolution: int) -> int:
    frames = _list_image_frames(video_dir)

    for idx in _evenly_spaced_indices(len(frames), frames_per_video):
        img = cv2.imread(str(frames[idx]))
        frame_count = _write_resized_frame(img, output_path, frame_count, resolution)

    return frame_count


def _extract_from_video_file(video_file: Path, output_path: Path, frame_count: int, frames_per_video: int, resolution: int) -> int:
    cap = cv2.VideoCapture(str(video_file))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx in _evenly_spaced_indices(total_frames, frames_per_video):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frame_count = _write_resized_frame(frame, output_path, frame_count, resolution)

    cap.release()
    return frame_count


def extract_frames(
    faceswap_dir: str,
    output_dir: str,
    num_videos: int = 500,
    frames_per_video: int = 4,
    resolution: int = 128,
    seed: int = 42,
):
    """
    Extract evenly-spaced frames from FaceSwap image directories or video files.

    Args:
        faceswap_dir: Path to FaceSwap media directory.
        output_dir: Directory to save extracted frames.
        num_videos: Number of video sources to sample from.
        frames_per_video: Number of frames to extract per video.
        resolution: Target resolution for output images.
        seed: Random seed for video selection.
    """
    faceswap_path = Path(faceswap_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_dirs = sorted([
        d for d in faceswap_path.iterdir()
        if d.is_dir() and _list_image_frames(d)
    ])
    direct_images = _list_image_frames(faceswap_path)
    video_files = _list_video_files(faceswap_path)

    if image_dirs:
        sources = image_dirs
        source_type = "image directories"
    elif video_files:
        sources = video_files
        source_type = "video files"
    elif direct_images:
        sources = [faceswap_path]
        source_type = "flat image directory"
    else:
        raise FileNotFoundError(f"No image frames or videos found in {faceswap_path}")

    print(f"Found {len(sources)} FaceSwap {source_type}")

    random.seed(seed)
    if len(sources) > num_videos:
        sources = random.sample(sources, num_videos)
    else:
        num_videos = len(sources)
        print(f"Only {num_videos} videos available, using all of them")

    frame_count = 0
    for source in tqdm(sources, desc="Extracting frames"):
        if source.is_file():
            frame_count = _extract_from_video_file(
                source, output_path, frame_count, frames_per_video, resolution
            )
        else:
            frame_count = _extract_from_image_dir(
                source, output_path, frame_count, frames_per_video, resolution
            )

    print(f"Done. Extracted {frame_count} frames to {output_path}")
    return frame_count


def prepare_faceforensics(
    output_dir: str = "data/faceswap",
    num_videos: int = 500,
    frames_per_video: int = 4,
    resolution: int = 128,
):
    """Full pipeline: download from Kaggle and extract frames."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    expected_frames = num_videos * frames_per_video
    existing = list(output_path.glob("*.png"))
    if len(existing) >= expected_frames:
        print(f"Already have {len(existing)} FaceSwap frames in {output_path}, skipping download and extraction.")
        return len(existing)

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
