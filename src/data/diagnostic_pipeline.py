"""
Diagnostic A' — unify the data-prep pipeline to test the resize-kernel leak.

Hypothesis being tested:
    The original prep pipeline introduces a label-correlated confound. FFHQ
    (real, label=0) is delivered pre-rendered at 128x128 via PIL. Stable
    Diffusion (OOD fake, label=1) is rendered at 512x512 then downsampled
    via PIL.LANCZOS. FaceSwap (in-dist fake, label=1) is downsampled from
    the Kaggle source frames via cv2.INTER_AREA. The cv2.INTER_AREA
    fingerprint is therefore 100% correlated with the in-dist fake label
    at training time, and ConstrainedCNN — which by design suppresses
    content and amplifies high-frequency residuals — could be locking onto
    that fingerprint instead of any real manipulation cue.

Diagnostic, in order:
    1. bit_identity_check() — confirm cv2.imread vs PIL.open produce
       byte-equivalent decoded pixels. If yes, the cv2-vs-PIL *encoder*
       split cannot be the leak (the model never sees encoder bytes).
    2. demo_resize_kernel_diff() — confirm cv2.INTER_AREA and PIL.LANCZOS
       on the *same* source produce different pixels. Establishes that
       the resize asymmetry exists at the pixel level.
    3. rerender_faceswap_pil() — re-extract FaceSwap from the Kaggle
       source through (PIL.open / cv2 video decode) → PIL.LANCZOS resize
       → PIL save. Output goes to a separate directory so the original
       cv2-rendered data is preserved for comparison.

After step 3, regenerate splits pointing at the unified FaceSwap dir
(but the *original* FFHQ + diffusion dirs — they already use PIL
pipelines, so this is a single-variable change), then retrain.

This module deliberately does NOT modify prepare_faceforensics.py. The
canonical prep stays intact so the original confounded data remains
reproducible for the writeup.
"""

import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Reuse helpers from the canonical prep — same source-finding logic,
# same seed/sampling contract, same per-source frame indexing.
from src.data.prepare_faceforensics import (
    VIDEO_SUFFIXES,
    _evenly_spaced_indices,
    _list_image_frames,
    _list_video_files,
    download_faceforensics_kaggle,
    find_faceswap_dir,
)


# ---------------------------------------------------------------------------
# Step 1: bit-identity check — does the cv2-vs-PIL encoder split leak?
# ---------------------------------------------------------------------------

def bit_identity_check(image_paths: Iterable[str]) -> List[Tuple[str, int]]:
    """
    For each PNG, decode via cv2 (BGR -> RGB) and via PIL (RGB), and report
    the max absolute pixel difference.

    Returns a list of (path, max_abs_diff). max_abs_diff == 0 means the
    decoders agree to the bit on this file, which proves the encoder
    split (cv2.imwrite vs PIL.save) cannot leak through to the model.
    """
    import cv2

    results = []
    for p in image_paths:
        cv_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if cv_bgr is None:
            raise FileNotFoundError(f"cv2 could not decode: {p}")
        cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        pil_rgb = np.array(Image.open(p).convert("RGB"))
        diff = int(np.abs(cv_rgb.astype(np.int32) - pil_rgb.astype(np.int32)).max())
        results.append((p, diff))
    return results


# ---------------------------------------------------------------------------
# Step 2: prove the resize-kernel asymmetry produces visibly different pixels
# ---------------------------------------------------------------------------

def demo_resize_kernel_diff(source_image_path: str, resolution: int = 128) -> dict:
    """
    Take a single high-res source image, resize it two ways, return a dict
    of pixel-level differences. Produces evidence that the cv2.INTER_AREA
    vs PIL.LANCZOS choice is not just metadata — it changes pixels.

    Returns:
        {
          "src_size": (H, W),
          "max_abs_diff_uint8": int,   # in [0, 255]
          "mean_abs_diff_uint8": float,
          "cv2_path": Path,            # written next to source for inspection
          "pil_path": Path,
        }
    """
    import cv2

    src = Path(source_image_path)
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cv2 could not decode: {src}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    cv2_resized = cv2.resize(bgr, (resolution, resolution), interpolation=cv2.INTER_AREA)
    cv2_resized_rgb = cv2.cvtColor(cv2_resized, cv2.COLOR_BGR2RGB)
    pil_resized = np.array(
        Image.fromarray(rgb).resize((resolution, resolution), Image.LANCZOS)
    )

    diff = np.abs(cv2_resized_rgb.astype(np.int32) - pil_resized.astype(np.int32))

    cv2_path = src.parent / f"{src.stem}_cv2_INTER_AREA.png"
    pil_path = src.parent / f"{src.stem}_pil_LANCZOS.png"
    Image.fromarray(cv2_resized_rgb).save(cv2_path, compress_level=1)
    Image.fromarray(pil_resized).save(pil_path, compress_level=1)

    return {
        "src_size": (h, w),
        "max_abs_diff_uint8": int(diff.max()),
        "mean_abs_diff_uint8": float(diff.mean()),
        "cv2_path": cv2_path,
        "pil_path": pil_path,
    }


# ---------------------------------------------------------------------------
# Step 3: re-extract FaceSwap from the Kaggle source through a PIL pipeline
# ---------------------------------------------------------------------------

def _rerender_one_source_pil(args):
    """
    Worker mirroring prepare_faceforensics._extract_one_source but using
    PIL.LANCZOS for resize and PIL.save for output.

    Video frames are still decoded via cv2.VideoCapture (BGR array), then
    handed to PIL for the resize. Decode is lossless; only the resize
    interpolation differs from the original prep.
    """
    src_idx, source_str, output_dir_str, frames_per_video, resolution = args
    source = Path(source_str)
    output_path = Path(output_dir_str)

    written = 0
    if source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
        import cv2
        cap = cv2.VideoCapture(str(source))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_idx, idx in enumerate(_evenly_spaced_indices(total, frames_per_video)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((resolution, resolution), Image.LANCZOS)
            out_file = output_path / f"{src_idx:05d}_{frame_idx:02d}.png"
            img.save(out_file, compress_level=1)
            written += 1
        cap.release()
    elif source.is_dir():
        frames = _list_image_frames(source)
        for frame_idx, idx in enumerate(_evenly_spaced_indices(len(frames), frames_per_video)):
            img = Image.open(frames[idx]).convert("RGB")
            img = img.resize((resolution, resolution), Image.LANCZOS)
            out_file = output_path / f"{src_idx:05d}_{frame_idx:02d}.png"
            img.save(out_file, compress_level=1)
            written += 1

    return written


def rerender_faceswap_pil(
    output_dir: str = "data_unified/faceswap",
    num_videos: int = 500,
    frames_per_video: int = 4,
    resolution: int = 128,
    seed: int = 42,
    num_workers: Optional[int] = None,
    kaggle_root: Optional[str] = None,
) -> int:
    """
    Re-extract FaceSwap frames from the Kaggle source using a PIL-only
    resize + save path. Same source set as the original prep (same seed,
    same num_videos, same frames_per_video) so each output frame
    corresponds 1:1 to a frame in the cv2-rendered data/faceswap/ dir.

    The cv2-rendered original at data/faceswap/ is left untouched.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    expected = num_videos * frames_per_video
    existing = list(output_path.glob("*.png"))
    if len(existing) >= expected:
        print(f"Already have {len(existing)} frames in {output_path}, skipping.")
        return len(existing)

    # Locate the FaceSwap directory inside the Kaggle download. If the
    # caller didn't pass kaggle_root, fall back to running the canonical
    # download (which is cached on subsequent calls).
    if kaggle_root is None:
        kaggle_root = download_faceforensics_kaggle()
    faceswap_dir = find_faceswap_dir(kaggle_root)
    print(f"FaceSwap source: {faceswap_dir}")

    # Mirror the source-listing logic from prepare_faceforensics.extract_frames
    # so the seeded sample and per-source iteration order match exactly.
    image_dirs = sorted([
        d for d in faceswap_dir.iterdir()
        if d.is_dir() and _list_image_frames(d)
    ])
    direct_images = _list_image_frames(faceswap_dir)
    video_files = _list_video_files(faceswap_dir)

    if image_dirs:
        sources = image_dirs
    elif video_files:
        sources = video_files
    elif direct_images:
        sources = [faceswap_dir]
    else:
        raise FileNotFoundError(f"No frames or videos found in {faceswap_dir}")

    random.seed(seed)
    if len(sources) > num_videos:
        sources = random.sample(sources, num_videos)
    else:
        num_videos = len(sources)

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 4) - 1)

    args_list = [
        (src_idx, str(source), str(output_path), frames_per_video, resolution)
        for src_idx, source in enumerate(sources)
    ]

    print(f"Re-rendering {len(args_list)} sources with PIL.LANCZOS ({num_workers} workers)...")

    total_written = 0
    with mp.Pool(processes=num_workers) as pool:
        for written in tqdm(
            pool.imap_unordered(_rerender_one_source_pil, args_list),
            total=len(args_list),
            desc="PIL re-render",
        ):
            total_written += written

    print(f"Done. Wrote {total_written} frames to {output_path}")
    return total_written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnostic A': re-extract FaceSwap through a PIL pipeline."
    )
    parser.add_argument("--output-dir", type=str, default="data_unified/faceswap")
    parser.add_argument("--num-videos", type=int, default=500)
    parser.add_argument("--frames-per-video", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    rerender_faceswap_pil(
        output_dir=args.output_dir,
        num_videos=args.num_videos,
        frames_per_video=args.frames_per_video,
        resolution=args.resolution,
        seed=args.seed,
        num_workers=args.num_workers,
    )
