"""
Frequency-domain visualization for deepfake analysis.

Generates DFT magnitude spectrum plots to reveal spectral artifacts
left by different image generation/manipulation methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Optional, List, Tuple


def compute_magnitude_spectrum(image: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """
    Compute the 2D DFT magnitude spectrum of a grayscale image.

    Args:
        image: Grayscale image (H, W) as uint8 or float.
        log_scale: Apply log(1 + magnitude) for visualization.

    Returns:
        Centered magnitude spectrum.
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    if log_scale:
        magnitude = np.log1p(magnitude)

    return magnitude


def compute_azimuthal_average(spectrum: np.ndarray) -> np.ndarray:
    """
    Compute radially averaged power spectrum (azimuthal average).
    Useful for comparing frequency content independent of orientation.
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = min(cy, cx)
    radial_mean = np.zeros(max_r)

    for radius in range(max_r):
        mask = r == radius
        if mask.any():
            radial_mean[radius] = spectrum[mask].mean()

    return radial_mean


def plot_spectrum_comparison(
    images: List[Tuple[np.ndarray, str]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4),
) -> plt.Figure:
    """
    Plot magnitude spectra side-by-side for multiple images.

    Args:
        images: List of (image, label) tuples.
        save_path: Optional path to save the figure.
        figsize: Figure size.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (img, label) in zip(axes, images):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        spectrum = compute_magnitude_spectrum(img)
        ax.imshow(spectrum, cmap="magma")
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_radial_profiles(
    images: List[Tuple[np.ndarray, str]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Plot radially-averaged power spectra for comparison across generators.

    Args:
        images: List of (image, label) tuples.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for img, label in images:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        spectrum = compute_magnitude_spectrum(img, log_scale=False)
        radial = compute_azimuthal_average(spectrum)
        freqs = np.arange(len(radial)) / len(radial)

        ax.plot(freqs, np.log1p(radial), label=label, linewidth=1.5)

    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Log Magnitude")
    ax.set_title("Radially-Averaged Power Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def batch_spectrum_analysis(
    image_dir: str,
    n_samples: int = 50,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute average magnitude spectrum over multiple images from a directory.
    Reveals systematic frequency artifacts of a generator.
    """
    image_dir = Path(image_dir)
    extensions = {".png", ".jpg", ".jpeg"}
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in extensions]
    image_files = image_files[:n_samples]

    spectra = []
    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (256, 256))
            spectrum = compute_magnitude_spectrum(img)
            spectra.append(spectrum)

    avg_spectrum = np.mean(spectra, axis=0)

    if save_path:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(avg_spectrum, cmap="magma")
        ax.set_title(f"Average Spectrum ({len(spectra)} images)")
        ax.axis("off")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return avg_spectrum
