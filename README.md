# Deepfake Cross-Generator Detection

## Overview

This project investigates **cross-generator generalization** in deepfake detection — specifically, whether forensic classifiers trained on one generation method (e.g., FaceSwap) can detect fakes produced by another (e.g., Stable Diffusion).

We replicate key findings from constrained-CNN forensic detectors (Bayar & Stamm, 2018) and evaluate their transfer performance across manipulation types using the FaceForensics++ benchmark and modern diffusion-based generators.

## Project Structure

```
deepfake-cross-generator/
├── README.md                    # This file
├── requirements.txt
├── notebooks/
│   ├── 01_frequency_analysis.ipynb    # Section 1: Frequency-domain analysis
│   └── 02_cross_generator_eval.ipynb  # Section 3: Cross-generator results
├── src/
│   ├── data/
│   │   ├── faceforensics_loader.py    # FaceForensics++ dataset utilities
│   │   └── diffusion_loader.py        # Diffusion-generated image loader
│   ├── models/
│   │   ├── constrained_cnn.py         # Bayar & Stamm constrained conv layer
│   │   └── baseline_resnet.py         # ResNet baseline for comparison
│   ├── training/
│   │   └── train.py                   # Training loop and config
│   ├── evaluation/
│   │   └── stratified_metrics.py      # Per-generator accuracy, AUC, EER
│   └── analysis/
│       └── frequency_visualization.py # DFT magnitude spectrum plots
├── results/
│   ├── figures/                        # Saved plots
│   └── metrics/                        # Saved metric CSVs
└── docs/
    └── notes.md                        # Reading notes and references
```

## Key Research Questions

1. **Frequency fingerprints**: Do different generators leave distinguishable spectral artifacts?
2. **Constrained convolutions**: Does the Bayar & Stamm preprocessing layer improve cross-generator robustness?
3. **Transfer gap**: How much accuracy is lost when evaluating on an unseen generator?

## Setup

```bash
pip install -r requirements.txt
```

### Datasets

- **FaceForensics++**: Request access at [faceforensics.com](https://github.com/ondyari/FaceForensics)
- **Diffusion-generated faces**: Generated using Stable Diffusion v1.5 / v2.1

## Usage

### Training

```bash
python src/training/train.py --model constrained_cnn --dataset faceforensics --manipulation Deepfakes
```

### Evaluation

```bash
python src/training/train.py --evaluate --checkpoint results/checkpoints/best.pth --test-set diffusion
```

### Notebooks

Run the Jupyter notebooks for visualization and analysis:

```bash
jupyter notebook notebooks/
```

## References

- Bayar, B., & Stamm, M. C. (2018). Constrained Convolutional Neural Networks: A New Approach Towards General Purpose Image Manipulation Detection. *IEEE TIFS*.
- Rössler, A., et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. *ICCV*.
- Wang, S. Y., et al. (2020). CNN-generated images are surprisingly easy to spot… for now. *CVPR*.

## License

Academic use only. This project is part of coursework research.
