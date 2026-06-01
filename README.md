# Spatial vs. Channel Attention in Encoder-Decoder Segmentation
### A Frequency-Domain Analysis of U-Net, Attention U-Net, and SEU-Net

**Course:** Deep Learning, 3 CFU | University of Bologna  
**Dataset:** ISIC 2018 Task 1 — Skin Lesion Segmentation  
**Author:** Ashmi Prasad

---

## Overview

This project compares three encoder-decoder segmentation architectures through a frequency-domain lens. Rather than stopping at standard overlap metrics (Dice, IoU), the evaluation includes:

- **Per-band spectral error** between predicted and ground truth masks using 2D Fourier decomposition (low, mid, high frequency bands)
- **Butterworth filtering experiments** that selectively remove low-frequency or high-frequency content from test images before inference, with no retraining

The central question is whether spatial attention (Attention U-Net) and channel attention (SEU-Net) differ in the type of frequency information they rely on.

---

## Models

| Model | Attention Type | Parameters |
|---|---|---|
| U-Net | None (baseline) | 31.0M |
| Attention U-Net | Spatial (attention gates at skip connections) | 31.4M |
| SEU-Net | Channel (squeeze-and-excitation blocks in encoder) | 31.2M |

All three share an identical backbone and are trained under identical conditions for fair comparison.

---

## Key Results

| Model | Dice (mean) | IoU (mean) | High-freq MSE |
|---|---|---|---|
| U-Net | 0.858 | 0.777 | 45.3 |
| Attention U-Net | 0.860 | 0.782 | 43.6 |
| SEU-Net | 0.862 | 0.787 | 42.3 |

**Main finding:** SEU-Net achieves the best overall Dice and lowest high-frequency spectral error. However, Attention U-Net is the most robust under high-pass filtering, suggesting spatial attention provides a specific advantage when only edge information is available.


- **Part 1** — Environment setup, data pipeline, model architectures, training (all 3 models, 2 seeds each)
- **Part 2** — Spatial metrics evaluation, FFT frequency analysis, radial band MSE
- **Part 3** — Butterworth filtering experiments, boundary overlays, frequency heatmaps, summary figures

---

## Dataset

The project uses the **ISIC 2018 Task 1** skin lesion segmentation dataset (2,594 images with binary masks). The dataset is not included in this repository due to its size (~12GB).

**To download:**

1. Go to [kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation](https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation)
2. Download and unzip
3. Update the paths in the CONFIG cell at the top of the notebook.

---

## Setup

The notebook was developed and run on **Google Colab Pro** with an NVIDIA A100-SXM4-40GB GPU. No additional installations are needed beyond the standard Colab environment.

Required libraries (all available in Colab by default):
- PyTorch
- torchvision
- NumPy
- matplotlib
- Pillow
- scipy

---
