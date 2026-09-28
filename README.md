# Prostate Cancer Grading from Whole-Slide Images (PANDA Challenge)

A PyTorch pipeline that predicts the **ISUP grade (0-5)** of prostate cancer from H&E-stained biopsy whole-slide images (WSIs). It uses the data from the Kaggle [Prostate cANcer graDe Assessment (PANDA) Challenge](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment) and fine-tunes an ImageNet-pretrained **ResNet-34** on tissue-tile mosaics built from each slide.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model & Training Details](#model--training-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Outputs](#outputs)
- [Results](#results)
- [Known Limitations & Ideas for Improvement](#known-limitations--ideas-for-improvement)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

Gleason/ISUP grading of prostate biopsies is time-consuming and subject to inter-observer variability. This project explores an automated approach:

1. Read a giant multi-resolution WSI (`.tiff`) with OpenSlide.
2. Cut it into fixed-size tiles and keep the tiles that contain the most tissue.
3. Stitch those tiles into a single compact image.
4. Classify that image into one of six ISUP grades with a CNN.

The whole workflow (data download, splitting, training, validation, and held-out testing) lives in a single notebook, `fullcode.ipynb`.

## Pipeline

```
Kaggle PANDA data
      │
      ▼
Stratified split by ISUP grade  (70% train / 15% val / 15% test)
      │
      ▼
WSI (.tiff) ──► read at level 1 ──► pad to multiple of 256 ──► 256×256 tiles
      │
      ▼
Rank tiles by darkness (sum of pixel values) ──► keep top 16 tissue-rich tiles
      │
      ▼
Stitch into 4×4 grid (1024×1024 RGB image)
      │
      ▼
Augment + normalize ──► ResNet-34 (6-way head) ──► ISUP grade prediction
```

**Tile selection.** A tile with a lower total pixel sum is darker, i.e. it contains more stained tissue and less white background. The 16 darkest tiles are selected. If a slide has fewer than 16 tiles, the grid is padded with blank white tiles.

## Repository Structure

```
.
├── fullcode.ipynb          # Full pipeline: data, model, training, evaluation
├── README.md
└── best_panda_resnet34.pth # Generated after training (best validation QWK)
```

## Requirements

- Python 3.9+
- A CUDA-capable GPU is strongly recommended (the script falls back to CPU, but training will be very slow; mixed-precision training uses `torch.cuda.amp`)
- OpenSlide system library (needed by `openslide-python`)
- A Kaggle account with the PANDA competition rules accepted

Python packages:

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Model, transforms, training |
| `openslide-python`, `tiffslide` | Reading multi-resolution WSIs |
| `numpy`, `pandas` | Array and metadata handling |
| `scikit-learn` | Splitting and metrics (QWK, report, confusion matrix) |
| `opencv-python`, `matplotlib` | Imaging / plotting utilities |
| `tqdm` | Progress bars |
| `kagglehub` | Automatic dataset download |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install the OpenSlide system library

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y openslide-tools

# macOS (Homebrew)
brew install openslide
```

On Windows, download the OpenSlide binaries from [openslide.org](https://openslide.org/download/) and add them to your `PATH`.

### 3. Install Python dependencies

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib tqdm \
            scikit-learn kagglehub openslide-python tiffslide
```

### 4. Set up Kaggle access

1. Join the [PANDA competition](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment) and accept its rules.
2. Create an API token in your Kaggle account settings and place `kaggle.json` in `~/.kaggle/` (or set the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables).

## Usage

Open and run the notebook:

```bash
jupyter notebook fullcode.ipynb
```

Or run it on Kaggle / Google Colab with a GPU runtime. When executed, the notebook will:

1. Download the PANDA dataset with `kagglehub.competition_download(...)`.
2. Build stratified train / validation / test splits.
3. Train the model for the configured number of epochs, saving the checkpoint with the best validation QWK.
4. Reload the best checkpoint and evaluate it once on the held-out test set.

> **Note:** The PANDA training set is very large (hundreds of GB). Make sure you have sufficient disk space and bandwidth before starting.

## Configuration

All hyperparameters are defined in the `Config` class at the top of the notebook:

| Parameter | Default | Description |
|---|---|---|
| `TILE_SIZE` | `256` | Side length (pixels) of each tile |
| `N_TILES` | `16` | Number of tissue tiles per slide (must be a perfect square) |
| `NUM_CLASSES` | `6` | ISUP grades 0-5 |
| `BATCH_SIZE` | `16` | Batch size |
| `EPOCHS` | `4` | Number of training epochs |
| `LR` | `3e-4` | Learning rate (AdamW) |
| `NUM_WORKERS` | `2` | DataLoader workers |
| `DEVICE` | `cuda` if available, else `cpu` | Compute device |
| `SEED` | `42` | Random seed for reproducibility |

## Model & Training Details

| Component | Choice |
|---|---|
| Backbone | ResNet-34, ImageNet-pretrained (`ResNet34_Weights.DEFAULT`) |
| Head | Final fully connected layer replaced with `Linear(512, 6)` |
| Input | 1024×1024 stitched mosaic of 16 tiles (256×256 each) |
| Loss | Cross-entropy |
| Optimizer | AdamW (`lr=3e-4`, `weight_decay=1e-4`) |
| Precision | Automatic mixed precision (AMP) with `GradScaler` |
| Augmentation (train) | Random horizontal and vertical flips |
| Normalization | ImageNet mean/std |
| Model selection | Best validation Quadratic Weighted Kappa |
| Data split | 70% train / 15% val / 15% test, stratified by `isup_grade` |

Tiles are extracted **on the fly** in the `Dataset`, so no preprocessed images need to be stored on disk.

## Evaluation Metrics

- **Quadratic Weighted Kappa (QWK)**: the official PANDA challenge metric. It penalizes predictions more heavily the further they are from the true grade.
- **Accuracy**
- **Cross-entropy loss**
- **Per-class precision / recall / F1** (`classification_report`)
- **Confusion matrix**

## Outputs

| Output | Description |
|---|---|
| `best_panda_resnet34.pth` | State dict of the model with the highest validation QWK |
| Console logs | Per-epoch train/validation loss, accuracy and QWK |
| Test report | Final loss, accuracy, QWK, classification report and confusion matrix on the held-out test set |

Loading the trained model for inference:

```python
import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet34()
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load("best_panda_resnet34.pth", map_location="cpu"))
model.eval()
```

## Results

> Fill in after running the notebook.

| Split | Loss | Accuracy | QWK |
|---|---|---|---|
| Validation | - | - | - |
| Test | - | - | - |

## Known Limitations & Ideas for Improvement

- **Short training.** Only 4 epochs are configured by default; more epochs with a learning-rate scheduler would likely help.
- **Simple tile selection.** Tiles are ranked by pixel-sum darkness, which can pick up pen marks or dark artifacts. A proper tissue mask (e.g. Otsu thresholding or HSV saturation) would be more robust.
- **Single resolution.** Only level 1 of the slide is used; multi-scale tiles could capture both context and cellular detail.
- **Classification loss only.** Since ISUP grades are ordinal, a regression or ordinal loss (with optimized thresholds) often improves QWK.
- **Label noise.** PANDA labels come from multiple sources (Radboud, Karolinska) with differing annotation quality; the provided segmentation masks are not used here.
- **Class imbalance.** Consider weighted sampling or class-weighted loss.
- **Stronger backbones and augmentation.** EfficientNet / ConvNeXt, color jitter, and stain augmentation are natural next steps.
- **Data loading speed.** On-the-fly WSI reading is I/O heavy; caching the stitched mosaics would speed up training.

## Acknowledgements

- [PANDA Challenge](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment) organizers and the underlying study: Bulten et al., *Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge*, Nature Medicine, 2022.
- [PyTorch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/), [OpenSlide](https://openslide.org/), [scikit-learn](https://scikit-learn.org/).

## License

Add your preferred license here (e.g. MIT). Note that use of the PANDA dataset is governed by the Kaggle competition rules.

## Disclaimer

This project is for research and educational purposes only. It is **not** a medical device and must not be used for clinical decision-making.
