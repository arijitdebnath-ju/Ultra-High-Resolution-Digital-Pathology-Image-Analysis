# Whole-Slide Image (WSI) Pathology Analysis Pipeline

A PyTorch-based pipeline that simulates end-to-end analysis of a **gigapixel whole-slide histopathology image**: tissue segmentation, patch extraction, deep-learning-based patch classification, and heatmap reconstruction for abnormality localization.

The notebook is self-contained and runs out of the box (e.g., on Kaggle/Colab) using a **synthetically generated slide**, so no external dataset or GPU is strictly required to try it out.

## Overview

Whole-slide images (WSIs) used in digital pathology are extremely large (often tens of thousands of pixels per side), far too big to feed directly into a CNN. The standard approach — reproduced here — is to:

1. Downsample the slide to a thumbnail for fast preprocessing.
2. Segment tissue from background (glass slide).
3. Tile the full-resolution slide into small patches, keeping only patches that overlap sufficient tissue.
4. Run a CNN classifier over the patches in batches.
5. Stitch the per-patch predictions back into a low-resolution probability heatmap.
6. Threshold the heatmap and draw bounding boxes around candidate abnormal regions.

## Pipeline Stages

| Stage | Component | Description |
|---|---|---|
| 0 | `Config` | Central place for all hyperparameters (patch size, stride, thresholds, batch size, device, etc.). |
| 1 | `TissueSegmenter` | Converts the thumbnail to HSV, applies Otsu thresholding on the saturation channel, and cleans the mask with morphological opening/closing to separate tissue from the white glass background. |
| 2 | `PatchExtractor` | Builds a coordinate grid over the full-resolution image and discards patches whose corresponding region in the tissue mask falls below a minimum tissue ratio. |
| 3 | `PathologyPatchDataset` | A lazy-loading `torch.utils.data.Dataset` that crops each patch on demand (rather than pre-loading all patches into memory) and applies the image transform. |
| 4 | `build_pathology_classifier` / `run_batch_inference` | A ResNet-18 backbone (ImageNet-pretrained) with a custom binary classification head (`Linear → ReLU → Dropout → Linear → Sigmoid`), run over the patch `DataLoader` with mixed-precision (`torch.amp.autocast`) inference. |
| 5 | `HeatmapReconstructor` | Reassembles per-patch probabilities into a 2D probability map, resizes it to the thumbnail resolution, overlays it as a color heatmap, and extracts bounding boxes of high-probability regions via contour detection. |
| 6 | `main()` | Orchestrates the full pipeline end-to-end and renders a 4-panel visualization. |

## Output

Running the notebook produces a 4-panel figure:

1. **Original Slide Thumbnail** – the downsampled RGB thumbnail.
2. **Tissue Foreground Mask** – binary mask separating tissue from background.
3. **Probability Heatmap** – per-region classifier confidence, reconstructed to thumbnail resolution.
4. **ROI Detection Overlay** – the heatmap blended onto the thumbnail with bounding boxes drawn around candidate abnormal regions.

Console output also reports slide dimensions, grid size, the fraction of tiles retained after tissue filtering, and the number of candidate ROIs detected.

## Requirements

```
torch
torchvision
opencv-python
numpy
matplotlib
tqdm
```

Install with:

```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm
```

A CUDA-capable GPU is used automatically if available (`Config.DEVICE`), otherwise the pipeline falls back to CPU.

## Usage

Run all cells in the notebook, or extract the code into a `.py` file and run:

```bash
python pathology_pipeline.py
```

### Using your own slide

By default the pipeline generates a **synthetic slide** for demonstration purposes. To run it on a real whole-slide image instead:

```python
class Config:
    USE_SYNTHETIC_IMAGE = False
    ...
```

and update the image path in `main()`:

```python
wsi_image = cv2.imread("path_to_slide.tif")
wsi_image = cv2.cvtColor(wsi_image, cv2.COLOR_BGR2RGB)
```

### Key configuration options (`Config`)

| Parameter | Default | Description |
|---|---|---|
| `PATCH_SIZE` | 256 | Size (pixels) of each extracted patch. |
| `STRIDE` | 256 | Step size between patches (equal to `PATCH_SIZE` ⇒ non-overlapping tiling). |
| `THUMB_SCALE` | 0.05 | Downsampling factor used to build the thumbnail for tissue segmentation. |
| `TISSUE_THRESHOLD` | 0.15 | Minimum fraction of tissue pixels required to keep a patch. |
| `BATCH_SIZE` | 64 | Inference batch size. |
| `NUM_WORKERS` | 2 | DataLoader worker processes. |
| `HEATMAP_ALPHA` | 0.45 | Blend weight of the heatmap overlay on the thumbnail. |

## ⚠️ Important Caveats

- **The classifier is not trained.** `build_pathology_classifier()` loads ImageNet-pretrained ResNet-18 weights but attaches a freshly initialized (random) classification head. The pipeline is therefore a **structural / engineering demonstration** of a scalable WSI inference pipeline — the heatmaps and bounding boxes it produces do **not** reflect real pathological findings.
- **The demo slide is synthetic.** `create_synthetic_wsi()` draws simple colored ellipses with noise to mimic tissue at a large scale; it is not derived from real histology.
- To make this pipeline clinically or scientifically meaningful, you would need to:
  1. Train (or fine-tune) the classification head — and likely the backbone — on labeled histopathology patches.
  2. Validate on held-out real WSIs.
  3. Add proper WSI I/O (e.g., via [OpenSlide](https://openslide.org/)) for formats like `.svs`/`.tif` rather than loading a whole image into memory with `cv2.imread`.

This project is intended as a **reference implementation / starting point** for building scalable, memory-efficient WSI inference pipelines in PyTorch, not as a diagnostic tool.

## License

Add a license of your choice (e.g., MIT) here.
