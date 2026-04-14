# Motion-DeepLab: Video Panoptic Segmentation on KITTI-STEP

A PyTorch exploration of Google's [Motion-DeepLab](https://arxiv.org/pdf/2102.11859), built as a university course project to understand the mechanics of video panoptic segmentation. 

Video panoptic segmentation is a complex, multi-stage problem. The goal of this project was to get familiar with the task by deconstructing a modern vps architeture. While the original paper evaluates several single-frame baselines, we chose to reproduce their multi-frame Motion-DeepLab as it offered a more complete picture of the problem and better documentation to work with.
---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Encoder — Modified ResNet-50](#encoder--modified-resnet-50)
  - [ASPP Module](#aspp-module)
  - [Dual Decoder](#dual-decoder)
  - [Multi-Head Predictions](#multi-head-predictions)
- [Training Pipeline](#training-pipeline)
  - [Stage 1 — Cityscapes Pretraining](#stage-1--cityscapes-pretraining)
  - [Stage 2 — KITTI-STEP Fine-tuning](#stage-2--kitti-step-fine-tuning)
  - [Data Augmentation](#data-augmentation)
  - [Loss Functions](#loss-functions)
  - [Optimization](#optimization)
- [Inference & Post-Processing](#inference--post-processing)
  - [Panoptic Decoding](#panoptic-decoding)
  - [Temporal Tracking](#temporal-tracking)
- [Evaluation](#evaluation)
  - [Metrics](#metrics)
  - [Performance Results](#performance-results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
  - [Prerequisites](#prerequisites)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation-1)
  - [Visualization](#visualization)
  - [Full Pipeline (Single Job)](#full-pipeline-single-job)
- [Slurm Cluster Usage](#slurm-cluster-usage)
- [References](#references)
- [License](#license)

---

## Overview

Historically, evaluating VPS models has been difficult. Early benchmarks (like Cityscapes-VPS) suffered from sparse labeling—only annotating a few frames within short video snippets. This made it impossible to evaluate long-term temporal consistency or pixel-precise tracking. Furthermore, existing metrics like Video Panoptic Quality (VPQ) often masked poor tracking performance behind good static segmentation scores.

Google Research introduced the STEP (Segmenting and Tracking Every Pixel) benchmark to solve this. The KITTI-STEP dataset provides continuous, dense, pixel-perfect annotations for long urban driving sequences. Alongside the dataset, they introduced the Segmentation and Tracking Quality (STQ) metric, which strictly decouples tracking performance from spatial segmentation to ensure models are fairly evaluated on both tasks independently.

Motion-DeepLab was built as one of the baselines to prove this new benchmark. It extends Panoptic-DeepLab with a **motion head** that predicts per-pixel displacement vectors from the current frame to the previous frame's instance centers. These motion offsets enable temporal association of instances, without relying on external optical flow networks.

### Key Features

- **7-channel input**: current RGB frame (3ch) + previous RGB frame (3ch) + previous-frame center heatmap (1ch)
- **Dual-decoder architecture**: separate decoders for semantic and instance tasks
- **Four prediction heads**: semantic logits, center heatmap, center offsets, motion offsets
- **Post-processing**: panoptic decoding and heatmap rendering
- **Two tracking backends**: official DeepLab2-aligned tracker
- **Two-stage training**: Cityscapes pretraining → KITTI-STEP fine-tuning
- **Mixed precision training** with gradient accumulation for large effective batch sizes

---

## Architecture

The model follows the Panoptic-DeepLab family design with a motion regression extension.

```
Input (B, 7, H, W)
  ├── Frame_t RGB (3ch)
  ├── Frame_{t-1} RGB (3ch)
  └── Prev Center Heatmap (1ch)
        │
        ▼
┌──────────────────────┐
│  ResNet-50 Backbone   │  (modified 7-ch conv1, ImageNet pretrained)
│  → res2, res3,        │
│    res4, res5          │
└──────────────────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
┌────────────────┐         ┌────────────────┐
│ Semantic Decoder│         │Instance Decoder │
│  ASPP + 2-stage │         │  ASPP + 2-stage │
│  upsampling     │         │  upsampling     │
│  (256 ch)       │         │  (128 ch)       │
└────────────────┘         └────────────────┘
        │                    │     │      │
        ▼                    ▼     ▼      ▼
   ┌─────────┐        ┌───────┐ ┌──────┐ ┌──────┐
   │Semantic │        │Center │ │Center│ │Motion│
   │ Head    │        │Heatmap│ │Offset│ │Offset│
   │(19 cls) │        │ (1ch) │ │(2ch) │ │(2ch) │
   └─────────┘        └───────┘ └──────┘ └──────┘

```

### Encoder — Modified ResNet-50

| Component | Details |
|-----------|---------|
| Backbone | `torchvision.models.resnet50` with ImageNet pretrained weights |
| Input Conv | Modified `conv1`: 7 input channels (original 3-ch weights split across RGB×2) |
| Output | Multi-scale features: `res2` (1/4), `res3` (1/8), `res4` (1/16), `res5` (1/32) |

The first convolutional layer is expanded from 3→7 channels. The pretrained ImageNet weights are copied to both RGB channel groups (divided by 2 for initialization stability), while the heatmap channel is zero-initialized.

### ASPP Module

Atrous Spatial Pyramid Pooling captures multi-scale context from `res5` features:

| Branch | Type | Dilation Rate |
|--------|------|:---:|
| Branch 1 | 1×1 Conv | — |
| Branch 2 | 3×3 Atrous Conv | 3 |
| Branch 3 | 3×3 Atrous Conv | 6 |
| Branch 4 | 3×3 Atrous Conv | 9 |
| Branch 5 | Global Avg Pooling + 1×1 Conv | — |
| Projection | 1×1 Conv + Dropout(0.1) | — |

All branches output 256 channels; concatenated (1280ch) then projected back to 256ch.

### Dual Decoder

Both decoders share the same architecture but differ in channel widths:

| Decoder | ASPP Channels | Low-Level Projections | Fusion Channels |
|---------|:---:|:---:|:---:|
| Semantic | 256 | res3→64, res2→32 | 256 |
| Instance | 256 | res3→32, res2→16 | 128 |

Each decoder performs two-stage progressive upsampling:
1. ASPP output → upsample to `res3` scale → concat with projected `res3` → 5×5 conv fusion
2. Fused features → upsample to `res2` scale → concat with projected `res2` → 5×5 conv fusion

### Multi-Head Predictions

| Head | Input | Conv | Output | Purpose |
|------|:---:|:---:|:---:|---------|
| Semantic | Semantic decoder (256ch) | 5×5→256→1×1 | 19 classes | Per-pixel class logits |
| Center Heatmap | Instance decoder (128ch) | 5×5→32→1×1 | 1 channel | Gaussian peaks at object centers |
| Center Offset | Instance decoder (128ch) | 5×5→32→1×1 | 2 channels (dy, dx) | Offset from pixel to its instance center |
| Motion Offset | Instance decoder (128ch) | 5×5→32→1×1 | 2 channels (dy, dx) | Displacement to same object's center in previous frame |
| Semantic Aux | `res4` features (1024ch) | 3×3→256→1×1 | 19 classes | Deep supervision at 1/16 scale |

All head outputs are bilinearly upsampled to input resolution. Offset predictions are scale-corrected to account for the encoder stride.

---

## Training Pipeline

### Stage 1 — Cityscapes Pretraining

Since Video Panoptic Segmentation extends Panoptic Segmentation, the problem can be reduced to fine-tuning a model designed for static images. While the original work pretrains the model on both Cityscapes and Mapillary Vistas using extensive data augmentation, our implementation obly pretrained the model on Cityscapes with only some of the data augmentation techniques.

- **Dataset**: Cityscapes `train` split (~2,975 images)
- **Mode**: Full-branch panoptic pretrain (semantic + center heatmap + center offset)
- **Input**: Current image duplicated as both frames, zero heatmap channel
- **Epochs**: 50 (configurable)
- **Labels**: Mapped from Cityscapes `labelId` or `trainId` to 19-class scheme. Instance IDs extracted from `*_gtFine_instanceIds.png` for thing classes (person, rider, car, truck, bus, train, motorcycle, bicycle).

### Stage 2 — KITTI-STEP Fine-tuning

- **Dataset**: KITTI-STEP `train` split (21 video sequences, ~8K pairs)
- **Input**: Consecutive frame pairs + previous-frame center heatmap
- **Epochs**: 200 (configurable)
- **Resume**: Initializes from Cityscapes pretrained checkpoint

### Data Augmentation

Official DeepLab2-style augmentation pipeline:

| Augmentation | Parameters |
|--------------|-----------|
| Random Scale | 0.5× to 2.0× (0.1 step increments) |
| Random Crop | 384 × 1248 |
| Padding | Zero-fill images, 255-fill semantic labels |
| Random Horizontal Flip | 50% probability |
| ImageNet Normalization | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

Paired augmentation ensures both the current and previous frames undergo identical geometric transforms.

### Loss Functions

The total loss combines five terms:

```
L_total = L_semantic
        + 0.4 × L_semantic_aux
        + 200 × L_center
        + 0.01 × L_offset
        + 0.01 × L_motion
```

| Loss | Type | Details |
|------|------|---------|
| `L_semantic` | Top-K Cross Entropy | OHEM: only top 20% hardest pixels contribute. Per-pixel weighting upweights small instances (area < 4096px → 3× weight). |
| `L_semantic_aux` | Top-K Cross Entropy | Same as above, computed on downsampled 1/16 scale predictions. Weight: 0.4. |
| `L_center` | MSE | Mean squared error between predicted and ground-truth Gaussian heatmaps (σ=8). Masked to "thing" pixels. |
| `L_offset` | L1 | Absolute error of center offset regression, weighted by instance mask. |
| `L_motion` | L1 | Absolute error of motion offset regression, weighted by matched instances. |

**Ground truth generation** is performed on-the-fly:
- Center heatmaps: Gaussian peaks (σ=8) at each instance centroid
- Center offsets: per-pixel (dy, dx) vector pointing from each thing pixel to its instance center
- Motion offsets: per-pixel (dy, dx) to the same instance's center in the previous frame
- Small instance weighting: instances with area < 4096 pixels get 3× semantic loss weight

### Optimization

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Base Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| LR Schedule | Polynomial decay (power=0.9) or Cosine Annealing |
| Batch Size | 8 |
| Gradient Accumulation | 4 steps (effective batch = 32) |
| Gradient Clipping | Max norm = 10.0 |
| Mixed Precision | AMP autocast + GradScaler |

---

## Inference & Post-Processing

### Panoptic Decoding

1. **Semantic prediction**: argmax over 19-class logits
2. **Center detection**: sigmoid activation → threshold (default 0.1) → NMS with 13×13 max-pool → top-K retention (K=200)
3. **Instance grouping**: each pixel assigned to closest detected center via predicted offsets (Vectorized tensor argmin over all center candidates)
4. **Panoptic merging**: semantic + instance maps combined using `label_divisor=1000` encoding (e.g., car instance 3 → `13×1000 + 3 = 13003`)
5. **Heatmap rendering**: Gaussian heatmap rendered from the decoded panoptic map → fed as 7th channel for the next frame

### Temporal Tracking

Temporal association is handled natively via motion-offset-based greedy matching, directly aligned with DeepLab2's `assign_instances_to_previous_tracks` logic:

* **Motion Propagation:** Instance centers are propagated to the previous frame using the model's predicted motion offsets.
* **Greedy Assignment:** Current centers are sorted by confidence and greedily matched to stored previous centers.
* **Distance Gating:** A match is only accepted if the squared distance between the projected center and the previous center is strictly less than the instance mask's bounding-box area.
* **Track Lifecycle:** Unmatched objects are assigned a new ID. Inactive tracks are stored in memory and pruned only after $\sigma=7$ frames without an update.
---

## Evaluation

### Metrics

Evaluation uses the **Segmentation and Tracking Quality (STQ)** metric ([Weber et al., 2021](https://arxiv.org/pdf/2102.11859)), the primary benchmark for KITTI-STEP.

| Metric | Description |
|--------|-------------|
| **STQ** | Geometric mean of AQ and IoU: `STQ = √(AQ × IoU)` |
| **AQ** (Association Quality) | Measures temporal consistency of instance tracks |
| **IoU** (Intersection over Union) | Mean IoU over semantic classes |

The STQ implementation follows the official DeepLab2 numpy reference (`segmentation_and_tracking_quality.py`).

### Performance Results

Best results achieved with the full pipeline (50-epoch Cityscapes pretrain → 200-epoch KITTI-STEP fine-tuning):

| Metric | Value |
|--------|:-----:|
| **STQ** | **0.477** |
| **AQ** | **0.429** |
| **IoU** | **0.531** |

Per-sequence breakdown (val split, 9 sequences):

| Sequence | Frames | STQ | AQ | IoU |
|:---:|:---:|:---:|:---:|:---:|
| 0002 | 233 | 0.396 | 0.347 | 0.453 |
| 0006 | 270 | 0.382 | 0.352 | 0.414 |
| 0007 | 800 | 0.531 | 0.657 | 0.429 |
| 0008 | 390 | 0.457 | 0.538 | 0.389 |
| 0010 | 294 | 0.427 | 0.437 | 0.417 |
| 0013 | 340 | 0.296 | 0.226 | 0.388 |
| 0014 | 106 | 0.424 | 0.424 | 0.424 |
| 0016 | 209 | 0.339 | 0.229 | 0.503 |
| 0018 | 339 | 0.437 | 0.479 | 0.398 |

## Setup & Usage

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- torchvision
- OpenCV (`cv2`)
- NumPy, SciPy
- A GPU with ≥8 GB VRAM

Install dependencies (if not using a pre-built environment module):

```bash
pip install torch torchvision opencv-python numpy scipy
```

### Dataset Preparation

#### KITTI-STEP

1. Download [KITTI-STEP](https://www.cvlibs.net/datasets/kitti/eval_step.php) images and panoptic maps.
2. Organize as follows (or symlink `images/` and `panoptic_maps/` into the project root):

```
project_root/
├── images/
│   ├── train/
│   │   ├── 0000/
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   └── ...
│   └── val/
│       ├── 0002/
│       └── ...
└── panoptic_maps/
    ├── train/
    │   ├── 0000/
    │   │   ├── 000000.png   # RGB-encoded: R=semantic, G+B=instance
    │   │   └── ...
    │   └── ...
    └── val/
        └── ...
```

Panoptic maps are 3-channel PNGs where:
- **Channel R** = semantic class ID (0–18, 255=ignore)
- **Channels G×256 + B** = instance ID (0=stuff/background)
- 
## References

1. **Motion-DeepLab**: Weber, M., et al. (2021). *STEP: Segmenting and Tracking Every Pixel*. NeurIPS 2021 Datasets and Benchmarks Track. [arXiv:2104.14462](https://arxiv.org/pdf/2102.11859)

2. **Panoptic-DeepLab**: Cheng, B., et al. (2020). *Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation*. CVPR 2020. [arXiv:1911.10194](https://arxiv.org/abs/1911.10194)

3. **DeepLab2**: Google Research. *DeepLab2: A TensorFlow Library for Deep Labeling*. [GitHub](https://github.com/google-research/deeplab2)

4. **KITTI-STEP Benchmark**: [https://www.cvlibs.net/datasets/kitti/eval_step.php](https://www.cvlibs.net/datasets/kitti/eval_step.php)

5. **Cityscapes Dataset**: Cordts, M., et al. (2016). *The Cityscapes Dataset for Semantic Urban Scene Understanding*. CVPR 2016. [https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)

---

## License

This project is for academic and research purposes. The codebase is an independent reimplementation. Original Motion-DeepLab and DeepLab2 are licensed under [Apache 2.0](https://github.com/google-research/deeplab2/blob/main/LICENSE) by Google Research.
