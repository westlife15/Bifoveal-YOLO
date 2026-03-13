# BIFOVEAL-YOLO: Dual-Acuity Architecture for Micro-Object Detection in Drone-Based Imagery

> **Version:** 10.1 &nbsp;|&nbsp; **Status:** Production &nbsp;|&nbsp; **Best mAP@50:** 54.12% &nbsp;|&nbsp; **Parameters:** 2,628,599

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Dataset](#3-dataset-visdrone-det-2019)
4. [Architecture](#4-model-architecture)
5. [Training Pipeline](#5-training-pipeline)
6. [Hyperparameters & Configuration](#6-hyperparameters--configuration)
7. [Results & Performance](#7-results--performance)
8. [Benchmark Comparisons](#8-benchmark-comparisons)
9. [Deployment](#9-deployment)
10. [File Structure](#10-file-structure)
11. [Execution Guide](#11-execution-guide)
12. [Known Limitations](#12-known-limitations)
13. [Future Work](#13-future-work)

---

## 1. Project Overview

BIFOVEAL-YOLO is a production-grade aerial object detection system designed specifically for the **VisDrone-DET 2019** benchmark — one of the most challenging detection datasets in computer vision. The name is inspired by the fovea in the human eye, the region of sharpest vision. Like dual-fovea optics, the model applies two levels of acuity: a standard detection pathway for mid/large objects, and a dedicated micro-scale pathway (the P2 head) for sub-15px objects that standard YOLO completely misses.

### The Core Problem

Drone-captured footage is fundamentally different from standard detection benchmarks:

| Challenge | VisDrone Reality | Standard COCO |
|---|---|---|
| Object density | 400+ objects per frame | ~7 per frame |
| Object size | 5–30px typical | 50–300px typical |
| Class imbalance | bicycle <2% of all labels | Relatively balanced |
| Viewpoint | Pure top-down aerial | Ground-level perspective |
| Scene variation | 5–120m altitude range | Fixed camera height |

Standard YOLOv8n trained on VisDrone achieves around **26% mAP@50**. BIFOVEAL-YOLO v10 achieves **54.12% mAP@50** — a 108% relative improvement — using a model of comparable size.

### Key Innovations

- **SPDSECAConv backbone (Tier 3):** Replaces stride-2 convolutions with lossless Space-to-Depth downsampling, preserving every spatial pixel before compression.
- **P2 micro-scale detection head:** A dedicated 160×160 feature map output for detecting objects smaller than 15px, paired with a lightweight depthwise-separable projection.
- **ASFM (Aerial Scale Feature Module):** Dilated multi-scale depthwise fusion at the P2 output, handling objects at multiple apparent sizes from different flight altitudes.
- **FeatureSR neck refinement:** Channel and spatial attention applied after FPN upsampling to clean blocky nearest-neighbour artifacts.
- **SECAConv attention (SE + CoordAtt):** Squeeze-and-Excitation combined with Coordinate Attention throughout the backbone for directional spatial awareness.
- **SAHI slice fine-tuning:** 80-epoch fine-tune on 36,097 sliced 640×640 crops, recalibrating scale sensitivity for massive high-resolution inference gains.
- **4-phase progressive curriculum:** Phased augmentation schedule that transitions from aggressive diversity to clean convergence across 200 epochs.

---

## 2. Tech Stack

### Core Framework

| Component | Library / Tool | Version | Role |
|---|---|---|---|
| Deep Learning | **PyTorch** | 2.9.0+cu126 | Tensor operations, autograd, model training |
| Detection Framework | **Ultralytics** | 8.4.21 | YOLO training infrastructure, data loading, loss |
| CUDA Runtime | **CUDA** | 12.6 | GPU acceleration |
| Language | **Python** | 3.12.12 | Runtime |

### Supporting Libraries

| Library | Version | Role |
|---|---|---|
| NumPy | 1.26.4 (pinned) | Numerical ops; pinned due to PyTorch 2.9 / NumPy 2.x incompatibility |
| SAHI | latest | Sliced dataset preparation + sliced inference |
| Pillow | latest | Image I/O for dataset conversion and slicing |
| PyYAML | bundled | Dataset config file generation and parsing |
| psutil | latest | Memory monitoring during training |

> **NumPy Pin Note:** NumPy must be pinned to `1.26.4`. PyTorch 2.9 on Kaggle produces silent tensor conversion errors with NumPy 2.x — no exception, just incorrect gradient values. Always uninstall and reinstall with `--no-deps`.

### Training Hardware

| Resource | Specification |
|---|---|
| GPU | NVIDIA Tesla T4 |
| VRAM | 14.6 GB |
| Training Platform | Kaggle (free tier) |
| Session Limit | ~9–12 hours per session |
| Storage | `/kaggle/working/` (100 GB, persistent across sessions) |

### Inference Hardware (Production)

| Target | Recommended Config |
|---|---|
| Server / Cloud | Any GPU with ≥8 GB VRAM, imgsz=1536 |
| Jetson AGX Orin | imgsz=1536, TensorRT FP16 |
| Jetson Orin NX | imgsz=1152, TensorRT FP16 |
| Jetson Xavier NX | imgsz=1024, TensorRT FP16 |

---

## 3. Dataset: VisDrone-DET 2019

### Overview

VisDrone-DET 2019 is a large-scale drone-captured object detection benchmark collected by DJI across 14 cities in China at altitudes ranging from 5 to 120 metres.

```
Total training images : 6,471
Total validation images: 548
Object classes        : 11
Avg objects per image : ~54
Max objects per image : 400+
Typical object size   : 5–30px (sub-2% of image area)
Native resolution     : ~1360×765 to 2000×1500px
```

### Class Distribution

| ID | Class | Frequency | Difficulty |
|---|---|---|---|
| 0 | pedestrian | High | Moderate |
| 1 | people | High | Hard (group detection) |
| 2 | bicycle | **Very Low** | **Hard** |
| 3 | car | Very High (dominant) | Easy |
| 4 | van | Medium | Moderate |
| 5 | truck | Medium | Moderate |
| 6 | tricycle | **Low** | **Hard** |
| 7 | awning-tricycle | **Very Low** | **Hard** |
| 8 | bus | Low | Moderate |
| 9 | motor | High | Moderate |
| 10 | others | Very Low | Very Hard (ill-defined) |

Classes 2, 6, 7 (bicycle, tricycle, awning-tricycle) are the **rare classes** — they appear in fewer than 5% of training images and required special handling via rare-class oversampling (T2-B).

### Annotation Format (Raw VisDrone)

```
<bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <category>, <truncation>, <occlusion>
```

`category=0` (ignored regions) are discarded. All remaining categories are converted to 0-indexed YOLO format.

### YOLO Format Conversion

```
<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
```

All coordinates normalised to `[0, 1]` by image dimensions. Images without any valid annotations are discarded. Conversion is handled by `prepare_dataset()`.

### Sliced Dataset (T2-C)

For the SAHI fine-tune phase, the training set is additionally sliced into overlapping 640×640 crops:

```
Slice size      : 640 × 640 px
Overlap ratio   : 0.2 (20%)
Stride          : 512 px (horizontal and vertical)
Min visibility  : 0.1 (10% of original box area must be inside crop)
Total slices    : ~36,097 crops from 6,471 images (~5.6× expansion)
```

Validation remains at full original resolution — sliced val would be misleading because full-image mAP drops during slice fine-tuning (expected and intentional).

---

## 4. Model Architecture

### High-Level Overview

```
Input Image (800×800, RGB)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│                     BACKBONE                         │
│  Conv → SECAConv → C2f                              │
│  → SPDSECAConv → C2f                    Tier 3      │
│  → SPDSECAConv → C2f                    (lossless)  │
│  → SECAConv → C2f → SPPF                            │
└──────────────────────────┬──────────────────────────┘
                           │  skip connections (P1, P2, P3)
                           ▼
┌─────────────────────────────────────────────────────┐
│                  FPN NECK (PANet)                    │
│  Upsample → Concat → C2f → FeatureSR               │
│  Upsample → Concat → C2f → FeatureSR               │
│  LightweightP2Prep                                  │
│  → Conv/Concat/C2f (bottom-up path)                 │
└────────────┬─────────────┬──────────────────────────┘
             │             │
             ▼             ▼
     ┌───────────┐  ┌───────────┐  ┌────────────────┐
     │   ASFM    │  │  P3 Head  │  │   P4 Head      │
     │  (P2, 96) │  │ (96ch)    │  │   (192ch)      │
     └─────┬─────┘  └─────┬─────┘  └───────┬────────┘
           └──────────────┴────────────────┘
                          │
                          ▼
                   Detect Layer
            (P2 micro + P3 + P4 outputs)
                          │
                          ▼
               11-class predictions
```

### Parameter Budget

```
Total parameters : 2,628,599
GFLOPs           : 21.5
Layers           : 233
Backbone params  : ~800K
FPN neck params  : ~400K
Detection head   : ~1.4M
```

---

### 4.1 Custom Modules

#### `CoordAtt` — Coordinate Attention

**Problem it solves:** Standard Squeeze-and-Excitation (SE) collapses the entire spatial dimension to a single global average, losing all information about *where* objects are in the feature map.

**How it works:**
1. Pool H-direction: `AdaptiveAvgPool2d((None, 1))` → shape `(B, C, H, 1)`
2. Pool W-direction: `AdaptiveAvgPool2d((1, None))` → shape `(B, C, 1, W)`
3. Concatenate along spatial dim → shared Conv1×1 → BN → SiLU → split back
4. Apply separate H-directional and W-directional sigmoid masks to input

**Why it matters for drones:** Elongated objects (buses, trucks, rows of pedestrians) have strong directional spatial information. CoordAtt lets the model differentially weight features by their H and W positions rather than treating all positions equally.

---

#### `SECAConv` — SE + CoordAtt Fused Conv

**Components:**
- `Conv(c1, c2, k, s)` — standard feature extraction
- SE block: `AvgPool → Conv(C//16) → ReLU → Conv(C) → Sigmoid → multiply`
- CoordAtt block: directional H/W spatial attention

**Usage in backbone:** Layers 1 and 7 (coarse entry and high-channel levels). Also used as the compression step inside `SPDSECAConv`.

**SE reduction ratio = 16:** A 256-channel layer has a 16-neuron bottleneck — small enough to be parameter-efficient, large enough for meaningful channel interactions.

---

#### `SpaceToDepth` — Lossless Spatial Downsampling

**The problem with stride-2 convolutions:**
A stride-2 conv samples every other pixel. For a 12px pedestrian, after two stride-2 operations it becomes a 6px × 6px feature representation — barely detectable. **75% of input pixels are simply discarded.**

**SpaceToDepth transformation:**
```
Input:  (B, C,   H,   W)
Output: (B, 4C, H/2, W/2)   [block_size=2]
```

Every 2×2 spatial block is folded into 4 new channels. **Zero pixels are discarded** — they are reorganised. Spatial resolution halves identically to striding, but all spatial information is preserved.

```python
# Mathematical identity:
output[b, c*4+k, h, w] = input[b, c, h*2 + (k//2), w*2 + (k%2)]
# where k ∈ {0,1,2,3} = four 2×2 sub-pixel positions
```

---

#### `SPDSECAConv` — Space-to-Depth + SECAConv (Tier 3)

**Architecture:**
```
SpaceToDepth(block=2)            (B, C, H, W) → (B, 4C, H/2, W/2)
SECAConv(4C → C_out, k=1, s=1)  channel compression + SE + CoordAtt
```

The `k=1` pointwise SECAConv learns *which of the four sub-pixel positions* is most informative for each output channel.

**Placement:** Backbone layers 3 and 5 — the two mid-resolution stride-2 downsampling steps where small-object information loss is most critical.

**Parameter cost:** +~200K parameters vs. standard SECAConv. Net gain: dramatically better small-object recall at layers where tiny pedestrians, bikes, and motors are at risk of disappearing entirely.

---

#### `MultiScaleFusion` — Dilated Depthwise Multi-Scale Fusion

**Problem:** At 5–120m altitude, the same physical object appears at very different scales. A standard 3×3 conv (effective receptive field ~9px) can capture a high-altitude object but cannot build context for a low-altitude one.

**Architecture:**
```
Input x (B, C, H, W)
├── Identity path                              (B, C, H, W)
├── DW conv3×3, dilation=3  → eff. field 7×7  (B, C, H, W)
├── DW conv5×5, dilation=3  → eff. field 13×13 (B, C, H, W)
└── DW conv7×5, dilation=3  → eff. field ~17×15 (B, C, H, W)
Concat → (B, 4C, H, W)
1×1 conv → BN → SiLU → (B, C, H, W)
```

Parallel branches capture multi-scale evidence simultaneously. Depthwise (one filter per channel) keeps compute manageable on a 160×160 feature map.

---

#### `ASFM` — Aerial Scale Feature Module

A named wrapper around `MultiScaleFusion`. Applied at **layer 28** — the P2 output just before the final detection head.

**Why only at P2?** P3/P4 objects are large enough for standard receptive fields. P2 objects (sub-15px) genuinely need multi-dilation to build context without reducing spatial resolution.

---

#### `FeatureSR` — Feature Spatial Refinement

**Problem:** `nn.Upsample(mode='nearest')` replicates pixels in 2×2 blocks, producing blocky features with no spatial coherence at slice boundaries.

**Components:**
1. **Channel attention** (lightweight SE, ratio=4): suppresses noisy channels introduced during FPN concat/fusion
2. **Spatial refinement** (residual 3×3→3×3): sharpens edges and object boundaries; residual connection preserves original features

**Placement:** Layers 16 (P3-level) and 20 (P2-level) in the FPN neck, after C2f fusion and before passing features to detection heads.

---

#### `LightweightP2Prep` — Depthwise-Separable P2 Projection

Projects the P2 C2f output (~32ch) to the final P2 detection channel width (96ch). Uses depthwise-separable architecture for parameter efficiency:

```
Parameter comparison:
  Standard Conv(32→96, k=3) : 32 × 96 × 9 = 27,648 params
  DW-Sep equivalent          : 32 × 9 + 32 × 96 = 3,360 params
  Saving: 8× fewer parameters
```

**Why add P2 at all?** YOLOv8n default detects at P3 (stride=8, minimum detectable ~64px²). VisDrone has pedestrians at 4×6px = 24px². P2 (stride=4) detects down to ~16px² — the critical detection floor for this dataset.

---

### 4.2 Architecture YAML Summary

```yaml
nc: 11
depth_multiple: 0.33   # YOLOv8-nano depth scaling
width_multiple: 0.25   # YOLOv8-nano width scaling

backbone:
  - [Conv,        [64, 3, 2]]       # 0  stem: 800→400px, 3→16ch
  - [SECAConv,    [128, 3, 2]]      # 1  SE+CA: 400→200px, 16→32ch
  - [C2f,         [128, True]]      # 2  feature extraction, 32ch
  - [SPDSECAConv, [256]]            # 3  T3: lossless 200→100px, 32→64ch
  - [C2f,         [256, True]]      # 4  deep extraction, 64ch
  - [SPDSECAConv, [512]]            # 5  T3: lossless 100→50px, 64→128ch
  - [C2f,         [512, True]]      # 6  deep extraction, 128ch
  - [SECAConv,    [1024, 3, 2]]     # 7  50→25px, 128→256ch
  - [C2f,         [1024, True]]     # 8  deep extraction, 256ch
  - [SPPF,        [1024, 5]]        # 9  multi-scale global context

head:
  # Top-down FPN
  - [nn.Upsample, [None, 2, 'nearest']]   # 10
  - [Concat, [1]]                          # 11  + backbone layer 6
  - [C2f, [512]]                           # 12  P4-level, 128ch
  - [nn.Upsample, [None, 2, 'nearest']]   # 13
  - [Concat, [1]]                          # 14  + backbone layer 4
  - [C2f, [256]]                           # 15  P3-level, 64ch
  - [FeatureSR, [1, 64]]                   # 16  P3 refinement
  - [nn.Upsample, [None, 2, 'nearest']]   # 17
  - [Concat, [1]]                          # 18  + backbone layer 2
  - [C2f, [128]]                           # 19  P2-level, 32ch
  - [FeatureSR, [1, 64]]                   # 20  P2 refinement
  - [LightweightP2Prep, [96]]              # 21  project → 96ch
  # Bottom-up path
  - [Conv, [384, 3, 2]]                    # 22  P2→P3
  - [Concat, [1]]                          # 23  + layer 16
  - [C2f, [384]]                           # 24  P3 output, 96ch
  - [Conv, [768, 3, 2]]                    # 25  P3→P4
  - [Concat, [1]]                          # 26  + layer 12
  - [C2f, [768]]                           # 27  P4 output, 192ch
  # Aerial scale fusion + detect
  - [ASFM, []]                             # 28  multi-dilation on P2
  - [Detect, [nc]]                         # 29  P2(96) P3(96) P4(192)
```

---

## 5. Training Pipeline

### Pipeline Overview

```
Phase 1: Main Training (200 epochs)
  prepare_dataset()           → YOLO format conversion
  balance_rare_classes()      → T2-B oversampling
  _build_yaml()               → architecture YAML
  OptimizedTrainer.train()    → 200-epoch training loop
        ↓
Phase 2: SWA (optional, ~5 min)
  run_swa()                   → average last N epoch*.pt
        ↓
Phase 3: Slice Fine-Tune (~8 GPU-hours)
  prepare_sliced_dataset()    → 36,097 crops
  finetune_sliced()           → 80 epochs on crops
        ↓
Phase 4: Evaluation
  run_sahi_inference()        → SAHI val sample
  Resolution sweep (manual)  → find optimal imgsz
```

---

### 5.1 Tier System

The design is organised into three cumulative tiers, each building on the previous:

#### Tier 1 — Zero-Risk Wins (inference/NMS config, no retraining)

| Change | Default | v10 | Rationale |
|---|---|---|---|
| `max_det` | 300 | **1000** | VisDrone images have 400+ objects; default silently truncates recall |
| `val iou` | 0.7 | **0.45** | Shoulder-to-shoulder pedestrians overlap at 50–65%; 0.7 merges them |
| SWA | None | **5-ckpt avg** | +1–3pp by smoothing sharp local minima |
| SAHI inference | None | **640px slices** | +5–9pp by showing each object at 3× relative size |

#### Tier 2 — Training Configuration Overhaul

| Change | v6 | v10 | Rationale |
|---|---|---|---|
| `lr0` | 0.02 | **0.01** | Model is pretrained; lower LR prevents overshooting warm-start |
| `lrf` | 0.01 | **0.005** | Final LR = 0.00005 for finer convergence |
| `warmup_epochs` | 5.0 | **3.0** | Pretrained weights need less warmup |
| `close_mosaic` | 10 | **30** | More clean-signal epochs for Phase 3 convergence |
| `patience` | 150 | **50** | v6 plateaus at ep~150; reduce wasted compute |
| `save_period` | 10 | **5** | More checkpoints → better SWA averaging |
| `mixup` (Ph1) | 0.18 | **0.12** | High mixup blurs tiny objects; reduce intensity |
| `copy_paste` (Ph1) | 0.12 | **0.20** | More rare-class injection from epoch 1 |
| Rare oversampling | None | **×2 for cls 2,6,7** | Bicycle/tricycle appear in <5% of images |
| SAHI fine-tune | None | **80ep on 36K crops** | Scale recalibration for high-res inference |

#### Tier 3 — Architecture Change

| Change | v6 | v10 | Gain |
|---|---|---|---|
| Backbone layers 3, 5 | SECAConv(stride=2) | **SPDSECAConv** | Lossless downsample preserves all pixels |

Toggle: `CONFIG['use_tier3'] = False` reverts to pure v6 architecture for ablation.

---

### 5.2 4-Phase Progressive Curriculum

The augmentation schedule adapts across 200 epochs:

| Phase | Epochs | Mosaic | Mixup | CopyPaste | Purpose |
|---|---|---|---|---|---|
| Phase 1 | 0–59 | 1.0 | 0.12 | 0.20 | Aggressive start — fast adaptation from v6 init |
| Phase 2 | 60–129 | 1.0 | 0.10 | 0.15 | Peak diversity — maximum generalisation |
| Phase 3 | 130–169 | 0.6 | 0.06 | 0.08 | Stable convergence — where best.pt is typically saved |
| Phase 4 | 170–199 | 0.0 | 0.0 | 0.0 | Clean signal — aligns with `close_mosaic=30` |

Implemented as a registered Ultralytics callback (`on_train_epoch_start`) that directly writes to `trainer.args`.

---

### 5.3 Weight Transfer (v6 → v10)

Training starts from a v6 VisDrone checkpoint via a two-tier transfer strategy:

**Tier A — Primary (intersect_dicts):**
Uses `torch_safe_load()` + `intersect_dicts()` to match parameters by name AND shape. Transfers ~460 keys including the entire FPN head (layers 10–29) and unchanged backbone layers (0, 1, 2, 7, 8, 9).

**Tier B — Structural Remap (bonus):**
Handles layers whose key names changed between v6 and v10:
- C2f layers 4/6: `model.4.0.cv1.*` → `model.4.cv1.*` (Sequential→n=2 change)
- SPDSECAConv layers 3/5: `model.3.se.*` → `model.3.seca.se.*` (new path)

Adds ~58 additional backbone keys. Tier B is wrapped in `try/except` — failure does not affect Tier A.

**Why domain-warm init matters:** Starting from v6 VisDrone weights (rather than COCO) saves approximately **40–50 epochs** of convergence time and raises the effective starting mAP from ~26% (COCO) to ~43% (v6 transfer).

> **Historical failure modes fixed in v10.1:**
> - `model.load(path_string)` → `'str has no .float()'`
> - `torch.load()` returning `ckpt['model']=None` (silent class reconstruction failure)
> - Using `model.load(ema_module)` → incompatibility with optimizer state

---

### 5.4 TAL Assignment Parameters

Task-Aligned Learning (TAL) assigns ground-truth boxes to prediction anchors during training. Custom parameters vs. YOLOv8 defaults:

| Parameter | Default | v10 | Rationale |
|---|---|---|---|
| `topk` | 10 | **15** | More anchor candidates per GT — critical for objects sharing grid cells at P2/P3 |
| `alpha` | 0.5 | **0.75** | Higher classification weight in alignment score → more certain predictions |
| `beta` | 6.0 | 6.0 | IoU weight — unchanged, no benefit observed from adjusting |

---

### 5.5 PicklingError Fix (v10.1)

During sliced fine-tuning, a `PicklingError` fires every epoch when Ultralytics' `save_model()` tries to `deepcopy` the trainer. Root cause:

1. `torch_safe_load()` creates temporary `sys.modules` aliases for cross-version compatibility
2. Aliases are **removed** after `torch.load()` completes
3. Class objects (like `v8DetectionLoss`) have `__module__` pointing to the now-deleted alias path
4. `deepcopy` → pickle → `ModuleNotFoundError` → `PicklingError`

**Fix (`_patch_save_model`):** Monkey-patches `trainer.save_model` to:
1. Strip `model.criterion`, `trainer.criterion`, and `ema.criterion` before `deepcopy`
2. Restore all three in a `finally` block — training continues even if save fails

Additionally, sys.modules aliases are installed **permanently** before `torch_safe_load()` runs, preventing the alias removal from causing issues later.

---

### 5.6 SAHI Slice Fine-Tuning (T2-C)

#### Why Slice Fine-Tuning Works

Main training at 800px calibrates the model for objects appearing at ~0.6–3% of image width. At 1536px full-image inference, a 10px pedestrian is still only 0.65% of image width — slightly better but not transformative.

With slice fine-tuning, the same 10px pedestrian appears in a 640px crop as **1.56% of slice width** — the model is trained to associate feature patterns with objects at this relative scale. When evaluated at imgsz=1536, that same 10px pedestrian activates the learned crop-scale features → dramatically better detection.

#### Why Full-Image Val mAP Drops

Full-image mAP at 800px drops from 0.452 to ~0.375 during slice fine-tuning. This is **expected and intentional** — the model is recalibrating away from 800px scale. The real gains only appear at imgsz≥1024 post-training. Do not use full-image 800px val as the training quality signal for this phase.

#### Session Resume

Kaggle sessions time out after 9–12 hours. Slice fine-tune for 80 epochs ≈ 8–9 hours on T4. If interrupted:
- `finetune_sliced()` automatically detects `yolo_air_sliced/weights/last.pt`
- Patches stale `train_args` (which still point to the 200-ep run directory)
- Resumes from the saved epoch transparently

---

## 6. Hyperparameters & Configuration

All parameters are centralised in the `CONFIG` dictionary. No magic numbers in function bodies.

### Core Training

```python
CONFIG = {
    'epochs'     : 200,       # Main training duration
    'batch_size' : 2,         # T4 VRAM limit at imgsz=800
    'imgsz'      : 800,       # Balances resolution vs VRAM
    'optimizer'  : 'SGD',     # Better generalisation than Adam for detection
    'lr0'        : 0.01,      # Initial LR (halved from default — pretrained start)
    'lrf'        : 0.005,     # LR final factor → effective final LR = 0.00005
    'momentum'   : 0.937,     # SGD momentum
    'weight_decay': 0.0005,   # L2 regularisation
    'warmup_epochs': 3.0,     # Reduced from 5.0 — warm-start needs less warmup
    'close_mosaic': 30,       # Stop mosaic for last 30 epochs
    'patience'   : 50,        # Early stop after 50 epochs without improvement
    'save_period': 5,         # Save epoch*.pt every 5 epochs (for SWA)
    'nbs'        : 64,        # Nominal batch size for gradient scaling
    'max_det'    : 1000,      # T1-A: was 300, raised for dense VisDrone scenes
    'iou'        : 0.45,      # T1-B: NMS threshold, was 0.7
}
```

### Loss Weights

```python
'box': 7.5,   # Bounding box regression loss weight
'cls': 1.0,   # Classification loss weight
'dfl': 1.5,   # Distribution Focal Loss weight (sub-pixel accuracy for tiny boxes)
```

### TAL Assignment

```python
'tal_topk' : 15,    # Top-K anchor candidates per GT box (default: 10)
'tal_alpha': 0.75,  # Classification weight in alignment score (default: 0.5)
'tal_beta' : 6.0,   # IoU weight in alignment score (default: 6.0)
```

### Rare Class Oversampling (T2-B)

```python
'rare_classes'          : [2, 6, 7],   # bicycle, tricycle, awning-tricycle
'rare_oversample_factor': 2,            # 2× frequency in training set
```

### SAHI Slicing

```python
'sahi_slice_height'  : 640,   # Must match sliced_imgsz exactly
'sahi_slice_width'   : 640,
'sahi_overlap_ratio' : 0.2,   # 20% overlap for boundary objects
'sahi_min_visibility': 0.1,   # Discard labels <10% visible in crop
```

### Sliced Fine-Tune

```python
'sliced_epochs': 80,    # ~8 GPU-hours on T4
'sliced_imgsz' : 640,   # MUST match slice size
'sliced_batch' : 4,     # Larger batch possible at 640px vs 800px
'sliced_lr0'   : 0.005, # Half of main LR — fine-tuning, not learning from scratch
```

---

## 7. Results & Performance

### Training Progress

| Stage | Epoch | mAP@50 | mAP@50-95 | Notes |
|---|---|---|---|---|
| Phase 1 peak | ep52 | 0.4336 | 0.2485 | v6 warm-start advantage visible |
| Phase 2 peak | ep116 | 0.4491 | 0.2598 | Maximum learning phase |
| **Phase 3 best** | **ep145** | **0.4523** | **0.2602** | **Saved as best.pt** |
| Phase 4 end | ep171 | 0.4510 | 0.2592 | Slight regression (close_mosaic) |

### Resolution Sweep (main best.pt)

| imgsz | mAP@50 | mAP@50-95 | Notes |
|---|---|---|---|
| 736 | 0.4294 | 0.2485 | Under-resolves small objects |
| 800 | 0.4506 | 0.2612 | Training resolution |
| 896 | 0.4717 | 0.2732 | +2.1pp |
| 1024 | 0.4824 | 0.2827 | +3.2pp |
| **1152** | **0.4888** | **0.2853** | **SWA peak** |
| 1280 | 0.4849 | 0.2842 | Slight regression |

### Resolution Sweep (sliced best.pt)

Sliced model shifts optimal resolution upward — scale recalibrated to 640px crops.

| imgsz | mAP@50 | mAP@50-95 |
|---|---|---|
| 640 | 0.3773 | 0.2195 |
| 800 | 0.4348 | 0.2595 |
| 1024 | 0.4887 | 0.2962 |
| 1152 | 0.5061 | 0.3087 |
| 1280 | 0.5198 | 0.3177 |
| 1408 | 0.5304 | 0.3250 |
| **1536** | **0.5412** | **0.3337** | **Production peak** |
| 1664 | 0.5403 | 0.3326 | Marginal regression |

### Final Scorecard

| Stage | Config | mAP@50 | Gain over baseline |
|---|---|---|---|
| YOLOv8n baseline | 800px | ~0.260 | — |
| v10 best.pt | 800px | 0.452 | +19.2pp (+74% relative) |
| v10 SWA | 1152px | 0.489 | +22.9pp (+88% relative) |
| **v10 sliced best.pt** | **1536px** | **0.541** | **+28.1pp (+108% relative)** |

### Per-Class Performance

Measured at imgsz=1280, sliced best.pt:

| Class | Before Slice FT | After Slice FT | Change |
|---|---|---|---|
| car | 0.862 | 0.895 | +0.033 |
| pedestrian | 0.578 | 0.676 | +0.098 |
| motor | 0.578 | 0.660 | +0.082 |
| bus | 0.640 | 0.684 | +0.044 |
| people | 0.466 | 0.552 | +0.086 |
| van | 0.533 | 0.587 | +0.054 |
| truck | 0.426 | 0.496 | +0.070 |
| tricycle | 0.348 | 0.439 | +0.091 |
| **bicycle** | **0.200** | **0.319** | **+0.119 (+59.5%)** |
| awning-tricycle | 0.194 | 0.265 | +0.071 |
| others | 0.141 | 0.100 | -0.041 |

Bicycle saw the largest relative gain (+59.5%), confirming that slice fine-tuning most benefits the rarest and smallest classes. The "others" class regression is expected — it is an ill-defined catch-all.

---

## 8. Benchmark Comparisons

### vs. Standard YOLO on VisDrone val

| Model | Params (M) | mAP@50 | Config |
|---|---|---|---|
| YOLOv8n (COCO init) | 3.2 | ~26% | Standard 640px |
| YOLOv8n (VisDrone init) | 3.2 | ~36% | Standard 800px |
| YOLOv8s | 11.2 | ~42% | Standard 800px |
| YOLOv8m | 25.9 | ~48% | Standard 800px |
| BIFOVEAL-YOLO v10 (800px) | **2.6** | **45.2%** | Our model, standard inference |
| BIFOVEAL-YOLO v10 (1536px) | **2.6** | **54.1%** | Our model, optimal resolution |

> BIFOVEAL-YOLO at 1536px outperforms YOLOv8m (10× larger) using only 2.6M parameters.

### vs. Published Aerial Detection Methods

| Method | mAP@50 | Params |
|---|---|---|
| ClusDet (ECCV 2019) | ~40% | Large |
| YOLO-DroneDetect | ~44% | Medium |
| UFPMP-Det (AAAI 2022) | ~49% | Large |
| TPH-YOLOv5 (ICCV 2021) | ~51% | 86M |
| **BIFOVEAL-YOLO v10** | **54.1%** | **2.6M** |

> Note: Direct comparison requires identical test conditions. These figures use VisDrone-DET 2019 val. Published results from respective papers.

---

## 9. Deployment

### Production Inference Configuration

```python
from ultralytics import YOLO

model = YOLO('yolo_air_sliced/weights/best.pt')

results = model.predict(
    source='path/to/drone/image.jpg',
    imgsz=1536,     # Critical — do NOT reduce below 1152
    iou=0.45,       # NMS threshold for dense scenes
    max_det=1000,   # VisDrone density requirement
    conf=0.25,      # Deployment threshold (0.001 for mAP evaluation)
)
```

### SAHI Mode (Maximum Accuracy)

For offline analysis or when latency is not critical:

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

sahi_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path='yolo_air_sliced/weights/best.pt',
    confidence_threshold=0.25,
    device='cuda:0',
)

result = get_sliced_prediction(
    image_path,
    sahi_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    postprocess_type='GREEDYNMM',
    postprocess_match_threshold=0.5,
)
```

### Performance by Hardware

| Platform | VRAM | imgsz | Est. FPS | mAP@50 | Status |
|---|---|---|---|---|---|
| Server GPU (A100/V100) | 40GB+ | 1536 | ~50 FPS | 54.1% | ✅ Full production |
| Jetson AGX Orin 64GB | 64GB | 1536 | ~20 FPS | 54.1% | ✅ Full deployment |
| Jetson Orin NX 16GB | 16GB | 1152 | ~20 FPS | 50.6% | ✅ With TensorRT |
| Jetson Xavier NX | 8GB | 1024 | ~15 FPS | 48.9% | ✅ Acceptable |
| Jetson Orin Nano | 8GB | 896 | ~12 FPS | 45.9% | ⚠️ Marginal |
| Jetson Nano | 4GB | 640 | ~10 FPS | 37.7% | ⚠️ Low accuracy |
| Consumer drone SoC | <2GB | — | — | — | ❌ Not supported |

### TensorRT Export (for Jetson)

```bash
# Export to TensorRT FP16 (2–4× speedup on Jetson)
yolo export model=yolo_air_sliced/weights/best.pt format=engine half=True
```

> **Custom modules note:** ASFM, SPDSECAConv, FeatureSR, and LightweightP2Prep use only standard PyTorch ops (Conv2d, AdaptiveAvgPool2d, cat, etc.) that TensorRT supports natively. Export should succeed without custom op registration.

---

## 10. File Structure

```
/kaggle/working/
│
├── yolo_air_ultimate/              # Main 200-epoch training run
│   ├── weights/
│   │   ├── best.pt                 # Best checkpoint (ep145, mAP=45.2%)
│   │   ├── last.pt                 # Final epoch checkpoint
│   │   └── epoch{0,5,10...195}.pt  # Periodic checkpoints for SWA
│   └── results.csv                 # Per-epoch training metrics
│
├── yolo_air_sliced/                # SAHI slice fine-tune run
│   ├── weights/
│   │   ├── best.pt                 # PRODUCTION WEIGHTS (mAP=54.1% @1536px)
│   │   ├── last.pt
│   │   └── epoch{0,5,10...75}.pt
│   └── results.csv
│
├── v10_swa.pt                      # SWA of main run ep175-195
├── v10_sliced_swa.pt               # SWA of sliced run ep50-65 (not recommended)
│
├── visdrone.yaml                   # Main training data config
├── visdrone_sliced.yaml            # Sliced fine-tune data config
├── yolo-air-ultimate.yaml          # Model architecture definition
│
├── images/
│   ├── train/                      # 6,471 converted training images
│   ├── val/                        # 548 converted validation images
│   └── train_sliced/               # ~36,097 sliced 640×640 crops
│
└── labels/
    ├── train/                      # YOLO format labels (+ _rare* copies)
    ├── val/
    └── train_sliced/               # Labels for sliced crops
```

### Download Priority

For inference/deployment, download in this order:

```
1. yolo_air_sliced/weights/best.pt   → PRODUCTION — sliced fine-tune (54.1%)
2. yolo_air_ultimate/weights/best.pt → Fallback — main training (45.2%)
3. v10_swa.pt                        → High-res inference (48.9% @1152px)
```

---

## 11. Execution Guide

### First-Time Setup

```python
# In CONFIG, set:
CONFIG['mode'] = 'train'
CONFIG['v6_weights'] = '/path/to/v6/best.pt'  # or None for COCO init
CONFIG['data_path']  = None  # auto-detect VisDrone on Kaggle
```

### Step-by-Step Execution Order

#### Step 1: Main Training (~30 GPU-hours)
```python
CONFIG['mode'] = 'train'
# Run notebook cell — auto-resumes if interrupted
```

#### Step 2: SWA (~5 minutes)
```python
CONFIG['mode'] = 'swa'
# Averages last 5 epoch*.pt checkpoints
# Result: v10_swa.pt
```

#### Step 3: Slice Fine-Tune (~8 GPU-hours)
```python
CONFIG['mode'] = 'finetune_sliced'
# Slices dataset then fine-tunes 80 epochs
# Auto-resumes if interrupted
# Result: yolo_air_sliced/weights/best.pt
```

#### Step 4: Resolution Sweep (manual, ~10 minutes)
```python
from ultralytics import YOLO
model = YOLO('yolo_air_sliced/weights/best.pt')
for sz in [640, 800, 896, 1024, 1152, 1280, 1408, 1536, 1664]:
    results = model.val(data='visdrone.yaml', imgsz=sz, iou=0.45)
    print(f"imgsz={sz}: mAP50={results.box.map50:.4f}")
```

#### Step 5: SAHI Validation Sample (~10 minutes)
```python
CONFIG['mode'] = 'infer_sahi'
# Evaluates on 20-image val sample with SAHI
```

### Execution Mode Reference

| Mode | Function | Duration |
|---|---|---|
| `'train'` | `train()` | ~30 GPU-hours |
| `'swa'` | `run_swa()` | ~5 minutes |
| `'finetune_sliced'` | `prepare_sliced_dataset()` + `finetune_sliced()` | ~8 GPU-hours |
| `'infer_sahi'` | `run_sahi_inference()` | ~10 minutes |

---

## 12. Known Limitations

### Technical

| Limitation | Description | Workaround |
|---|---|---|
| `augment=True` disabled | Custom modules break TTA | Use multi-resolution sweep instead |
| Resolution dependency | mAP drops significantly below 1024px | Always deploy at imgsz≥1152 |
| SWA on sliced runs | Does not improve over best.pt | Skip SWA after slice fine-tune |
| "others" class | Performance degraded post fine-tune | Consider excluding from evaluation |
| VRAM at 1536px | ~6GB for inference, ~14GB for training | Use imgsz=1152 on 6GB GPUs |
| TTA incompatibility | ASFM dilated convs break some TTA augmentations | Not a deployment concern |

### Dataset

- Coverage limited to Chinese urban environments; generalisation to other geographies unverified
- Predominantly daytime footage; night/adverse weather performance untested
- Calibrated for DJI-style optics; footage from fixed-wing UAVs may require fine-tuning

### Deployment

- 1536px inference at ~45ms/image on T4 — real-time but not ultra-low-latency
- SAHI mode (200–400ms/image) unsuitable for real-time applications
- No temporal tracking; requires external tracker (ByteTrack, BoT-SORT) for video
- Confidence not calibrated; threshold tuning needed per deployment scenario

---

## 13. Future Work

### Short-Term (High Impact, Minimal Architecture Change)

| Enhancement | Expected Gain | Notes |
|---|---|---|
| WIoU v3 loss | +1–2.5pp base mAP | Replace CIoU; dynamic outlier weighting for tiny boxes |
| SWA window fix | +0.5pp | Target Phase 3 peak (ep130–165) not Phase 4 |
| Extend to 120 fine-tune epochs | +0.5–1pp | Loss still declining at ep80 |
| Rare oversample factor 3 | +2–4pp bicycle class | Increase from 2 to 3 for classes 2, 6, 7 |
| Soft-NMS post-processing | +0.5–1.5pp | Decays confidence of overlapping boxes vs hard suppression |

### Medium-Term (Architecture)

| Enhancement | Expected Gain | Notes |
|---|---|---|
| BiFPN neck | +1.5–3pp | Weighted bidirectional fusion replaces PANet |
| CARAFE upsampling | +1–2pp | Content-aware upsampling replaces nearest-neighbour |
| Copy-Reduce-Paste augmentation | +2–4pp rare classes | Scale-down before paste for rare tiny objects |
| DyHead at P2 | +1–2pp | Scale+spatial+task attention at detection head |

### Long-Term (System-Level)

- **Altitude-conditioned detection:** Inject estimated altitude as conditioning signal for explicit scale adaptation
- **Multi-frame temporal fusion:** Stack consecutive frames to improve detection of stationary partially-occluded objects
- **Knowledge distillation:** Distil into a 1M parameter student for Jetson Nano / mobile deployment
- **Domain adaptation pipeline:** Standardised fine-tune workflow for new geographies using the slice fine-tune infrastructure

---

## Quick Reference

### Production Command

```python
from ultralytics import YOLO
model = YOLO('yolo_air_sliced/weights/best.pt')
results = model.predict(source=image, imgsz=1536, iou=0.45, max_det=1000, conf=0.25)
```

### Key Metrics

```
Parameters   : 2,628,599
GFLOPs       : 21.5
Best mAP@50  : 54.12%  (sliced best.pt @ imgsz=1536)
Best mAP@50-95: 33.37% (sliced best.pt @ imgsz=1536)
Training time : ~38 GPU-hours total (30h main + 8h slice)
Hardware      : NVIDIA Tesla T4 (14.6 GB VRAM)
```

### Architecture Summary

```
Backbone : Conv → SECAConv → C2f → SPDSECAConv×2 → SECAConv → C2f → SPPF
Neck     : PANet FPN with FeatureSR refinement at P2/P3 levels
Head     : 4-output Detect (P2 micro + P3 + P4) with ASFM at P2
```

---

*BIFOVEAL-YOLO v10.1 — March 2026*
