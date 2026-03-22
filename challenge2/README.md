# Challenge 2: NorgesGruppen Data — Grocery Object Detection

## Overview
Detect and classify grocery products on Norwegian store shelf images. Train a detector, submit as ZIP with code running in GPU-enabled sandbox.

## Format
- **Submission:** ZIP file with `run.py` at root
- **Max model weights:** 420 MB
- **Max files:** 50
- **Execution timeout:** 300 seconds (5 min)

## Training Data
- **COCO dataset:** 248 shelf images, ~22,700 annotations, 356 product categories
- **Product reference images:** 327 products with multi-angle photos (main, front, back, left, right, top, bottom)
- **Format:** bbox as `[x, y, width, height]` in COCO format

## Sandbox Environment
- **GPU:** NVIDIA L4, 24GB VRAM
- **Pre-installed:** PyTorch 2.6.0, ultralytics 8.1.0, onnxruntime-gpu, OpenCV, albumentations, timm, supervision
- **No pip install at runtime**
- **BLOCKED modules:** os, sys, subprocess, pickle, requests, urllib, multiprocessing, threading, eval(), exec()

## run.py Contract
```bash
python run.py --images /data/images/ --output /output/predictions.json
```
- Input: JPEG files at `/data/images/` (format: `img_XXXXX.jpg`)
- Output: JSON array of `{bbox, category_id, confidence}`

## Scoring
| Component | Weight | What |
|-----------|--------|------|
| Detection mAP@0.5 | **70%** | Localization accuracy (category ignored) |
| Classification mAP@0.5 | **30%** | Correct category_id + IoU ≥ 0.5 |

- Detection-only submissions score **up to 0.70**
- Score range: 0–1.0

## Strategy

### Phase 1: Detection (get the 70%)
Fine-tune YOLOv8/v11 on the 248 shelf images. All 22,700 annotations as a single "product" class. This alone can score up to 0.70.

### Phase 2: Classification (get the remaining 30%)
Use 327 product reference images (multi-angle) for few-shot classification:
1. Crop detected boxes from YOLO
2. Embed with CLIP or DINOv2
3. Match to pre-computed reference image embeddings (nearest neighbor)
4. 356 categories but 327 with reference images — handle gap with fallback

### Size Budget
- YOLOv8m: ~50 MB
- CLIP ViT-B/32: ~340 MB
- Reference embeddings: <5 MB
- **Total: ~395 MB** (under 420 MB limit)

### Key Gotchas
- `os` module is blocked — can't use `os.listdir()`. Use `pathlib` instead.
- `subprocess` blocked — no shell commands
- 300s for full inference on all test images — need efficient pipeline
- 16-bit vs 8-bit trap from DM i AI history — watch for image format issues

## Priority: HIGH
CV is a strength. Detection well-understood. Classification with reference images is the interesting ML challenge.
