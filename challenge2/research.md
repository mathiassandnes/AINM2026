# Winning NM i AI 2026: Grocery shelf detection strategy guide

**A two-stage pipeline — YOLOv8m for detection with DINOv2 embedding-based classification — is the highest-impact strategy for this competition.** The 0.7/0.3 detection-classification scoring split means detection quality is 2.3× more valuable per mAP point, but the reference product images are an underexploited asset that can dramatically boost classification. With only 248 training images across 357 classes (~0.7 images per class on average), an end-to-end 357-class YOLO model will severely underperform on classification. The two-stage approach decouples these problems: detect "products" generically at high recall, then classify each crop by matching DINOv2 embeddings against the 2,289 reference images. Given the L4 GPU's generous compute budget (300 seconds for ~248 images), this pipeline runs with room to spare for TTA and multi-model ensembling via Weighted Boxes Fusion.

---

## Strategy ranking by expected score impact

Before diving deep, here is the prioritized action list. Each strategy is ranked by its expected contribution to the final score (0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5):

| Rank | Strategy | Score impact | Effort |
|------|----------|-------------|--------|
| 1 | Train at imgsz=1280 (not 640) | +0.03–0.06 | Low |
| 2 | Two-stage: YOLO detect → DINOv2 classify | +0.02–0.05 | Medium |
| 3 | YOLOv8m/l as primary detector (not n/s) | +0.02–0.04 | Low |
| 4 | Strong augmentation (mosaic, copy-paste, HSV) | +0.02–0.03 | Low |
| 5 | WBF ensemble of 2–3 YOLO models | +0.01–0.03 | Medium |
| 6 | Test-time augmentation (flip + multi-scale) | +0.01–0.02 | Low |
| 7 | Unknown product handling via confidence gating | +0.005–0.02 | Medium |
| 8 | NMS IoU threshold tuning for dense scenes | +0.005–0.015 | Trivial |
| 9 | Prediction confidence floor at 0.001 | +0.005–0.01 | Trivial |
| 10 | Copy-paste reference products onto shelf backgrounds | +0.01–0.02 | High |

---

## Choosing the right model architecture

### YOLOv8 variants: size, speed, and the overfitting cliff

The 420 MB model limit and 300-second inference window are generous — even YOLOv8x processes 248 images in under 2 seconds at 640px on an L4 GPU. The real constraint is **overfitting on 248 images**, not compute.

| Variant | Params | COCO mAP@50-95 | FP16 ONNX (357 classes) | L4 inference (ms/img, 640px) |
|---------|--------|----------------|------------------------|------------------------------|
| YOLOv8n | 3.4M | 37.3 | ~7 MB | ~1.5 |
| YOLOv8s | 11.3M | 44.9 | ~23 MB | ~2.0 |
| **YOLOv8m** | **26.1M** | **50.2** | **~52 MB** | **~3.0** |
| YOLOv8l | 43.9M | 52.9 | ~88 MB | ~4.0 |
| YOLOv8x | 68.5M | 53.9 | ~137 MB | ~6.0 |

**YOLOv8m is the sweet spot.** It captures 93% of YOLOv8x's COCO accuracy at 38% of the parameters. With only 248 training images, YOLOv8x's 68M parameters risk memorizing the training set. The marginal gain from m→l→x on COCO (+2.7 and +1.0 mAP) disappears when data is scarce. YOLOv8m provides enough capacity for 357 classes — the detection head adds negligible parameters when scaling from 80 to 357 classes (only **+0.2M parameters**, or ~1.2% increase). Train both YOLOv8m and YOLOv8l, then ensemble them — this outperforms a single YOLOv8x while reducing overfitting risk.

### RT-DETR: available but not recommended

RT-DETR-L (53.0% AP, ~45M params) is available in ultralytics 8.1.0 via `RTDETR("rtdetr-l.pt")`, but it has three problems for this scenario: transformer attention mechanisms consume significantly more VRAM during training (limiting batch size on L4), AMP training can produce NaN outputs, and its advantages in NMS-free inference matter less when you have 300 seconds of compute. **Stick with YOLOv8.**

---

## The two-stage pipeline: detection then classification

### Why end-to-end 357-class YOLO fails here

With 22,700 annotations across 357 classes, the average class has ~63 annotations — but the distribution is long-tailed, meaning many classes have fewer than 10 examples. YOLO's classification head learns per-class features from training data alone. It cannot use the reference product images at all. This is a fundamental limitation: **your most valuable asset (7 reference views per product) goes unused in end-to-end training.**

### The architecture that wins

```
Shelf image → YOLOv8m (class-agnostic or 4 super-categories) → Crop detections
    → DINOv2 ViT-B embed each crop → FAISS cosine similarity → Product class ID
```

**Stage 1 — Detection:** Train YOLOv8m to detect "product" as a single class (or 4 classes matching the store sections: Egg, Frokost, Knekkebrod, Varmedrikke). Class-agnostic detection with dense shelf data is well-studied — YOLOv8 achieves **F1=89% on SKU-110K** (a 1.7M-annotation shelf detection benchmark) with just 25 training epochs. With far fewer classes to distinguish, the detector achieves higher recall on the 248 training images.

**Stage 2 — Classification:** Extract DINOv2 embeddings from each detection crop, then find the nearest reference embedding via cosine similarity. DINOv2's self-supervised training on 142M images produces features that generalize remarkably well without fine-tuning — its KoLeo regularization loss specifically optimizes for nearest-neighbor retrieval quality, improving retrieval by **8%+ over standard ViT features**.

### DINOv2 model selection and availability

DINOv2 is confirmed available in **timm 0.9.12** (added in v0.9.0). Load it directly:

```python
import timm
model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
```

| DINOv2 variant | Params | Embedding dim | FP16 size | Inference (100 crops, 224×224) |
|---------------|--------|--------------|-----------|-------------------------------|
| ViT-S/14 | 21M | 384 | ~42 MB | ~0.15s |
| **ViT-B/14** | **86M** | **768** | **~172 MB** | **~0.4s** |
| ViT-L/14 | 304M | 1024 | ~609 MB | ~1.2s |

**DINOv2 ViT-B/14 at FP16 (~172 MB) is the recommendation.** Combined with YOLOv8m FP16 (~52 MB), the total is ~224 MB — well within the 420 MB limit, leaving room for a second YOLO model. ViT-S/14 works as a lighter fallback at 42 MB.

### Reference embedding database

Pre-compute embeddings for all 327 products × 7 views = 2,289 reference images. Store as a NumPy file:

- **ViT-B (768-dim):** 2,289 × 768 × 4 bytes = **7.0 MB** — negligible storage
- At inference, FAISS `IndexFlatIP` on 2,289 L2-normalized vectors completes search in **<1ms per query**

For classification, use **top-1 nearest neighbor** with cosine similarity. Each shelf image produces ~50–150 detection crops; embedding and matching all crops takes **0.5–1.0 seconds per image**. For 248 images, the entire classification stage completes in ~125–250 seconds, leaving ample headroom within the 300-second budget alongside YOLO detection.

### Complete inference pipeline code

```python
import timm, torch, numpy as np, faiss
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms

# Load models
yolo = YOLO('yolov8m_best.pt')
embed_model = timm.create_model('vit_base_patch14_dinov2.lvd142m', 
                                 pretrained=True, num_classes=0).eval().cuda().half()

# Pre-computed reference database
ref_embeddings = np.load('ref_embeddings.npy')  # (2289, 768), L2-normalized
ref_labels = np.load('ref_labels.npy')           # (2289,) product IDs
index = faiss.IndexFlatIP(768)
index.add(ref_embeddings.astype(np.float32))

crop_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

for img_path in test_images:
    # Stage 1: Detect products
    results = yolo(img_path, conf=0.15, iou=0.65, imgsz=1280)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    det_confs = results[0].boxes.conf.cpu().numpy()
    
    # Stage 2: Classify each crop
    img = Image.open(img_path).convert('RGB')
    crops = [crop_transform(img.crop(b)) for b in boxes]
    if len(crops) == 0:
        continue
    batch = torch.stack(crops).cuda().half()
    
    with torch.no_grad():
        embeddings = embed_model(batch).cpu().float().numpy()
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    D, I = index.search(embeddings.astype(np.float32), k=1)
    predicted_classes = ref_labels[I[:, 0]]
    similarity_scores = D[:, 0]
    
    # Unknown product gating: low similarity → class 356
    predicted_classes[similarity_scores < 0.35] = 356
```

---

## Training the detection stage: hyperparameters and augmentation

### Optimal YOLOv8 training configuration

The critical insight for 248 images: **use heavy augmentation, freeze-then-unfreeze training, and large image resolution.** Products on shelves are small and densely packed — training at 1280px rather than 640px yields **+2–5 mAP points** on dense detection benchmarks.

```python
from ultralytics import YOLO

# Stage 1: Frozen backbone (10 epochs)
model = YOLO('yolov8m.pt')
model.train(
    data='grocery.yaml',
    epochs=10,
    imgsz=1280,
    batch=4,              # Conservative for 1280px on L4 24GB
    freeze=10,            # Freeze backbone layers
    lr0=0.001,
    optimizer='AdamW',
    amp=True,
    cache='ram',          # Only 248 images — fits in RAM easily
)

# Stage 2: Full fine-tuning (150+ epochs)
model = YOLO('runs/detect/train/weights/best.pt')
model.train(
    data='grocery.yaml',
    epochs=200,
    imgsz=1280,
    batch=4,
    lr0=0.01,
    lrf=0.01,             # Cosine decay to 0.0001
    optimizer='SGD',
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    
    # Augmentation (critical)
    mosaic=1.0,
    mixup=0.1,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0005,
    fliplr=0.5,
    flipud=0.0,           # Never flip shelves vertically
    close_mosaic=20,      # Disable mosaic last 20 epochs
    
    # Regularization
    patience=30,          # Early stopping
    amp=True,
    cls=1.5,              # Increase classification loss weight for 357 classes
    cache='ram',
)
```

### Dataset format conversion

Ultralytics expects YOLO-format labels, not COCO JSON. Convert with:

```python
import json, os

with open('annotations.json') as f:
    coco = json.load(f)

img_lookup = {img['id']: img for img in coco['images']}
os.makedirs('labels/train', exist_ok=True)

for ann in coco['annotations']:
    img = img_lookup[ann['image_id']]
    x, y, w, h = ann['bbox']  # COCO format: x,y,w,h (absolute)
    # Convert to YOLO: x_center, y_center, width, height (normalized)
    xc = (x + w/2) / img['width']
    yc = (y + h/2) / img['height']
    wn = w / img['width']
    hn = h / img['height']
    
    label_file = os.path.join('labels/train', 
                              os.path.splitext(img['file_name'])[0] + '.txt')
    with open(label_file, 'a') as f:
        f.write(f"{ann['category_id']} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
```

### Why these augmentation choices matter for shelves

**Mosaic (1.0)** is the single most impactful augmentation for small datasets. It combines 4 images into a grid, effectively quadrupling data diversity and forcing the model to handle products at different scales and positions. With 248 images, some mosaics will reuse images — this is expected and still beneficial.

**Copy-paste augmentation** deserves special attention. The "Simple Copy-Paste" paper showed **+10 AP improvement in low-data regimes**. For this competition, you can paste cropped reference product images onto shelf backgrounds. YOLOv8's native `copy_paste` parameter requires segmentation masks, so implement it externally using albumentations or a custom data loader that crops product regions from reference images and composites them onto shelf scenes.

**Close_mosaic=20** is critical: disabling mosaic for the final 20 epochs lets the model calibrate on full-resolution, undistorted images, improving bounding box precision. This matters because mAP@0.5 still requires 50% IoU — imprecise boxes from mosaic-trained models lose points.

**flipud=0.0** is non-negotiable: shelves have gravity. Vertical flips create impossible scenes that confuse the model.

---

## Ensembling and test-time augmentation within constraints

### Multi-model ensemble budget

The 420 MB limit accommodates impressive ensembles in FP16:

| Ensemble configuration | Total size | Expected gain |
|----------------------|-----------|--------------|
| YOLOv8m + YOLOv8l + DINOv2-B (all FP16) | ~312 MB | Baseline two-stage |
| YOLOv8m + YOLOv8l + YOLOv8s + DINOv2-B (all FP16) | ~335 MB | +0.01–0.02 |
| YOLOv8x + YOLOv8m + DINOv2-B (all FP16) | ~361 MB | +0.01–0.03 |

Train YOLOv8m and YOLOv8l with different random seeds or augmentation configs, then merge their detections with **Weighted Boxes Fusion (WBF)**. WBF averages overlapping box coordinates weighted by confidence — unlike NMS, it uses all predictions rather than discarding overlapping ones, producing better-localized boxes. WBF won top places in Open Images and COCO detection challenges.

```python
from ensemble_boxes import weighted_boxes_fusion

# Normalize boxes to [0,1], run both models
boxes_list = [boxes_model_m / img_size, boxes_model_l / img_size]
scores_list = [scores_model_m, scores_model_l]
labels_list = [labels_model_m, labels_model_l]

boxes, scores, labels = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    weights=[0.4, 0.6],    # Weight larger model higher
    iou_thr=0.5,           # Tune on validation: 0.4-0.6
    skip_box_thr=0.0001    # Keep all predictions
)
```

**Important:** The `ensemble_boxes` package must be pre-installed or bundled — check the sandbox. If unavailable, implement a simple WBF from scratch (the algorithm is ~50 lines of Python).

### Test-time augmentation: free performance

YOLOv8's built-in TTA applies horizontal flip + multi-scale inference at three resolutions. Enable it with `augment=True`:

```python
results = model.predict(source=img_path, augment=True, conf=0.001, iou=0.65)
```

TTA adds **+1.0–1.5 mAP@0.5 points** based on YOLOv5/v8 COCO benchmarks, at a cost of 2–3× inference time. Since YOLOv8m at 1280px takes ~8ms per image on L4, TTA brings it to ~24ms — still only **6 seconds total for 248 images**. This is essentially free performance.

### FP16 export for inference speed

```python
model = YOLO('best.pt')
model.export(format='onnx', imgsz=1280, half=True, simplify=True, opset=17)
```

FP16 halves model size with **<0.1% mAP loss**. The L4 GPU has native FP16 Tensor Core support. For maximum speed, export to TensorRT `.engine` format, which provides 2–5× speedup over PyTorch inference.

---

## Handling the unknown product class (ID 356)

Category 356 ("unknown_product") is a proper evaluation class — **you must predict it to avoid losing mAP on both detection and classification**. Missing all unknown products penalizes detection_mAP (false negatives) and misclassifying them as known products penalizes classification_mAP (false positives for the wrong class, false negatives for unknown).

### The embedding similarity gating approach

In the two-stage pipeline, unknown product handling emerges naturally: if a detection crop's maximum cosine similarity to any reference embedding falls below a threshold, classify it as unknown. This is more principled than YOLO's approach — standard YOLO models assign **0.95–0.99 confidence to OOD objects** while randomly labeling them as known classes.

**Threshold calibration:** On your validation set, plot similarity score distributions for correct matches versus incorrect/unknown matches. The optimal threshold typically falls between **0.25–0.45** cosine similarity. Start at 0.35 and tune. A helpful heuristic: if the top-1 and top-2 nearest neighbors belong to different products and both have similar scores, the crop is likely unknown.

### For end-to-end YOLO (fallback strategy)

If using a single 357-class YOLO model, include unknown_product as class 356 in training. Ensure your training annotations label unknown products correctly. The model will learn the visual pattern of "unusual product" — which may generalize poorly to novel unknowns. Supplement with an entropy-based filter: if the softmax distribution over classes is nearly uniform (high entropy), reclassify as unknown.

---

## Exploiting reference images with metric learning

### When frozen DINOv2 k-NN isn't enough: ArcFace fine-tuning

Frozen DINOv2 embeddings provide a strong baseline (estimated **70–85% Top-1 accuracy** on shelf crops based on RP2K retrieval benchmarks). To push further, fine-tune a projection head with ArcFace loss on augmented reference images:

```python
import timm, torch, torch.nn as nn

# Frozen DINOv2 backbone
backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', 
                              pretrained=True, num_classes=0).eval().cuda()
for p in backbone.parameters():
    p.requires_grad = False

# Trainable ArcFace head
class ArcFaceClassifier(nn.Module):
    def __init__(self, embed_dim=768, proj_dim=256, num_classes=357, s=64, m=0.5):
        super().__init__()
        self.proj = nn.Linear(embed_dim, proj_dim)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, proj_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
    
    def forward(self, features, labels=None):
        x = self.proj(features)
        x = nn.functional.normalize(x)
        W = nn.functional.normalize(self.weight)
        cosine = x @ W.t()
        if labels is not None:  # Training: add angular margin
            one_hot = torch.zeros_like(cosine).scatter_(1, labels.unsqueeze(1), 1)
            theta = torch.acos(torch.clamp(cosine, -1+1e-7, 1-1e-7))
            cosine = torch.cos(theta + one_hot * self.m)
        return self.s * cosine

head = ArcFaceClassifier().cuda()
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
```

Train this head on the 2,289 reference images (heavily augmented — random crop, color jitter, rotation, scale) plus any shelf crops you extract from the 248 training images. **ArcFace handles class imbalance naturally** and produces embeddings optimized for open-set recognition, which helps with the unknown product class.

### The RetailKLIP precedent

RetailKLIP (2023) fine-tuned CLIP's vision backbone with ArcFace on retail products and achieved strong zero-shot recognition on unseen retail datasets. Their key technique — blockwise learning rate decay (top ViT layers get lr=2e-4, decaying 0.7× per block downward) — prevents catastrophic forgetting of pretrained features while adapting to the retail domain.

---

## Competition-specific scoring optimizations

### The 0.7/0.3 split demands recall-first thinking

A +0.01 improvement in detection mAP contributes **+0.007** to the final score, while the same improvement in classification contributes only +0.003. This means detection improvements are **2.33× more valuable**. Practical implications:

- **Use low confidence thresholds (0.001) for prediction output.** mAP evaluation sweeps all thresholds via the precision-recall curve. High confidence thresholds filter out potential true positives.
- **Tune NMS IoU threshold to 0.6–0.7 for dense shelves.** Standard 0.45 aggressively suppresses adjacent products — the #1 failure mode in retail detection. Products side-by-side on shelves can have high IoU when detected with slightly imprecise boxes.
- **mAP@0.5 is lenient on localization.** A box overlapping 50% with ground truth counts as correct. Prioritize finding every product over pixel-perfect boxes.
- **Cross-validate on local splits first.** With 3 submissions/day, you cannot afford trial-and-error. Split 248 images into 5 stratified folds (stratified by store section) and evaluate locally. Submit only when local mAP improves by ≥0.005.

### Per-section specialization

The 4 store sections (Egg, Frokost, Knekkebrod, Varmedrikke) likely have distinct product categories. If you can identify which section a test image belongs to (via a simple classifier or metadata), restrict the classification search space to only products from that section. This reduces confusion between visually similar products across sections and speeds up FAISS search.

---

## Common pitfalls in retail shelf detection

**Dense NMS suppression** is the most common failure. Adjacent products with slightly imprecise boxes get IoU > 0.45 and one gets suppressed. Always use higher NMS IoU thresholds (0.6–0.7) or Soft-NMS, which decays confidence of overlapping boxes with a Gaussian function rather than hard deletion.

**Overfitting to shelf layouts** happens quickly with 248 images. The model memorizes product positions rather than learning product appearance. Strong spatial augmentation (mosaic, translation, scale) disrupts positional priors.

**Ignoring the `close_mosaic` parameter** leads to poor bounding box calibration. Mosaic creates artificial boundaries between merged images; training final epochs without mosaic lets the model adjust to clean single-image predictions.

**Training and inference resolution mismatch** causes silent accuracy drops. If you train at 1280px, inference must also use 1280px. Mixing resolutions between training and inference degrades mAP by 2–5 points.

**Not analyzing class distribution** before training wastes effort. Run a simple frequency analysis on your COCO annotations — identify which of the 357 classes have fewer than 10 examples, and consider image-level oversampling or copy-paste augmentation specifically for those rare classes.

---

## Recommended 2–3 week implementation roadmap

**Week 1 — Foundation (days 1–7)**

Days 1–2: Set up data pipeline. Convert COCO annotations to YOLO format. Analyze class distribution. Create 5-fold stratified validation splits by store section. Pre-compute DINOv2 reference embeddings for all 327 products × 7 views.

Days 3–4: Train baseline YOLOv8m end-to-end (357 classes, imgsz=1280, strong augmentation). Evaluate on local validation. Submit once to calibrate local-vs-leaderboard correlation. Expected baseline: ~0.45–0.55 final score.

Days 5–7: Implement two-stage pipeline. Train class-agnostic YOLOv8m detector. Wire up DINOv2 embedding extraction + FAISS k-NN classification. Test unknown product thresholding. Submit and compare against baseline. Expected: +0.03–0.08 improvement.

**Week 2 — Optimization (days 8–14)**

Days 8–9: Train YOLOv8l as second detector. Implement WBF ensemble of YOLOv8m + YOLOv8l. Add TTA to inference. Evaluate ensemble locally.

Days 10–11: Implement copy-paste augmentation using reference product images pasted onto shelf backgrounds. Retrain detectors with copy-paste enabled. Experiment with `cls` loss weight (try 1.0, 1.5, 2.0).

Days 12–13: Fine-tune ArcFace projection head on DINOv2 features using reference images + augmented shelf crops. Compare k-NN vs ArcFace classification accuracy. Tune unknown product similarity threshold on validation.

Day 14: Consolidate best configuration. Submit optimized pipeline. Expected: ~0.60–0.75 final score.

**Week 3 — Polish and edge cases (days 15–21)**

Days 15–16: Try pseudo-labeling. Run best model on test images, use high-confidence predictions (>0.7) as additional training data, retrain. Evaluate if this helps or hurts on validation.

Days 17–18: Experiment with per-section models or per-section classification search spaces. Try adding YOLOv8s as a third ensemble member. Optimize all thresholds (NMS IoU, confidence, unknown similarity, WBF IoU).

Days 19–20: Export all models to FP16 ONNX or TensorRT. Verify end-to-end pipeline runs within 300 seconds. Run final ablation studies.

Day 21: Final submission with best validated configuration. Reserve 1–2 submissions for last-minute threshold adjustments.

---

## Conclusion

The winning formula for this competition has three pillars. **First**, train detection at 1280px with aggressive augmentation — this alone separates competitive from mediocre submissions in dense shelf detection. **Second**, exploit the reference images through DINOv2 embedding matching — this transforms a hopeless 357-class-with-248-images classification problem into a tractable retrieval task where frozen foundation model features shine. **Third**, ensemble multiple detector sizes with WBF and apply TTA — these are essentially free mAP points given the generous compute budget. The scoring formula's 0.7 detection weighting means every improvement in recall matters most, so tune for finding every product rather than being certain about what it is. The two-stage pipeline naturally handles the unknown product class through similarity gating, avoiding the well-documented failure mode where YOLO assigns near-perfect confidence to out-of-distribution objects. Budget your three daily submissions carefully — invest Week 1 in establishing that your local validation correlates with the leaderboard, then trust local evaluation for rapid experimentation.