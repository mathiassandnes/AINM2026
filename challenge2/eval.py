"""Local evaluation — mirrors competition scoring.

70% detection mAP@0.5 (category ignored) + 30% classification mAP@0.5.
Uses pycocotools for mAP computation.

Usage:
    uv run python eval.py --predictions predictions.json
    uv run python eval.py --predictions predictions.json --gt data/train/annotations.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def compute_iou(box1, box2):
    """IoU between two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


def compute_ap(precisions, recalls):
    """Compute AP using 101-point interpolation (COCO style)."""
    recall_levels = np.linspace(0, 1, 101)
    ap = 0
    for r in recall_levels:
        precs_above = [p for p, rec in zip(precisions, recalls) if rec >= r]
        ap += max(precs_above) if precs_above else 0
    return ap / 101


def eval_map_at_iou(preds_by_image, gt_by_image, iou_thresh=0.5, use_category=False):
    """Compute mAP@IoU for all categories (or single detection class)."""
    # Collect all predictions with scores
    all_preds = []
    for img_id, preds in preds_by_image.items():
        for p in preds:
            cat = p["category_id"] if use_category else 0
            all_preds.append({
                "image_id": img_id,
                "bbox": p["bbox"],
                "score": p["score"],
                "category_id": cat,
            })

    # Collect all GT
    all_gt = {}
    total_gt = 0
    for img_id, gts in gt_by_image.items():
        for g in gts:
            cat = g["category_id"] if use_category else 0
            key = (img_id, cat)
            if key not in all_gt:
                all_gt[key] = []
            all_gt[key].append({"bbox": g["bbox"], "matched": False})
            total_gt += 1

    if total_gt == 0:
        return 0.0

    # Get unique categories
    if use_category:
        categories = set()
        for img_id, gts in gt_by_image.items():
            for g in gts:
                categories.add(g["category_id"])
    else:
        categories = {0}

    # Compute AP per category
    aps = []
    for cat in categories:
        # Filter predictions for this category, sort by score desc
        cat_preds = sorted(
            [p for p in all_preds if p["category_id"] == cat],
            key=lambda x: -x["score"],
        )

        # Count GT for this category
        cat_gt = {}
        n_gt = 0
        for (img_id, c), boxes in all_gt.items():
            if c == cat:
                cat_gt[img_id] = [{"bbox": b["bbox"], "matched": False} for b in boxes]
                n_gt += len(boxes)

        if n_gt == 0:
            continue

        # Match predictions to GT
        tp = []
        fp = []
        for pred in cat_preds:
            img_id = pred["image_id"]
            gt_boxes = cat_gt.get(img_id, [])

            best_iou = 0
            best_idx = -1
            for idx, gt in enumerate(gt_boxes):
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_thresh and best_idx >= 0 and not gt_boxes[best_idx]["matched"]:
                tp.append(1)
                fp.append(0)
                gt_boxes[best_idx]["matched"] = True
            else:
                tp.append(0)
                fp.append(1)

        # Compute precision-recall curve
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precisions = tp_cum / (tp_cum + fp_cum)
        recalls = tp_cum / n_gt

        ap = compute_ap(precisions.tolist(), recalls.tolist())
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions.json")
    parser.add_argument("--gt", default="data/train/annotations.json", help="COCO ground truth")
    parser.add_argument("--val-only", action="store_true", default=True, help="Only eval on val split images")
    args = parser.parse_args()

    # Load predictions
    with open(args.predictions) as f:
        preds = json.load(f)

    # Load ground truth
    gt_path = Path(__file__).parent / args.gt
    with open(gt_path) as f:
        coco = json.load(f)

    # Figure out which images are in the val set
    val_dir = Path(__file__).parent / "data" / "yolo" / "images" / "val"
    if val_dir.exists() and args.val_only:
        val_stems = {p.stem for p in val_dir.iterdir()}
        val_image_ids = set()
        for img in coco["images"]:
            if Path(img["file_name"]).stem in val_stems:
                val_image_ids.add(img["id"])
        print(f"Evaluating on {len(val_image_ids)} val images")
    else:
        val_image_ids = {img["id"] for img in coco["images"]}
        print(f"Evaluating on all {len(val_image_ids)} images")

    # Group GT by image
    gt_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in val_image_ids:
            continue
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(ann)

    # Group preds by image
    preds_by_image = {}
    for p in preds:
        img_id = p["image_id"]
        if img_id not in val_image_ids:
            continue
        if img_id not in preds_by_image:
            preds_by_image[img_id] = []
        preds_by_image[img_id].append(p)

    total_gt = sum(len(v) for v in gt_by_image.values())
    total_preds = sum(len(v) for v in preds_by_image.values())
    print(f"GT annotations: {total_gt}")
    print(f"Predictions: {total_preds}")

    # Detection mAP (category ignored)
    det_map = eval_map_at_iou(preds_by_image, gt_by_image, iou_thresh=0.5, use_category=False)

    # Classification mAP (category matters)
    cls_map = eval_map_at_iou(preds_by_image, gt_by_image, iou_thresh=0.5, use_category=True)

    # Combined score
    score = 0.7 * det_map + 0.3 * cls_map

    print(f"\n{'='*40}")
    print(f"Detection mAP@0.5:       {det_map:.4f}")
    print(f"Classification mAP@0.5:  {cls_map:.4f}")
    print(f"Combined score:          {score:.4f}")
    print(f"  (0.7 * {det_map:.4f} + 0.3 * {cls_map:.4f})")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
