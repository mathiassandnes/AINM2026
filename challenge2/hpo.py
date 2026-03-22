"""HPO sweep for detection + classification hyperparameters.

Runs the inference pipeline on local val set with different parameter combos
and evaluates using eval.py logic.

Usage:
    uv run python hpo.py
    uv run python hpo.py --det-only   # skip classification (faster)
"""

import argparse
import json
import time
from itertools import product as cartesian
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

# Import eval functions
from eval import eval_map_at_iou

# Import run.py functions
from run import (
    det_preprocess, det_postprocess, nms,
    _crop_one, DINO_MEAN, DINO_STD, DINO_SIZE,
)

SCRIPT_DIR = Path(__file__).parent


def load_gt(val_dir, ann_path):
    """Load ground truth for val images."""
    with open(ann_path) as f:
        coco = json.load(f)

    val_stems = {p.stem for p in val_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    val_image_ids = set()
    for img in coco["images"]:
        if Path(img["file_name"]).stem in val_stems:
            val_image_ids.add(img["id"])

    gt_by_image = {}
    for ann in coco["annotations"]:
        if ann["image_id"] in val_image_ids:
            gt_by_image.setdefault(ann["image_id"], []).append(ann)

    return gt_by_image


def run_detection(det_session, det_input_name, val_dir, conf_thresh, iou_thresh):
    """Run detection on val images, return raw boxes per image."""
    results = {}
    for img_path in sorted(val_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png") or img_path.name.startswith("."):
            continue
        image_id = int(img_path.stem.split("_")[-1])
        arr, pil_img, orig_w, orig_h, scale, pad_x, pad_y = det_preprocess(str(img_path))
        outputs = det_session.run(None, {det_input_name: arr})
        boxes, scores = det_postprocess(outputs[0], orig_w, orig_h, scale, pad_x, pad_y,
                                         conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        results[image_id] = {
            "boxes": boxes, "scores": scores,
            "pil_img": pil_img,
        }
    return results


def classify_detections(cls_session, cls_input_name, ref_embeddings, ref_cat_ids,
                        det_results, classify_top_k, unknown_thresh, chunk_size=16):
    """Classify detected boxes."""
    predictions = []
    for image_id, res in det_results.items():
        boxes, scores = res["boxes"], res["scores"]
        pil_img = res["pil_img"]
        if len(boxes) == 0:
            continue

        # Classify top-K by confidence, rest get category_id=0
        n = len(boxes)
        sort_idx = scores.argsort()[::-1]
        top_k = min(classify_top_k, n)
        cat_ids = [0] * n

        # Process top-K crops
        top_indices = sort_idx[:top_k]
        for chunk_start in range(0, len(top_indices), chunk_size):
            chunk_idx = top_indices[chunk_start:chunk_start + chunk_size]
            crops = np.stack([_crop_one(pil_img, boxes[i]) for i in chunk_idx])
            crops = ((crops - DINO_MEAN) / DINO_STD).astype(np.float32)
            feats = cls_session.run(None, {cls_input_name: crops})[0]
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = feats / (norms + 1e-8)
            sims = feats @ ref_embeddings.T
            best_idx = sims.argmax(axis=1)
            best_sim = sims.max(axis=1)
            for j, (idx, sim) in enumerate(zip(best_idx, best_sim)):
                orig_i = chunk_idx[j]
                if sim >= unknown_thresh:
                    cat_ids[orig_i] = int(ref_cat_ids[idx])
                else:
                    cat_ids[orig_i] = 355  # unknown_product

        for j in range(n):
            x1, y1, x2, y2 = boxes[j]
            predictions.append({
                "image_id": image_id,
                "category_id": cat_ids[j],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(scores[j]),
            })
    return predictions


def det_only_predictions(det_results):
    """Convert detection results to predictions with category_id=0."""
    predictions = []
    for image_id, res in det_results.items():
        for j in range(len(res["boxes"])):
            x1, y1, x2, y2 = res["boxes"][j]
            predictions.append({
                "image_id": image_id,
                "category_id": 0,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(res["scores"][j]),
            })
    return predictions


def evaluate(predictions, gt_by_image):
    """Run eval and return scores."""
    preds_by_image = {}
    for p in predictions:
        preds_by_image.setdefault(p["image_id"], []).append(p)

    det_map = eval_map_at_iou(preds_by_image, gt_by_image, iou_thresh=0.5, use_category=False)
    cls_map = eval_map_at_iou(preds_by_image, gt_by_image, iou_thresh=0.5, use_category=True)
    score = 0.7 * det_map + 0.3 * cls_map
    return det_map, cls_map, score, len(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-only", action="store_true")
    args = parser.parse_args()

    val_dir = SCRIPT_DIR / "data" / "yolo" / "images" / "val"
    ann_path = SCRIPT_DIR / "data" / "train" / "annotations.json"

    print("Loading ground truth...")
    gt_by_image = load_gt(val_dir, ann_path)
    total_gt = sum(len(v) for v in gt_by_image.values())
    print(f"Val images: {len(gt_by_image)}, GT annotations: {total_gt}")

    print("Loading detection model...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    det_session = ort.InferenceSession(str(SCRIPT_DIR / "best.onnx"), providers=providers)
    det_input_name = det_session.get_inputs()[0].name

    # Load classifier if needed
    cls_session = None
    if not args.det_only:
        cls_path = SCRIPT_DIR / "dino_embed.onnx"
        emb_path = SCRIPT_DIR / "ref_embeddings.npy"
        ids_path = SCRIPT_DIR / "ref_cat_ids.json"
        if cls_path.exists() and emb_path.exists() and ids_path.exists():
            print("Loading classification model...")
            cls_session = ort.InferenceSession(str(cls_path), providers=providers)
            cls_input_name = cls_session.get_inputs()[0].name
            ref_embeddings = np.load(str(emb_path))
            with open(str(ids_path)) as f:
                ref_cat_ids = np.array(json.load(f), dtype=np.int32)
            print(f"Classifier loaded: {len(ref_cat_ids)} products, dim={ref_embeddings.shape[1]}")

    # ── HPO Grid ──
    conf_thresholds = [0.001, 0.01, 0.05, 0.1, 0.15, 0.25]
    nms_iou_thresholds = [0.5, 0.6, 0.65, 0.7, 0.8]

    if cls_session:
        unknown_thresholds = [0.15, 0.25, 0.35, 0.45, 0.55]
        classify_top_ks = [50, 100, 200, 9999]
    else:
        unknown_thresholds = [0.35]
        classify_top_ks = [0]

    print(f"\n{'='*90}")
    print(f"{'conf':>6} {'nms_iou':>7} {'unk_th':>6} {'top_k':>5} | {'det_mAP':>8} {'cls_mAP':>8} {'score':>8} {'#preds':>7} {'time':>6}")
    print(f"{'='*90}")

    best_score = 0
    best_params = {}

    for conf_t, nms_iou in cartesian(conf_thresholds, nms_iou_thresholds):
        t0 = time.time()
        det_results = run_detection(det_session, det_input_name, val_dir, conf_t, nms_iou)
        det_time = time.time() - t0

        if args.det_only or cls_session is None:
            preds = det_only_predictions(det_results)
            det_map, cls_map, score, n_preds = evaluate(preds, gt_by_image)
            print(f"{conf_t:>6.3f} {nms_iou:>7.2f} {'--':>6} {'--':>5} | {det_map:>8.4f} {cls_map:>8.4f} {score:>8.4f} {n_preds:>7} {det_time:>5.1f}s")
            if score > best_score:
                best_score = score
                best_params = {"conf": conf_t, "nms_iou": nms_iou}
        else:
            for unk_t, top_k in cartesian(unknown_thresholds, classify_top_ks):
                t1 = time.time()
                preds = classify_detections(
                    cls_session, cls_input_name, ref_embeddings, ref_cat_ids,
                    det_results, top_k, unk_t)
                cls_time = time.time() - t1
                det_map, cls_map, score, n_preds = evaluate(preds, gt_by_image)
                total_time = det_time + cls_time
                print(f"{conf_t:>6.3f} {nms_iou:>7.2f} {unk_t:>6.2f} {top_k:>5} | {det_map:>8.4f} {cls_map:>8.4f} {score:>8.4f} {n_preds:>7} {total_time:>5.1f}s")
                if score > best_score:
                    best_score = score
                    best_params = {"conf": conf_t, "nms_iou": nms_iou, "unknown": unk_t, "top_k": top_k}

    print(f"\n{'='*90}")
    print(f"BEST: score={best_score:.4f} params={best_params}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
