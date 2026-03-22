import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort


def preprocess(img_path, input_size=1280):
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    canvas.paste(img_resized, (pad_x, pad_y))
    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return arr, orig_w, orig_h, scale, pad_x, pad_y


def postprocess(output, orig_w, orig_h, scale, pad_x, pad_y,
                conf_thresh=0.01, iou_thresh=0.65):
    # output shape: [1, 360, N] where 360 = 4 bbox + 356 classes
    preds = output[0].T  # [N, 360]

    # Split bbox and class scores
    boxes_raw = preds[:, :4]  # cx, cy, w, h
    class_scores = preds[:, 4:]  # (N, 356)

    # Get best class and confidence for each detection
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    # Filter by confidence
    mask = confidences > conf_thresh
    boxes_raw = boxes_raw[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes_raw) == 0:
        return np.zeros((0, 4)), np.zeros(0, dtype=np.int32), np.zeros(0)

    # Convert to x1y1x2y2 in original image coords
    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = np.clip((cx - w / 2 - pad_x) / scale, 0, orig_w)
    y1 = np.clip((cy - h / 2 - pad_y) / scale, 0, orig_h)
    x2 = np.clip((cx + w / 2 - pad_x) / scale, 0, orig_w)
    y2 = np.clip((cy + h / 2 - pad_y) / scale, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS per class
    keep = multiclass_nms(boxes, confidences, class_ids, iou_thresh)
    return boxes[keep], class_ids[keep], confidences[keep]


def multiclass_nms(boxes, scores, class_ids, iou_thresh):
    """NMS applied per class."""
    keep = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_indices = np.where(cls_mask)[0]
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        cls_keep = nms(cls_boxes, cls_scores, iou_thresh)
        keep.extend(cls_indices[cls_keep].tolist())

    return keep


def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(
        str(script_dir / "yolo_mc.onnx"), providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"Model loaded: yolo_mc.onnx")

    predictions = []

    for img in sorted(Path(args.input).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png") or img.name.startswith("."):
            continue
        image_id = int(img.stem.split("_")[-1])

        arr, orig_w, orig_h, scale, pad_x, pad_y = preprocess(str(img))
        outputs = session.run(None, {input_name: arr})
        boxes, class_ids, scores = postprocess(
            outputs[0], orig_w, orig_h, scale, pad_x, pad_y)

        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j]
            predictions.append({
                "image_id": image_id,
                "category_id": int(class_ids[j]),
                "bbox": [round(float(x1), 1), round(float(y1), 1),
                         round(float(x2 - x1), 1), round(float(y2 - y1), 1)],
                "score": round(float(scores[j]), 3),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Done: {len(predictions)} predictions")


if __name__ == "__main__":
    main()
