import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort


def det_preprocess(img_path, input_size=1280):
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
    return arr, img, orig_w, orig_h, scale, pad_x, pad_y


def det_postprocess(output, orig_w, orig_h, scale, pad_x, pad_y,
                    conf_thresh=0.01, iou_thresh=0.65):
    preds = output[0].T
    mask = preds[:, 4] > conf_thresh
    preds = preds[mask]
    if len(preds) == 0:
        return np.zeros((0, 4)), np.zeros(0)

    cx, cy, w, h, conf = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3], preds[:, 4]
    x1 = np.clip((cx - w / 2 - pad_x) / scale, 0, orig_w)
    y1 = np.clip((cy - h / 2 - pad_y) / scale, 0, orig_h)
    x2 = np.clip((cx + w / 2 - pad_x) / scale, 0, orig_w)
    y2 = np.clip((cy + h / 2 - pad_y) / scale, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms(boxes, conf, iou_thresh)
    return boxes[keep], conf[keep]


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


CLS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
CLS_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
CLS_SIZE = 518


def classify_detections(cls_session, cls_input_name, pil_img, boxes):
    n = len(boxes)
    cat_ids = np.zeros(n, dtype=np.int32)

    chunk_size = 32
    for cs in range(0, n, chunk_size):
        end = min(cs + chunk_size, n)
        crops = []
        for i in range(cs, end):
            x1, y1, x2, y2 = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(pil_img.width, x2), min(pil_img.height, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                crop = pil_img.crop((0, 0, 50, 50))
            else:
                crop = pil_img.crop((x1, y1, x2, y2))
            crop = crop.resize((CLS_SIZE, CLS_SIZE), Image.BILINEAR)
            arr = np.array(crop).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            crops.append(arr)
        crops = np.stack(crops)
        crops = ((crops - CLS_MEAN) / CLS_STD).astype(np.float32)
        logits = cls_session.run(None, {cls_input_name: crops})[0]
        cat_ids[cs:end] = logits.argmax(axis=1)
        del crops, logits

    return cat_ids.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    det_session = ort.InferenceSession(
        str(script_dir / "best.onnx"), providers=providers)
    det_input_name = det_session.get_inputs()[0].name

    cls_session = ort.InferenceSession(
        str(script_dir / "classifier.onnx"), providers=providers)
    cls_input_name = cls_session.get_inputs()[0].name

    predictions = []

    for img in sorted(Path(args.input).iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png") or img.name.startswith("."):
            continue
        image_id = int(img.stem.split("_")[-1])

        arr, pil_img, orig_w, orig_h, scale, pad_x, pad_y = det_preprocess(str(img))
        outputs = det_session.run(None, {det_input_name: arr})
        boxes, scores = det_postprocess(outputs[0], orig_w, orig_h, scale, pad_x, pad_y)
        del arr, outputs

        if len(boxes) == 0:
            continue

        if len(boxes) > 300:
            top_k = scores.argsort()[::-1][:300]
            boxes, scores = boxes[top_k], scores[top_k]

        cat_ids = classify_detections(cls_session, cls_input_name, pil_img, boxes)
        del pil_img

        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j]
            predictions.append({
                "image_id": image_id,
                "category_id": cat_ids[j],
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
