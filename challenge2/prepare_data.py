"""Convert COCO annotations to YOLO format for training.

Expects:
  data/NM_NGD_coco_dataset/annotations.json  (COCO format)
  data/NM_NGD_coco_dataset/images/            (shelf images)

Produces:
  data/yolo/images/train/
  data/yolo/images/val/
  data/yolo/labels/train/
  data/yolo/labels/val/
  data/yolo/data.yaml
"""

import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

DATA_ROOT = Path(__file__).parent / "data"


def find_coco_json():
    """Find the COCO annotation file."""
    candidates = [
        DATA_ROOT / "NM_NGD_product_images" / "train" / "annotations.json",
        DATA_ROOT / "NM_NGD_coco_dataset" / "annotations.json",
        DATA_ROOT / "train" / "annotations.json",
        DATA_ROOT / "annotations.json",
    ]
    # Also search for any json in data dir
    for p in DATA_ROOT.rglob("*.json"):
        if p.name == "annotations.json":
            candidates.append(p)

    for c in candidates:
        if c.exists():
            # Verify it's COCO format
            with open(c) as f:
                data = json.load(f)
            if isinstance(data, dict) and "annotations" in data and "images" in data:
                print(f"Found COCO annotations: {c}")
                return c, data
    raise FileNotFoundError(
        f"No COCO annotation file found. Looked in:\n"
        + "\n".join(f"  {c}" for c in candidates[:5])
    )


def coco_to_yolo(coco_data, output_dir: Path, images_src: Path, val_split=0.15):
    """Convert COCO annotations to YOLO format with train/val split."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build lookups
    images_by_id = {img["id"]: img for img in coco_data["images"]}
    categories = {cat["id"]: cat for cat in coco_data["categories"]}

    # Group annotations by image
    anns_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # Category IDs are already 0-355 contiguous — use them directly as YOLO class IDs
    # This way model output class == COCO category_id, no remapping needed at inference
    all_cat_ids = sorted(categories.keys())
    cat_id_to_yolo = {cid: cid for cid in all_cat_ids}  # identity mapping
    num_classes = max(all_cat_ids) + 1  # 356

    print(f"Images: {len(images_by_id)}")
    print(f"Annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {num_classes}")

    # Count annotations per category
    cat_counts = Counter()
    for ann in coco_data["annotations"]:
        cat_counts[ann["category_id"]] += 1
    print(f"Top 10 categories by count:")
    for cid, count in cat_counts.most_common(10):
        name = categories.get(cid, {}).get("name", "?")
        print(f"  {cid} ({name}): {count}")

    # Train/val split by image
    image_ids = list(images_by_id.keys())
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * (1 - val_split))
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])

    print(f"Train: {len(train_ids)} images, Val: {len(val_ids)} images")

    # Create directories
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Convert
    import shutil

    for img_id, img_info in images_by_id.items():
        split = "train" if img_id in train_ids else "val"
        w, h = img_info["width"], img_info["height"]
        fname = img_info["file_name"]

        # Copy image
        src = images_src / fname
        if not src.exists():
            # Try without subdirectory
            src = images_src / Path(fname).name
        if src.exists():
            shutil.copy2(src, output_dir / "images" / split / Path(fname).name)

        # Write YOLO label
        anns = anns_by_image.get(img_id, [])
        label_path = output_dir / "labels" / split / (Path(fname).stem + ".txt")

        lines = []
        for ann in anns:
            bbox = ann["bbox"]  # [x, y, width, height] in pixels
            cat_id = ann["category_id"]
            yolo_class = cat_id_to_yolo[cat_id]

            # Convert COCO [x,y,w,h] to YOLO [cx, cy, w, h] normalized
            cx = (bbox[0] + bbox[2] / 2) / w
            cy = (bbox[1] + bbox[3] / 2) / h
            bw = bbox[2] / w
            bh = bbox[3] / h

            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            bw = max(0, min(1, bw))
            bh = max(0, min(1, bh))

            lines.append(f"{yolo_class} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    # Write data.yaml
    # Build category names list
    cat_names = [categories[cid].get("name", str(cid)) for cid in all_cat_ids]

    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

nc: {num_classes}
names: {cat_names}
"""
    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)

    # Also save the category mapping for inference
    mapping = {
        "cat_id_to_yolo": cat_id_to_yolo,
        "yolo_to_cat_id": {v: k for k, v in cat_id_to_yolo.items()},
        "categories": {cid: categories[cid] for cid in all_cat_ids},
    }
    with open(output_dir / "category_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"YOLO dataset written to {output_dir}")
    print(f"Config: {output_dir / 'data.yaml'}")
    return output_dir / "data.yaml"


def main():
    coco_path, coco_data = find_coco_json()

    # Find images directory (sibling of annotations file)
    images_src = coco_path.parent / "images"
    if not images_src.exists():
        # Try parent
        images_src = coco_path.parent
        jpg_count = len(list(images_src.glob("*.jpg")))
        if jpg_count == 0:
            print(f"WARNING: No images found at {images_src}")
            print("Looking for images...")
            for d in DATA_ROOT.rglob("*.jpg"):
                images_src = d.parent
                print(f"Found images in: {images_src}")
                break

    print(f"Images source: {images_src}")
    yolo_dir = DATA_ROOT / "yolo"
    coco_to_yolo(coco_data, yolo_dir, images_src, val_split=0.15)


if __name__ == "__main__":
    main()
