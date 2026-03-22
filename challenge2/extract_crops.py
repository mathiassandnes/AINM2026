"""Extract labeled crops from GT bounding boxes for classifier training.

Groups annotations by image to avoid re-opening images.

Usage:
    uv run python extract_crops.py
    uv run python extract_crops.py --include-refs
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image


SCRIPT_DIR = Path(__file__).parent
MIN_CROP_SIZE = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-refs", action="store_true")
    parser.add_argument("--output-dir", default="data/crops")
    parser.add_argument("--pad-ratio", type=float, default=0.05)
    args = parser.parse_args()

    ann_path = SCRIPT_DIR / "data" / "train" / "annotations.json"
    img_dir = SCRIPT_DIR / "data" / "train" / "images"
    out_dir = SCRIPT_DIR / args.output_dir

    with open(ann_path) as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    categories = {c["id"]: c["name"] for c in coco["categories"]}

    # Create dirs
    for cat_id in categories:
        (out_dir / str(cat_id)).mkdir(parents=True, exist_ok=True)

    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # Extract crops — one image open per image
    cat_counts = Counter()
    skipped = 0
    for img_idx, (image_id, anns) in enumerate(anns_by_image.items()):
        img_file = img_dir / id_to_file[image_id]
        if not img_file.exists():
            continue
        img = Image.open(img_file).convert("RGB")
        iw, ih = img.size

        for ann in anns:
            cat_id = ann["category_id"]
            x, y, w, h = ann["bbox"]
            if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
                skipped += 1
                continue
            pad_x = w * args.pad_ratio
            pad_y = h * args.pad_ratio
            x1 = max(0, int(x - pad_x))
            y1 = max(0, int(y - pad_y))
            x2 = min(iw, int(x + w + pad_x))
            y2 = min(ih, int(y + h + pad_y))
            crop = img.crop((x1, y1, x2, y2))
            crop_path = out_dir / str(cat_id) / f"gt_{image_id}_{ann['id']}.jpg"
            crop.save(crop_path, quality=85)
            cat_counts[cat_id] += 1

        if (img_idx + 1) % 50 == 0:
            print(f"  {img_idx + 1}/{len(anns_by_image)} images, {sum(cat_counts.values())} crops", flush=True)

    print(f"Extracted {sum(cat_counts.values())} GT crops ({skipped} skipped)", flush=True)

    if args.include_refs:
        ref_dir = SCRIPT_DIR / "data" / "NM_NGD_product_images"
        meta_path = ref_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            name_to_catid = {c["name"]: c["id"] for c in coco["categories"]}
            ref_count = 0
            for p in meta["products"]:
                if p["product_name"] not in name_to_catid or not p["has_images"]:
                    continue
                cat_id = name_to_catid[p["product_name"]]
                product_dir = ref_dir / p["product_code"]
                if not product_dir.exists():
                    continue
                for img_file in product_dir.glob("*.jpg"):
                    crop = Image.open(img_file).convert("RGB")
                    out_path = out_dir / str(cat_id) / f"ref_{p['product_code']}_{img_file.stem}.jpg"
                    crop.save(out_path, quality=85)
                    cat_counts[cat_id] += 1
                    ref_count += 1
            print(f"Added {ref_count} reference images", flush=True)

    counts = sorted(cat_counts.values())
    print(f"Categories: {len(cat_counts)}, Total: {sum(counts)}", flush=True)
    print(f"Min: {counts[0]}, Max: {counts[-1]}, Median: {counts[len(counts)//2]}", flush=True)
    print(f"<5 crops: {sum(1 for c in counts if c < 5)}, <10: {sum(1 for c in counts if c < 10)}", flush=True)


if __name__ == "__main__":
    main()
