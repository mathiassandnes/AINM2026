"""Re-embed reference images with all views (not just main+front).

Keep individual view embeddings instead of averaging — shelf crops may match
a specific angle better than an averaged representation.

Usage:
    uv run python embed_references_v2.py
    uv run python embed_references_v2.py --avg   # average per product (fewer embeddings)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import timm
import timm.data
from PIL import Image


SCRIPT_DIR = Path(__file__).parent
VIEWS = ["main", "front", "back", "left", "right", "top", "bottom"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--avg", action="store_true", help="Average views per product")
    parser.add_argument("--views", nargs="+", default=VIEWS)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load COCO categories
    ann_path = SCRIPT_DIR / "data" / "train" / "annotations.json"
    with open(ann_path) as f:
        coco = json.load(f)
    name_to_catid = {c["name"]: c["id"] for c in coco["categories"]}

    # Load product metadata
    ref_dir = SCRIPT_DIR / "data" / "NM_NGD_product_images"
    meta_path = ref_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    code_to_catid = {}
    for p in meta["products"]:
        if p["product_name"] in name_to_catid and p["has_images"]:
            code_to_catid[p["product_code"]] = name_to_catid[p["product_name"]]
    print(f"Products with images and category match: {len(code_to_catid)}")

    # Load model
    print(f"Loading model: {args.model}")
    model = timm.create_model(args.model, pretrained=True, num_classes=0)
    model = model.to(device).eval()
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    # Embed all views
    all_embeddings = []
    all_cat_ids = []

    for code, cat_id in sorted(code_to_catid.items(), key=lambda x: x[1]):
        product_dir = ref_dir / code
        if not product_dir.exists():
            continue

        img_paths = []
        for view in args.views:
            img_path = product_dir / f"{view}.jpg"
            if img_path.exists():
                img_paths.append(img_path)

        if not img_paths:
            continue

        # Embed batch
        imgs = []
        for p in img_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
            except Exception as e:
                print(f"  Skipping {p}: {e}")
                continue

        if not imgs:
            continue

        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model(batch).cpu().numpy()

        if args.avg:
            avg = feats.mean(axis=0)
            avg = avg / np.linalg.norm(avg)
            all_embeddings.append(avg)
            all_cat_ids.append(cat_id)
        else:
            # Keep each view as separate embedding
            for feat in feats:
                feat = feat / np.linalg.norm(feat)
                all_embeddings.append(feat)
                all_cat_ids.append(cat_id)

    embeddings = np.stack(all_embeddings).astype(np.float32)
    cat_ids = np.array(all_cat_ids, dtype=np.int32)

    unique_products = len(set(all_cat_ids))
    print(f"Embedded {unique_products} products, {len(cat_ids)} total vectors, dim={embeddings.shape[1]}")

    # Save
    np.save(SCRIPT_DIR / "ref_embeddings.npy", embeddings)
    with open(SCRIPT_DIR / "ref_cat_ids.json", "w") as f:
        json.dump(cat_ids.tolist(), f)

    print(f"Saved ref_embeddings.npy ({embeddings.nbytes / 1024:.1f} KB) and ref_cat_ids.json")


if __name__ == "__main__":
    main()
