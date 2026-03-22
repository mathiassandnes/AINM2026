"""Precompute reference image embeddings for product classification.

Uses a timm model (available in sandbox) to embed all product reference images.
The resulting embeddings + mapping are bundled in the submission ZIP.

Usage:
    uv run python embed_references.py
    uv run python embed_references.py --model vit_small_patch14_dinov2.lvd142m
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def get_model(model_name: str, device: str):
    """Load a timm model as a feature extractor."""
    import timm

    model = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 = feature extractor
    model = model.to(device).eval()

    # Get model-specific transforms
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    return model, transform


def embed_images(model, transform, image_paths: list, device: str, batch_size: int = 32) -> np.ndarray:
    """Embed a list of images, return (N, D) array."""
    all_feats = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))
            except Exception as e:
                print(f"  Skipping {p}: {e}")
                # Use zeros as placeholder
                imgs.append(torch.zeros(3, 224, 224))

        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model(batch)
        all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m", help="timm model name")
    parser.add_argument("--ref-dir", default="data/NM_NGD_product_images", help="Product images dir")
    parser.add_argument("--output", default="reference_embeddings.npz", help="Output file")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load COCO categories
    ann_path = Path(__file__).parent / "data" / "train" / "annotations.json"
    with open(ann_path) as f:
        coco = json.load(f)
    name_to_catid = {c["name"]: c["id"] for c in coco["categories"]}

    # Load product metadata
    meta_path = Path(__file__).parent / args.ref_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    # Build product_code -> category_id mapping
    code_to_catid = {}
    for p in meta["products"]:
        if p["product_name"] in name_to_catid and p["has_images"]:
            code_to_catid[p["product_code"]] = name_to_catid[p["product_name"]]

    print(f"Products with images and category match: {len(code_to_catid)}")

    # Load model
    print(f"Loading model: {args.model}")
    model, transform = get_model(args.model, device)

    # Embed all reference images, grouped by category
    ref_dir = Path(__file__).parent / args.ref_dir
    all_embeddings = []  # (N, D)
    all_category_ids = []  # (N,) — which category each embedding belongs to
    views_to_use = ["main", "front"]  # Most relevant views for shelf matching

    for code, cat_id in sorted(code_to_catid.items(), key=lambda x: x[1]):
        product_dir = ref_dir / code
        if not product_dir.exists():
            continue

        # Collect reference images for this product
        img_paths = []
        for view in views_to_use:
            img_path = product_dir / f"{view}.jpg"
            if img_path.exists():
                img_paths.append(img_path)

        # Fallback: use whatever is available
        if not img_paths:
            for f in product_dir.glob("*.jpg"):
                img_paths.append(f)
                if len(img_paths) >= 2:
                    break

        if not img_paths:
            continue

        feats = embed_images(model, transform, img_paths, device)
        # Average the views into one embedding per product
        avg_feat = feats.mean(axis=0)
        avg_feat = avg_feat / np.linalg.norm(avg_feat)  # L2 normalize

        all_embeddings.append(avg_feat)
        all_category_ids.append(cat_id)

    embeddings = np.stack(all_embeddings)  # (num_products, D)
    category_ids = np.array(all_category_ids)  # (num_products,)

    print(f"Embedded {len(category_ids)} products, feature dim: {embeddings.shape[1]}")

    # Save
    output_path = Path(__file__).parent / args.output
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        category_ids=category_ids,
    )
    print(f"Saved to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()