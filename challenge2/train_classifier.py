"""Fine-tune DINOv2-S classifier on extracted product crops.

Uses timm 0.9.12 (matches sandbox) with DINOv2 ViT-Small backbone.
Adds a linear classification head for 357 classes.

Usage:
    uv run python train_classifier.py
    uv run python train_classifier.py --epochs 30 --lr 1e-4
    uv run python train_classifier.py --freeze-backbone  # faster, linear probe only
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
NUM_CLASSES = 357  # 0-355 + unknown_product (356)


class CropDataset(Dataset):
    def __init__(self, samples, transform, is_train=True):
        self.samples = samples  # list of (path, category_id)
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cat_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, cat_id


def build_transforms(input_size=518):
    """Build train/val transforms matching DINOv2 preprocessing."""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = T.Compose([
        T.RandomResizedCrop(input_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        normalize,
    ])

    val_transform = T.Compose([
        T.Resize(input_size),
        T.CenterCrop(input_size),
        T.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def load_samples(crops_dir, val_ratio=0.15):
    """Load all crop samples, split into train/val."""
    all_samples = []
    for cat_dir in sorted(crops_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_id = int(cat_dir.name)
        for img_path in cat_dir.glob("*.jpg"):
            all_samples.append((str(img_path), cat_id))

    random.seed(42)
    random.shuffle(all_samples)

    # Split by image_id to avoid leakage (GT crops from same image in train+val)
    # Group by source image
    gt_images = {}
    ref_samples = []
    for path, cat_id in all_samples:
        fname = Path(path).name
        if fname.startswith("gt_"):
            # gt_IMAGEID_ANNID.jpg
            parts = fname.split("_")
            img_id = parts[1]
            gt_images.setdefault(img_id, []).append((path, cat_id))
        else:
            ref_samples.append((path, cat_id))

    # Split GT images into train/val
    img_ids = list(gt_images.keys())
    random.shuffle(img_ids)
    n_val = max(1, int(len(img_ids) * val_ratio))
    val_img_ids = set(img_ids[:n_val])

    train_samples = []
    val_samples = []
    for img_id, samples in gt_images.items():
        if img_id in val_img_ids:
            val_samples.extend(samples)
        else:
            train_samples.extend(samples)

    # Add ref images to train only
    train_samples.extend(ref_samples)

    return train_samples, val_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dir", default="data/crops")
    parser.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1,
                        help="LR multiplier for backbone (vs head)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Only train the classification head")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--output", default="classifier.pt")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    crops_dir = SCRIPT_DIR / args.crops_dir
    if not crops_dir.exists():
        print(f"Crops dir not found: {crops_dir}")
        print("Run extract_crops.py first!")
        return

    # Load data
    train_samples, val_samples = load_samples(crops_dir)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_transform, val_transform = build_transforms(args.input_size)
    train_ds = CropDataset(train_samples, train_transform, is_train=True)
    val_ds = CropDataset(val_samples, val_transform, is_train=False)

    # Weighted sampler for class imbalance
    train_labels = [s[1] for s in train_samples]
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Build model
    import timm
    print(f"Loading backbone: {args.model}")
    backbone = timm.create_model(args.model, pretrained=True, num_classes=0)
    feat_dim = backbone.num_features
    print(f"Feature dim: {feat_dim}")

    model = nn.Sequential(
        backbone,
        nn.LayerNorm(feat_dim),
        nn.Dropout(0.3),
        nn.Linear(feat_dim, NUM_CLASSES),
    )
    model = model.to(device)

    if args.freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen — training head only")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01,
        )
    else:
        # Differential LR: lower for backbone, higher for head
        backbone_params = list(backbone.parameters())
        head_params = [p for n, p in model.named_parameters() if not n.startswith("0.")]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
            {"params": head_params, "lr": args.lr},
        ], weight_decay=0.01)
        print(f"Fine-tuning: backbone LR={args.lr * args.backbone_lr_scale:.1e}, head LR={args.lr:.1e}")

    # Label smoothing helps with noisy GT crops
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += len(labels)

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(labels)

        train_acc = train_correct / train_total if train_total else 0
        val_acc = val_correct / val_total if val_total else 0
        train_avg_loss = train_loss / train_total if train_total else 0
        val_avg_loss = val_loss / val_total if val_total else 0

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), SCRIPT_DIR / args.output)

        lr = optimizer.param_groups[-1]["lr"]
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train_loss={train_avg_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_avg_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={lr:.1e} {'*' if is_best else ''}")

    print(f"\nBest val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Saved to {args.output}")
    print(f"\nNext: run export_classifier.py to convert to ONNX")


if __name__ == "__main__":
    main()
