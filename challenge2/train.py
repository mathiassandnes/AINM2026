"""Train YOLOv8 on NorgesGruppen grocery shelf dataset.

Usage:
    python train.py                          # Full multi-class (356 classes)
    python train.py --single-class           # Single-class detection only
    python train.py --epochs 50 --batch 8    # Custom params
    python train.py --resume                 # Resume interrupted training

Run prepare_data.py first to convert COCO -> YOLO format.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(
    data_yaml: str = "data/yolo/data.yaml",
    model: str = "yolov8m.pt",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 1280,
    single_class: bool = False,
    resume: bool = False,
    name: str = "grocery",
):
    data_path = Path(__file__).parent / data_yaml

    if not data_path.exists():
        print(f"Data config not found: {data_path}")
        print("Run prepare_data.py first!")
        return

    if resume:
        # Find last checkpoint
        runs_dir = Path(__file__).parent / "runs" / "detect"
        last_pt = runs_dir / name / "weights" / "last.pt"
        if last_pt.exists():
            print(f"Resuming from {last_pt}")
            model_obj = YOLO(str(last_pt))
        else:
            print(f"No checkpoint found at {last_pt}, starting fresh")
            model_obj = YOLO(model)
    else:
        model_obj = YOLO(model)

    # Training config — pragmatic competition settings
    results = model_obj.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        name=name,
        # Augmentation — shelf images benefit from these
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,       # slight rotation (shelves aren't perfectly straight)
        translate=0.1,
        scale=0.3,
        flipud=0.0,        # don't flip vertically — products have orientation
        fliplr=0.5,        # horizontal flip is fine
        mosaic=1.0,        # mosaic augmentation helps with dense scenes
        mixup=0.1,
        # Training params
        patience=20,        # early stopping
        save_period=10,     # save checkpoint every 10 epochs
        single_cls=single_class,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Output
        project=str(Path(__file__).parent / "runs" / "detect"),
        exist_ok=True,
        verbose=True,
    )

    print(f"\nTraining complete!")
    print(f"Best model: runs/detect/{name}/weights/best.pt")
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on grocery data")
    parser.add_argument("--data", default="data/yolo/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8m.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--single-class", action="store_true", help="Train single-class detector")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--name", default="grocery", help="Run name")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        single_class=args.single_class,
        resume=args.resume,
        name=args.name,
    )


if __name__ == "__main__":
    main()
