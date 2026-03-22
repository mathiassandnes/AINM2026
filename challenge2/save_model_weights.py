"""Save timm model weights to a standalone .pth file for sandbox deployment.

The sandbox can't download models at runtime, so we bundle the weights.

Usage: uv run python save_model_weights.py
"""

import torch
import timm
from pathlib import Path


def main():
    model_name = "vit_small_patch14_dinov2.lvd142m"
    model = timm.create_model(model_name, pretrained=True, num_classes=0)

    output_path = Path(__file__).parent / f"{model_name.replace('.', '_')}.pth"
    torch.save(model.state_dict(), output_path)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved {model_name} weights to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()