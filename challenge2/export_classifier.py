"""Export trained classifier to ONNX for sandbox submission.

Usage:
    uv run python export_classifier.py
    uv run python export_classifier.py --checkpoint classifier.pt --output classifier.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).parent
NUM_CLASSES = 357
INPUT_SIZE = 518


def build_model(model_name="vit_small_patch14_dinov2.lvd142m"):
    import timm
    backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    feat_dim = backbone.num_features
    model = nn.Sequential(
        backbone,
        nn.LayerNorm(feat_dim),
        nn.Dropout(0.0),  # disable dropout for inference
        nn.Linear(feat_dim, NUM_CLASSES),
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="classifier.pt")
    parser.add_argument("--model", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--output", default="classifier.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    print(f"Building model: {args.model}")
    model = build_model(args.model)

    print(f"Loading weights: {args.checkpoint}")
    state_dict = torch.load(SCRIPT_DIR / args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    # Test forward pass
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape} (should be [1, {NUM_CLASSES}])")

    # Export to ONNX
    output_path = SCRIPT_DIR / args.output
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Exported to {output_path} ({size_mb:.1f} MB)")

    # Verify with ONNX Runtime
    import onnxruntime as ort
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]
    diff = np.abs(out.numpy() - onnx_out).max()
    print(f"ONNX verification: max diff = {diff:.6f} (should be ~0)")

    print(f"\nDone! Update run.py to use classifier.onnx instead of dino_embed.onnx + ref_embeddings.npy")


if __name__ == "__main__":
    main()
