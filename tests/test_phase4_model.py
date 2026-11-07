
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.edgedepth_unet import build_model


def main():
    print("\nEdgeDepth AI - Phase 4 Model Test")
    print("---------------------------------")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    batch_size = 2
    height = 32
    width = 40

    # Synthetic RGB guidance.
    rgb = torch.rand(batch_size, 3, height, width)

    # Synthetic low-resolution depth.
    depth = torch.rand(batch_size, 1, height, width) * 10.0

    # Synthetic validity mask.
    mask = torch.ones(batch_size, 1, height, width)

    # Mark a small region invalid.
    mask[:, :, 10:15, 10:15] = 0.0
    depth = depth * mask

    # Combine the five input channels.
    model_input = torch.cat(
        [rgb, depth, mask],
        dim=1
    ).to(device)

    for scale in (2, 4, 8):
        print(f"\nTesting {scale}x model...")

        model = build_model(
            scale_factor=scale,
            base_channels=16
        ).to(device)

        model.eval()

        with torch.no_grad():
            result = model(model_input)

        expected_height = height * scale
        expected_width = width * scale

        assert result["depth"].shape == (
            batch_size, 1, expected_height, expected_width
        )

        assert result["boundary"].shape == (
            batch_size, 1, expected_height, expected_width
        )

        assert result["residual"].shape == (
            batch_size, 1, expected_height, expected_width
        )

        assert torch.isfinite(result["depth"]).all()
        assert torch.isfinite(result["boundary"]).all()

        assert result["boundary"].min() >= 0
        assert result["boundary"].max() <= 1

        print("Depth output:", tuple(result["depth"].shape))
        print("Boundary output:", tuple(result["boundary"].shape))
        print("Residual output:", tuple(result["residual"].shape))
        print("Boundary range:",
              float(result["boundary"].min()),
              "to",
              float(result["boundary"].max()))

        del model
        del result

    # Test that the model supports backpropagation.
    print("\nTesting gradient computation...")

    model = build_model(
        scale_factor=2,
        base_channels=16
    ).to(device)

    model.train()

    result = model(model_input)

    loss = result["depth"].mean() + result["boundary"].mean()
    loss.backward()

    gradients_exist = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert gradients_exist

    print("Gradient computation: Passed")
    print("\nALL PHASE 4 MODEL TESTS PASSED!")


if __name__ == "__main__":
    main()
