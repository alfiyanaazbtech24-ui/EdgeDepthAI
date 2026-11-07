
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.engine import (
    train_one_epoch,
    validate_one_epoch,
)


class TinyDepthModel(nn.Module):
    """Small model used only to test the training engine."""

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            5,
            1,
            kernel_size=1
        )

    def forward(self, x):
        low_resolution = self.conv(x)

        prediction = F.interpolate(
            low_resolution,
            size=(8, 8),
            mode="bilinear",
            align_corners=False
        )

        return {"depth": prediction}


def make_test_batches():
    batches = []

    for _ in range(2):
        batches.append({
            "input": torch.rand(1, 5, 4, 4),
            "target": torch.rand(1, 1, 8, 8),
            "target_mask": torch.ones(1, 1, 8, 8),
        })

    return batches


def test_training_engine():
    torch.manual_seed(42)

    device = torch.device("cpu")

    model = TinyDepthModel().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    initial_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    train_results = train_one_epoch(
        model=model,
        data_loader=make_test_batches(),
        optimizer=optimizer,
        device=device,
    )

    assert train_results["batches"] == 2
    assert train_results["valid_pixels"] == 128
    assert torch.isfinite(
        torch.tensor(train_results["loss"])
    )
    assert torch.isfinite(
        torch.tensor(train_results["rmse"])
    )

    parameters_changed = any(
        not torch.equal(before, after)
        for before, after in zip(
            initial_parameters,
            model.parameters()
        )
    )

    assert parameters_changed

    print("Training loss:", train_results["loss"])
    print("Training RMSE:", train_results["rmse"])
    print("Training engine test PASSED")

    parameters_before_validation = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    validation_results = validate_one_epoch(
        model=model,
        data_loader=make_test_batches(),
        device=device,
    )

    assert validation_results["batches"] == 2
    assert validation_results["valid_pixels"] == 128

    for before, after in zip(
        parameters_before_validation,
        model.parameters()
    ):
        assert torch.equal(before, after)

    print("Validation loss:", validation_results["loss"])
    print("Validation RMSE:", validation_results["rmse"])
    print("Validation engine test PASSED")


if __name__ == "__main__":
    test_training_engine()
    print("\nAll training engine tests PASSED.")
