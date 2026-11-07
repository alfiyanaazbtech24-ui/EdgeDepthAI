import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from training.losses import (
    masked_l1_loss,
    masked_rmse,
)


def test_masked_l1_loss():
    prediction = torch.tensor(
        [[[[1.0, 2.0],
           [3.0, 4.0]]]],
        requires_grad=True
    )

    target = torch.tensor(
        [[[[2.0, 4.0],
           [100.0, 8.0]]]]
    )

    mask = torch.tensor(
        [[[[1.0, 1.0],
           [0.0, 1.0]]]]
    )

    loss = masked_l1_loss(
        prediction,
        target,
        mask
    )

    # Valid absolute errors: 1, 2, 4
    expected = 7.0 / 3.0

    assert torch.isclose(
        loss,
        torch.tensor(expected),
        atol=1e-6
    )

    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()

    # The invalid pixel must receive zero gradient.
    assert prediction.grad[0, 0, 1, 0] == 0

    print("Masked L1 loss:", loss.item())
    print("Masked L1 test PASSED")


def test_masked_rmse():
    prediction = torch.tensor(
        [[[[1.0, 2.0],
           [3.0, 4.0]]]]
    )

    target = torch.tensor(
        [[[[2.0, 4.0],
           [100.0, 8.0]]]]
    )

    mask = torch.tensor(
        [[[[1.0, 1.0],
           [0.0, 1.0]]]]
    )

    rmse = masked_rmse(
        prediction,
        target,
        mask
    )

    expected = torch.sqrt(
        torch.tensor((1.0 + 4.0 + 16.0) / 3.0)
    )

    assert torch.isclose(
        rmse,
        expected,
        atol=1e-6
    )

    print("Masked RMSE:", rmse.item())
    print("Masked RMSE test PASSED")


def test_empty_mask():
    prediction = torch.randn(
        1, 1, 4, 4,
        requires_grad=True
    )

    target = torch.randn(1, 1, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)

    loss = masked_l1_loss(
        prediction,
        target,
        mask
    )

    assert loss.item() == 0.0

    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()

    print("Empty mask test PASSED")


def test_invalid_values():
    prediction = torch.tensor(
        [[[[1.0, float("nan")],
           [3.0, 4.0]]]]
    )

    target = torch.tensor(
        [[[[2.0, 5.0],
           [float("inf"), 8.0]]]]
    )

    mask = torch.ones_like(target)

    loss = masked_l1_loss(
        prediction,
        target,
        mask
    )

    # Only the first and last pixels are valid.
    assert torch.isclose(
        loss,
        torch.tensor(2.5),
        atol=1e-6
    )

    print("Invalid values test PASSED")


if __name__ == "__main__":
    test_masked_l1_loss()
    test_masked_rmse()
    test_empty_mask()
    test_invalid_values()

    print("\nAll loss tests PASSED.")