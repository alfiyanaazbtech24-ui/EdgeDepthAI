
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from training.data_pipeline import (
    deterministic_split,
    create_nyu_dataloaders,
    create_middlebury_dataloaders,
)


NYU_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "NYU_Depth_V2"
    / "nyu_depth_v2_labeled.mat"
)

MIDDLEBURY_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "Middlebury"
    / "MiddEval3"
    / "trainingH"
)


def test_deterministic_split():
    items = list(range(100))

    train_a, val_a = deterministic_split(
        items,
        train_fraction=0.9,
        seed=42,
    )

    train_b, val_b = deterministic_split(
        items,
        train_fraction=0.9,
        seed=42,
    )

    assert train_a == train_b
    assert val_a == val_b

    assert len(train_a) == 90
    assert len(val_a) == 10

    assert set(train_a).isdisjoint(set(val_a))
    assert set(train_a + val_a) == set(items)

    print("Deterministic split test PASSED")


def test_nyu_loaders():
    train_loader, val_loader = create_nyu_dataloaders(
        mat_path=NYU_PATH,
        scale_factor=2,
        batch_size=2,
        train_fraction=0.8,
        seed=42,
        max_samples=10,
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("\nNYU data loader test")
    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Training input:", tuple(train_batch["input"].shape))
    print("Training target:", tuple(train_batch["target"].shape))

    assert len(train_loader.dataset) == 8
    assert len(val_loader.dataset) == 2

    assert train_batch["input"].shape == (2, 5, 320, 240)
    assert train_batch["target"].shape == (2, 1, 640, 480)

    assert val_batch["input"].shape[1] == 5
    assert val_batch["target"].shape[1] == 1

    assert train_batch["target_unit"] == ["meters", "meters"]
    assert val_batch["target_type"] == ["depth", "depth"]

    assert torch.isfinite(train_batch["input"]).all()
    assert torch.isfinite(train_batch["target"]).all()

    print("NYU data loader test PASSED")


def test_middlebury_loaders():
    train_loader, val_loader = create_middlebury_dataloaders(
        root_dir=MIDDLEBURY_PATH,
        scale_factor=2,
        train_fraction=0.8,
        seed=42,
        max_samples=5,
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("\nMiddlebury data loader test")
    print("Training scenes:", len(train_loader.dataset))
    print("Validation scenes:", len(val_loader.dataset))
    print("Training input:", tuple(train_batch["input"].shape))
    print("Training target:", tuple(train_batch["target"].shape))

    assert len(train_loader.dataset) == 4
    assert len(val_loader.dataset) == 1

    assert train_batch["input"].shape[0] == 1
    assert train_batch["input"].shape[1] == 5

    assert train_batch["target"].shape[0] == 1
    assert train_batch["target"].shape[1] == 1

    assert train_batch["target_type"] == ["disparity"]
    assert train_batch["target_unit"] == ["pixels"]

    assert torch.isfinite(train_batch["input"]).all()
    assert torch.isfinite(train_batch["target"]).all()

    print("Middlebury data loader test PASSED")


if __name__ == "__main__":
    test_deterministic_split()
    test_nyu_loaders()
    test_middlebury_loaders()

    print("\nAll data pipeline tests PASSED.")
