
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from training.checkpoints import (
    save_checkpoint,
    load_checkpoint,
    save_training_history,
    load_training_history,
)


def test_checkpoint_save_and_load():
    torch.manual_seed(42)

    with tempfile.TemporaryDirectory() as temporary_dir:
        checkpoint_path = (
            Path(temporary_dir) / "test_checkpoint.pth"
        )

        model = nn.Conv2d(5, 1, kernel_size=1)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )

        # Perform one update so the optimizer has state to save.
        input_tensor = torch.rand(1, 5, 4, 4)
        target = torch.rand(1, 1, 4, 4)

        prediction = model(input_tensor)
        loss = torch.abs(prediction - target).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        expected_parameters = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

        config = {
            "dataset": "NYU_Depth_V2",
            "scale_factor": 2,
            "learning_rate": 0.001,
        }

        history = [
            {
                "epoch": 1,
                "train_l1": 0.35,
                "val_l1": 0.32,
                "val_rmse": 0.45,
            }
        ]

        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=1,
            best_validation_l1=0.32,
            config=config,
            history=history,
        )

        assert checkpoint_path.is_file()
        assert not Path(
            str(checkpoint_path) + ".tmp"
        ).exists()

        # Change model parameters to verify restoration.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10.0)

        loaded = load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            device="cpu",
        )

        assert loaded["epoch"] == 1
        assert loaded["best_validation_l1"] == 0.32
        assert loaded["config"] == config
        assert loaded["history"] == history

        for key, value in model.state_dict().items():
            assert torch.equal(
                value,
                expected_parameters[key]
            )

        print("Checkpoint save/load test PASSED")


def test_history_save_and_load():
    with tempfile.TemporaryDirectory() as temporary_dir:
        history_path = (
            Path(temporary_dir) / "history.json"
        )

        history = [
            {
                "epoch": 1,
                "train_l1": 0.35,
                "val_l1": 0.32,
                "val_rmse": 0.45,
            },
            {
                "epoch": 2,
                "train_l1": 0.30,
                "val_l1": 0.28,
                "val_rmse": 0.40,
            },
        ]

        save_training_history(
            history_path,
            history
        )

        loaded = load_training_history(
            history_path
        )

        assert loaded == history

        print("Training history save/load test PASSED")


if __name__ == "__main__":
    test_checkpoint_save_and_load()
    test_history_save_and_load()

    print("\nAll checkpoint tests PASSED.")
