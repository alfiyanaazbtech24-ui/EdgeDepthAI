
"""
Checkpoint and training-history utilities for EdgeDepth AI.
"""

import json
import os
from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_validation_l1,
    config,
    history,
):
    """
    Save a complete training checkpoint.

    Writes to a temporary file first, then replaces the
    destination to reduce the risk of a partially written file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": int(epoch),
        "best_validation_l1": float(best_validation_l1),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),
        "config": config,
        "history": history,
    }

    temporary_path = path.with_name(path.name + ".tmp")

    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu",
):
    """
    Load model and optional optimizer state.

    Only load checkpoint files that you trust.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    required_keys = (
        "epoch",
        "best_validation_l1",
        "model_state_dict",
        "config",
        "history",
    )

    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(
                f"Checkpoint is missing required key: {key}"
            )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer_state = checkpoint.get(
            "optimizer_state_dict"
        )

        if optimizer_state is None:
            raise ValueError(
                "Checkpoint does not contain optimizer state."
            )

        optimizer.load_state_dict(optimizer_state)

    return checkpoint


def save_training_history(path, history):
    """Save epoch metrics as readable JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(path.name + ".tmp")

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                history,
                file,
                indent=4,
                allow_nan=False,
            )

        os.replace(temporary_path, path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_training_history(path):
    """Load a previously saved JSON training history."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, list):
        raise ValueError(
            "Training history must contain a JSON list."
        )

    return history
