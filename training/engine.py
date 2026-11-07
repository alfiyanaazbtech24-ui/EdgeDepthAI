
"""
Training and validation engine for EdgeDepth AI.

Supports:
    NYU Depth V2: depth in meters
    Middlebury: disparity in pixels

Both datasets use the same masked loss implementation,
but must be trained in separate runs.
"""

import torch

from training.losses import (
    masked_l1_loss,
    masked_rmse,
)


def _prepare_batch(batch, device):
    """Move model input, target, and validity mask to the device."""

    model_input = batch["input"].to(
        device=device,
        dtype=torch.float32
    )

    target = batch["target"].to(
        device=device,
        dtype=torch.float32
    )

    target_mask = batch["target_mask"].to(
        device=device,
        dtype=torch.float32
    )

    return model_input, target, target_mask


def _calculate_valid_count(prediction, target, mask):
    """Count pixels eligible for loss and metric calculations."""

    valid = (
        (mask > 0)
        & torch.isfinite(mask)
        & torch.isfinite(target)
        & torch.isfinite(prediction)
    )

    return valid.sum().item()


def _run_epoch(
    model,
    data_loader,
    device,
    optimizer=None,
    max_grad_norm=1.0,
):
    """
    Run one training or validation epoch.

    If optimizer is supplied, model parameters are updated.
    Otherwise, the model is evaluated without gradient tracking.
    """

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_absolute_error = 0.0
    total_squared_error = 0.0
    total_valid_pixels = 0
    total_batches = 0

    for batch in data_loader:
        model_input, target, target_mask = _prepare_batch(
            batch,
            device
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model(model_input)

            if "depth" not in output:
                raise KeyError(
                    "Model output must contain a 'depth' tensor."
                )

            prediction = output["depth"]

            if prediction.shape != target.shape:
                raise ValueError(
                    f"Prediction shape {tuple(prediction.shape)} "
                    f"does not match target shape {tuple(target.shape)}"
                )

            valid_count = _calculate_valid_count(
                prediction,
                target,
                target_mask
            )

            if valid_count == 0:
                continue

            loss = masked_l1_loss(
                prediction,
                target,
                target_mask
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm
                )

                optimizer.step()

        # Calculate aggregate metrics using all valid pixels.
        valid = (
            (target_mask > 0)
            & torch.isfinite(target_mask)
            & torch.isfinite(target)
            & torch.isfinite(prediction)
        )

        safe_prediction = torch.where(
            valid,
            prediction.detach(),
            torch.zeros_like(prediction)
        )

        safe_target = torch.where(
            valid,
            target,
            torch.zeros_like(target)
        )

        error = safe_prediction - safe_target

        total_absolute_error += (
            torch.abs(error).sum().item()
        )

        total_squared_error += (
            (error ** 2).sum().item()
        )

        total_valid_pixels += valid_count
        total_batches += 1

    if total_valid_pixels == 0:
        raise ValueError(
            "No valid target pixels were found in this epoch."
        )

    mean_l1 = (
        total_absolute_error / total_valid_pixels
    )

    rmse = (
        total_squared_error / total_valid_pixels
    ) ** 0.5

    return {
        "loss": mean_l1,
        "l1": mean_l1,
        "rmse": rmse,
        "valid_pixels": total_valid_pixels,
        "batches": total_batches,
    }


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    max_grad_norm=1.0,
):
    """Train the model for one epoch."""

    return _run_epoch(
        model=model,
        data_loader=data_loader,
        device=device,
        optimizer=optimizer,
        max_grad_norm=max_grad_norm,
    )


def validate_one_epoch(
    model,
    data_loader,
    device,
):
    """Evaluate the model without updating its parameters."""

    with torch.no_grad():
        return _run_epoch(
            model=model,
            data_loader=data_loader,
            device=device,
            optimizer=None,
        )
