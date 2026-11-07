"""
Masked loss functions for EdgeDepth AI.

These functions support both:
    NYU Depth V2: metric depth in meters
    Middlebury: disparity in pixels

Invalid target pixels are excluded from loss calculations.
"""

import torch


def _get_valid_mask(prediction, target, mask):
    """Build a validity mask and validate tensor dimensions."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match "
            f"target shape {target.shape}"
        )

    if mask.shape != target.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match "
            f"target shape {target.shape}"
        )

    valid = (
        (mask > 0)
        & torch.isfinite(mask)
        & torch.isfinite(target)
        & torch.isfinite(prediction)
    )

    return valid


def masked_l1_loss(prediction, target, mask):
    """
    Calculate mean absolute error over valid pixels only.

    Returns a differentiable zero if no valid pixels exist.
    """

    valid = _get_valid_mask(prediction, target, mask)

    safe_prediction = torch.where(
        valid,
        prediction,
        torch.zeros_like(prediction)
    )

    safe_target = torch.where(
        valid,
        target,
        torch.zeros_like(target)
    )

    absolute_error = torch.abs(
        safe_prediction - safe_target
    )

    valid_count = valid.sum()

    loss = absolute_error.sum() / valid_count.clamp(min=1)

    return loss


def masked_rmse(prediction, target, mask):
    """
    Calculate root mean squared error over valid pixels only.

    This is an evaluation metric, not the default training loss.
    """

    valid = _get_valid_mask(prediction, target, mask)

    safe_prediction = torch.where(
        valid,
        prediction,
        torch.zeros_like(prediction)
    )

    safe_target = torch.where(
        valid,
        target,
        torch.zeros_like(target)
    )

    squared_error = (
        safe_prediction - safe_target
    ) ** 2

    valid_count = valid.sum()

    mse = (
        squared_error.sum()
        / valid_count.clamp(min=1)
    )

    return torch.sqrt(mse)