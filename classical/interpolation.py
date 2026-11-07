
import cv2
import numpy as np


INTERPOLATION_METHODS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


def validate_scale(scale):
    """
    Validate supported upsampling factors.
    """

    if scale not in (2, 4, 8):
        raise ValueError(
            "Upsampling factor must be 2, 4 or 8."
        )


def get_output_size(depth, scale):
    """
    Calculate output dimensions.
    """

    height, width = depth.shape

    return (
        width * scale,
        height * scale
    )


def interpolate_depth(
    depth,
    scale=2,
    method="bicubic",
    valid_mask=None
):
    """
    Upsample a depth map.

    Parameters:
        depth: H x W float32 depth map
        scale: 2, 4 or 8
        method: nearest, bilinear or bicubic
        valid_mask: optional H x W validity mask

    Returns:
        reconstructed: high-resolution float32 depth
        confidence: high-resolution float32 confidence
    """

    validate_scale(scale)

    if method not in INTERPOLATION_METHODS:
        raise ValueError(
            f"Unknown interpolation method: {method}"
        )

    depth = np.asarray(depth, dtype=np.float32)

    if depth.ndim != 2:
        raise ValueError(
            "Depth map must be two-dimensional."
        )

    if depth.size == 0:
        raise ValueError("Depth map is empty.")

    if valid_mask is None:

        valid_mask = (
            np.isfinite(depth) &
            (depth > 0)
        )

    else:

        valid_mask = (
            np.asarray(valid_mask, dtype=bool) &
            np.isfinite(depth) &
            (depth > 0)
        )

    if valid_mask.shape != depth.shape:
        raise ValueError(
            "Validity mask dimensions must match depth."
        )

    if not np.any(valid_mask):
        raise ValueError(
            "No valid depth pixels available."
        )

    output_size = get_output_size(depth, scale)

    # Use zero internally for invalid pixels.
    safe_depth = np.where(
        valid_mask,
        depth,
        0.0
    ).astype(np.float32)

    method_code = INTERPOLATION_METHODS[method]

    if method == "nearest":

        reconstructed = cv2.resize(
            safe_depth,
            output_size,
            interpolation=method_code
        )

        confidence = cv2.resize(
            valid_mask.astype(np.float32),
            output_size,
            interpolation=cv2.INTER_NEAREST
        )

    else:

        # Normalized interpolation:
        # Resize depth numerator and validity weights
        # independently to prevent invalid pixels
        # from introducing artificial zero depth.

        numerator = cv2.resize(
            safe_depth,
            output_size,
            interpolation=method_code
        )

        weights = cv2.resize(
            valid_mask.astype(np.float32),
            output_size,
            interpolation=method_code
        )

        # Negative cubic interpolation weights can occur
        # around discontinuities. Clamp the denominator.
        denominator = np.maximum(weights, 1e-6)

        reconstructed = numerator / denominator

        confidence = np.clip(
            weights,
            0.0,
            1.0
        )

        # Pixels without meaningful support remain invalid.
        reconstructed[weights <= 1e-6] = 0.0

    reconstructed = np.nan_to_num(
        reconstructed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    confidence = np.nan_to_num(
        confidence,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    return reconstructed, confidence


def nearest_neighbor(depth, scale=2, valid_mask=None):
    """
    Nearest-neighbour interpolation.
    """

    return interpolate_depth(
        depth,
        scale,
        "nearest",
        valid_mask
    )


def bilinear(depth, scale=2, valid_mask=None):
    """
    Bilinear interpolation.
    """

    return interpolate_depth(
        depth,
        scale,
        "bilinear",
        valid_mask
    )


def bicubic(depth, scale=2, valid_mask=None):
    """
    Bicubic interpolation.
    """

    return interpolate_depth(
        depth,
        scale,
        "bicubic",
        valid_mask
    )
