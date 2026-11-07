
import time
import numpy as np

from classical.interpolation import (
    nearest_neighbor,
    bilinear,
    bicubic
)


METHODS = {
    "nearest": nearest_neighbor,
    "bilinear": bilinear,
    "bicubic": bicubic
}


def reconstruct(
    depth,
    scale=2,
    method="bicubic",
    valid_mask=None
):
    """
    Common depth reconstruction interface.

    Parameters:
        depth: Input depth map.
        scale: 2, 4 or 8.
        method: Selected interpolation method.
        valid_mask: Optional depth validity mask.

    Returns:
        Dictionary containing:
            depth
            confidence
            method
            scale
            processing_time
            output_shape
    """

    if method not in METHODS:
        raise ValueError(
            f"Unsupported method: {method}"
        )

    start_time = time.perf_counter()

    depth = np.asarray(
        depth,
        dtype=np.float32
    )

    reconstructed, confidence = METHODS[method](
        depth,
        scale=scale,
        valid_mask=valid_mask
    )

    processing_time = (
        time.perf_counter() - start_time
    )

    return {
        "depth": reconstructed,
        "confidence": confidence,
        "method": method,
        "scale": scale,
        "processing_time": processing_time,
        "output_shape": list(reconstructed.shape)
    }


def get_available_methods():
    """
    Return currently implemented algorithms.
    """

    return list(METHODS.keys())
