
import cv2
import numpy as np


def detect_rgb_edges(
    rgb,
    low_threshold=50,
    high_threshold=150
):
    """
    Detect edges in an RGB image using Canny.

    Returns:
        Boolean edge map.
    """

    if rgb is None:
        raise ValueError("RGB image is required.")

    rgb = np.asarray(rgb)

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Expected an RGB image.")

    rgb = rgb[:, :, :3].astype(np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(
        gray,
        int(low_threshold),
        int(high_threshold)
    )

    return edges > 0


def detect_depth_edges(depth, threshold=None):
    """
    Detect depth discontinuities using Sobel gradients.

    Returns:
        Boolean edge map and gradient magnitude.
    """

    depth = np.asarray(depth, dtype=np.float32)

    if depth.ndim != 2:
        raise ValueError("Depth must be a 2D array.")

    valid = np.isfinite(depth) & (depth > 0)

    if not np.any(valid):
        raise ValueError("No valid depth pixels.")

    safe_depth = depth.copy()
    safe_depth[~valid] = 0.0

    grad_x = cv2.Sobel(
        safe_depth,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    grad_y = cv2.Sobel(
        safe_depth,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(grad_x, grad_y)

    if threshold is None:
        valid_gradients = magnitude[valid]

        threshold = float(
            np.percentile(valid_gradients, 90)
        )

    edges = (magnitude >= threshold) & valid

    return edges, magnitude


def get_edge_strength(rgb):
    """
    Calculate normalized Sobel edge strength from RGB guidance.

    Returns:
        Float32 edge strength map in [0, 1].
    """

    rgb = np.asarray(rgb)

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Expected RGB guidance image.")

    gray = cv2.cvtColor(
        rgb[:, :, :3].astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )

    gray = gray.astype(np.float32) / 255.0

    grad_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    grad_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(grad_x, grad_y)

    maximum = float(magnitude.max())

    if maximum > 1e-8:
        magnitude /= maximum

    return np.clip(magnitude, 0.0, 1.0).astype(np.float32)
