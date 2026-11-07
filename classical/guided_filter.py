
import cv2
import numpy as np


def guided_filter(
    guide,
    source,
    radius=8,
    eps=0.01,
    valid_mask=None
):
    """
    Grayscale guided filter.

    Args:
        guide: RGB or grayscale guidance image.
        source: 2D depth map.
        radius: Local filter radius.
        eps: Regularization parameter.
        valid_mask: Optional boolean depth validity mask.

    Returns:
        Refined float32 depth map.
    """

    source = np.asarray(source, dtype=np.float32)

    if source.ndim != 2:
        raise ValueError("Source depth must be a 2D array.")

    if guide is None:
        raise ValueError("A guidance image is required.")

    guide = np.asarray(guide)

    if guide.ndim == 3:
        if guide.shape[2] == 4:
            guide = guide[:, :, :3]

        guide = cv2.cvtColor(
            guide.astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        )

    if guide.ndim != 2:
        raise ValueError("Guide must be a grayscale or RGB image.")

    if guide.shape != source.shape:
        raise ValueError(
            "Guide and source depth must have identical dimensions."
        )

    if radius < 1:
        raise ValueError("Radius must be at least 1.")

    if eps <= 0:
        raise ValueError("Epsilon must be positive.")

    guide = guide.astype(np.float32)

    if guide.max() > 1.0:
        guide /= 255.0

    valid = np.isfinite(source) & (source > 0)

    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    if not np.any(valid):
        raise ValueError("No valid depth pixels available.")

    # Replace invalid values for stable local calculations.
    source_filled = source.copy()
    source_filled[~valid] = 0.0

    kernel_size = 2 * int(radius) + 1

    def box_mean(array):
        return cv2.boxFilter(
            array,
            ddepth=-1,
            ksize=(kernel_size, kernel_size),
            normalize=True,
            borderType=cv2.BORDER_REFLECT
        )

    # Local means.
    mean_i = box_mean(guide)
    mean_p = box_mean(source_filled)

    mean_ii = box_mean(guide * guide)
    mean_ip = box_mean(guide * source_filled)

    variance_i = mean_ii - mean_i * mean_i
    covariance_ip = mean_ip - mean_i * mean_p

    a = covariance_ip / (variance_i + float(eps))
    b = mean_p - a * mean_i

    mean_a = box_mean(a)
    mean_b = box_mean(b)

    output = mean_a * guide + mean_b

    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    # Do not silently turn invalid measurements into valid measurements.
    output[~valid] = 0.0

    return output
