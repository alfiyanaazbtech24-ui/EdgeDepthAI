
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def bilateral_refine(
    depth,
    valid_mask=None,
    diameter=7,
    sigma_color=0.1,
    sigma_space=7.0
):
    """
    Edge-preserving bilateral filtering for a depth map.

    Args:
        depth: 2D float32 depth map.
        valid_mask: Optional boolean validity mask.
        diameter: Bilateral filter neighbourhood diameter.
        sigma_color: Depth range sensitivity.
        sigma_space: Spatial neighbourhood sensitivity.

    Returns:
        Refined float32 depth map.
    """
    depth = np.asarray(depth, dtype=np.float32)

    if depth.ndim != 2:
        raise ValueError("Depth map must be a 2D array.")

    valid = np.isfinite(depth) & (depth > 0)

    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    if not np.any(valid):
        raise ValueError("Cannot filter a depth map with no valid pixels.")

    # Temporarily fill invalid pixels using nearest valid neighbours.
    nearest_indices = distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True
    )

    filled = depth.copy()
    filled[~valid] = depth[tuple(nearest_indices)][~valid]

    filtered = cv2.bilateralFilter(
        filled,
        d=int(diameter),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space)
    )

    filtered = np.nan_to_num(
        filtered,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    # Keep original invalid pixels invalid.
    filtered[~valid] = 0.0

    return filtered
