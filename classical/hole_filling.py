
import numpy as np
from scipy.ndimage import distance_transform_edt


def fill_depth_holes(
    depth,
    valid_mask=None,
    max_distance=None,
    return_mask=False
):
    """
    Fill invalid depth pixels using nearest valid neighbours.

    Args:
        depth: 2D depth map.
        valid_mask: Optional boolean mask.
        max_distance: Maximum allowed distance in pixels.
                      None means fill all holes.
        return_mask: Return a mask identifying newly filled pixels.

    Returns:
        Filled depth map, or (filled_depth, filled_mask).
    """

    depth = np.asarray(depth, dtype=np.float32)

    if depth.ndim != 2:
        raise ValueError("Depth map must be 2D.")

    valid = np.isfinite(depth) & (depth > 0)

    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    if not np.any(valid):
        raise ValueError("Cannot fill holes: no valid depth pixels exist.")

    missing = ~valid

    filled = depth.copy()
    filled_mask = np.zeros(depth.shape, dtype=bool)

    if not np.any(missing):
        if return_mask:
            return filled, filled_mask
        return filled

    distances, nearest_indices = distance_transform_edt(
        missing,
        return_distances=True,
        return_indices=True
    )

    nearest_values = depth[tuple(nearest_indices)]

    allowed = missing.copy()

    if max_distance is not None:
        if max_distance < 0:
            raise ValueError("max_distance cannot be negative.")

        allowed &= distances <= float(max_distance)

    filled[allowed] = nearest_values[allowed]
    filled_mask[allowed] = True

    filled = np.nan_to_num(
        filled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    if return_mask:
        return filled, filled_mask

    return filled
