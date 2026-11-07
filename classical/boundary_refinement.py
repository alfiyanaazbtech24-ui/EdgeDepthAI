
import numpy as np

from classical.guided_filter import guided_filter
from backend.edge_detection import get_edge_strength


def refine_boundaries(
    rgb,
    depth,
    valid_mask=None,
    radius=8,
    eps=0.01,
    edge_preservation=0.85
):
    """
    Refine depth near RGB object boundaries.

    Strong RGB edges receive more weight from the original depth.
    Smooth image regions receive more weight from guided filtering.

    Args:
        rgb: RGB guidance image.
        depth: 2D depth map.
        valid_mask: Optional validity mask.
        radius: Guided filter radius.
        eps: Guided filter regularization.
        edge_preservation: Weight assigned to original depth at edges.

    Returns:
        Refined float32 depth map.
    """

    depth = np.asarray(depth, dtype=np.float32)

    if depth.ndim != 2:
        raise ValueError("Depth must be a 2D array.")

    rgb = np.asarray(rgb)

    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB and depth dimensions must match.")

    if not 0.0 <= edge_preservation <= 1.0:
        raise ValueError("edge_preservation must be between 0 and 1.")

    valid = np.isfinite(depth) & (depth > 0)

    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    if not np.any(valid):
        raise ValueError("No valid depth pixels available.")

    guided = guided_filter(
        guide=rgb,
        source=depth,
        radius=radius,
        eps=eps,
        valid_mask=valid
    )

    edge_strength = get_edge_strength(rgb)

    # Higher edge strength means greater preservation of original depth.
    original_weight = edge_strength * float(edge_preservation)

    refined = (
        original_weight * depth
        + (1.0 - original_weight) * guided
    )

    refined = np.nan_to_num(
        refined,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    refined[~valid] = 0.0

    return refined
