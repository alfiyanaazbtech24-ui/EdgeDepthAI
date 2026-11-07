
import time
import numpy as np

from classical.bilateral import bilateral_refine
from classical.guided_filter import guided_filter
from classical.hole_filling import fill_depth_holes
from classical.boundary_refinement import refine_boundaries
from backend.edge_detection import (
    detect_rgb_edges,
    detect_depth_edges
)


def refine_depth_map(
    rgb,
    depth,
    valid_mask=None,
    use_bilateral=True,
    use_guided=True,
    use_hole_filling=True,
    use_boundary=True,
    bilateral_diameter=7,
    bilateral_sigma_color=0.1,
    bilateral_sigma_space=7.0,
    guided_radius=8,
    guided_eps=0.01,
    hole_max_distance=None,
    edge_preservation=0.85
):
    """
    Complete edge-aware depth refinement pipeline.

    Args:
        rgb: RGB guidance image.
        depth: 2D depth map.
        valid_mask: Optional validity mask.
        use_bilateral: Enable bilateral filtering.
        use_guided: Enable guided filtering.
        use_hole_filling: Enable hole filling.
        use_boundary: Enable boundary refinement.

    Returns:
        Dictionary containing refined depth and diagnostics.
    """

    start_time = time.perf_counter()

    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=np.float32)

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RGB image must have three channels.")

    if depth.ndim != 2:
        raise ValueError("Depth map must be 2D.")

    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            "RGB and depth dimensions must match. "
            "Resize the RGB image or depth map before refinement."
        )

    valid = np.isfinite(depth) & (depth > 0)

    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    if not np.any(valid):
        raise ValueError("Input depth map has no valid pixels.")

    working_depth = depth.copy()
    working_depth[~valid] = 0.0

    original_valid_count = int(valid.sum())

    # Step 1: Bilateral filtering.
    if use_bilateral:
        working_depth = bilateral_refine(
            working_depth,
            valid_mask=valid,
            diameter=bilateral_diameter,
            sigma_color=bilateral_sigma_color,
            sigma_space=bilateral_sigma_space
        )

    # Step 2: RGB-guided filtering.
    if use_guided:
        working_depth = guided_filter(
            guide=rgb[:, :, :3],
            source=working_depth,
            radius=guided_radius,
            eps=guided_eps,
            valid_mask=valid
        )

    # Step 3: Fill missing depth values.
    if use_hole_filling:
        working_depth, filled_mask = fill_depth_holes(
            working_depth,
            valid_mask=valid,
            max_distance=hole_max_distance,
            return_mask=True
        )
    else:
        filled_mask = np.zeros(depth.shape, dtype=bool)

    # After hole filling, newly filled pixels are usable.
    refined_valid = valid | filled_mask

    # Step 4: Refine object boundaries.
    if use_boundary:
        working_depth = refine_boundaries(
            rgb=rgb[:, :, :3],
            depth=working_depth,
            valid_mask=refined_valid,
            radius=guided_radius,
            eps=guided_eps,
            edge_preservation=edge_preservation
        )

    # Diagnostics.
    rgb_edges = detect_rgb_edges(rgb[:, :, :3])

    depth_edges, depth_gradient = detect_depth_edges(
        working_depth,
        threshold=None
    )

    working_depth = np.nan_to_num(
        working_depth,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    elapsed = time.perf_counter() - start_time

    valid_values = working_depth[refined_valid]

    if valid_values.size:
        depth_min = float(valid_values.min())
        depth_max = float(valid_values.max())
        depth_mean = float(valid_values.mean())
    else:
        depth_min = 0.0
        depth_max = 0.0
        depth_mean = 0.0

    return {
        "depth": working_depth,
        "valid_mask": refined_valid,
        "filled_mask": filled_mask,
        "rgb_edges": rgb_edges,
        "depth_edges": depth_edges,
        "depth_gradient": depth_gradient,
        "processing_time": float(elapsed),
        "statistics": {
            "original_valid_pixels": original_valid_count,
            "filled_pixels": int(filled_mask.sum()),
            "final_valid_pixels": int(refined_valid.sum()),
            "minimum_depth": depth_min,
            "maximum_depth": depth_max,
            "mean_depth": depth_mean
        }
    }
