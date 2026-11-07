import sys
from pathlib import Path

# Add the project root to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from classical.bilateral import bilateral_refine
from classical.guided_filter import guided_filter
from classical.hole_filling import fill_depth_holes
from classical.boundary_refinement import refine_boundaries

from backend.edge_detection import (
    detect_rgb_edges,
    detect_depth_edges
)

from backend.depth_refinement import refine_depth_map


def main():
    # Create a synthetic RGB image with two regions.
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    rgb[:, :32] = [255, 0, 0]
    rgb[:, 32:] = [0, 255, 0]

    # Create a synthetic depth map.
    depth = np.ones((64, 64), dtype=np.float32) * 5.0

    # Make the right side a different depth.
    depth[:, 32:] = 8.0

    # Introduce invalid depth pixels.
    depth[20:25, 20:25] = 0.0
    depth[40:45, 40:45] = np.nan

    valid = np.isfinite(depth) & (depth > 0)

    print("\nTesting Phase 3 algorithms")
    print("--------------------------")

    # Test bilateral filtering.
    bilateral = bilateral_refine(
        depth,
        valid_mask=valid
    )
    print("Bilateral filtering:", bilateral.shape)

    # Test guided filtering.
    guided = guided_filter(
        rgb,
        depth,
        valid_mask=valid
    )
    print("Guided filtering:", guided.shape)

    # Test hole filling.
    filled, filled_mask = fill_depth_holes(
        depth,
        valid_mask=valid,
        return_mask=True
    )
    print("Hole filling:", filled.shape)
    print("Filled pixels:", int(filled_mask.sum()))

    # Test boundary refinement.
    boundary = refine_boundaries(
        rgb,
        filled
    )
    print("Boundary refinement:", boundary.shape)

    # Test RGB edge detection.
    rgb_edges = detect_rgb_edges(rgb)
    print("RGB edge detection:", rgb_edges.shape)

    # Test depth edge detection.
    depth_edges, gradient = detect_depth_edges(filled)
    print("Depth edge detection:", depth_edges.shape)
    print("Depth gradient:", gradient.shape)

    # Test complete pipeline.
    result = refine_depth_map(
        rgb=rgb,
        depth=depth,
        valid_mask=valid,
        use_bilateral=True,
        use_guided=True,
        use_hole_filling=True,
        use_boundary=True
    )

    output = result["depth"]

    print("\nComplete refinement pipeline")
    print("-----------------------------")
    print("Output shape:", output.shape)
    print("NaN present:", np.isnan(output).any())
    print("Infinite values:", np.isinf(output).any())
    print("Processing time:", result["processing_time"])
    print("Statistics:", result["statistics"])

    # Basic correctness checks.
    assert output.shape == depth.shape
    assert np.isfinite(output).all()
    assert result["filled_mask"].sum() > 0
    assert result["valid_mask"].all()

    print("\nALL PHASE 3 TESTS PASSED!")


if __name__ == "__main__":
    main()
