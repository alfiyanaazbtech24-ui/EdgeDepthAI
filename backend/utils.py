
import cv2
import numpy as np

from pathlib import Path
from datetime import datetime


def ensure_directory(directory):
    """
    Create directory if it does not exist.
    """

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    return directory


def save_depth_npy(depth, output_path):
    """
    Save full-precision floating-point depth.
    """

    depth = np.asarray(depth, dtype=np.float32)

    if depth.ndim != 2:
        raise ValueError(
            "Depth map must be two-dimensional."
        )

    output_path = Path(output_path)

    ensure_directory(output_path.parent)

    np.save(output_path, depth)

    return output_path


def save_depth_png(depth, output_path, valid_mask=None):
    """
    Save a normalized 16-bit visualization.

    Important:
        This PNG is a visualization, not a metric-preserving
        replacement for the original float32 depth map.
    """

    depth = np.asarray(depth, dtype=np.float32)

    if valid_mask is None:

        valid_mask = (
            np.isfinite(depth) &
            (depth > 0)
        )

    if not np.any(valid_mask):
        raise ValueError(
            "No valid depth values available for visualization."
        )

    minimum = float(np.min(depth[valid_mask]))
    maximum = float(np.max(depth[valid_mask]))

    visualization = np.zeros(
        depth.shape,
        dtype=np.uint16
    )

    if maximum > minimum:

        normalized = (
            depth[valid_mask] - minimum
        ) / (maximum - minimum)

        visualization[valid_mask] = (
            normalized * 65535
        ).astype(np.uint16)

    else:

        visualization[valid_mask] = 65535

    output_path = Path(output_path)

    ensure_directory(output_path.parent)

    success = cv2.imwrite(
        str(output_path),
        visualization
    )

    if not success:
        raise IOError(
            "Unable to save depth visualization."
        )

    return output_path


def save_confidence(confidence, output_path):
    """
    Save confidence map as a floating-point NPY file.
    """

    confidence = np.asarray(
        confidence,
        dtype=np.float32
    )

    output_path = Path(output_path)

    ensure_directory(output_path.parent)

    np.save(output_path, confidence)

    return output_path


def generate_result_id():
    """
    Generate a unique timestamp-based result ID.
    """

    return datetime.now().strftime(
        "EDG_%Y%m%d_%H%M%S_%f"
    )


def calculate_depth_statistics(depth, valid_mask=None):
    """
    Calculate descriptive statistics for valid pixels.
    """

    depth = np.asarray(depth, dtype=np.float32)

    if valid_mask is None:

        valid_mask = (
            np.isfinite(depth) &
            (depth > 0)
        )

    if not np.any(valid_mask):

        return {
            "valid_pixels": 0,
            "invalid_pixels": int(depth.size),
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None
        }

    values = depth[valid_mask]

    return {
        "valid_pixels": int(values.size),
        "invalid_pixels": int(depth.size - values.size),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values))
    }
