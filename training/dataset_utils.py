"""Shared utilities for EdgeDepth AI dataset loaders."""

from pathlib import Path

import cv2
import numpy as np


SUPPORTED_SCALES = (2, 4, 8)


def validate_scale_factor(scale_factor):
    """Validate the supported upsampling factors."""
    if scale_factor not in SUPPORTED_SCALES:
        raise ValueError(
            f"scale_factor must be one of {SUPPORTED_SCALES}"
        )


def crop_to_scale(array, scale_factor):
    """
    Crop the bottom and right edges so dimensions are
    divisible by the requested scale factor.
    """
    validate_scale_factor(scale_factor)

    height, width = array.shape[:2]

    new_height = (height // scale_factor) * scale_factor
    new_width = (width // scale_factor) * scale_factor

    if new_height == 0 or new_width == 0:
        raise ValueError("Image is too small for the selected scale.")

    return array[:new_height, :new_width, ...]


def resize_rgb_to_low_resolution(rgb, scale_factor):
    """Resize an RGB image to its low-resolution input size."""
    height, width = rgb.shape[:2]

    low_width = width // scale_factor
    low_height = height // scale_factor

    resized = cv2.resize(
        rgb,
        (low_width, low_height),
        interpolation=cv2.INTER_AREA
    )

    return np.ascontiguousarray(
        resized.astype(np.float32) / 255.0
    )


def resize_nearest(array, scale_factor):
    """Resize a single-channel array using nearest-neighbor."""
    height, width = array.shape[:2]

    result = cv2.resize(
        array,
        (width // scale_factor, height // scale_factor),
        interpolation=cv2.INTER_NEAREST
    )

    return np.ascontiguousarray(result)


def make_model_input(rgb_low, depth_low, mask_low):
    """
    Create the five-channel input expected by EdgeDepthUNet.

    Channels:
        0-2: RGB
        3:   Low-resolution depth/disparity
        4:   Validity mask
    """
    rgb_chw = np.transpose(rgb_low, (2, 0, 1))

    depth_chw = depth_low[np.newaxis, ...].astype(np.float32)
    mask_chw = mask_low[np.newaxis, ...].astype(np.float32)

    model_input = np.concatenate(
        [rgb_chw, depth_chw, mask_chw],
        axis=0
    )

    return np.ascontiguousarray(model_input, dtype=np.float32)