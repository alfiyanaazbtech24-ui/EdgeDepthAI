"""
Middlebury Eval 3 dataset loader for EdgeDepth AI.

Uses:
    im0.png       Left RGB image
    im1.png       Right RGB image
    disp0GT.pfm   Ground-truth disparity
    mask0nocc.png Non-occluded validity mask
    calib.txt     Stereo calibration

Target:
    Disparity in pixels (NOT metric depth).
"""

from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from training.dataset_utils import (
    validate_scale_factor,
    crop_to_scale,
    resize_rgb_to_low_resolution,
    resize_nearest,
    make_model_input,
)


def read_pfm(path):
    """
    Read a Middlebury PFM file.

    Returns:
        image: vertically corrected float32 array
        scale: PFM header scale value

    The scale is retained as metadata. It is not applied to
    disparity values because the supplied Middlebury files
    have values consistent with the calibration disparity
    range when interpreted directly as pixels.
    """
    path = Path(path)

    with path.open("rb") as file:
        header = file.readline().decode("ascii").strip()

        if header not in ("PF", "Pf"):
            raise ValueError(f"Invalid PFM header: {header}")

        color = header == "PF"

        dimensions = file.readline().decode("ascii").strip()

        while dimensions.startswith("#") or not dimensions:
            dimensions = file.readline().decode("ascii").strip()

        match = re.match(r"^(\d+)\s+(\d+)$", dimensions)

        if match is None:
            raise ValueError(
                f"Invalid PFM dimensions: {dimensions}"
            )

        width, height = map(int, match.groups())

        scale = float(file.readline().decode("ascii").strip())

        if scale == 0:
            raise ValueError("PFM scale cannot be zero.")

        dtype = "<f4" if scale < 0 else ">f4"

        data = np.fromfile(file, dtype=dtype)

    channels = 3 if color else 1
    expected_values = width * height * channels

    if data.size != expected_values:
        raise ValueError(
            f"PFM data size mismatch: expected {expected_values}, "
            f"received {data.size}"
        )

    if color:
        image = data.reshape(height, width, 3)
    else:
        image = data.reshape(height, width)

    # PFM stores rows from bottom to top.
    image = np.flipud(image).copy()

    return image.astype(np.float32), scale


def read_middlebury_calibration(path):
    """Read scalar and camera calibration values from calib.txt."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    calibration = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key in ("cam0", "cam1"):
                numbers = re.findall(
                    r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                    value
                )

                calibration[key] = [
                    float(number) for number in numbers
                ]
            else:
                try:
                    calibration[key] = float(value)
                except ValueError:
                    calibration[key] = value

    if "cam0" in calibration and len(calibration["cam0"]) >= 9:
        calibration["focal_length"] = calibration["cam0"][0]

    return calibration


class MiddleburyDataset(Dataset):
    """Middlebury Eval 3 trainingH dataset."""

    def __init__(
        self,
        root_dir,
        scale_factor=2,
        scenes=None,
        max_samples=None,
    ):
        self.root_dir = Path(root_dir)
        self.scale_factor = int(scale_factor)

        validate_scale_factor(self.scale_factor)

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"Middlebury directory not found: {self.root_dir}"
            )

        if scenes is None:
            scene_dirs = sorted(
                path for path in self.root_dir.iterdir()
                if path.is_dir()
            )
        else:
            scene_dirs = [
                self.root_dir / scene for scene in scenes
            ]

        self.scenes = []

        required_files = (
            "im0.png",
            "im1.png",
            "disp0GT.pfm",
            "mask0nocc.png",
            "calib.txt",
        )

        for scene_dir in scene_dirs:
            if not scene_dir.is_dir():
                raise FileNotFoundError(scene_dir)

            missing = [
                filename
                for filename in required_files
                if not (scene_dir / filename).is_file()
            ]

            if missing:
                raise FileNotFoundError(
                    f"Missing files in {scene_dir.name}: {missing}"
                )

            self.scenes.append(scene_dir)

        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive.")

            self.scenes = self.scenes[:max_samples]

        if not self.scenes:
            raise ValueError("No valid Middlebury scenes found.")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        scene_dir = self.scenes[index]

        # Read left and right stereo images.
        left_bgr = cv2.imread(
            str(scene_dir / "im0.png"),
            cv2.IMREAD_COLOR
        )

        right_bgr = cv2.imread(
            str(scene_dir / "im1.png"),
            cv2.IMREAD_COLOR
        )

        if left_bgr is None or right_bgr is None:
            raise RuntimeError(
                f"Could not read stereo images: {scene_dir}"
            )

        left_rgb = cv2.cvtColor(
            left_bgr,
            cv2.COLOR_BGR2RGB
        )

        right_rgb = cv2.cvtColor(
            right_bgr,
            cv2.COLOR_BGR2RGB
        )

        # Read disparity and validity mask.
        disparity, pfm_scale = read_pfm(
            scene_dir / "disp0GT.pfm"
        )

        mask = cv2.imread(
            str(scene_dir / "mask0nocc.png"),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise RuntimeError(
                f"Could not read validity mask: {scene_dir}"
            )

        if disparity.ndim != 2:
            raise ValueError(
                "Expected a single-channel disparity PFM."
            )

        if (
            left_rgb.shape[:2] != disparity.shape
            or right_rgb.shape[:2] != disparity.shape
            or mask.shape != disparity.shape
        ):
            raise ValueError(
                f"Image/disparity/mask dimensions differ "
                f"in scene {scene_dir.name}"
            )

        # Crop all arrays to dimensions divisible by scale.
        left_rgb = crop_to_scale(
            left_rgb, self.scale_factor
        )

        right_rgb = crop_to_scale(
            right_rgb, self.scale_factor
        )

        disparity = crop_to_scale(
            disparity, self.scale_factor
        )

        mask = crop_to_scale(
            mask, self.scale_factor
        )

        # Strict non-occluded validity:
        # 255 = valid non-occluded
        # 128 = occluded
        # 0   = invalid
        valid_mask = (
            (mask == 255)
            & np.isfinite(disparity)
            & (disparity > 0)
        )

        disparity = np.where(
            valid_mask,
            disparity,
            0.0
        ).astype(np.float32)

        # Low-resolution left RGB.
        rgb_low = resize_rgb_to_low_resolution(
            left_rgb,
            self.scale_factor
        )

        # Create sparse low-resolution disparity input.
        disparity_low = resize_nearest(
            disparity,
            self.scale_factor
        )

        mask_low = resize_nearest(
            valid_mask.astype(np.uint8),
            self.scale_factor
        )

        mask_low = (
            (mask_low > 0)
            & np.isfinite(disparity_low)
            & (disparity_low > 0)
        ).astype(np.float32)

        disparity_low = np.where(
            mask_low > 0,
            disparity_low,
            0.0
        ).astype(np.float32)

        # Five-channel input for EdgeDepthUNet.
        model_input = make_model_input(
            rgb_low,
            disparity_low,
            mask_low
        )

        # Dense disparity target.
        target = disparity[np.newaxis, ...]
        target_mask = valid_mask[np.newaxis, ...]

        # Right image is retained for stereo-related experiments.
        right_rgb = np.transpose(
            right_rgb.astype(np.float32) / 255.0,
            (2, 0, 1)
        )

        calibration = read_middlebury_calibration(
            scene_dir / "calib.txt"
        )

        return {
            "input": torch.from_numpy(model_input),

            "target": torch.from_numpy(
                np.ascontiguousarray(target, dtype=np.float32)
            ),

            "target_mask": torch.from_numpy(
                np.ascontiguousarray(
                    target_mask.astype(np.float32)
                )
            ),

            "right_rgb": torch.from_numpy(
                np.ascontiguousarray(right_rgb, dtype=np.float32)
            ),

            "dataset_name": "Middlebury",

            "scene_name": scene_dir.name,

            "target_type": "disparity",

            "target_unit": "pixels",

            "pfm_scale": float(pfm_scale),

            "calibration": calibration,
        }