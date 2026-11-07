"""
NYU Depth V2 dataset loader for EdgeDepth AI.

The loader uses:
    images:    RGB images
    depths:    Dense metric-depth supervision
    rawDepths: Sparse/raw depth measurements

The model receives:
    RGB + low-resolution raw depth + validity mask

The target is:
    Full-resolution dense metric depth in meters.
"""

from pathlib import Path
import os

import cv2
import h5py
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


class NYUDepthDataset(Dataset):
    """Lazy-loading NYU Depth V2 dataset."""

    def __init__(
        self,
        mat_path,
        scale_factor=2,
        indices=None,
        max_samples=None,
    ):
        self.mat_path = Path(mat_path)
        self.scale_factor = int(scale_factor)

        validate_scale_factor(self.scale_factor)

        if not self.mat_path.is_file():
            raise FileNotFoundError(
                f"NYU MAT file not found: {self.mat_path}"
            )

        # Read metadata without keeping the file open.
        with h5py.File(self.mat_path, "r") as file:
            required_keys = ("images", "depths", "rawDepths")

            for key in required_keys:
                if key not in file:
                    raise KeyError(
                        f"NYU MAT file is missing dataset: {key}"
                    )

            image_shape = file["images"].shape
            depth_shape = file["depths"].shape
            raw_shape = file["rawDepths"].shape

            if len(image_shape) != 4 or image_shape[1] != 3:
                raise ValueError(
                    f"Unexpected NYU image shape: {image_shape}"
                )

            if len(depth_shape) != 3 or len(raw_shape) != 3:
                raise ValueError(
                    "NYU depth arrays must have shape [N, H, W]."
                )

            if image_shape[0] != depth_shape[0]:
                raise ValueError("NYU image/depth sample counts differ.")

            if depth_shape != raw_shape:
                raise ValueError("NYU dense and raw depth shapes differ.")

            self.total_samples = int(image_shape[0])
            self.height = int(depth_shape[1])
            self.width = int(depth_shape[2])

        if indices is None:
            selected_indices = list(range(self.total_samples))
        else:
            selected_indices = [int(i) for i in indices]

        for index in selected_indices:
            if index < 0 or index >= self.total_samples:
                raise IndexError(
                    f"NYU sample index out of range: {index}"
                )

        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive.")
            selected_indices = selected_indices[:max_samples]

        if not selected_indices:
            raise ValueError("NYU dataset contains no selected samples.")

        self.indices = selected_indices

        # HDF5 is opened lazily, separately in each worker process.
        self._file = None
        self._file_pid = None

    def __len__(self):
        return len(self.indices)

    def _get_file(self):
        """Get an HDF5 handle for the current process."""
        current_pid = os.getpid()

        if self._file is None or self._file_pid != current_pid:
            self.close()

            self._file = h5py.File(
                self.mat_path,
                "r"
            )
            self._file_pid = current_pid

        return self._file

    def close(self):
        """Close the current HDF5 handle."""
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
                self._file_pid = None

    def __getstate__(self):
        """Avoid serializing an open HDF5 handle to workers."""
        state = self.__dict__.copy()
        state["_file"] = None
        state["_file_pid"] = None
        return state

    def __del__(self):
        self.close()

    def __getitem__(self, item):
        sample_index = self.indices[item]

        file = self._get_file()

        # NYU storage:
        # images: [N, 3, H, W]
        # depths: [N, H, W]
        # rawDepths: [N, H, W]

        rgb = np.asarray(
            file["images"][sample_index],
            dtype=np.uint8
        )

        dense_depth = np.asarray(
            file["depths"][sample_index],
            dtype=np.float32
        )

        raw_depth = np.asarray(
            file["rawDepths"][sample_index],
            dtype=np.float32
        )

        # Convert RGB from CHW to HWC.
        rgb = np.transpose(rgb, (1, 2, 0))

        # Ensure all arrays have matching spatial dimensions.
        if rgb.shape[:2] != dense_depth.shape:
            raise ValueError(
                f"RGB/depth size mismatch at sample {sample_index}"
            )

        # Crop dimensions to multiples of scale_factor.
        rgb = crop_to_scale(rgb, self.scale_factor)
        dense_depth = crop_to_scale(
            dense_depth, self.scale_factor
        )
        raw_depth = crop_to_scale(
            raw_depth, self.scale_factor
        )

        # Dense target validity.
        target_mask = (
            np.isfinite(dense_depth) &
            (dense_depth > 0)
        )

        dense_depth = np.where(
            target_mask,
            dense_depth,
            0.0
        ).astype(np.float32)

        # Raw depth validity.
        raw_mask = (
            np.isfinite(raw_depth) &
            (raw_depth > 0)
        )

        raw_depth = np.where(
            raw_mask,
            raw_depth,
            0.0
        ).astype(np.float32)

        # Low-resolution RGB.
        rgb_low = resize_rgb_to_low_resolution(
            rgb,
            self.scale_factor
        )

        # Low-resolution sparse depth and validity.
        depth_low = resize_nearest(
            raw_depth,
            self.scale_factor
        )

        mask_low = resize_nearest(
            raw_mask.astype(np.uint8),
            self.scale_factor
        )

        # Remove any invalid values after resizing.
        mask_low = (
            np.isfinite(depth_low) &
            (depth_low > 0) &
            (mask_low > 0)
        ).astype(np.float32)

        depth_low = np.where(
            mask_low > 0,
            depth_low,
            0.0
        ).astype(np.float32)

        # Construct model input: [5, H/scale, W/scale].
        model_input = make_model_input(
            rgb_low,
            depth_low,
            mask_low
        )

        # Construct dense target: [1, H, W].
        target = dense_depth[np.newaxis, ...]
        target_mask = target_mask[np.newaxis, ...]

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
            "dataset_name": "NYU_Depth_V2",
            "target_type": "depth",
            "target_unit": "meters",
            "sample_index": sample_index,
        }