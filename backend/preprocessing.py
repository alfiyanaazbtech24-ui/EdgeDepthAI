
import cv2
import numpy as np

from pathlib import Path
from PIL import Image


SUPPORTED_RGB = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff"
}

SUPPORTED_DEPTH = {
    ".png", ".jpg", ".jpeg",
    ".tif", ".tiff", ".npy", ".pfm"
}


def read_pfm(path):
    """Read a PFM depth/disparity file."""

    with open(path, "rb") as file:
        header = file.readline().decode("ascii").strip()

        if header == "PF":
            channels = 3
        elif header == "Pf":
            channels = 1
        else:
            raise ValueError("Invalid PFM header.")

        dimensions = file.readline().decode("ascii").strip()

        while dimensions.startswith("#"):
            dimensions = file.readline().decode("ascii").strip()

        width, height = map(int, dimensions.split())

        scale = float(file.readline().decode("ascii").strip())

        if scale == 0:
            raise ValueError("Invalid PFM scale.")

        endian = "<" if scale < 0 else ">"

        data = np.fromfile(file, endian + "f")
        data *= abs(scale)

        expected = width * height * channels

        if data.size != expected:
            raise ValueError("PFM data size does not match dimensions.")

        shape = (
            (height, width)
            if channels == 1
            else (height, width, channels)
        )

        data = np.reshape(data, shape)
        data = np.flipud(data).copy()

        if channels == 3:
            data = data[:, :, :3]

        return data.astype(np.float32)


def load_rgb(path):
    """Load RGB guidance image."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"RGB image not found: {path}")

    if path.suffix.lower() not in SUPPORTED_RGB:
        raise ValueError("Unsupported RGB image format.")

    image = Image.open(path).convert("RGB")

    return np.asarray(image, dtype=np.uint8)


def load_depth(path):
    """
    Load a single-channel depth or disparity map.

    Supports PNG, JPG, TIFF, NPY and PFM.

    Returns:
        depth: float32 H x W array
        valid_mask: bool H x W array
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Depth map not found: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_DEPTH:
        raise ValueError(f"Unsupported depth format: {extension}")

    if extension == ".npy":
        depth = np.load(path, allow_pickle=False)

    elif extension == ".pfm":
        depth = read_pfm(path)

    else:
        depth = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED
        )

        if depth is None:
            raise ValueError("Unable to decode depth image.")

    # Accept grayscale images saved with RGB/RGBA channels.
    if depth.ndim == 3:

        if depth.shape[2] not in (3, 4):
            raise ValueError(
                "Depth image must contain one grayscale channel."
            )

        channels = depth[:, :, :3].astype(np.float32)

        difference = np.max(
            np.abs(channels - channels[:, :, :1])
        )

        if difference > 2:
            raise ValueError(
                "Colorized depth images are not supported. "
                "Upload a true grayscale depth map."
            )

        depth = channels[:, :, 0]

    if depth.ndim != 2:
        raise ValueError(
            "Depth data must be a two-dimensional array."
        )

    depth = depth.astype(np.float32)

    valid_mask = (
        np.isfinite(depth) &
        (depth > 0)
    )

    depth = np.where(
        valid_mask,
        depth,
        0.0
    ).astype(np.float32)

    if depth.size == 0:
        raise ValueError("Depth map is empty.")

    if not np.any(valid_mask):
        raise ValueError(
            "Depth map contains no valid depth measurements."
        )

    return depth, valid_mask


def validate_rgb_depth(rgb, depth):
    """Validate RGB and depth dimensions."""

    if rgb is not None:

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                "RGB guidance must have three channels."
            )

        if rgb.shape[:2] != depth.shape[:2]:
            raise ValueError(
                "RGB and depth dimensions must match."
            )

    return True


def create_validity_mask(depth):
    """Generate validity mask."""

    return (
        np.isfinite(depth) &
        (depth > 0)
    )


def normalize_depth(depth, valid_mask=None):
    """Normalize valid depth values for visualization/model input."""

    depth = np.asarray(depth, dtype=np.float32)

    if valid_mask is None:
        valid_mask = create_validity_mask(depth)

    if not np.any(valid_mask):
        return np.zeros_like(depth, dtype=np.float32)

    minimum = np.min(depth[valid_mask])
    maximum = np.max(depth[valid_mask])

    result = np.zeros_like(depth, dtype=np.float32)

    if maximum > minimum:
        result[valid_mask] = (
            depth[valid_mask] - minimum
        ) / (maximum - minimum)

    return result
