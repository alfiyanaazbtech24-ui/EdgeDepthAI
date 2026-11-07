import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from training.middlebury_dataset import (
    read_pfm,
    read_middlebury_calibration,
)


ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "Middlebury"
    / "MiddEval3"
    / "trainingH"
)


for scene_dir in sorted(ROOT.iterdir()):
    if not scene_dir.is_dir():
        continue

    pfm_path = scene_dir / "disp0GT.pfm"
    mask_path = scene_dir / "mask0nocc.png"
    calib_path = scene_dir / "calib.txt"

    if not all(p.is_file() for p in (
        pfm_path, mask_path, calib_path
    )):
        continue

    disparity, scale = read_pfm(pfm_path)

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    valid = (
        (mask == 255)
        & np.isfinite(disparity)
        & (disparity > 0)
    )

    values = disparity[valid]

    if values.size == 0:
        print(scene_dir.name, "NO VALID PIXELS")
        continue

    calibration = read_middlebury_calibration(calib_path)

    p1, p50, p99 = np.percentile(
        values, [1, 50, 99]
    )

    print(f"\nScene: {scene_dir.name}")
    print(f"PFM header scale: {scale}")
    print(f"Valid pixels: {values.size}")
    print(f"Minimum: {values.min():.3f}")
    print(f"1st percentile: {p1:.3f}")
    print(f"Median: {p50:.3f}")
    print(f"99th percentile: {p99:.3f}")
    print(f"Maximum: {values.max():.3f}")
    print(
        f"Calibration vmin/vmax: "
        f"{calibration.get('vmin')} / "
        f"{calibration.get('vmax')}"
    )