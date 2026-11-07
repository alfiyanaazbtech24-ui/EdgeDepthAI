
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from training.middlebury_dataset import (
    MiddleburyDataset,
    read_pfm,
)


DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "Middlebury"
    / "MiddEval3"
    / "trainingH"
)


def test_pfm_reader():
    pfm_path = (
        DATASET_ROOT
        / "Adirondack"
        / "disp0GT.pfm"
    )

    disparity, scale = read_pfm(pfm_path)

    finite_percentage = (
        torch.isfinite(torch.from_numpy(disparity))
        .float()
        .mean()
        .item() * 100
    )

    print("\nPFM reader test")
    print("Shape:", disparity.shape)
    print("Scale:", scale)
    print("Finite percentage:", finite_percentage)

    assert disparity.shape == (992, 1436)
    assert scale < 0
    assert torch.isfinite(
        torch.from_numpy(disparity)
    ).any()

    print("PFM reader PASSED")


def test_middlebury_dataset():
    for scale in (2, 4, 8):
        dataset = MiddleburyDataset(
            root_dir=DATASET_ROOT,
            scale_factor=scale,
            scenes=["Adirondack"],
        )

        sample = dataset[0]

        model_input = sample["input"]
        target = sample["target"]
        target_mask = sample["target_mask"]
        right_rgb = sample["right_rgb"]

        expected_height = (992 // scale) * scale
        expected_width = (1436 // scale) * scale

        print(f"\nTesting Middlebury scale: {scale}")
        print("Input:", tuple(model_input.shape))
        print("Target:", tuple(target.shape))
        print("Mask:", tuple(target_mask.shape))
        print("Right RGB:", tuple(right_rgb.shape))

        assert model_input.shape[0] == 5

        assert target.shape == (
            1,
            expected_height,
            expected_width
        )

        assert target_mask.shape == target.shape

        assert right_rgb.shape == (
            3,
            expected_height,
            expected_width
        )

        assert target.shape[-2] == model_input.shape[-2] * scale
        assert target.shape[-1] == model_input.shape[-1] * scale

        assert torch.isfinite(model_input).all()
        assert torch.isfinite(target).all()
        assert torch.isfinite(right_rgb).all()

        assert model_input[0:3].min() >= 0
        assert model_input[0:3].max() <= 1

        assert target_mask.sum() > 0

        assert sample["target_type"] == "disparity"
        assert sample["target_unit"] == "pixels"
        assert sample["scene_name"] == "Adirondack"

        calibration = sample["calibration"]

        assert abs(calibration["focal_length"] - 2076.037) < 0.01
        assert abs(calibration["baseline"] - 176.252) < 0.01
        assert abs(calibration["doffs"] - 106.542) < 0.01

        print("Valid disparity pixels:", int(target_mask.sum()))
        print("Calibration:", calibration)
        print(f"Middlebury scale {scale} PASSED")


if __name__ == "__main__":
    test_pfm_reader()
    test_middlebury_dataset()
