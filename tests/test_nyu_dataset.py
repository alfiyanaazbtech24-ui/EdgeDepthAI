import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch

from training.nyu_dataset import NYUDepthDataset
from models.edgedepth_unet import build_model


MAT_PATH = (
    "datasets/NYU_Depth_V2/nyu_depth_v2_labeled.mat"
)


def test_nyu_dataset():
    for scale in (2, 4, 8):
        dataset = NYUDepthDataset(
            mat_path=MAT_PATH,
            scale_factor=scale,
            max_samples=3,
        )

        sample = dataset[0]

        model_input = sample["input"]
        target = sample["target"]
        target_mask = sample["target_mask"]

        print(f"\nTesting scale factor: {scale}")

        print("Input:", tuple(model_input.shape))
        print("Target:", tuple(target.shape))
        print("Target mask:", tuple(target_mask.shape))

        assert model_input.shape[0] == 5

        assert target.shape[0] == 1
        assert target_mask.shape == target.shape

        assert target.shape[-2] == model_input.shape[-2] * scale
        assert target.shape[-1] == model_input.shape[-1] * scale

        assert torch.isfinite(model_input).all()
        assert torch.isfinite(target).all()

        assert model_input[0:3].min() >= 0
        assert model_input[0:3].max() <= 1

        assert set(torch.unique(model_input[4]).tolist()).issubset(
            {0.0, 1.0}
        )

        assert target_mask.sum() > 0

        assert sample["target_type"] == "depth"
        assert sample["target_unit"] == "meters"

        # Check compatibility with the existing model.
        model = build_model(
            scale_factor=scale,
            base_channels=16
        )

        model.eval()

        with torch.no_grad():
            output = model(model_input.unsqueeze(0))

        assert output["depth"].shape == target.unsqueeze(0).shape
        assert torch.isfinite(output["depth"]).all()

        print("Model output:", tuple(output["depth"].shape))
        print("NYU scale", scale, "PASSED")

        dataset.close()


if __name__ == "__main__":
    test_nyu_dataset()