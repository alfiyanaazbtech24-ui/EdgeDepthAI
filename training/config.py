
"""
Experiment configuration for EdgeDepth AI.

NYU and Middlebury configurations are deliberately separate:
- NYU targets are metric depth in meters.
- Middlebury targets are disparity in pixels.
"""

from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TrainingConfig:
    dataset: str
    data_path: str
    output_dir: str

    target_type: str
    target_unit: str

    scale_factor: int = 2
    batch_size: int = 2
    epochs: int = 20
    learning_rate: float = 0.0001
    base_channels: int = 16

    train_fraction: float = 0.9
    seed: int = 42
    num_workers: int = 0

    # None means use the full available dataset.
    # Set a small integer for a smoke test.
    max_samples: int | None = None

    device: str = "cpu"

    def validate(self):
        """Validate configuration values before training."""

        if self.dataset not in ("nyu", "middlebury"):
            raise ValueError(
                "dataset must be 'nyu' or 'middlebury'."
            )

        if self.scale_factor not in (2, 4, 8):
            raise ValueError(
                "scale_factor must be 2, 4, or 8."
            )

        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if self.dataset == "middlebury" and self.batch_size != 1:
            raise ValueError(
                "Middlebury currently requires batch_size=1 "
                "because scene image dimensions vary."
            )

        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if self.base_channels < 1:
            raise ValueError("base_channels must be positive.")

        if not 0 < self.train_fraction < 1:
            raise ValueError(
                "train_fraction must be between 0 and 1."
            )

        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        if self.max_samples is not None and self.max_samples < 2:
            raise ValueError(
                "max_samples must be at least 2, or None."
            )

        if self.dataset == "nyu":
            if self.target_type != "depth":
                raise ValueError(
                    "NYU target_type must be 'depth'."
                )

            if self.target_unit != "meters":
                raise ValueError(
                    "NYU target_unit must be 'meters'."
                )

        if self.dataset == "middlebury":
            if self.target_type != "disparity":
                raise ValueError(
                    "Middlebury target_type must be 'disparity'."
                )

            if self.target_unit != "pixels":
                raise ValueError(
                    "Middlebury target_unit must be 'pixels'."
                )

        data_path = Path(self.data_path)

        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {data_path}"
            )

        Path(self.output_dir).mkdir(
            parents=True,
            exist_ok=True
        )

        return self

    def to_dict(self):
        """Return configuration as a JSON-serializable dictionary."""
        return asdict(self)


def create_nyu_config(
    max_samples=None,
    scale_factor=2,
):
    """Create a configuration for NYU Depth V2."""

    return TrainingConfig(
        dataset="nyu",
        data_path=(
            "datasets/NYU_Depth_V2/"
            "nyu_depth_v2_labeled.mat"
        ),
        output_dir="outputs/training/nyu",

        target_type="depth",
        target_unit="meters",

        scale_factor=scale_factor,
        batch_size=2,
        epochs=20,
        learning_rate=0.0001,
        base_channels=16,

        train_fraction=0.9,
        seed=42,
        num_workers=0,
        max_samples=max_samples,
        device="cpu",
    )


def create_middlebury_config(
    max_samples=None,
    scale_factor=2,
):
    """Create a configuration for Middlebury."""

    return TrainingConfig(
        dataset="middlebury",
        data_path="datasets/Middlebury/MiddEval3",
        output_dir="outputs/training/middlebury",

        target_type="disparity",
        target_unit="pixels",

        scale_factor=scale_factor,
        batch_size=1,
        epochs=20,
        learning_rate=0.0001,
        base_channels=16,

        train_fraction=0.8,
        seed=42,
        num_workers=0,
        max_samples=max_samples,
        device="cpu",
    )
