
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config import (
    create_nyu_config,
    create_middlebury_config,
)


def test_nyu_config():
    config = create_nyu_config(max_samples=10)

    config.validate()

    assert config.dataset == "nyu"
    assert config.target_type == "depth"
    assert config.target_unit == "meters"
    assert config.batch_size == 2
    assert config.max_samples == 10

    print("NYU configuration test PASSED")


def test_middlebury_config():
    config = create_middlebury_config(max_samples=5)

    config.validate()

    assert config.dataset == "middlebury"
    assert config.target_type == "disparity"
    assert config.target_unit == "pixels"
    assert config.batch_size == 1
    assert config.max_samples == 5

    print("Middlebury configuration test PASSED")


def test_invalid_configuration():
    config = create_nyu_config(max_samples=10)

    config.target_unit = "pixels"

    try:
        config.validate()
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid NYU target unit was not rejected."
        )

    print("Invalid configuration rejection test PASSED")


if __name__ == "__main__":
    test_nyu_config()
    test_middlebury_config()
    test_invalid_configuration()

    print("\nAll training configuration tests PASSED.")
