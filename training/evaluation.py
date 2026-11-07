
"""
Evaluation framework for EdgeDepth AI.

Compares:
    1. EdgeDepth AI
    2. Nearest-neighbor interpolation
    3. Bilinear interpolation
    4. Bicubic interpolation

NYU metrics are in meters.
Middlebury metrics are in disparity pixels.

Evaluation uses the deterministic validation split recorded
in the training checkpoint configuration.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.edgedepth_unet import build_model

from classical.interpolation import interpolate_depth

from training.checkpoints import load_checkpoint

from training.data_pipeline import (
    create_nyu_dataloaders,
    create_middlebury_dataloaders,
)


METHODS = (
    "edge_depth_ai",
    "nearest",
    "bilinear",
    "bicubic",
)


def create_validation_loader(config):
    """Recreate the validation split used during training."""

    common = {
        "scale_factor": config["scale_factor"],
        "train_fraction": config["train_fraction"],
        "seed": config["seed"],
        "num_workers": 0,
        "max_samples": config.get("max_samples"),
    }

    if config["dataset"] == "nyu":
        _, validation_loader = create_nyu_dataloaders(
            mat_path=config["data_path"],
            batch_size=config["batch_size"],
            **common,
        )

    elif config["dataset"] == "middlebury":
        _, validation_loader = create_middlebury_dataloaders(
            root_dir=config["data_path"],
            **common,
        )

    else:
        raise ValueError(
            f"Unsupported checkpoint dataset: {config['dataset']}"
        )

    return validation_loader



def create_metric_accumulators():
    return {
        method: {
            "absolute_error_sum": 0.0,
            "squared_error_sum": 0.0,
            "evaluated_pixels": 0,
            "ground_truth_valid_pixels": 0,
            "coverage_pixels": 0,
        }
        for method in METHODS
    }



def update_metrics(
    accumulator,
    prediction,
    target,
    target_mask,
    common_valid_mask,
):
    """Accumulate errors on common pixels and track individual coverage."""

    valid_target = (
        (target_mask > 0)
        & np.isfinite(target_mask)
        & np.isfinite(target)
        & (target > 0)
    )

    valid_prediction = (
        valid_target
        & np.isfinite(prediction)
        & (prediction > 0)
    )

    accumulator["ground_truth_valid_pixels"] += int(
        valid_target.sum()
    )

    accumulator["coverage_pixels"] += int(
        valid_prediction.sum()
    )

    if not np.any(common_valid_mask):
        return

    error = (
        prediction[common_valid_mask].astype(np.float64)
        - target[common_valid_mask].astype(np.float64)
    )

    accumulator["absolute_error_sum"] += float(
        np.abs(error).sum()
    )

    accumulator["squared_error_sum"] += float(
        np.square(error).sum()
    )

    accumulator["evaluated_pixels"] += int(
        common_valid_mask.sum()
    )


def finalize_metrics(accumulators):
    results = {}

    for method, values in accumulators.items():
        count = values["evaluated_pixels"]
        ground_truth_count = values["ground_truth_valid_pixels"]
        coverage_count = values["coverage_pixels"]

        if count == 0:
            raise ValueError(
                f"No common valid pixels available for {method}."
            )

        if ground_truth_count == 0:
            raise ValueError(
                f"No valid ground-truth pixels available for {method}."
            )

        mae = values["absolute_error_sum"] / count

        rmse = np.sqrt(
            values["squared_error_sum"] / count
        )

        coverage = coverage_count / ground_truth_count

        results[method] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "evaluated_pixels": count,
            "ground_truth_valid_pixels": ground_truth_count,
            "coverage_pixels": coverage_count,
            "coverage": float(coverage),
        }

    return results




def evaluate_model(model, validation_loader, device, scale_factor):
    model.eval()
    accumulators = create_metric_accumulators()

    with torch.no_grad():
        for batch in validation_loader:
            model_input = batch["input"].to(
                device,
                dtype=torch.float32,
            )

            target_batch = batch["target"].cpu().numpy()
            mask_batch = batch["target_mask"].cpu().numpy()

            model_output = model(model_input)
            model_predictions = model_output["depth"].cpu().numpy()

            if model_predictions.shape != target_batch.shape:
                raise ValueError(
                    "Model prediction and target shapes do not match: "
                    f"{model_predictions.shape} vs {target_batch.shape}"
                )

            batch_size = model_predictions.shape[0]

            for index in range(batch_size):
                target = target_batch[index, 0]
                target_mask = mask_batch[index, 0]

                predictions = {
                    "edge_depth_ai": model_predictions[index, 0]
                }

                low_resolution_depth = (
                    model_input[index, 3].cpu().numpy()
                )

                low_resolution_mask = (
                    model_input[index, 4].cpu().numpy() > 0
                )

                for method in ("nearest", "bilinear", "bicubic"):
                    baseline_prediction, _ = interpolate_depth(
                        depth=low_resolution_depth,
                        scale=scale_factor,
                        method=method,
                        valid_mask=low_resolution_mask,
                    )

                    if baseline_prediction.shape != target.shape:
                        raise ValueError(
                            f"{method} prediction shape "
                            f"{baseline_prediction.shape} does not match "
                            f"target shape {target.shape}"
                        )

                    predictions[method] = baseline_prediction

                valid_target = (
                    (target_mask > 0)
                    & np.isfinite(target_mask)
                    & np.isfinite(target)
                    & (target > 0)
                )

                common_valid_mask = valid_target.copy()

                for prediction in predictions.values():
                    common_valid_mask &= (
                        np.isfinite(prediction)
                        & (prediction > 0)
                    )

                for method, prediction in predictions.items():
                    update_metrics(
                        accumulator=accumulators[method],
                        prediction=prediction,
                        target=target,
                        target_mask=target_mask,
                        common_valid_mask=common_valid_mask,
                    )

    return finalize_metrics(accumulators)



def evaluate_checkpoint(
    checkpoint_path,
    output_path=None,
    device="cpu",
):
    """Load a trained checkpoint and evaluate its validation split."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device(device)

    # Read saved experiment configuration before constructing the model.
    saved_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    config = saved_checkpoint.get("config")

    if not isinstance(config, dict):
        raise ValueError(
            "Checkpoint does not contain a valid configuration."
        )

    required_config = (
        "dataset",
        "data_path",
        "scale_factor",
        "base_channels",
        "train_fraction",
        "seed",
        "batch_size",
        "target_type",
        "target_unit",
    )

    for key in required_config:
        if key not in config:
            raise KeyError(
                f"Checkpoint configuration is missing: {key}"
            )

    model = build_model(
        scale_factor=config["scale_factor"],
        base_channels=config["base_channels"],
    ).to(device)

    # Restore the exact trained weights.
    load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        device=device,
    )

    validation_loader = create_validation_loader(config)

    print("\nEvaluation")
    print("----------")
    print("Dataset:", config["dataset"])
    print("Target:", config["target_type"])
    print("Unit:", config["target_unit"])
    print("Scale factor:", config["scale_factor"])
    print("Validation batches:", len(validation_loader))
    print("Checkpoint:", checkpoint_path)

    results = evaluate_model(
        model=model,
        validation_loader=validation_loader,
        device=device,
        scale_factor=config["scale_factor"],
    )

    report = {
        "dataset": config["dataset"],
        "target_type": config["target_type"],
        "target_unit": config["target_unit"],
        "scale_factor": config["scale_factor"],
        "checkpoint": str(checkpoint_path),
        "metrics": results,
    }

    print("\nEvaluation results")
    print("------------------")

    for method, metrics in results.items():
        print(f"\n{method}")
        print(f"  MAE: {metrics['mae']:.6f} {config['target_unit']}")
        print(f"  RMSE: {metrics['rmse']:.6f} {config['target_unit']}")
        print(f"  Coverage: {metrics['coverage'] * 100:.2f}%")
        print(
            "  Evaluated pixels:",
            metrics["evaluated_pixels"],
        )

    if output_path is None:
        output_path = (
            Path("outputs")
            / "evaluation"
            / f"{config['dataset']}_evaluation.json"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False,
        )

    print("\nEvaluation report saved:", output_path)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an EdgeDepth AI checkpoint."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best.pth or another training checkpoint.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report output path.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Evaluation device.",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but it is not available."
        )

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
