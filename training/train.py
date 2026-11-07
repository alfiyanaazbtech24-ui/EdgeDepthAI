
"""
Main training entry point for EdgeDepth AI.

Examples from the project root:

    python -m training.train --dataset nyu --max-samples 10 --epochs 1

    python -m training.train --dataset middlebury --max-samples 5 --epochs 1

    python -m training.train --dataset nyu --epochs 20

    python -m training.train --dataset nyu --resume outputs/training/nyu/latest.pth
"""

import argparse
import math
from pathlib import Path

import torch

from models.edgedepth_unet import build_model

from training.config import (
    create_nyu_config,
    create_middlebury_config,
)

from training.data_pipeline import (
    create_nyu_dataloaders,
    create_middlebury_dataloaders,
)

from training.engine import (
    train_one_epoch,
    validate_one_epoch,
)

from training.checkpoints import (
    save_checkpoint,
    load_checkpoint,
    save_training_history,
)


def parse_arguments():
    """Read command-line training options."""

    parser = argparse.ArgumentParser(
        description="Train the EdgeDepth AI depth refinement model."
    )

    parser.add_argument(
        "--dataset",
        choices=("nyu", "middlebury"),
        required=True,
        help="Dataset to use for this training run.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs. Defaults to the configuration value.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the dataset size for a smoke test.",
    )

    parser.add_argument(
        "--scale-factor",
        type=int,
        choices=(2, 4, 8),
        default=2,
        help="Upsampling scale factor.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training.",
    )

    return parser.parse_args()


def create_experiment_config(args):
    """Create the configuration for the selected dataset."""

    if args.dataset == "nyu":
        config = create_nyu_config(
            max_samples=args.max_samples,
            scale_factor=args.scale_factor,
        )
    else:
        config = create_middlebury_config(
            max_samples=args.max_samples,
            scale_factor=args.scale_factor,
        )

    if args.epochs is not None:
        config.epochs = args.epochs

    config.validate()

    return config


def create_experiment_loaders(config):
    """Create dataset-specific training and validation loaders."""

    if config.dataset == "nyu":
        return create_nyu_dataloaders(
            mat_path=config.data_path,
            scale_factor=config.scale_factor,
            batch_size=config.batch_size,
            train_fraction=config.train_fraction,
            seed=config.seed,
            num_workers=config.num_workers,
            max_samples=config.max_samples,
        )

    return create_middlebury_dataloaders(
        root_dir=config.data_path,
        scale_factor=config.scale_factor,
        train_fraction=config.train_fraction,
        seed=config.seed,
        num_workers=config.num_workers,
        max_samples=config.max_samples,
    )


def validate_resume_checkpoint(checkpoint, config):
    """Prevent resuming with an incompatible experiment setup."""

    saved_config = checkpoint.get("config", {})

    compatibility_fields = (
        "dataset",
        "target_type",
        "target_unit",
        "scale_factor",
        "base_channels",
    )

    for field in compatibility_fields:
        saved_value = saved_config.get(field)
        current_value = config.to_dict().get(field)

        if saved_value != current_value:
            raise ValueError(
                f"Cannot resume: configuration mismatch for "
                f"'{field}'. Checkpoint has {saved_value!r}; "
                f"current run has {current_value!r}."
            )


def run_training(config, resume_path=None):
    """Run the complete training and validation process."""

    torch.manual_seed(config.seed)

    device = torch.device(config.device)

    print("\nEdgeDepth AI training")
    print("---------------------")
    print("Dataset:", config.dataset)
    print("Target:", config.target_type, "in", config.target_unit)
    print("Scale factor:", config.scale_factor)
    print("Device:", device)
    print("Epochs:", config.epochs)
    print("Maximum samples:", config.max_samples)

    train_loader, validation_loader = (
        create_experiment_loaders(config)
    )

    print("\nData loaders created successfully.")
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))

    model = build_model(
        scale_factor=config.scale_factor,
        base_channels=config.base_channels,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_checkpoint_path = output_dir / "latest.pth"
    best_checkpoint_path = output_dir / "best.pth"
    history_path = output_dir / "history.json"

    start_epoch = 1
    best_validation_l1 = math.inf
    history = []

    if resume_path is not None:
        print("\nLoading checkpoint:", resume_path)

        checkpoint = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )

        validate_resume_checkpoint(
            checkpoint,
            config,
        )

        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_l1 = float(
            checkpoint["best_validation_l1"]
        )
        history = checkpoint["history"]

        print("Resuming from epoch:", start_epoch)
        print("Previous best validation L1:", best_validation_l1)

    if start_epoch > config.epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {start_epoch - 1}, "
            f"which is not below the requested total of "
            f"{config.epochs} epochs. Increase --epochs to resume."
        )

    for epoch in range(start_epoch, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        print("---------------------")

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        validation_metrics = validate_one_epoch(
            model=model,
            data_loader=validation_loader,
            device=device,
        )

        epoch_record = {
            "epoch": epoch,
            "train_l1": float(train_metrics["l1"]),
            "train_rmse": float(train_metrics["rmse"]),
            "validation_l1": float(validation_metrics["l1"]),
            "validation_rmse": float(validation_metrics["rmse"]),
            "train_valid_pixels": int(
                train_metrics["valid_pixels"]
            ),
            "validation_valid_pixels": int(
                validation_metrics["valid_pixels"]
            ),
        }

        history.append(epoch_record)

        print(
            f"Training L1:       {epoch_record['train_l1']:.6f} "
            f"{config.target_unit}"
        )
        print(
            f"Training RMSE:     {epoch_record['train_rmse']:.6f} "
            f"{config.target_unit}"
        )
        print(
            f"Validation L1:     {epoch_record['validation_l1']:.6f} "
            f"{config.target_unit}"
        )
        print(
            f"Validation RMSE:   {epoch_record['validation_rmse']:.6f} "
            f"{config.target_unit}"
        )

        is_best = (
            epoch_record["validation_l1"] < best_validation_l1
        )

        if is_best:
            best_validation_l1 = epoch_record["validation_l1"]

        checkpoint_arguments = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "best_validation_l1": best_validation_l1,
            "config": config.to_dict(),
            "history": history,
        }

        save_checkpoint(
            path=latest_checkpoint_path,
            **checkpoint_arguments,
        )

        if is_best:
            save_checkpoint(
                path=best_checkpoint_path,
                **checkpoint_arguments,
            )
            print("New best checkpoint saved.")

        save_training_history(
            path=history_path,
            history=history,
        )

        print("Latest checkpoint saved.")
        print("Training history saved.")

    print("\nTraining finished.")
    print("Best validation L1:", best_validation_l1)
    print("Output directory:", output_dir)

    return history


def main():
    args = parse_arguments()

    config = create_experiment_config(args)

    run_training(
        config=config,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
