
"""
Reproducible data splits and PyTorch DataLoaders for EdgeDepth AI.

NYU:
    Random sample-level split, default 90% train / 10% validation.

Middlebury:
    Random scene-level split, default 80% train / 20% validation.

Middlebury uses batch_size=1 to support potentially different
image dimensions between scenes.
"""
import h5py

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.nyu_dataset import NYUDepthDataset
from training.middlebury_dataset import MiddleburyDataset


def deterministic_split(items, train_fraction, seed=42):
    """
    Split a sequence into reproducible training and validation lists.

    The input items are not modified.
    """
    items = list(items)

    if len(items) < 2:
        raise ValueError(
            "At least two items are required for a train/validation split."
        )

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "train_fraction must be greater than 0 and less than 1."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    shuffled_indices = torch.randperm(
        len(items),
        generator=generator
    ).tolist()

    train_count = int(len(items) * train_fraction)

    train_count = max(
        1,
        min(train_count, len(items) - 1)
    )

    train_items = [
        items[i] for i in shuffled_indices[:train_count]
    ]

    validation_items = [
        items[i] for i in shuffled_indices[train_count:]
    ]

    return train_items, validation_items


def create_nyu_dataloaders(
    mat_path,
    scale_factor=2,
    batch_size=2,
    train_fraction=0.9,
    seed=42,
    num_workers=0,
    max_samples=None,
):
    """Create NYU training and validation DataLoaders."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    with h5py.File(mat_path, "r") as file:
        total_samples = int(file["images"].shape[0])

    selected_indices = list(range(total_samples))

    if max_samples is not None:
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2.")

        selected_indices = selected_indices[:max_samples]

    train_indices, validation_indices = deterministic_split(
        selected_indices,
        train_fraction=train_fraction,
        seed=seed,
    )

    train_dataset = NYUDepthDataset(
        mat_path=mat_path,
        scale_factor=scale_factor,
        indices=train_indices,
    )

    validation_dataset = NYUDepthDataset(
        mat_path=mat_path,
        scale_factor=scale_factor,
        indices=validation_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, validation_loader


def middlebury_scene_family(scene_name):
    """Return the shared family name for related Middlebury scenes."""

    related_families = {
        "motorcycle": {"Motorcycle", "MotorcycleE"},
        "piano": {"Piano", "PianoL"},
        "playtable": {"Playtable", "PlaytableP"},
    }

    for family, variants in related_families.items():
        if scene_name in variants:
            return family

    return scene_name



def create_middlebury_dataloaders(
    root_dir,
    scale_factor=2,
    train_fraction=0.8,
    seed=42,
    num_workers=0,
    max_samples=None,
):
    """Create Middlebury DataLoaders with scene-family separation.

    Related scene variants are always assigned to the same partition.
    Only labelled scenes under trainingH are used.
    """

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    root_dir = Path(root_dir)

    if not root_dir.is_dir():
        raise FileNotFoundError(
            f"Middlebury root directory does not exist: {root_dir}"
        )

    if root_dir.name.lower() == "trainingh":
        scene_root = root_dir
    else:
        scene_root = root_dir / "trainingH"

    if not scene_root.is_dir():
        raise FileNotFoundError(
            "Middlebury training directory was not found: "
            f"{scene_root}"
        )

    scene_names = sorted(
        path.name
        for path in scene_root.iterdir()
        if path.is_dir()
    )

    if len(scene_names) < 2:
        raise ValueError(
            "At least two Middlebury training scenes are required."
        )

    if max_samples is not None:
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2.")

        scene_names = scene_names[:max_samples]

    # Group related scenes before splitting.
    families = {}

    for scene_name in scene_names:
        family = middlebury_scene_family(scene_name)
        families.setdefault(family, []).append(scene_name)

    family_names = sorted(families.keys())

    if len(family_names) < 2:
        raise ValueError(
            "At least two independent Middlebury scene families "
            "are required for training and validation."
        )

    train_families, validation_families = deterministic_split(
        family_names,
        train_fraction=train_fraction,
        seed=seed,
    )

    # Expand each family back into its scene names.
    train_scenes = sorted(
        scene
        for family in train_families
        for scene in families[family]
    )

    validation_scenes = sorted(
        scene
        for family in validation_families
        for scene in families[family]
    )

    print(f"Middlebury labelled scenes selected: {len(scene_names)}")
    print("Training families:", train_families)
    print("Validation families:", validation_families)
    print("Training scenes:", train_scenes)
    print("Validation scenes:", validation_scenes)

    # Explicit safety check: no family can appear in both partitions.
    train_family_set = {
        middlebury_scene_family(scene)
        for scene in train_scenes
    }

    validation_family_set = {
        middlebury_scene_family(scene)
        for scene in validation_scenes
    }

    if train_family_set.intersection(validation_family_set):
        raise RuntimeError(
            "Scene-family leakage detected between training "
            "and validation."
        )

    train_dataset = MiddleburyDataset(
        root_dir=scene_root,
        scale_factor=scale_factor,
        scenes=train_scenes,
    )

    validation_dataset = MiddleburyDataset(
        root_dir=scene_root,
        scale_factor=scale_factor,
        scenes=validation_scenes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, validation_loader
