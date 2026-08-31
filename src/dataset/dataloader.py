from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from src.dataset.image_dataset import BinaryImageDataset

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "dataset.yml"


def load_config(config_path=DEFAULT_CONFIG_PATH):
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataset(config, split):
    return BinaryImageDataset(
        real_dir=config[f"real_img_{split}_ds_path"],
        aigen_dir=config[f"aigen_img_{split}_ds_path"],
    )


def build_dataloader(config, split, shuffle, num_workers=0, collate_fn=None, pin_memory=False):
    return DataLoader(
        build_dataset(config, split),
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def build_dataloaders(config_path=DEFAULT_CONFIG_PATH, num_workers=0, collate_fn=None, pin_memory=False):
    config = load_config(config_path)
    train_loader = build_dataloader(
        config, "train", shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=pin_memory,
    )
    val_loader = build_dataloader(
        config, "val", shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=pin_memory,
    )
    return train_loader, val_loader
