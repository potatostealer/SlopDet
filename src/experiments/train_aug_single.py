"""Single-device training with on-the-fly augmentation: LoRA-adapted Siglip2 + QFormer + MLP.

Run from the repo root:
    python -m src.experiments.train_aug_single [--config src/configs/training.yml]

Identical to train_single.py except for the collates: the datasets in
dataset.yml are clean originals, and every loaded image is degraded in the
collate (src/dataset/online_augment.py) as configured by the online_augment
block of the training config -- per image p_none untouched / p_single one
random single augmentation / p_multi one random chain, with the parameters of
augment.yml. Training images are re-drawn every epoch from a stream private to
each DataLoader worker (and DDP rank); validation images get a fixed
augmentation keyed on their file name (blake2b(seed:stem)), so val metrics are
comparable across epochs and runs. online_augment.run_name replaces run_name so
the TensorBoard and checkpoint dirs do not collide with the clean baseline.
"""

import os

from src.dataset.collate import Siglip2AugmentCollate, Siglip2Collate
from src.dataset.dataloader import build_dataloader
from src.dataset.dataloader import load_config as load_dataset_config
from src.dataset.online_augment import build_online_augmenter
from src.experiments.train_single import load_training_config, parse_args, run


def apply_online_augment_overrides(cfg: dict) -> None:
    """online_augment.run_name (when set) replaces run_name, before the logger
    and checkpoint dirs are derived from it."""
    run_name = (cfg.get("online_augment") or {}).get("run_name")
    if run_name:
        cfg["run_name"] = run_name


def build_aug_loaders(cfg: dict):
    """train_single.build_loaders with augmenting collates: a per-process stream
    for train, a per-file deterministic seed (or none / a stream) for val."""
    data_cfg = cfg["data"]
    ds_cfg = load_dataset_config(data_cfg["dataset_config"])
    if data_cfg["batch_size_override"]:
        ds_cfg["batch_size"] = data_cfg["batch_size_override"]

    augmenter, aug_cfg = build_online_augmenter(cfg.get("online_augment"))
    checkpoint_path = cfg["model"]["checkpoint_path"]
    train_collate = Siglip2AugmentCollate(checkpoint_path, augmenter)
    if aug_cfg.val_mode == "deterministic":
        val_collate = Siglip2AugmentCollate(checkpoint_path, augmenter, deterministic_seed=aug_cfg.seed)
    elif aug_cfg.val_mode == "stream":
        val_collate = Siglip2AugmentCollate(checkpoint_path, augmenter)
    else:
        val_collate = Siglip2Collate(checkpoint_path)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(
            f"online augment: {augmenter.describe()}; val={aug_cfg.val_mode}"
            f"{f' (seed {aug_cfg.seed})' if aug_cfg.val_mode == 'deterministic' else ''}; "
            f"params from {aug_cfg.params_config}"
        )

    train_loader = build_dataloader(
        ds_cfg, "train", shuffle=True, num_workers=data_cfg["num_workers"],
        collate_fn=train_collate, pin_memory=data_cfg["pin_memory"],
    )
    val_loader = build_dataloader(
        ds_cfg, "val", shuffle=False, num_workers=data_cfg["num_workers"],
        collate_fn=val_collate, pin_memory=data_cfg["pin_memory"],
    )
    return train_loader, val_loader


def main():
    args = parse_args()
    cfg = load_training_config(args)
    apply_online_augment_overrides(cfg)
    run(cfg, args, loader_builder=build_aug_loaders)


if __name__ == "__main__":
    main()
