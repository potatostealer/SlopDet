"""Multi-GPU DDP training with on-the-fly augmentation: LoRA-adapted Siglip2 + QFormer + MLP.

Run from the repo root:
    python -m src.experiments.train_aug_ddp [--config src/configs/training.yml]

train_ddp.py with the augmenting collates of train_aug_single.py (see there and
src/dataset/online_augment.py for the mixture). GPU selection, sharding and
metric reduction are exactly train_ddp.py's: Lightning re-launches this module
(python -m src.experiments.train_aug_ddp) once per rank. Each rank's DataLoader
workers are seeded per (worker, global rank), so every worker on every rank
draws its own augmentation stream; validation augmentations are keyed on the
file name, so DistributedSampler sharding does not change what a file looks like.
"""

from src.experiments.train_aug_single import apply_online_augment_overrides, build_aug_loaders
from src.experiments.train_ddp import run
from src.experiments.train_single import load_training_config, parse_args


def main():
    args = parse_args()
    cfg = load_training_config(args)
    apply_online_augment_overrides(cfg)
    run(cfg, args, loader_builder=build_aug_loaders)


if __name__ == "__main__":
    main()
