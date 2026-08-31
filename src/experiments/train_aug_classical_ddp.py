"""Multi-GPU DDP training with on-the-fly augmentation and classical forensic features fused into the QFormer input.

Run from the repo root:
    python -m src.experiments.train_aug_classical_single_ddp [--config src/configs/training.yml]

train_ddp.py's GPU selection, sharding and metric reduction with the loaders and model of
train_aug_classical_single.py (see there for the architecture and the standardisation statistics). Lightning
re-launches this module once per rank; the classical features are extracted in every rank's DataLoader workers, so
data.num_workers is per rank. The standardisation statistics are computed by rank 0 before it enters trainer.fit --
i.e. before the other ranks are launched -- and loaded from classical.stats.dir/<run_name>.npz by every rank.
"""

from src.experiments.train_aug_classical_single import (
    apply_classical_overrides,
    build_classical_loaders,
    build_classical_model,
)
from src.experiments.train_ddp import run
from src.experiments.train_single import load_training_config, parse_args


def main():
    args = parse_args()
    cfg = load_training_config(args)
    apply_classical_overrides(cfg)
    run(cfg, args, loader_builder=build_classical_loaders, model_builder=build_classical_model)


if __name__ == "__main__":
    main()
