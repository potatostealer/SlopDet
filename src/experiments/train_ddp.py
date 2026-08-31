"""Multi-GPU DDP training: LoRA-adapted Siglip2 vision encoder + QFormer + MLP.

Run from the repo root:
    python -m src.experiments.train_ddp [--config src/configs/training.yml]

Physical GPUs are chosen via ddp.cuda_visible_devices in the config (default
4 devices); an externally set CUDA_VISIBLE_DEVICES takes precedence. Each
visible device becomes one DDP rank: Lightning re-launches this script once
per rank and swaps the dataloader samplers for DistributedSampler, so every
rank sees a distinct shard of the train and val sets, and batch_size is per
device (global batch = batch_size * num_devices).

Metrics: step-level train loss/acc come from the rank-0 batch; the epoch-level
train accuracy and both val metrics are reduced across all ranks on epoch
completion (torchmetrics syncs at compute, val/loss logs with sync_dist=True).
"""

import math
import os

import lightning as L
import torch

from src.experiments.train_single import (
    LoraQFormerDetector,
    build_loaders,
    build_logger_and_checkpoint,
    load_training_config,
    parse_args,
)


def run(cfg: dict, args, loader_builder=build_loaders, model_builder=LoraQFormerDetector):
    """Pick the GPUs, seed, build loaders / model / logger / trainer and fit.

    Split from main() so variants (train_aug_ddp.py) can adjust cfg between
    loading it and deriving the logger / checkpoint dirs, pass a loader_builder
    whose collates augment on the fly, and a model_builder (cfg -> LightningModule)
    for another architecture (train_aug_classical_ddp.py). Everything
    in model_builder runs once per rank, before the process group exists.
    """
    trainer_cfg = cfg["trainer"]

    # Must be set before anything initializes CUDA. setdefault: an externally
    # set CUDA_VISIBLE_DEVICES wins over the config value.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg["ddp"]["cuda_visible_devices"]))
    num_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    L.seed_everything(cfg["seed"], workers=True)
    torch.set_float32_matmul_precision("high")

    train_loader, val_loader = loader_builder(cfg)
    model = model_builder(cfg)
    logger, checkpoint_cb = build_logger_and_checkpoint(cfg)

    per_device_bs = train_loader.batch_size
    # DistributedSampler pads to equal per-rank counts, hence the double ceil.
    steps_per_epoch = math.ceil(math.ceil(len(train_loader.dataset) / num_devices) / per_device_bs)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(
            f"ddp: {num_devices} devices (CUDA_VISIBLE_DEVICES="
            f"{os.environ['CUDA_VISIBLE_DEVICES']}); per-device batch size {per_device_bs}, "
            f"global batch size {per_device_bs * num_devices}"
        )
        print(
            f"steps per epoch (per device): {steps_per_epoch}; "
            f"epochs: {trainer_cfg['num_epochs']}"
        )
    logger.log_hyperparams(  # rank-zero-only internally
        {
            "steps_per_epoch": steps_per_epoch,
            "num_epochs": trainer_cfg["num_epochs"],
            "num_devices": num_devices,
            "global_batch_size": per_device_bs * num_devices,
        }
    )

    trainer = L.Trainer(
        max_epochs=trainer_cfg["num_epochs"],
        accelerator="gpu",
        devices=num_devices,
        strategy="ddp",
        precision=trainer_cfg["precision"],
        logger=logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=trainer_cfg["log_every_n_steps"],
        gradient_clip_val=trainer_cfg["gradient_clip_val"],
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches,
    )
    trainer.fit(model, train_loader, val_loader)


def main(loader_builder=build_loaders):
    args = parse_args()
    cfg = load_training_config(args)
    run(cfg, args, loader_builder=loader_builder)


if __name__ == "__main__":
    main()
