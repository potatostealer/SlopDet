"""Single-device training: LoRA-adapted Siglip2 vision encoder + QFormer + MLP.

Run from the repo root:
    python -m src.experiments.train_single [--config src/configs/training.yml]

The Siglip2 vision tower is frozen except for the LoRA adapters; the QFormer
and MLP classifier are fully trainable. Features are taken from
last_hidden_state (the post_layernorm output), bypassing the pooling head.
"""

import argparse
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryAccuracy
from transformers import Siglip2VisionModel

from src.dataset.collate import Siglip2Collate
from src.dataset.dataloader import build_dataloader
from src.dataset.dataloader import load_config as load_dataset_config
from src.modules.attention import QFormer
from src.modules.classifier import MLPClassifier
from src.modules.lora_adapter import (
    apply_lora_to_siglip2_vision,
    count_lora_parameters,
    mark_only_lora_trainable,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "training.yml"


def build_lora_vision(cfg: dict) -> Siglip2VisionModel:
    """The frozen Siglip2 vision tower with trainable LoRA adapters on the attention projections."""
    vision = Siglip2VisionModel.from_pretrained(
        cfg["model"]["checkpoint_path"],
        attn_implementation=cfg["model"]["attn_implementation"],
    )
    vision.use_head = False
    del vision.head  # pooling head is bypassed entirely
    # Freeze before adapting so the freshly created LoRA params stay trainable.
    vision.requires_grad_(False)
    apply_lora_to_siglip2_vision(
        vision,
        targets=cfg["lora"]["targets"],
        r=cfg["lora"]["r"],
        alpha=cfg["lora"]["alpha"],
        dropout=cfg["lora"]["dropout"],
    )
    mark_only_lora_trainable(vision)
    # from_pretrained returns the model in eval mode and Lightning preserves
    # submodule modes at fit start; switch back so LoRA dropout is active
    # during training (Siglip2 itself has no mode-sensitive layers).
    vision.train()
    return vision


def build_qformer(cfg: dict, dim: int) -> QFormer:
    """The qformer: block over dim-wide tokens."""
    qf = cfg["qformer"]
    return QFormer(
        m=qf["m"],
        n_layers=qf["n_layers"],
        n_heads=qf["n_heads"],
        inner_dim=dim,
        k=qf["k"],
        dropout=qf["dropout"],
        mlp_ratio=qf["mlp_ratio"],
        grad_checkpointing=qf["grad_checkpointing"],
    )


def build_classifier(cfg: dict, dim: int) -> MLPClassifier:
    """The classifier: block on dim-wide pooled latents."""
    clf = cfg["classifier"]
    return MLPClassifier(in_dim=dim, hidden_dims=clf["hidden_dims"], dropout=clf["dropout"])


class LoraQFormerDetector(L.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.vision = build_lora_vision(cfg)
        self.build_head(cfg, self.vision.config.hidden_size)

        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

    def build_head(self, cfg: dict, vision_dim: int) -> None:
        """Everything after the vision tower: a QFormer + MLP reading the vision_dim-wide
        patch tokens. Variants override this to put modules in front of the QFormer
        or to run it at another width (train_aug_classical_single.py)."""
        self.qformer = build_qformer(cfg, vision_dim)
        self.classifier = build_classifier(cfg, vision_dim)

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes):
        feats = self.vision(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        ).last_hidden_state  # (B, T, D), post_layernorm output
        latents = self.qformer(feats, key_padding_mask=pixel_attention_mask.bool())
        return self.classifier(latents.mean(dim=1))  # mean over m latents == squeeze for m=1

    def forward_batch(self, batch: dict) -> torch.Tensor:
        """Logits for one collate batch (variants whose batches carry extra inputs override this)."""
        return self(batch["pixel_values"], batch["pixel_attention_mask"], batch["spatial_shapes"])

    def _step(self, batch, acc):
        logits = self.forward_batch(batch)
        loss = self.criterion(logits, batch["labels"])
        acc.update(torch.sigmoid(logits), batch["labels"].long())
        return loss

    def training_step(self, batch, batch_idx):
        # No sync_dist: under DDP the step-level values shown/logged are the
        # rank-0 batch. The epoch-level train/acc is still globally reduced
        # because torchmetrics syncs its state across ranks at compute time.
        loss = self._step(batch, self.train_acc)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Both val metrics are reduced across ranks when validation completes:
        # sync_dist averages the loss, torchmetrics syncs val_acc at compute.
        loss = self._step(batch, self.val_acc)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        opt = self.hparams["optimizer"]
        return torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=float(opt["lr"]),
            weight_decay=float(opt["weight_decay"]),
            betas=tuple(opt["betas"]),
        )

    def param_groups(self) -> dict:
        """{name: module} of the trainable modules after the vision tower, for the fit-start summary."""
        return {"qformer": self.qformer, "classifier": self.classifier}

    def on_fit_start(self):
        if self.trainer.is_global_zero:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            lora = count_lora_parameters(self.vision)
            frozen = sum(p.numel() for p in self.vision.parameters() if not p.requires_grad)
            groups = " ".join(
                f"{name}={sum(p.numel() for p in module.parameters()):,}"
                for name, module in self.param_groups().items()
            )
            print(
                f"params: total={total:,} trainable={trainable:,} "
                f"(lora={lora:,} {groups}) "
                f"frozen_vision={frozen:,}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def load_training_config(args) -> dict:
    """Load training.yml and apply the CLI overrides."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    trainer_cfg = cfg["trainer"]
    data_cfg = cfg["data"]
    if args.accelerator is not None:
        trainer_cfg["accelerator"] = args.accelerator
        if args.accelerator == "cpu":
            trainer_cfg["devices"] = 1  # a GPU index list is invalid for the CPU accelerator
    if args.precision is not None:
        trainer_cfg["precision"] = args.precision
    if args.epochs is not None:
        trainer_cfg["num_epochs"] = args.epochs
    if args.batch_size is not None:
        data_cfg["batch_size_override"] = args.batch_size
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
    return cfg


def build_loaders(cfg):
    """Plain DataLoaders (no custom sampler): under DDP Lightning replaces the
    samplers with DistributedSampler automatically, so each rank sees a
    distinct shard of both splits and batch_size is per device."""
    data_cfg = cfg["data"]
    ds_cfg = load_dataset_config(data_cfg["dataset_config"])
    if data_cfg["batch_size_override"]:
        ds_cfg["batch_size"] = data_cfg["batch_size_override"]
    collate = Siglip2Collate(cfg["model"]["checkpoint_path"])
    train_loader = build_dataloader(
        ds_cfg, "train", shuffle=True, num_workers=data_cfg["num_workers"],
        collate_fn=collate, pin_memory=data_cfg["pin_memory"],
    )
    val_loader = build_dataloader(
        ds_cfg, "val", shuffle=False, num_workers=data_cfg["num_workers"],
        collate_fn=collate, pin_memory=data_cfg["pin_memory"],
    )
    return train_loader, val_loader


def build_logger_and_checkpoint(cfg):
    trainer_cfg = cfg["trainer"]
    logger = TensorBoardLogger(save_dir=trainer_cfg["log_dir"], name=cfg["run_name"])
    checkpoint_cb = ModelCheckpoint(
        dirpath=Path(trainer_cfg["ckpt_dir"]) / cfg["run_name"],
        filename="epoch{epoch:02d}",  # keep "val/acc" out: the slash would create subdirs
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    return logger, checkpoint_cb


def run(cfg: dict, args, loader_builder=build_loaders, model_builder=LoraQFormerDetector):
    """Seed, build loaders / model / logger / trainer and fit.

    Split from main() so variants (train_aug_single.py) can adjust cfg between
    loading it and deriving the logger / checkpoint dirs, pass a loader_builder
    whose collates augment on the fly, and a model_builder (cfg -> LightningModule)
    for another architecture (train_aug_classical_single.py).
    """
    trainer_cfg = cfg["trainer"]

    L.seed_everything(cfg["seed"], workers=True)
    torch.set_float32_matmul_precision("high")

    train_loader, val_loader = loader_builder(cfg)
    model = model_builder(cfg)
    logger, checkpoint_cb = build_logger_and_checkpoint(cfg)

    steps_per_epoch = len(train_loader)
    print(
        f"steps per epoch: {steps_per_epoch} train, {len(val_loader)} val; "
        f"epochs: {trainer_cfg['num_epochs']}"
    )
    logger.log_hyperparams(
        {"steps_per_epoch": steps_per_epoch, "num_epochs": trainer_cfg["num_epochs"]}
    )

    trainer = L.Trainer(
        max_epochs=trainer_cfg["num_epochs"],
        accelerator=trainer_cfg["accelerator"],
        devices=trainer_cfg["devices"],
        strategy=trainer_cfg["strategy"],
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
