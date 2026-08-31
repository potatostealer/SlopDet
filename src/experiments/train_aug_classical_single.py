"""Single-device training with on-the-fly augmentation and classical forensic features fused into the QFormer input.

Run from the repo root:
    python -m src.experiments.train_aug_classical_single [--config src/configs/training.yml]

train_aug_single.py (same streaming augmentation, same LoRA-adapted Siglip2 tower, same QFormer + MLP head) with one
addition per image: the augmented image, at its native resolution, also goes through ClassicalFeatureExtractor
(classical_forensics.py) inside the DataLoader workers (src/dataset/classical_collate.py), so every batch carries
batch["classical"] = {family: (B, P, D), "global": (B, 16), "patch_meta": (B, P, 4)} next to the Siglip2 inputs.

In the model (LoraQFormerClassicalDetector) the Siglip2 tower runs untouched up to last_hidden_state (B, 256, 1152);
those tokens are projected by a shallow MLP (1152 -> h -> d_model, LayerNorm) and the classical features are turned into
P * len(families) + 1 tokens of the same width by ClassicalTokenizer (fusion_tokenizer.py: per-family standardisation and
projection MLPs, family embeddings, patch-position embedding, one global degradation token, family dropout while
training). The two are concatenated along the token axis -- the classical tokens are never padding, the Siglip2 padding
mask is extended accordingly -- and the QFormer + MLP head read the concatenation exactly as before. d_model defaults
to the Siglip2 width, so the qformer: / classifier: blocks of training.yml carry over unchanged.

Standardisation: the raw families are nowhere near unit scale while the tokenizer clips standardised values to
+-classical.clamp, so before fitting the per-dimension mean / std of every family are estimated on
classical.stats.num_images augmented training images and cached in classical.stats.dir/<run_name>.npz (computed on
rank 0 when missing, loaded otherwise; validated against the extractor configuration). They end up in the checkpoint as
buffers of the tokenizer, so eval.py only needs the checkpoint.

The extraction is CPU-bound (~0.3 s per 1024 px image): give the loaders many workers (data.num_workers / --num-workers,
32 on the 256-core box). classical.run_name replaces run_name (after online_augment.run_name), so TensorBoard, checkpoint
and stats paths do not collide with the plain augmented run.
"""

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.modules.classical_forensics import FEATURE_VERSION, ClassicalFeatureExtractor
from src.modules.fusion_tokenizer import ClassicalTokenizer, FeatureStandardizer, load_stats, save_stats
from src.dataset.augment import _build
from src.dataset.classical_collate import Siglip2ClassicalAugmentCollate, Siglip2ClassicalCollate
from src.dataset.dataloader import build_dataloader, build_dataset
from src.dataset.dataloader import load_config as load_dataset_config
from src.dataset.online_augment import build_online_augmenter
from src.experiments.train_aug_single import apply_online_augment_overrides
from src.experiments.train_single import (
    LoraQFormerDetector,
    build_classifier,
    build_qformer,
    load_training_config,
    parse_args,
    run,
)

FAMILIES = ("spectral", "dct", "residual", "wavelet", "color")  # ClassicalFeatureExtractor's per-patch families


# --------------------------------------------------------------------------- #
#  config
# --------------------------------------------------------------------------- #


@dataclass
class SiglipProjCfg:
    """classical.siglip_proj: the shallow MLP on the Siglip2 last_hidden_state tokens."""

    hidden_mult: float = 1.0  # hidden width = hidden_mult * d_model
    dropout: float = 0.0


@dataclass
class StatsCfg:
    """classical.stats: where the standardisation statistics live and how they are estimated."""

    dir: str = "logs/classical_stats"  # <dir>/<run_name>.npz
    num_images: int = 2048  # augmented train images the mean / std are estimated on
    recompute: bool = False  # true = ignore an existing file (rank 0 rewrites it)


@dataclass
class ClassicalCfg:
    """The classical: block of training.yml."""

    patch: int = 256  # extractor crop size (snapped down to a multiple of 16, floored at 64)
    n_rich: int = 4  # rich-texture crops per image
    n_poor: int = 4  # poor-texture crops per image; n_rich + n_poor patch tokens per family
    families: List[str] = field(default_factory=lambda: list(FAMILIES))
    d_model: Optional[int] = None  # fusion width the QFormer runs at; null = Siglip2 hidden size
    hidden_mult: int = 2  # ClassicalTokenizer per-family projection hidden width = hidden_mult * d_model
    family_dropout: float = 0.15  # P(whole family -> mask token) per image while training
    clamp: float = 8.0  # standardised features are clipped to [-clamp, clamp]
    siglip_proj: SiglipProjCfg = field(default_factory=SiglipProjCfg)
    stats: StatsCfg = field(default_factory=StatsCfg)
    run_name: Optional[str] = None  # replaces run_name / online_augment.run_name when set


def load_classical_cfg(cfg: dict) -> ClassicalCfg:
    """cfg["classical"] -> ClassicalCfg (unknown keys are an error, like the other config blocks)."""
    c = _build(ClassicalCfg, cfg.get("classical"), "classical")
    fams = list(c.families or [])
    unknown = [f for f in fams if f not in FAMILIES]
    if not fams or unknown or len(set(fams)) != len(fams):
        raise ValueError(f"classical.families must be a non-empty subset of {list(FAMILIES)} without repeats, got {fams}")
    c.families = fams
    if c.n_rich < 0 or c.n_poor < 0 or c.n_rich + c.n_poor < 1:
        raise ValueError(f"classical.n_rich / n_poor must be >= 0 with at least one patch, got {c.n_rich} / {c.n_poor}")
    if c.patch < 64:
        raise ValueError(f"classical.patch must be >= 64, got {c.patch}")
    if c.d_model is not None and int(c.d_model) < 1:
        raise ValueError(f"classical.d_model must be null or a positive int, got {c.d_model}")
    if not 0.0 <= float(c.family_dropout) < 1.0:
        raise ValueError(f"classical.family_dropout must be in [0, 1), got {c.family_dropout}")
    if int(c.stats.num_images) < 1:
        raise ValueError(f"classical.stats.num_images must be >= 1, got {c.stats.num_images}")
    return c


def is_rank_zero() -> bool:
    """Before the process group exists only Lightning's LOCAL_RANK tells the ranks apart (single node)."""
    return os.environ.get("LOCAL_RANK", "0") == "0"


def apply_classical_overrides(cfg: dict) -> None:
    """online_augment.run_name, then classical.run_name (when set), replace run_name before the logger,
    checkpoint and stats paths are derived from it."""
    apply_online_augment_overrides(cfg)
    run_name = (cfg.get("classical") or {}).get("run_name")
    if run_name:
        cfg["run_name"] = run_name


def build_extractor(ccfg: ClassicalCfg) -> ClassicalFeatureExtractor:
    return ClassicalFeatureExtractor(
        patch=int(ccfg.patch), n_rich=int(ccfg.n_rich), n_poor=int(ccfg.n_poor), families=tuple(ccfg.families)
    )


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #


def build_classical_loaders(cfg: dict):
    """train_aug_single.build_aug_loaders with the classical collates: the same augmentation policy (per-process
    stream for train; deterministic per file / none / stream for val) and every batch also carries batch["classical"]
    computed on the augmented image."""
    data_cfg = cfg["data"]
    ds_cfg = load_dataset_config(data_cfg["dataset_config"])
    if data_cfg["batch_size_override"]:
        ds_cfg["batch_size"] = data_cfg["batch_size_override"]

    ccfg = load_classical_cfg(cfg)
    extractor = build_extractor(ccfg)
    augmenter, aug_cfg = build_online_augmenter(cfg.get("online_augment"))
    checkpoint_path = cfg["model"]["checkpoint_path"]
    train_collate = Siglip2ClassicalAugmentCollate(checkpoint_path, augmenter, extractor)
    if aug_cfg.val_mode == "deterministic":
        val_collate = Siglip2ClassicalAugmentCollate(
            checkpoint_path, augmenter, extractor, deterministic_seed=aug_cfg.seed
        )
    elif aug_cfg.val_mode == "stream":
        val_collate = Siglip2ClassicalAugmentCollate(checkpoint_path, augmenter, extractor)
    else:
        val_collate = Siglip2ClassicalCollate(checkpoint_path, extractor)
    if is_rank_zero():
        print(
            f"online augment: {augmenter.describe()}; val={aug_cfg.val_mode}"
            f"{f' (seed {aug_cfg.seed})' if aug_cfg.val_mode == 'deterministic' else ''}; "
            f"params from {aug_cfg.params_config}"
        )
        print(
            f"classical features: families={ccfg.families} patch={ccfg.patch} "
            f"n_rich={ccfg.n_rich} n_poor={ccfg.n_poor} -> dims {extractor.dims()}; "
            f"extracted in {data_cfg['num_workers']} loader worker(s)"
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


# --------------------------------------------------------------------------- #
#  standardisation statistics
# --------------------------------------------------------------------------- #


def stats_path(cfg: dict) -> Path:
    return Path(load_classical_cfg(cfg).stats.dir) / f"{cfg['run_name']}.npz"


def stats_meta(ccfg: ClassicalCfg, extractor: ClassicalFeatureExtractor) -> dict:
    """What a stats file must agree on to be reused: the extractor configuration and its dims (a
    mismatch is an error: another run's file) and the feature version (a mismatch means the feature
    definitions changed since the file was written: it is recomputed)."""
    return {
        "families": list(ccfg.families),
        "patch": int(ccfg.patch),
        "n_rich": int(ccfg.n_rich),
        "n_poor": int(ccfg.n_poor),
        "dims": {k: int(v) for k, v in extractor.dims().items()},
        "version": int(FEATURE_VERSION),
    }


def compute_stats(cfg: dict, ccfg: ClassicalCfg, extractor: ClassicalFeatureExtractor) -> tuple:
    """Per-dimension mean / std of every family over classical.stats.num_images augmented train images, drawn
    with the training collate (per-process augmentation stream) from a loader of its own, seeded from cfg["seed"]
    so the run's training shuffle is not consumed. Returns (stats, images seen)."""
    data_cfg = cfg["data"]
    ds_cfg = load_dataset_config(data_cfg["dataset_config"])
    if data_cfg["batch_size_override"]:
        ds_cfg["batch_size"] = data_cfg["batch_size_override"]
    augmenter, _ = build_online_augmenter(cfg.get("online_augment"))
    collate = Siglip2ClassicalAugmentCollate(cfg["model"]["checkpoint_path"], augmenter, extractor)
    dataset = build_dataset(ds_cfg, "train")
    num_images = min(int(ccfg.stats.num_images), len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=ds_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        collate_fn=collate,
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
    )
    num_batches = math.ceil(num_images / ds_cfg["batch_size"])
    standardizer = FeatureStandardizer(extractor.dims())
    seen = 0
    bar = tqdm(total=num_batches, desc="classical stats", unit="batch", dynamic_ncols=True)
    for i, batch in enumerate(loader):
        standardizer.update(batch["classical"])
        seen += int(batch["labels"].shape[0])
        bar.update(1)
        if i + 1 >= num_batches:
            break
    bar.close()
    return standardizer.finalize(), seen


def load_or_compute_stats(cfg: dict, ccfg: ClassicalCfg, extractor: ClassicalFeatureExtractor,
                          wait_timeout: float = 3600.0) -> dict:
    """The stats for this run: classical.stats.dir/<run_name>.npz when present (and neither recompute nor
    written by an older FEATURE_VERSION), else computed by rank 0 and written atomically. Other ranks never
    compute; Lightning's launcher starts them only after rank 0 enters trainer.fit (i.e. after this
    returned), and should they start earlier anyway they wait for the file (DDP broadcasts rank 0's buffers
    at init regardless). A file whose extractor configuration differs from the config is an error."""
    path = stats_path(cfg)
    meta = stats_meta(ccfg, extractor)
    rank_zero = is_rank_zero()

    if not rank_zero:
        deadline = time.monotonic() + wait_timeout
        while not path.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError(f"rank {os.environ.get('LOCAL_RANK')}: classical stats file never appeared: {path}")
            time.sleep(2.0)

    if path.is_file():
        stats, saved = load_stats(path)
        config_keys = [k for k in meta if k != "version"]
        saved_meta, expected = {k: saved.get(k) for k in config_keys}, {k: meta[k] for k in config_keys}
        if saved_meta != expected:
            raise ValueError(
                f"classical stats {path} were computed for {saved_meta}, the config asks for {expected}: "
                "set classical.stats.recompute: true, delete the file, or change classical.run_name"
            )
        stale = saved.get("version") != meta["version"]
        if not rank_zero or not (ccfg.stats.recompute or stale):
            if rank_zero:
                print(f"classical stats: loaded {path} ({saved.get('num_images', '?')} images)")
            return stats
        if stale:
            print(
                f"classical stats: {path} was written by feature version {saved.get('version')}, "
                f"the extractor is at {meta['version']}: recomputing"
            )

    t0 = time.perf_counter()
    stats, seen = compute_stats(cfg, ccfg, extractor)
    save_stats(path, stats, {**meta, "num_images": seen, "seed": int(cfg["seed"])})
    print(f"classical stats: {seen} augmented train images in {time.perf_counter() - t0:.0f}s -> {path}")
    return stats


# --------------------------------------------------------------------------- #
#  model
# --------------------------------------------------------------------------- #


class LoraQFormerClassicalDetector(LoraQFormerDetector):
    """LoraQFormerDetector whose QFormer reads [siglip_proj(last_hidden_state) ; ClassicalTokenizer(classical)].

    The vision tower, LoRA, loss, metrics, logging and optimiser are inherited; only what sits between
    last_hidden_state and the QFormer changes, plus the width the QFormer + MLP run at (classical.d_model).
    """

    def build_head(self, cfg: dict, vision_dim: int) -> None:
        ccfg = load_classical_cfg(cfg)
        extractor = build_extractor(ccfg)  # only for dims() / n_patches: the features arrive with the batch
        dims = extractor.dims()
        d_model = int(ccfg.d_model or vision_dim)
        n_heads = int(cfg["qformer"]["n_heads"])
        if d_model % n_heads:
            raise ValueError(f"classical.d_model={d_model} must be divisible by qformer.n_heads={n_heads}")
        hidden = max(1, int(round(float(ccfg.siglip_proj.hidden_mult) * d_model)))

        # shallow MLP on the Siglip2 patch tokens; the LayerNorm matches the tokenizer's out_norm so
        # both token streams enter the QFormer at the same scale
        self.siglip_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(ccfg.siglip_proj.dropout)),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )
        self.classical_tokenizer = ClassicalTokenizer(
            dims,
            d_model=d_model,
            hidden_mult=int(ccfg.hidden_mult),
            family_dropout=float(ccfg.family_dropout),
            clamp=float(ccfg.clamp),
        )
        self.n_classical_tokens = extractor.n_patches * len(self.classical_tokenizer.families) + 1
        self.d_model = d_model
        self.qformer = build_qformer(cfg, d_model)
        self.classifier = build_classifier(cfg, d_model)

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes, classical=None):
        if classical is None:
            raise ValueError(
                "LoraQFormerClassicalDetector needs the classical features of the batch (batch['classical'] from a "
                "Siglip2Classical*Collate, src/dataset/classical_collate.py); this caller passed only the Siglip2 inputs"
            )
        feats = self.vision(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        ).last_hidden_state  # (B, T, D), post_layernorm output -- Siglip2 untouched up to here
        siglip_tokens = self.siglip_proj(feats)  # (B, T, d_model)
        classical_tokens = self.classical_tokenizer(classical)  # (B, P * families + 1, d_model)
        if classical_tokens.shape[1] != self.n_classical_tokens:
            raise ValueError(
                f"got {classical_tokens.shape[1]} classical tokens, the model was built for {self.n_classical_tokens} "
                "(collate extractor and checkpoint disagree on patches / families)"
            )
        tokens = torch.cat([siglip_tokens, classical_tokens], dim=1)
        keep = pixel_attention_mask.bool()
        mask = torch.cat([keep, keep.new_ones((keep.shape[0], classical_tokens.shape[1]))], dim=1)
        latents = self.qformer(tokens, key_padding_mask=mask)
        return self.classifier(latents.mean(dim=1))  # mean over m latents == squeeze for m=1

    def forward_batch(self, batch: dict) -> torch.Tensor:
        return self(batch["pixel_values"], batch["pixel_attention_mask"], batch["spatial_shapes"], batch["classical"])

    def param_groups(self) -> dict:
        return {
            "siglip_proj": self.siglip_proj,
            "classical_tokenizer": self.classical_tokenizer,
            "qformer": self.qformer,
            "classifier": self.classifier,
        }


def build_classical_model(cfg: dict) -> LoraQFormerClassicalDetector:
    """The model_builder for run(): standardisation statistics first (rank 0 may spend a while extracting
    features here), then the model with those statistics loaded into the tokenizer's buffers."""
    ccfg = load_classical_cfg(cfg)
    extractor = build_extractor(ccfg)
    stats = load_or_compute_stats(cfg, ccfg, extractor)
    model = LoraQFormerClassicalDetector(cfg)
    model.classical_tokenizer.set_stats(stats)
    if is_rank_zero():
        vision_dim = model.vision.config.hidden_size
        print(
            f"classical fusion: siglip {vision_dim} -> {model.siglip_proj[0].out_features} -> {model.d_model} "
            f"(256 tokens) + classical {model.n_classical_tokens} tokens "
            f"({extractor.n_patches} patches x {len(model.classical_tokenizer.families)} families + global) "
            f"-> QFormer at d={model.d_model}"
        )
    return model


def main():
    args = parse_args()
    cfg = load_training_config(args)
    apply_classical_overrides(cfg)
    run(cfg, args, loader_builder=build_classical_loaders, model_builder=build_classical_model)


if __name__ == "__main__":
    main()
