"""On-the-fly image degradation: one augmentation outcome per image, drawn from a fixed mixture.

The offline pipeline (src/dataset/augment.py) writes 9 variants of every image to disk. This module applies the
same augmentation functions inside the data pipeline instead, so the datasets stay clean originals and no disk is
spent on variants. Per image, independently:

    p_none    nothing
    p_single  exactly one of jpeg | blur | resize | noise | jitter | crop, chosen uniformly; its parameters are
              drawn uniformly from the option lists (jitter factors are continuous uniform in [1 - x, 1 + x])
    p_multi   one chain: a subset of size multi.min_size..multi.max_size drawn uniformly over ALL such subsets
              (57 for 2..6, so the chain-size pmf is 15/20/15/6/1 of 57 -- a 3-chain is likelier than a 6-chain),
              applied in a uniformly random order, each step's parameters drawn as above

The subset draw deliberately differs from augment.build_plans(), which picks the chain size first (uniform over
2..6) and only then a subset of that size.

Nothing here owns a random stream: every call takes a random.Random, so the collate decides the policy -- a
per-process stream for training (new draws every epoch), a per-file seed for a validation set that must look the
same in every epoch and run (see val_rng).
"""

import itertools
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image

from src.dataset.augment import (
    AUG_ORDER,
    MultiCfg,
    ParamsCfg,
    Step,
    _build,
    apply_step,
    image_seed,
    sample_step,
    validate_multi,
    validate_params,
)

VAL_MODES = ("deterministic", "none", "stream")


@dataclass
class OnlineAugmentCfg:
    """The online_augment: block of training.yml."""

    p_none: float = 0.10
    p_single: float = 0.40
    p_multi: float = 0.50
    val_mode: str = "deterministic"  # deterministic | none | stream
    seed: int = 1234  # deterministic val plans only
    run_name: Optional[str] = None  # replaces the top-level run_name when set
    params_config: str = "src/configs/augment.yml"  # its augment.params / augment.multi blocks are read


def load_online_augment_cfg(block: Optional[dict]) -> OnlineAugmentCfg:
    """online_augment: mapping -> OnlineAugmentCfg (unknown keys are an error, like augment.yml)."""
    cfg = _build(OnlineAugmentCfg, block, "online_augment")
    probs = (cfg.p_none, cfg.p_single, cfg.p_multi)
    if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError(f"online_augment: p_none + p_single + p_multi must be 1 and all >= 0, got {probs}")
    if cfg.val_mode not in VAL_MODES:
        raise ValueError(f"online_augment.val_mode must be one of {VAL_MODES}, got {cfg.val_mode!r}")
    return cfg


def load_params(path: str) -> tuple:
    """(ParamsCfg, MultiCfg) from the augment.params / augment.multi blocks of an augment.yml.

    Not augment.load_augment_config(): that validates the offline-only keys too (input_dir must exist).
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    block = raw.get("augment") if isinstance(raw, dict) else None
    if not isinstance(block, dict):
        raise ValueError(f"{path}: expected a top-level augment: mapping")
    params = _build(ParamsCfg, block.get("params"), "augment.params")
    multi = _build(MultiCfg, block.get("multi"), "augment.multi")
    validate_params(params)
    validate_multi(multi)
    return params, multi


class OnlineAugmenter:
    """Samples and applies one augmentation outcome per image.

    Plain attributes only (pickles cleanly to DataLoader workers) and no mutable state, so one instance can be
    shared by several collates.
    """

    def __init__(self, p_none: float, p_single: float, p_multi: float, params: ParamsCfg,
                 min_size: int = 2, max_size: int = len(AUG_ORDER)):
        self.p_none, self.p_single, self.p_multi = float(p_none), float(p_single), float(p_multi)
        self.params = params
        self.min_size, self.max_size = int(min_size), int(max_size)
        # every subset of AUG_ORDER with min_size <= size <= max_size: 57 for 2..6
        self.subsets = tuple(
            c for k in range(self.min_size, self.max_size + 1) for c in itertools.combinations(AUG_ORDER, k)
        )
        if self.p_multi > 0 and not self.subsets:
            raise ValueError("online_augment: p_multi > 0 but multi.min_size..max_size yields no subsets")

    def sample_plan(self, rng: random.Random) -> list:
        """The steps for one image: [] (identity), [one Step], or a chain of 2+ Steps in application order."""
        u = rng.random()
        if u < self.p_none:
            return []
        if u < self.p_none + self.p_single:
            return [sample_step(rng.choice(AUG_ORDER), rng, self.params)]
        names = list(rng.choice(self.subsets))  # uniform over all subsets ...
        rng.shuffle(names)  # ... then a uniform permutation of the chosen one
        return [sample_step(name, rng, self.params) for name in names]

    def apply(self, img: Image.Image, steps: list) -> Image.Image:
        for step in steps:
            img = apply_step(img, step, self.params)
        return img

    def __call__(self, img: Image.Image, rng: random.Random) -> Image.Image:
        return self.apply(img, self.sample_plan(rng))

    def describe(self) -> str:
        return (
            f"p_none={self.p_none} p_single={self.p_single} p_multi={self.p_multi} "
            f"({len(self.subsets)} subsets of size {self.min_size}..{self.max_size})"
        )


def plan_label(steps: list) -> str:
    """Human-readable chain, e.g. 'jpeg(quality=50) -> blur(sigma=1.0)'; 'identity' for no steps."""
    return " -> ".join(step.label() for step in steps) if steps else "identity"


def val_rng(path, seed: int) -> random.Random:
    """A stream that depends only on the file name and seed -- augment.py's keying, blake2b(seed : stem) -- so a
    validation image is augmented identically in every epoch, run, rank and worker count."""
    return random.Random(image_seed(Path(path).stem, seed))


def build_online_augmenter(block: Optional[dict]) -> tuple:
    """(OnlineAugmenter, OnlineAugmentCfg) from the online_augment: block of a training config."""
    cfg = load_online_augment_cfg(block)
    params, multi = load_params(cfg.params_config)
    augmenter = OnlineAugmenter(cfg.p_none, cfg.p_single, cfg.p_multi, params, multi.min_size, multi.max_size)
    return augmenter, cfg


__all__ = [
    "OnlineAugmentCfg",
    "OnlineAugmenter",
    "Step",
    "VAL_MODES",
    "build_online_augmenter",
    "load_online_augment_cfg",
    "load_params",
    "plan_label",
    "val_rng",
]
