"""Batch collation: open images and run the Siglip2 NaFlex image processor."""

import hashlib
import os
import random

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import get_worker_info
from transformers import Siglip2ImageProcessor

from src.dataset.online_augment import OnlineAugmenter, val_rng


class Siglip2Collate:
    """collate_fn for BinaryImageDataset batches of (image_path, label).

    Plain-attribute class so it pickles cleanly to DataLoader workers; the
    processor is built once here rather than per batch.
    """

    def __init__(self, checkpoint_path: str):
        self.processor = Siglip2ImageProcessor.from_pretrained(checkpoint_path)

    def __call__(self, batch: list) -> dict:
        paths, labels = zip(*batch)
        return self.encode([Image.open(path).convert("RGB") for path in paths], labels)

    def encode(self, images: list, labels) -> dict:
        """PIL images + labels -> the model's batch dict."""
        enc = self.processor(images=images, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"],                    # (B, 256, 768)
            "pixel_attention_mask": enc["pixel_attention_mask"],    # (B, 256)
            "spatial_shapes": enc["spatial_shapes"],                # (B, 2)
            "labels": torch.tensor(labels, dtype=torch.float32),    # float for BCE
        }


def _global_rank() -> int:
    """DDP rank of this process: the process group when it is up, else the LOCAL_RANK
    Lightning's launcher exports to every rank (single node: local == global), else 0."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _stream_seed() -> int:
    """Seed for this process's training augmentation stream.

    Python's global `random` is what Lightning re-seeds per (base seed, worker id,
    global rank) in every DataLoader worker (seed_everything(workers=True) ->
    pl_worker_init_function), so one draw from it already differs per worker and
    rank. Rank and worker id are hashed in as well for the case nothing re-seeded
    the process: num_workers=0 under DDP, where every rank sits on the identical
    seed_everything state.
    """
    info = get_worker_info()
    worker_id = -1 if info is None else info.id
    key = f"{random.getrandbits(64)}:{_global_rank()}:{worker_id}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


class Siglip2AugmentCollate(Siglip2Collate):
    """Siglip2Collate that augments every image (src/dataset/online_augment.py) before the processor.

    deterministic_seed=None (training): each image draws from a stream private to
    the process running the collate. It is created lazily on the first batch and
    re-created after a fork (pid check: DataLoader workers are forked, so a stream
    made in the parent would otherwise be inherited identically by every worker),
    so workers, ranks and epochs all see different augmentations.

    deterministic_seed=int (validation): random.Random(image_seed(stem, seed)) per
    file, so a validation image looks the same in every epoch, run, rank and
    worker count and val metrics stay comparable.
    """

    def __init__(self, checkpoint_path: str, augmenter: OnlineAugmenter, deterministic_seed=None):
        super().__init__(checkpoint_path)
        self.augmenter = augmenter
        self.deterministic_seed = deterministic_seed
        self._stream = None
        self._stream_pid = None

    def _stream_rng(self) -> random.Random:
        pid = os.getpid()
        if self._stream is None or self._stream_pid != pid:
            self._stream, self._stream_pid = random.Random(_stream_seed()), pid
        return self._stream

    def _rng_for(self, path) -> random.Random:
        if self.deterministic_seed is None:
            return self._stream_rng()
        return val_rng(path, self.deterministic_seed)

    def __call__(self, batch: list) -> dict:
        paths, labels = zip(*batch)
        images = [self.augmenter(Image.open(path).convert("RGB"), self._rng_for(path)) for path in paths]
        return self.encode(images, labels)
