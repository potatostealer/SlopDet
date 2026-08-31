"""Collates that add ClassicalFeatureExtractor features (classical_forensics.py) to the Siglip2 batch.

The SigLIP2 side is untouched: Siglip2Collate / Siglip2AugmentCollate open (and augment) the images and run the NaFlex
processor exactly as before. In addition, every image -- the same PIL image that goes into the processor, i.e. AFTER the
augmentation and at its native resolution (the processor's resize would destroy the traces the extractor looks for) --
is converted to an HxWx3 uint8 array and run through the extractor, and the per-image outputs are stacked with
fusion_tokenizer.collate_classical into

    batch["classical"] = {family: (B, P, D_family) ..., "global": (B, 16), "patch_meta": (B, P, 4)}   float32

plus "valid" (B,) float32: 1 where the extractor succeeded, 0 where it raised -- next to the usual pixel_values /
pixel_attention_mask / spatial_shapes / labels. An image whose extraction raises (a LinAlgError on a degenerate patch
took a full run down once) gets zero features and valid = 0: ClassicalTokenizer then replaces its classical tokens by
its mask token, the "missing modality" it is trained on through family dropout, and FeatureStandardizer skips it; every
such failure is reported with warnings.warn and the file path. The extraction is numpy / scipy on the CPU (~0.3 s per
1024 px image, ~0.07 s per 200 px thumbnail) and runs inside the DataLoader workers, so the classical training scripts
want many of them (data.num_workers / --num-workers).
"""

import warnings

import numpy as np
import torch

from src.modules.classical_forensics import ClassicalFeatureExtractor
from src.modules.fusion_tokenizer import collate_classical
from src.dataset.collate import Siglip2AugmentCollate, Siglip2Collate


def empty_features(extractor: ClassicalFeatureExtractor) -> dict:
    """All-zero extractor output of the right shapes (the placeholder for a failed extraction)."""
    dims = extractor.dims()
    out = {k: np.zeros((extractor.n_patches, d), np.float32) for k, d in dims.items() if k != "global"}
    out["global"] = np.zeros(dims["global"], np.float32)
    out["patch_meta"] = np.zeros((extractor.n_patches, 4), np.float32)
    return out


def extract_classical(extractor: ClassicalFeatureExtractor, images: list, paths=None) -> dict:
    """PIL RGB images -> the batched {family: tensor} dict of their classical features + "valid" (B,).

    One image raising inside the extractor must not kill a multi-hour run: it gets empty_features()
    and valid = 0 (masked downstream, see the module docstring), and the failure is warned with the
    file path (`paths`, when known) and the exception.
    """
    feats, valid = [], []
    for i, img in enumerate(images):
        try:
            feats.append(extractor(np.asarray(img)))
            valid.append(1.0)
        except Exception as e:  # noqa: BLE001 -- any numerical failure (LinAlgError, ValueError, ...)
            where = paths[i] if paths is not None else f"image {i} of size {img.size}"
            warnings.warn(f"classical features failed for {where}: {e!r}; its classical tokens are masked")
            feats.append(empty_features(extractor))
            valid.append(0.0)
    batch = collate_classical(feats)
    batch["valid"] = torch.tensor(valid, dtype=torch.float32)
    return batch


class _ClassicalEncode:
    """Mixin: extend Siglip2Collate.encode (which both __call__ paths end in) with batch["classical"].
    Must precede the Siglip2 collate class in the MRO; the subclass sets self.extractor."""

    extractor: ClassicalFeatureExtractor
    _paths = None  # the batch's file names while __call__ runs, for the failure warning

    def __call__(self, batch: list) -> dict:
        self._paths = [path for path, _ in batch]
        try:
            return super().__call__(batch)
        finally:
            self._paths = None

    def encode(self, images: list, labels) -> dict:
        batch = super().encode(images, labels)
        batch["classical"] = extract_classical(self.extractor, images, self._paths)
        return batch


class Siglip2ClassicalCollate(_ClassicalEncode, Siglip2Collate):
    """Siglip2Collate (clean images) + classical features. Plain attributes only, so it pickles to workers."""

    def __init__(self, checkpoint_path: str, extractor: ClassicalFeatureExtractor):
        super().__init__(checkpoint_path)
        self.extractor = extractor


class Siglip2ClassicalAugmentCollate(_ClassicalEncode, Siglip2AugmentCollate):
    """Siglip2AugmentCollate (per-process stream or per-file deterministic augmentation, see there) + classical
    features of the augmented image."""

    def __init__(self, checkpoint_path: str, augmenter, extractor: ClassicalFeatureExtractor, deterministic_seed=None):
        super().__init__(checkpoint_path, augmenter, deterministic_seed=deterministic_seed)
        self.extractor = extractor


__all__ = [
    "Siglip2ClassicalAugmentCollate",
    "Siglip2ClassicalCollate",
    "empty_features",
    "extract_classical",
]
