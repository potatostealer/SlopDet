"""
fusion_tokenizer.py
===================
Turns the numpy output of `ClassicalFeatureExtractor` into tokens living in the
same space as the LoRA-adapted SigLIP 2 patch tokens, plus a reference
attention-pooling head so the interface is unambiguous.

    classical_feats (numpy, per image)
        -> collate_classical(...)               # -> batched tensors
        -> ClassicalTokenizer                   # -> (B, P*G+1, d_model)
        -> cat with siglip_tokens               # -> (B, N+P*G+1, d_model)
        -> AttentionPooler -> MLP -> logit

Design notes are inline; the two that matter most:
  * one projection MLP *per family*, shared across patches -- the families have
    completely different statistics and dimensionalities, and a single MLP over
    the 1.3k-dim concatenation both wastes parameters and lets the largest
    family dominate the input scale;
  * family dropout -- entire families are masked out at random during training,
    which is the cheapest available defence against the model latching onto one
    fragile cue (and buys graceful degradation when a cue is destroyed by
    compression at test time).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  collation
# --------------------------------------------------------------------------- #


def collate_classical(batch: Sequence[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Stack a list of extractor outputs into batched float tensors."""
    keys = batch[0].keys()
    return {k: torch.from_numpy(np.stack([b[k] for b in batch])).float() for k in keys}


# --------------------------------------------------------------------------- #
#  standardisation statistics
# --------------------------------------------------------------------------- #


class FeatureStandardizer:
    """
    Running per-dimension mean / std of every family, all patches of all images
    pooled, i.e. the `stats` ClassicalTokenizer standardises with.

    The raw families are nowhere near unit scale (DCT quantiser estimates go up
    to 16, wavelet kurtoses to ~100) while the tokenizer clamps standardised
    values to +-clamp, so identity statistics would clip real information.

        std = FeatureStandardizer(extractor.dims())
        for feats in ...: std.update(feats)      # one extractor output (numpy)
        for batch in ...: std.update(batch["classical"])   # or a collate_classical() batch
        stats = std.finalize()                   # {family: {"mean": (D,), "std": (D,)}}

    "patch_meta" is not standardised (its entries are already in [0, 1]) and is
    ignored; images flagged invalid by a batch's "valid" (B,) entry (failed
    extraction, see src/dataset/classical_collate.py) are skipped. Sums are
    accumulated in float64.
    """

    def __init__(self, dims: Dict[str, int]):
        self.dims = dict(dims)
        self.count = {k: 0 for k in self.dims}
        self.sum = {k: np.zeros(d, dtype=np.float64) for k, d in self.dims.items()}
        self.sumsq = {k: np.zeros(d, dtype=np.float64) for k, d in self.dims.items()}

    def update(self, feats: Dict[str, "np.ndarray | torch.Tensor"]) -> None:
        valid = feats.get("valid")
        if valid is not None:
            valid = np.asarray(valid.detach().cpu().numpy() if isinstance(valid, torch.Tensor) else valid) > 0.5
        for k, d in self.dims.items():
            x = feats[k]
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            x = np.asarray(x, dtype=np.float64)
            if valid is not None:
                x = x[valid]  # batched input: the first axis is the image
            x = x.reshape(-1, d)  # (P, D) / (B, P, D) / (D,) / (B, D) -> rows
            self.count[k] += x.shape[0]
            self.sum[k] += x.sum(0)
            self.sumsq[k] += np.square(x).sum(0)

    def finalize(self) -> Dict[str, Dict[str, np.ndarray]]:
        out = {}
        for k in self.dims:
            n = max(self.count[k], 1)
            mean = self.sum[k] / n
            var = np.maximum(self.sumsq[k] / n - mean * mean, 0.0)
            out[k] = {"mean": mean.astype(np.float32), "std": np.sqrt(var).astype(np.float32)}
        return out


def save_stats(path, stats: Dict[str, Dict[str, np.ndarray]], meta: Optional[dict] = None) -> None:
    """stats (+ a JSON-able meta dict) -> one .npz, written atomically (tmp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, v in stats.items():
        arrays[f"{k}.mean"] = np.asarray(v["mean"], dtype=np.float32)
        arrays[f"{k}.std"] = np.asarray(v["std"], dtype=np.float32)
    arrays["__meta__"] = np.array(json.dumps(meta or {}))
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_stats(path) -> Tuple[Dict[str, Dict[str, np.ndarray]], dict]:
    """The inverse of save_stats: (stats, meta)."""
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    with np.load(path) as z:
        meta = json.loads(str(z["__meta__"])) if "__meta__" in z.files else {}
        for name in z.files:
            if name.endswith(".mean"):
                k = name[: -len(".mean")]
                stats[k] = {"mean": z[name], "std": z[f"{k}.std"]}
    return stats, meta


# --------------------------------------------------------------------------- #
#  tokenizer
# --------------------------------------------------------------------------- #


class ClassicalTokenizer(nn.Module):
    """
    dims: {"spectral": 134, "dct": 244, ..., "global": 16} from extractor.dims()
    stats: {family: {"mean": (D,), "std": (D,)}} from FeatureStandardizer.finalize()
    """

    def __init__(
        self,
        dims: Dict[str, int],
        d_model: int = 768,
        hidden_mult: int = 2,
        meta_dim: int = 4,
        family_dropout: float = 0.15,
        clamp: float = 8.0,
        stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ):
        super().__init__()
        self.families: List[str] = [k for k in dims if k != "global"]
        self.d_model = d_model
        self.family_dropout = family_dropout
        self.clamp = clamp

        self.proj = nn.ModuleDict()
        for k, d in dims.items():
            h = hidden_mult * d_model
            self.proj[k] = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, h),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(h, d_model),
            )
            # identity until set_stats() / a checkpoint fills them in
            self.register_buffer(f"mean_{k}", torch.zeros(d))
            self.register_buffer(f"std_{k}", torch.ones(d))
        if stats is not None:
            self.set_stats(stats)

        # one learned embedding per family (+1 for the global/degradation token)
        self.fam_emb = nn.Parameter(torch.zeros(len(dims), d_model))
        self.meta_proj = nn.Linear(meta_dim, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.fam_emb, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.out_norm = nn.LayerNorm(d_model)

    def set_stats(self, stats: Dict[str, Dict[str, np.ndarray]]) -> None:
        """Load per-family standardisation statistics (FeatureStandardizer.finalize() / load_stats())
        into the mean_* / std_* buffers; std is floored at 1e-3. Every family incl. "global" is
        required, so a family silently left at identity (and clipped by `clamp`) cannot happen."""
        for k in self.proj:
            if k not in stats:
                raise KeyError(f"classical stats: no statistics for family {k!r} (have {sorted(stats)})")
            mean = torch.as_tensor(stats[k]["mean"], dtype=torch.float32).reshape(-1)
            std = torch.as_tensor(stats[k]["std"], dtype=torch.float32).reshape(-1).clamp_min(1e-3)
            buf_mean, buf_std = getattr(self, f"mean_{k}"), getattr(self, f"std_{k}")
            if mean.shape != buf_mean.shape or std.shape != buf_std.shape:
                raise ValueError(
                    f"classical stats: family {k!r} has {tuple(mean.shape)} entries, the tokenizer expects "
                    f"{tuple(buf_mean.shape)} (different extractor families / version?)"
                )
            with torch.no_grad():
                buf_mean.copy_(mean.to(buf_mean.device))
                buf_std.copy_(std.to(buf_std.device))

    def _standardize(self, x: torch.Tensor, k: str) -> torch.Tensor:
        m = getattr(self, f"mean_{k}")
        s = getattr(self, f"std_{k}")
        return ((x - m) / s).clamp(-self.clamp, self.clamp)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        feats[family] : (B, P, D)   feats["global"] : (B, Dg)
        feats["patch_meta"] : (B, P, 4)
        feats["valid"] : (B,) optional, 0 = the extractor failed on that image: every one of its
                         tokens (global included) becomes the mask token, exactly like a family
                         dropped out during training, so the model handles it as a missing modality
        returns (B, P*len(families) + 1, d_model)
        """
        meta = self.meta_proj(feats["patch_meta"])                   # (B, P, d)
        toks, B = [], meta.shape[0]
        valid = feats.get("valid")
        valid = None if valid is None else valid.to(meta.dtype).view(B, 1, 1)

        for i, k in enumerate(self.families):
            x = self._standardize(feats[k], k)
            t = self.proj[k](x) + self.fam_emb[i] + meta             # (B, P, d)
            keep = None
            if self.training and self.family_dropout > 0:
                keep = (torch.rand(B, 1, 1, device=t.device) > self.family_dropout).float()
            if valid is not None:
                keep = valid if keep is None else keep * valid
            if keep is not None:
                t = keep * t + (1 - keep) * (self.mask_token + self.fam_emb[i])
            toks.append(t)

        g = self._standardize(feats["global"], "global")
        gt = (self.proj["global"](g) + self.fam_emb[-1]).unsqueeze(1)  # (B, 1, d)
        if valid is not None:
            gt = valid * gt + (1 - valid) * (self.mask_token + self.fam_emb[-1])
        toks.append(gt)
        return self.out_norm(torch.cat(toks, dim=1))

