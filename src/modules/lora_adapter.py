"""Minimal in-house LoRA (no PEFT) for the Siglip2 vision encoder.

Wraps a chosen subset of the attention projections (q_proj / k_proj / v_proj /
out_proj) of every Siglip2EncoderLayer under a Siglip2VisionModel.
"""

import math
from typing import Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

LORA_TARGET_CHOICES = ("q_proj", "k_proj", "v_proj", "out_proj")


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank residual.

    y = base(x) + (lora_B @ lora_A @ x) * alpha / r

    lora_B starts at zero, so the wrapped layer is exactly the base layer at
    initialization. Dropout applies only to the LoRA branch input.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}")
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        factory = {"dtype": base.weight.dtype, "device": base.weight.device}
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, **factory))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + lora * self.scaling

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}"


def apply_lora_to_siglip2_vision(
    vision_model: nn.Module,
    targets: Sequence[str],
    r: int,
    alpha: float,
    dropout: float = 0.0,
) -> int:
    """Wrap the chosen projections of every vision encoder layer in-place.

    Walks vision_model.encoder.layers[*].self_attn by attribute path rather
    than matching parameter names: the Siglip2 text tower has byte-identical
    layer names, so name matching on a full Siglip2Model would silently adapt
    both towers. Pass a Siglip2VisionModel (or model.vision_model of a full
    Siglip2Model). Returns the number of projections replaced.
    """
    targets = tuple(targets)
    if not targets:
        raise ValueError("LoRA targets must be a non-empty subset of "
                         f"{LORA_TARGET_CHOICES}")
    invalid = [t for t in targets if t not in LORA_TARGET_CHOICES]
    if invalid:
        raise ValueError(f"Invalid LoRA targets {invalid}; "
                         f"choose from {LORA_TARGET_CHOICES}")
    replaced = 0
    for layer in vision_model.encoder.layers:
        attn = layer.self_attn
        for name in targets:
            proj = getattr(attn, name)
            if isinstance(proj, LoRALinear):
                raise ValueError(f"{name} already wrapped by LoRALinear; "
                                 "apply_lora_to_siglip2_vision was called twice")
            if not isinstance(proj, nn.Linear):
                raise TypeError(f"{name} is {type(proj).__name__}, expected nn.Linear")
            setattr(attn, name, LoRALinear(proj, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
    return replaced


def lora_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    for name, param in module.named_parameters():
        if "lora_" in name:
            yield param


def mark_only_lora_trainable(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        param.requires_grad_("lora_" in name)


def count_lora_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(module))
