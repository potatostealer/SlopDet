"""MLP head mapping pooled QFormer features to a single binary logit."""

from typing import Sequence

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """in_dim -> *hidden_dims -> 1 MLP.

    Returns raw logits: pair with BCEWithLogitsLoss for training and apply
    sigmoid outside for probabilities (BCELoss on post-sigmoid outputs is
    disallowed under autocast).
    """

    def __init__(self, in_dim: int = 1152, hidden_dims: Sequence[int] = (512, 64),
                 dropout: float = 0.0):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_dim) -> (B,)
        return self.net(x).squeeze(-1)
