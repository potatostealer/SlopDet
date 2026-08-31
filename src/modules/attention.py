import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SelfAttentionBlock(nn.Module):
    """Pre-LN multi-head self-attention with flash attention and additive residual.

    The attention output feeds an expanding MLP (d_model -> d_model * mlp_ratio
    -> d_model) in place of the usual square output projection, so attention and
    feed-forward share a single residual and no norm sits between them.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.dropout = dropout

    def forward(self, x, key_padding_mask=None):
        # x: (B, seq_len, D)
        # key_padding_mask: (B, seq_len) bool, True = real token, False = padding
        B, T, D = x.shape
        h = self.norm(x)  # pre layer norm
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        # (B, T, D) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]  # broadcast over heads and queries
        # dispatches to the FlashAttention kernel when available (and no mask is given)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return x + self.mlp(out)  # additive residual

class CrossAttentionBlock(nn.Module):
    """Pre-LN multi-head cross-attention with flash attention and additive residual.

    q attends to x; the residual is on the query stream. As in
    SelfAttentionBlock, an expanding MLP (d_model -> d_model * mlp_ratio ->
    d_model) replaces the square output projection, fusing attention and
    feed-forward into one residual branch.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_x = nn.LayerNorm(d_model)
        self.to_q = nn.Linear(d_model, d_model)
        self.to_kv = nn.Linear(d_model, 2 * d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.dropout = dropout

    def forward(self, q, x, key_padding_mask=None):
        # q: (B, Tq, D)  queries
        # x: (B, Tk, D)  context being attended to
        # key_padding_mask: (B, Tk) bool, True = real token, False = padding
        B, Tq, D = q.shape
        Tk = x.shape[1]
        hq = self.norm_q(q)              # pre layer norm (query stream)
        hx = self.norm_x(x)              # pre layer norm (context)
        query = self.to_q(hq)
        key, value = self.to_kv(hx).chunk(2, dim=-1)
        # -> (B, n_heads, T, head_dim)
        query = query.view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]  # broadcast over heads and queries
        # dispatches to the FlashAttention kernel when available (and no mask is given)
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, Tq, D)
        return q + self.mlp(out)        # additive residual on the query stream

class QFormer(nn.Module):
    """
    Compresses x into m latent tokens, reading at several depths.
    """

    def __init__(self, m: int, n_layers: int, n_heads: int,
                 inner_dim: int, k: int,
                 dropout: float = 0.0, mlp_ratio: int = 4,
                 grad_checkpointing: bool = True):
        super().__init__()
        assert n_layers % k == 0, \
            f"n_layers={n_layers} must be a multiple of k={k} so the last " \
            f"layer is a cross-attention read (trailing self-attention " \
            f"output would never reach the latents)"
        self.inner_dim = inner_dim
        self.k = k
        self.mlp_ratio = mlp_ratio
        self.grad_checkpointing = grad_checkpointing
        self.latents = nn.Parameter(torch.randn(m, inner_dim) * 0.02)
        layers = []
        for i in range(1, n_layers + 1):
            if i % self.k == 0:
                layers.append(CrossAttentionBlock(inner_dim, n_heads, mlp_ratio=mlp_ratio, dropout=dropout))
            else:
                layers.append(SelfAttentionBlock(inner_dim, n_heads, mlp_ratio=mlp_ratio, dropout=dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, key_padding_mask=None):
        # x: (B, T, D)  ->  (B, m, D)
        # key_padding_mask: (B, T) bool, True = real token, False = padding
        B = x.shape[0]
        q = self.latents.unsqueeze(0).expand(B, -1, -1)
        use_ckpt = self.grad_checkpointing and self.training and torch.is_grad_enabled()
        for i, layer in enumerate(self.layers, start=1):
            if use_ckpt:
                if i % self.k == 0:
                    q = checkpoint(layer, q, x, key_padding_mask, use_reentrant=False)
                else:
                    x = checkpoint(layer, x, key_padding_mask, use_reentrant=False)
            else:
                if i % self.k == 0:
                    q = layer(q, x, key_padding_mask)
                else:
                    x = layer(x, key_padding_mask)
        return q
    