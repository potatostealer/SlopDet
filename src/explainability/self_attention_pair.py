#!/usr/bin/env python3
"""Real vs AI pairs: what the SigLIP2 vision encoder's self-attention (LoRA-adapted, same checkpoint) attends to.

qformer_attention_pair.py with a different heat. Every one of the 27 Siglip2Attention modules of the vision tower
is hooked, and the per-token map is the attention each token RECEIVES: the softmax weight from query token q to
key token k, averaged over the real (non-padding) query tokens, the 16 heads and the 27 layers -- so, like the
QFormer vector, it sums to 1 over the tokens and the two can be compared directly. --agg rollout instead chains
the head-averaged layer matrices with the 0.5 * identity residual of attention rollout (Abnar & Zuidema 2020) and
averages the final matrix over the real queries.

The weights are recomputed from each module's input (a forward hook) with its own q_proj / k_proj / v_proj /
out_proj -- the LoRALinear wrappers of the checkpoint, so the LoRA deltas are included -- and checked by pushing
attn @ V through out_proj and comparing with the module's actual output. Padded keys are masked (exactly 0
attention) and padded queries are left out of the averages.

Same pairs, plans, figures and summary.csv as qformer_attention_pair.py (bottom row = canvas without overlay);
summary.csv additionally holds the outer-ring attention share of every layer (ring_l00..ring_l26), and the stdout
summary shows how that share evolves with depth for real vs AI.

Run from the repo root:
    python self_attention_pair.py                        # 10 pairs x (clean + 3 augs) -> comparisons_self_attention/
    python self_attention_pair.py -n 2 --n-aug 1 --device 7
    python self_attention_pair.py --agg rollout --out comparisons_self_attention_rollout
"""

import functools
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qformer_attention import padded_canvas  # noqa: E402
from qformer_attention_pair import LABELS, Detector, border_mass, build_parser, run_pairs  # noqa: E402
from src.modules.lora_adapter import LORA_TARGET_CHOICES, LoRALinear  # noqa: E402

DEFAULT_OUT = HERE / "comparisons_self_attention"


class SelfAttentionDetector(Detector):
    """Detector whose per-token heat comes from the vision encoder's self-attention instead of the QFormer."""

    def __init__(self, ckpt, device, agg="mean"):
        super().__init__(ckpt, device)
        self.hook.handle.remove()  # the base class's QFormer hook is not used here
        self.agg = agg
        self.modules = [layer.self_attn for layer in self.model.vision.encoder.layers]
        self.n_layers = len(self.modules)
        self.n_heads = self.modules[0].num_heads
        self.ring, self.ring_label = 0, "the outer ring"  # which ring the border statistics refer to
        self.captured = [None] * self.n_layers
        self.handles = [
            m.register_forward_hook(functools.partial(self._capture, index=i), with_kwargs=True)
            for i, m in enumerate(self.modules)
        ]
        lora = [name for name in LORA_TARGET_CHOICES if isinstance(getattr(self.modules[0], name), LoRALinear)]
        n_wrapped = sum(isinstance(getattr(m, n), LoRALinear) for m in self.modules for n in LORA_TARGET_CHOICES)
        self.note = (f"hooked {self.n_layers} self-attention layers x {self.n_heads} heads (head_dim "
                     f"{self.modules[0].head_dim}); LoRA on {', '.join(lora)} = {n_wrapped} wrapped projections; "
                     f"aggregation: {agg}")

    def _capture(self, module, args, kwargs, output, index):
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        self.captured[index] = (hidden, output[0])

    def close(self):
        for handle in self.handles:
            handle.remove()

    def vis_key_mask(self, key_mask, shapes):
        """(B, T) bool: the keys the VISUALISATION softmax runs over; None = the model's own (all real tokens)."""
        return None

    def query_mask(self, key_mask, shapes):
        """(B, T) bool: the query rows averaged into the per-token heat (default: every real token)."""
        return key_mask

    def keep_box(self, hp, wp):
        """(r0, c0, r1, c1) patch box drawn dashed on the figure when the visualisation keeps only part of the grid."""
        return None

    @torch.no_grad()
    def layer_weights(self, module, hidden, key_mask, vis_key_mask=None):
        """Siglip2Attention.forward made explicit. Returns the (B, heads, T, T) softmax weights for the
        visualisation (over vis_key_mask when given, else the model's own over key_mask) and the module output the
        TRUE weights give, for the consistency check -- the model's inference is never touched."""
        B, T, _ = hidden.shape
        shape = (B, T, module.num_heads, module.head_dim)
        q = module.q_proj(hidden).view(shape).transpose(1, 2)
        k = module.k_proj(hidden).view(shape).transpose(1, 2)
        v = module.v_proj(hidden).view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) * module.scale
        true_scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = true_scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        out = module.out_proj((attn @ v).transpose(1, 2).reshape(B, T, -1))
        if vis_key_mask is not None:
            vis_scores = scores.masked_fill(~vis_key_mask[:, None, None, :], float("-inf"))
            attn = vis_scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        return attn, out

    @torch.no_grad()
    def run(self, images):
        batch = self.collate.encode(images, [0] * len(images))
        pv = batch["pixel_values"].to(self.device)
        mask = batch["pixel_attention_mask"].to(self.device)
        shapes = batch["spatial_shapes"].to(self.device)
        logits = self.model(pv, mask, shapes)
        probs = torch.sigmoid(logits.float()).reshape(-1).cpu().numpy()

        B, T = mask.shape
        key_mask = mask.bool()
        vis_mask = self.vis_key_mask(key_mask, shapes)
        q_mask = self.query_mask(key_mask, shapes)
        per_layer = torch.zeros(B, self.n_layers, T, dtype=torch.float32)  # received attention per layer
        rollout = None
        eye = torch.eye(T, device=self.device)
        for l, module in enumerate(self.modules):
            hidden, out_ref = self.captured[l]
            attn, out = self.layer_weights(module, hidden, key_mask, vis_mask)
            self.worst_err = max(self.worst_err, float((out - out_ref).abs().max() / out_ref.abs().max()))
            A = attn.float().mean(dim=1)  # (B, T, T), mean over the heads
            if self.agg == "rollout":
                A = 0.5 * A + 0.5 * eye
                A = A / A.sum(dim=-1, keepdim=True)
                rollout = A if rollout is None else A @ rollout
                A = rollout
            for i in range(B):
                per_layer[i, l] = A[i][q_mask[i]].mean(dim=0).cpu()  # over the selected queries
        aggregate = per_layer[:, -1] if self.agg == "rollout" else per_layer.mean(dim=1)

        items = []
        for i, img in enumerate(images):
            hp, wp = shapes[i].tolist()
            canvas, ids = padded_canvas(batch["pixel_values"][i], hp, wp)
            vec = aggregate[i].numpy()
            items.append({
                "width": img.size[0], "height": img.size[1], "hp": hp, "wp": wp, "canvas": canvas, "ids": ids,
                "attn": vec, "prob": float(probs[i]), "n_pad": T - hp * wp, "keep_box": self.keep_box(hp, wp),
                "extra": {f"ring_l{l:02d}": f"{border_mass(per_layer[i, l].numpy(), hp, wp, self.ring)[0]:.4f}"
                          for l in range(self.n_layers)},
            })
        return items


def add_agg_argument(parser):
    parser.add_argument("--agg", choices=("mean", "rollout"), default="mean",
                        help="mean = attention received, averaged over queries, heads and layers; "
                             "rollout = attention rollout through the layers, averaged over queries")
    return parser


def heat_labels(args, suffix=""):
    """(heat_name, heat_desc(detector)) for the figures; suffix is appended to the description."""
    heat_name = "received self-attention" if args.agg == "mean" else "attention rollout"

    def heat_desc(det):
        if args.agg == "mean":
            base = (f"SigLIP2 vision self-attention received per token, mean over queries, {det.n_heads} heads "
                    f"and {det.n_layers} layers (LoRA applied)")
        else:
            base = f"SigLIP2 vision attention rollout over {det.n_layers} layers ({det.n_heads} heads averaged), mean over queries"
        return base + suffix

    return heat_name, heat_desc


def print_layer_table(rows, what):
    layer_keys = sorted(k for k in rows[0] if k.startswith("ring_l"))
    print(f"\n{what} by layer (mean over images):")
    print("        " + " ".join(f"{k[6:]:>4}" for k in layer_keys))
    for label in LABELS:
        sel = [r for r in rows if r["label"] == label]
        print(f"  {label:<5} " + " ".join(f"{np.mean([float(r[k]) for r in sel]):>4.0%}" for k in layer_keys))


def main():
    args = add_agg_argument(build_parser(__doc__, DEFAULT_OUT)).parse_args()
    heat_name, heat_desc = heat_labels(args)
    rows = run_pairs(args, functools.partial(SelfAttentionDetector, agg=args.agg), heat_name, heat_desc)
    print_layer_table(rows, "outer-ring share of the received attention")
    return 0


if __name__ == "__main__":
    sys.exit(main())
