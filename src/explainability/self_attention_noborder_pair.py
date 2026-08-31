#!/usr/bin/env python3
"""self_attention_pair.py with the border tokens removed from the attention softmax -- visualisation only.

Hypothesis: the encoder parks information in non-informative tokens along the image border (the outer patches
soak up ~half of all self-attention in self_attention_pair.py), so the attention over the remaining tokens is
hidden behind that. Here every hook drops the logits of the keys in the outermost --rings (default 2) rings of
patches -- the outermost ring and the second one, for each image's own h_p x w_p grid, plus the padding as always
-- BEFORE the softmax, so the weights are renormalised over the interior tokens. The model's own inference is not
touched: the forward pass runs exactly as in training / eval, the hooks only read each attention module's input;
the consistency check still compares the TRUE attention (attn @ V through out_proj) with the module's output.

--queries all (default) averages the interior-only attention over every real query token, like
self_attention_pair.py; --queries interior also drops the border tokens as queries, i.e. what the interior looks
at among itself. The ring statistic in the titles / summary.csv / stdout refers to the outermost KEPT ring (the
new border after dropping --rings rings), to see whether the border focus simply moves inwards; the dashed box on
the heat panels marks the kept region.

Run from the repo root:
    python self_attention_noborder_pair.py                       # 10 pairs -> comparisons_self_attention_noborder/
    python self_attention_noborder_pair.py --rings 1 --queries interior -n 2 --n-aug 1 --device 7
"""

import functools
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qformer_attention_pair import build_parser, run_pairs  # noqa: E402
from self_attention_pair import SelfAttentionDetector, add_agg_argument, heat_labels, print_layer_table  # noqa: E402

DEFAULT_OUT = HERE / "comparisons_self_attention_noborder"


class NoBorderSelfAttentionDetector(SelfAttentionDetector):
    """SelfAttentionDetector whose visualisation softmax ignores the `rings` outermost rings of patches."""

    def __init__(self, ckpt, device, agg="mean", rings=2, queries="all"):
        super().__init__(ckpt, device, agg)
        self.rings, self.queries = rings, queries
        self.ring = rings  # statistics: the outermost ring that is still in the softmax
        self.ring_label = f"the outermost kept ring (ring {rings + 1})"
        self.note += (f"; visualisation drops the {rings} outermost rings of key tokens before the softmax, "
                      f"queries = {queries}")

    def interior(self, key_mask, shapes):
        """(B, T) bool: real tokens whose patch is at least `rings` patches away from every edge of its grid."""
        B, T = key_mask.shape
        keep = torch.zeros_like(key_mask)
        for i in range(B):
            hp, wp = shapes[i].tolist()
            if hp <= 2 * self.rings or wp <= 2 * self.rings:
                raise ValueError(f"grid {hp}x{wp} has no interior after dropping {self.rings} rings; use --rings smaller")
            t = torch.arange(hp * wp, device=key_mask.device)
            r, c = t // wp, t % wp
            depth = torch.minimum(torch.minimum(r, hp - 1 - r), torch.minimum(c, wp - 1 - c))
            keep[i, : hp * wp] = depth >= self.rings
        return keep & key_mask

    def vis_key_mask(self, key_mask, shapes):
        return self.interior(key_mask, shapes)

    def query_mask(self, key_mask, shapes):
        return self.interior(key_mask, shapes) if self.queries == "interior" else key_mask

    def keep_box(self, hp, wp):
        return (self.rings, self.rings, hp - self.rings, wp - self.rings)


def main():
    parser = add_agg_argument(build_parser(__doc__, DEFAULT_OUT))
    parser.add_argument("--rings", type=int, default=2, help="outermost rings of patches dropped from the softmax")
    parser.add_argument("--queries", choices=("all", "interior"), default="all",
                        help="which query tokens are averaged: all real tokens, or only the kept interior")
    args = parser.parse_args()
    if args.rings < 1:
        parser.error("--rings must be >= 1")
    heat_name, heat_desc = heat_labels(args, suffix=f"; {args.rings} outer rings dropped from the softmax, "
                                                    f"{args.queries} queries")
    make = functools.partial(NoBorderSelfAttentionDetector, agg=args.agg, rings=args.rings, queries=args.queries)
    rows = run_pairs(args, make, heat_name, heat_desc)
    print_layer_table(rows, f"share of the attention on the outermost kept ring (ring {args.rings + 1})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
