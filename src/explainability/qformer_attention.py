#!/usr/bin/env python3
"""Where does the QFormer look? Per-head attention of its final cross-attention layer, mapped back onto the image.

Correspondence (established and checked in inspect_siglip.py):

    token t in 0..255  <->  patch (row = t // w_p, col = t % w_p) of the processor's canvas (h_p*16 x w_p*16 px)
                            for t < h_p*w_p; every later token is zero padding (masked, so its attention is exactly 0)
    canvas patch (r, c)  <->  the original-image rectangle  x in [c*W/w_p, (c+1)*W/w_p),  y in [r*H/h_p, (r+1)*H/h_p)
                            (the canvas is one uniform per-side rescale of the whole image: no crop, no offset)

So a (256,) attention vector is read by dropping the padding and reshaping to (h_p, w_p); it is drawn either on the
canvas at 16 px per patch ("token space", with the padded tokens laid out where they sit in raster order, i.e. as
extra rows under the image) or nearest-resized to (H, W) ("original space", the exact inverse of the patch mapping;
--interp bilinear smooths it instead).

The QFormer (src/modules/attention.py) is [self, self, cross] with m=1 latent query and 16 heads. Its cross layer
uses F.scaled_dot_product_attention, which does not return weights, so a forward pre-hook captures the layer's
inputs (q, x, key_padding_mask) and the weights softmax(QK^T / sqrt(d) + mask) are recomputed with the layer's own
to_q / to_kv; the script checks that attn @ V reproduces the SDPA output. The keys are the SigLIP tokens after the
two self-attention layers -- still one token per patch position, so the mapping above applies unchanged.

Outputs (--out, default dim_samples/_attention/):
    <sample>__tokens.png     original | rescaled + padded canvas | canvas + head-mean heat (top-k patches boxed),
                             then the 16 heads on the canvas
    <sample>__original.png   rescaled + padded canvas | original | original + reversed head-mean heat (boxes),
                             then the 16 heads reversed onto the original
    attention.npz            raw per-head weights (N, heads, 256), spatial shapes, sizes, p(AI), file names
plus, on stdout, the top-k patches of the head-mean per image with their canvas and original rectangles.

Run from the repo root (after sample_dims.py):
    python qformer_attention.py                                  # ckpt from src/configs/eval.yml, freest GPU
    python qformer_attention.py --ckpt logs/checkpoints/lora_qformer_online_aug/epoch01.ckpt --limit 4 --device 7
    python qformer_attention.py --interp bilinear --top-k 8 --shared-scale
"""

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from matplotlib import colors, patches as mpatches  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inspect_siglip import IMAGE_EXTS, denormalise, pick_device, unpatchify  # noqa: E402
from src.dataset.collate import Siglip2Collate  # noqa: E402
from src.experiments.eval import describe_model, load_model, resolve_device  # noqa: E402
from src.modules.attention import CrossAttentionBlock  # noqa: E402

DEFAULT_SAMPLES = HERE / "dim_samples"
DEFAULT_EVAL_CFG = HERE / "src" / "configs" / "eval.yml"
PATCH = 16
INK, INK_MUTED, GRID = "#333333", "#777777", "#ffffff"


# ----------------------------------------------------------------------------- attention weights
def cross_attention_weights(layer: CrossAttentionBlock, q, x, key_padding_mask):
    """What layer.forward feeds to scaled_dot_product_attention, made explicit: (B, heads, Tq, Tk) softmax weights.
    Also returns V so the caller can check attn @ V against the SDPA output."""
    B, Tq, D = q.shape
    Tk = x.shape[1]
    query = layer.to_q(layer.norm_q(q)).view(B, Tq, layer.n_heads, layer.head_dim).transpose(1, 2)
    key, value = layer.to_kv(layer.norm_x(x)).chunk(2, dim=-1)
    key = key.view(B, Tk, layer.n_heads, layer.head_dim).transpose(1, 2)
    value = value.view(B, Tk, layer.n_heads, layer.head_dim).transpose(1, 2)
    scores = query @ key.transpose(-1, -2) / math.sqrt(layer.head_dim)
    if key_padding_mask is not None:
        scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
    return scores.softmax(dim=-1), (query, key, value)


class FinalCrossAttention:
    """Forward pre-hook on the QFormer's last layer: keeps (q, x, key_padding_mask) of the latest call."""

    def __init__(self, qformer):
        self.layer = qformer.layers[-1]
        if not isinstance(self.layer, CrossAttentionBlock):
            raise TypeError(f"last QFormer layer is {type(self.layer).__name__}, expected CrossAttentionBlock")
        self.args = None
        self.handle = self.layer.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        self.args = args  # QFormer.forward calls layer(q, x, key_padding_mask) positionally

    @torch.no_grad()
    def weights(self):
        q, x, mask = self.args
        attn, (query, key, value) = cross_attention_weights(self.layer, q, x, mask)
        # the layer's own computation, for a consistency check
        sdpa = F.scaled_dot_product_attention(query, key, value, attn_mask=mask[:, None, None, :] if mask is not None else None)
        err = float((attn @ value - sdpa).abs().max() / sdpa.abs().max())
        return attn, err


# ----------------------------------------------------------------------------- geometry
def token_layout(hp, wp, n_tokens):
    """Raster layout of all n_tokens tokens in a w_p-wide grid: (rows, w_p) array of token ids, -1 where no token."""
    rows = math.ceil(n_tokens / wp)
    ids = np.full((rows, wp), -1, dtype=np.int64)
    ids.flat[:n_tokens] = np.arange(n_tokens)
    return ids


def padded_canvas(pixel_values, hp, wp):
    """The 256 token rows drawn as an image: the h_p x w_p canvas, then the zero-padding tokens (mid grey after
    de-normalising) as extra rows below it; cells that hold no token are white."""
    n_tokens = pixel_values.shape[0]
    ids = token_layout(hp, wp, n_tokens)
    rows = ids.shape[0]
    full = torch.zeros(rows * wp, pixel_values.shape[1], dtype=pixel_values.dtype)
    full[:n_tokens] = pixel_values
    canvas = denormalise(unpatchify(full, rows, wp, PATCH)).permute(1, 2, 0).numpy().copy()
    for r, c in zip(*np.where(ids < 0)):
        canvas[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = 255
    return canvas, ids


def heat_in_token_space(attn_256, ids):
    """(256,) attention -> (rows, w_p) array following the token layout; NaN where no token."""
    heat = np.full(ids.shape, np.nan)
    valid = ids >= 0
    heat[valid] = attn_256[ids[valid]]
    return heat


def heat_in_original_space(attn_256, hp, wp, height, width, mode):
    """Drop the padding, reshape to the patch grid and resize to the original image: nearest = each patch is exactly
    its rectangle of the original; bilinear = smoothed."""
    grid = torch.as_tensor(attn_256[: hp * wp], dtype=torch.float32).reshape(1, 1, hp, wp)
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(grid, size=(height, width), mode=mode, **kwargs)[0, 0].numpy()


def top_patches(attn_256, hp, wp, k):
    """The k highest tokens among the real patches: [(token, row, col, score)] best first."""
    real = attn_256[: hp * wp]
    order = np.argsort(-real)[:k]
    return [(int(t), int(t // wp), int(t % wp), float(real[t])) for t in order]


# ----------------------------------------------------------------------------- drawing
def overlay(ax, base, heat, extent, cmap, alpha_max, vmax=None, dim=0.0):
    """base (H, W, 3) uint8 (darkened by `dim` so the heat stands out) then heat (h, w) with per-cell alpha
    proportional to the value; returns the mappable for a colorbar."""
    if dim > 0:
        base = (base.astype(np.float32) * (1.0 - dim)).astype(np.uint8)
    ax.imshow(base, extent=extent, interpolation="nearest" if base.shape[0] <= 512 else "antialiased")
    finite = np.nan_to_num(heat, nan=0.0)
    vmax = float(np.max(finite)) if vmax is None else vmax
    norm = colors.Normalize(0.0, vmax if vmax > 0 else 1.0)
    level = np.clip(norm(finite), 0, 1)
    rgba = cmap(level)
    rgba[..., 3] = alpha_max * level
    rgba[np.isnan(heat), 3] = 0.0
    ax.imshow(rgba, extent=extent, interpolation="nearest")
    return ScalarMappable(norm=norm, cmap=cmap)


def draw_grid(ax, xs, ys, alpha=0.25):
    ax.vlines(xs, ys[0], ys[-1], colors=GRID, linewidth=0.4, alpha=alpha)
    ax.hlines(ys, xs[0], xs[-1], colors=GRID, linewidth=0.4, alpha=alpha)


def draw_boxes(ax, boxes, label=True):
    """boxes: [(x, y, w, h, rank)] in the axis' data units."""
    for x, y, w, h, rank in boxes:
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, edgecolor="white", linewidth=2.2))
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, edgecolor=INK, linewidth=1.0))
        if label:
            ax.text(x + w / 2, y + h / 2, str(rank), ha="center", va="center", fontsize=8, color=INK,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))


def tidy(ax, title, size=None):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")


def figure(sample, attn, head_mean, tops, args, space):
    """One figure: comparison row (both images + heat) and the 16 heads. space = 'tokens' | 'original'."""
    hp, wp, H, W = sample["hp"], sample["wp"], sample["height"], sample["width"]
    canvas, ids = sample["canvas"], sample["ids"]
    rows = ids.shape[0]
    n_pad = attn.shape[-1] - hp * wp
    n_heads = attn.shape[0]
    n_cols = 4
    n_rows = math.ceil(n_heads / n_cols)
    cmap = plt.get_cmap(args.cmap)
    canvas_extent = (0, wp * PATCH, rows * PATCH, 0)
    orig_extent = (0, W, H, 0)
    canvas_xs, canvas_ys = np.arange(wp + 1) * PATCH, np.arange(rows + 1) * PATCH
    orig_xs, orig_ys = np.arange(wp + 1) * W / wp, np.arange(hp + 1) * H / hp
    aspect_canvas, aspect_orig = rows / wp, H / W  # height / width of what each panel shows
    aspect_heat = aspect_canvas if space == "tokens" else aspect_orig

    # figure height from the panel aspects: 3 panels across the top, n_cols across each head row
    fig_w = 16.0
    title_h = 0.45  # inches reserved above every panel for its two-line title
    h_top = fig_w / 3 * max(aspect_canvas, aspect_orig) + title_h
    h_row = fig_w / n_cols * aspect_heat + title_h
    head_h = 0.8
    fig_h = head_h + h_top + n_rows * h_row
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs0 = fig.add_gridspec(2, 1, height_ratios=[h_top, n_rows * h_row], left=0.01, right=0.99,
                           top=1 - head_h / fig_h, bottom=0.005, hspace=0.06)
    gs_top = gs0[0].subgridspec(1, 3, wspace=0.04)
    gs_grid = gs0[1].subgridspec(n_rows, n_cols, wspace=0.04, hspace=title_h / (h_row - title_h) * 1.1)

    def draw_canvas(ax, title):
        ax.imshow(canvas, extent=canvas_extent, interpolation="nearest")
        if not args.no_grid:
            draw_grid(ax, canvas_xs, canvas_ys)
        ax.axhline(hp * PATCH, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
        if rows > hp:
            ax.text(wp * PATCH / 2, hp * PATCH + (rows - hp) * PATCH / 2, f"{n_pad} padding tokens",
                    ha="center", va="center", fontsize=8, color=INK)
        tidy(ax, title)

    def draw_original(ax, title):
        ax.imshow(sample["image"], extent=orig_extent)
        if not args.no_grid:
            draw_grid(ax, orig_xs, orig_ys)
        tidy(ax, title)

    def heat_and_boxes(ax, vec, vmax, boxes=None):
        if space == "tokens":
            heat = heat_in_token_space(vec, ids)
            mappable = overlay(ax, canvas, heat, canvas_extent, cmap, args.alpha, vmax, args.dim)
            if not args.no_grid:
                draw_grid(ax, canvas_xs, canvas_ys)
            ax.axhline(hp * PATCH, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
        else:
            heat = heat_in_original_space(vec, hp, wp, H, W, args.interp)
            mappable = overlay(ax, sample["image"], heat, orig_extent, cmap, args.alpha, vmax, args.dim)
            if not args.no_grid:
                draw_grid(ax, orig_xs, orig_ys)
        if boxes:
            draw_boxes(ax, boxes)
        return mappable

    def boxes_for(tops_):
        if space == "tokens":
            return [(c * PATCH, r * PATCH, PATCH, PATCH, i + 1) for i, (_, r, c, _) in enumerate(tops_)]
        return [(c * W / wp, r * H / hp, W / wp, H / hp, i + 1) for i, (_, r, c, _) in enumerate(tops_)]

    canvas_title = (f"with rescale + padding: {wp * PATCH}x{hp * PATCH} px canvas\n"
                    f"= {hp} x {wp} patches + {n_pad} padding tokens (grey)")
    orig_title = (f"without: original {W}x{H} px\n"
                  f"grid = patch footprints, {W / wp:.0f}x{H / hp:.0f} px each")
    ax_a, ax_b, ax_c = (fig.add_subplot(gs_top[0, i]) for i in range(3))
    if space == "tokens":
        draw_original(ax_a, orig_title)
        draw_canvas(ax_b, canvas_title)
        mappable = heat_and_boxes(ax_c, head_mean, None, boxes_for(tops))
        tidy(ax_c, f"canvas + mean of {n_heads} heads (max {head_mean.max():.3f})\n"
                   f"boxes = top {len(tops)} patches; photo dimmed {args.dim:.0%}")
    else:
        draw_canvas(ax_a, canvas_title)
        draw_original(ax_b, orig_title)
        mappable = heat_and_boxes(ax_c, head_mean, None, boxes_for(tops))
        tidy(ax_c, f"original + mean of {n_heads} heads, {args.interp} un-scaling\n"
                   f"boxes = top {len(tops)} patches; photo dimmed {args.dim:.0%}")
    cbar = fig.colorbar(mappable, ax=ax_c, fraction=0.035, pad=0.02)
    cbar.set_label("attention weight (softmax over the tokens)", fontsize=7, color=INK_MUTED)
    cbar.ax.tick_params(labelsize=7, colors=INK_MUTED)

    shared_vmax = float(attn.max()) if args.shared_scale else None
    for h in range(n_heads):
        ax = fig.add_subplot(gs_grid[h // n_cols, h % n_cols])
        vec = attn[h]
        t, r, c, s = top_patches(vec, hp, wp, 1)[0]
        heat_and_boxes(ax, vec, shared_vmax, boxes_for([(t, r, c, s)]))
        tidy(ax, f"head {h}: max {s:.2f} at patch ({r}, {c}) = token {t}")

    where = "token space: the 256 tokens as laid out for the model" if space == "tokens" else "original image space"
    fig.suptitle(
        f"{sample['name']}   {W}x{H} -> {wp * PATCH}x{hp * PATCH} ({hp}x{wp} = {hp * wp} patches, "
        f"{n_pad} padded), scale {wp * PATCH / W:.3f} x {hp * PATCH / H:.3f}   p(AI) = {sample['prob']:.3f}\n"
        f"final QFormer cross-attention (latent query -> SigLIP tokens), {where}; "
        f"each panel scaled to {'the image-wide max' if args.shared_scale else 'its own max'}",
        fontsize=10, color=INK, x=0.01, y=0.995, ha="left", va="top")
    return fig


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default=None, help="Lightning checkpoint (default: model.ckpt_path of --eval-config)")
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CFG)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES, help="dir written by sample_dims.py")
    parser.add_argument("--out", type=Path, default=None, help="default: <samples>/_attention")
    parser.add_argument("--limit", type=int, default=None, help="only the first N sample files")
    parser.add_argument("--device", default="auto", help="cpu | cuda:N | N | auto (CUDA device with most free memory)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5, help="patches boxed on the head-mean panels")
    parser.add_argument("--interp", choices=("nearest", "bilinear"), default="nearest",
                        help="how the patch grid is un-scaled onto the original (nearest = exact patch rectangles)")
    parser.add_argument("--cmap", default="Oranges", help="single-hue sequential colormap for the heat")
    parser.add_argument("--alpha", type=float, default=0.9, help="opacity of the heat at its maximum")
    parser.add_argument("--dim", type=float, default=0.4, help="darken the photo under the heat by this fraction")
    parser.add_argument("--match", default=None, help="only sample files whose name contains this substring")
    parser.add_argument("--shared-scale", action="store_true", help="scale the 16 head panels to one image-wide max")
    parser.add_argument("--no-grid", action="store_true", help="no patch grid lines")
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()

    if args.ckpt is None:
        with open(args.eval_config) as f:
            args.ckpt = yaml.safe_load(f)["model"]["ckpt_path"]
    out = args.out or args.samples / "_attention"
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in args.samples.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"no sample images in {args.samples}; run sample_dims.py first")
    if args.match:
        paths = [p for p in paths if args.match in p.name]
    paths = paths[: args.limit] if args.limit else paths
    if not paths:
        raise SystemExit("no sample files left after --match / --limit")

    device = resolve_device(str(pick_device(args.device)) if args.device == "auto" else args.device)
    t0 = time.perf_counter()
    model, hparams, info = load_model({"ckpt_path": args.ckpt, "siglip_checkpoint_path": None}, device)
    print(f"model: {describe_model(hparams)}\nckpt:  {info['ckpt_path']} (epoch {info['epoch']}, step {info['global_step']}) "
          f"loaded on {device} in {time.perf_counter() - t0:.1f}s")
    hook = FinalCrossAttention(model.qformer)
    n_heads = hook.layer.n_heads
    print(f"final QFormer layer: {type(hook.layer).__name__} with {n_heads} heads, head_dim {hook.layer.head_dim}, "
          f"m = {model.qformer.latents.shape[0]} latent quer{'y' if model.qformer.latents.shape[0] == 1 else 'ies'}")

    collate = Siglip2Collate(hparams["model"]["checkpoint_path"])
    samples, worst_err, pad_leak = [], 0.0, 0.0
    for start in range(0, len(paths), args.batch_size):
        chunk = paths[start:start + args.batch_size]
        images = [Image.open(p).convert("RGB") for p in chunk]
        batch = collate.encode(images, [1] * len(images))
        with torch.no_grad():
            logits = model(batch["pixel_values"].to(device), batch["pixel_attention_mask"].to(device),
                           batch["spatial_shapes"].to(device))
        probs = torch.sigmoid(logits.float()).reshape(-1).cpu()
        attn, err = hook.weights()  # (B, heads, m, 256)
        worst_err = max(worst_err, err)
        attn = attn.mean(dim=2).float().cpu().numpy()  # mean over the m latent queries (m = 1: identity)
        for i, (path, img) in enumerate(zip(chunk, images)):
            hp, wp = batch["spatial_shapes"][i].tolist()
            pad_leak = max(pad_leak, float(attn[i, :, hp * wp:].max()) if hp * wp < attn.shape[-1] else 0.0)
            canvas, ids = padded_canvas(batch["pixel_values"][i], hp, wp)
            samples.append({
                "name": path.name, "path": str(path), "image": np.asarray(img), "width": img.size[0],
                "height": img.size[1], "hp": hp, "wp": wp, "canvas": canvas, "ids": ids,
                "attn": attn[i], "prob": float(probs[i]),
            })
    hook.handle.remove()
    print(f"\nrecomputed weights vs the layer's SDPA output: max rel err {worst_err:.1e}; "
          f"max attention on a padding token: {pad_leak:.1e}\n")

    for s in samples:
        head_mean = s["attn"].mean(axis=0)
        tops = top_patches(head_mean, s["hp"], s["wp"], args.top_k)
        H, W, hp, wp = s["height"], s["width"], s["hp"], s["wp"]
        print(f"{s['name']}   {W}x{H} -> {hp}x{wp} grid   p(AI) = {s['prob']:.3f}   "
              f"head maxima: " + " ".join(f"{s['attn'][h].max():.2f}" for h in range(n_heads)))
        for rank, (t, r, c, score) in enumerate(tops, start=1):
            print(f"   #{rank} token {t:>3} = patch ({r:>2}, {c:>2})  mean attn {score:.3f}   "
                  f"canvas px x {c * PATCH}-{(c + 1) * PATCH}, y {r * PATCH}-{(r + 1) * PATCH}   "
                  f"original px x {c * W / wp:.0f}-{(c + 1) * W / wp:.0f}, y {r * H / hp:.0f}-{(r + 1) * H / hp:.0f}")
        for space in ("tokens", "original"):
            fig = figure(s, s["attn"], head_mean, tops, args, space)
            fig.savefig(out / f"{Path(s['name']).stem}__{space}.png", dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)

    np.savez(
        out / "attention.npz",
        names=np.array([s["name"] for s in samples]), attention=np.stack([s["attn"] for s in samples]),
        spatial_shapes=np.array([(s["hp"], s["wp"]) for s in samples]),
        sizes=np.array([(s["width"], s["height"]) for s in samples]), prob_ai=np.array([s["prob"] for s in samples]),
    )
    peak = np.stack([s["attn"].max(axis=1) for s in samples]).mean(axis=0)
    print(f"\nmean over images of each head's max attention (1/256 = {1 / 256:.4f} would be uniform): "
          + " ".join(f"h{h} {v:.2f}" for h, v in enumerate(peak)))
    # how much of the attention lands on the outermost ring of patches, vs that ring's share of the patches
    border_mass, border_share = [], []
    for s in samples:
        hp, wp = s["hp"], s["wp"]
        ring = np.zeros((hp, wp), dtype=bool)
        ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
        grid = s["attn"].mean(axis=0)[: hp * wp].reshape(hp, wp)
        border_mass.append(grid[ring].sum())
        border_share.append(ring.mean())
    print(f"head-mean attention mass on the outermost patch ring: {np.mean(border_mass):.1%} on average "
          f"(min {np.min(border_mass):.1%}, max {np.max(border_mass):.1%}) vs the ring's share of the patches "
          f"{np.mean(border_share):.1%}")
    print(f"wrote {2 * len(samples)} figures + attention.npz to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
