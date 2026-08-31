#!/usr/bin/env python3
"""Real vs AI pairs: where the QFormer looks at each image, clean and under the same augmentations.

ai_gen_all_val and real_all_val share 8,000 file stems (<16 hex>.png there, <16 hex>.jpg here: the Open Images
photo and the AI image generated for it). This script draws N such pairs, runs the detector on both images with
qformer_attention.py's machinery (final cross-attention weights recomputed from a forward pre-hook, then averaged
over the heads) and writes, per pair,

    <stem>__clean.png         real | AI, as they are
    <stem>__aug1..augK.png    the same pair after one augmentation plan drawn the way the training collate draws
                              them (src/dataset/online_augment.py mixture with the training checkpoint's
                              online_augment block; identity plans are re-drawn) and applied IDENTICALLY to both
                              images: same steps, same order, same parameters, same noise seed

Each figure is the __tokens view of qformer_attention.py: top row = the rescaled + padded canvas the model sees with
the head-mean attention (colour scale shared by the pair, top-k patches boxed), bottom row = the same canvas without
the overlay. summary.csv collects p(AI), the patch grid, the top patch and the attention mass on the outermost patch
ring for every image of every figure.

Run from the repo root:
    python qformer_attention_pair.py                        # 10 pairs x (clean + 3 augmentations) -> comparisons/
    python qformer_attention_pair.py -n 2 --n-aug 1 --device 7
    python qformer_attention_pair.py --stems 0001ced78587af6a 00056e3e7311ae2f --seed 1
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from matplotlib import patches as mpatches  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.utils.inspect_siglip import IMAGE_EXTS, pick_device  # noqa: E402
from qformer_attention import (  # noqa: E402
    INK, INK_MUTED, PATCH, FinalCrossAttention, draw_boxes, draw_grid, heat_in_token_space, overlay,
    padded_canvas, tidy, top_patches,
)
from src.dataset.augment import image_seed  # noqa: E402
from src.dataset.collate import Siglip2Collate  # noqa: E402
from src.dataset.online_augment import build_online_augmenter, plan_label  # noqa: E402
from src.experiments.eval import describe_model, load_model, resolve_device  # noqa: E402

DEFAULT_REAL = "data/real_val"   # real / AI val directories with stem-paired files (--real-dir / --ai-dir override)
DEFAULT_AI = "data/ai_gen_val"
DEFAULT_EVAL_CFG = HERE / "src" / "configs" / "eval.yml"
DEFAULT_TRAINING_CFG = HERE / "src" / "configs" / "training.yml"
DEFAULT_OUT = HERE / "comparisons"
LABELS = ("real", "AI")


# ----------------------------------------------------------------------------- pairs and plans
def find_pairs(real_dir, ai_dir):
    """{stem: (real path, AI path)} for every stem present in both directories."""
    def by_stem(directory):
        files = {}
        for p in sorted(Path(directory).iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                files.setdefault(p.stem, p)  # first extension wins (sorted: .jpg before .png)
        return files
    real, ai = by_stem(real_dir), by_stem(ai_dir)
    return {stem: (real[stem], ai[stem]) for stem in sorted(set(real) & set(ai))}


def choose_pairs(pairs, n, seed, stems=None):
    if stems:
        missing = [s for s in stems if s not in pairs]
        if missing:
            raise SystemExit(f"no pair for stems {missing}")
        return [(s, pairs[s]) for s in stems]
    chosen = random.Random(seed).sample(sorted(pairs), min(n, len(pairs)))
    return [(s, pairs[s]) for s in chosen]


def draw_plan(augmenter, stem, k, seed):
    """One non-identity plan for augmentation k of this pair, reproducible from (stem, k, seed) alone."""
    rng = random.Random(image_seed(f"{stem}/aug{k}", seed))
    plan = augmenter.sample_plan(rng)
    while not plan:  # identity (p_none) is what the __clean figure already shows
        plan = augmenter.sample_plan(rng)
    return plan


def ring_cells(hp, wp, ring=0):
    """(hp, wp) bool mask of the patches whose distance to the nearest edge is exactly `ring` (0 = outermost)."""
    r, c = np.arange(hp)[:, None], np.arange(wp)[None, :]
    depth = np.minimum(np.minimum(r, hp - 1 - r), np.minimum(c, wp - 1 - c))
    return depth == ring


def border_mass(attn_256, hp, wp, ring=0):
    """Share of the attention on ring `ring` of the patch grid (0 = outermost), and that ring's share of the patches."""
    cells = ring_cells(hp, wp, ring)
    grid = attn_256[: hp * wp].reshape(hp, wp)
    return float(grid[cells].sum()), float(cells.mean())


# ----------------------------------------------------------------------------- model
class Detector:
    """Checkpoint + collate + hook: PIL images -> per-image dicts with the head-mean attention and p(AI)."""

    def __init__(self, ckpt, device):
        self.model, self.hparams, self.info = load_model({"ckpt_path": ckpt, "siglip_checkpoint_path": None}, device)
        self.device = device
        self.hook = FinalCrossAttention(self.model.qformer)
        self.n_heads = self.hook.layer.n_heads
        self.collate = Siglip2Collate(self.hparams["model"]["checkpoint_path"])
        self.worst_err = 0.0

    def close(self):
        self.hook.handle.remove()

    @torch.no_grad()
    def run(self, images):
        batch = self.collate.encode(images, [0] * len(images))
        logits = self.model(batch["pixel_values"].to(self.device), batch["pixel_attention_mask"].to(self.device),
                            batch["spatial_shapes"].to(self.device))
        probs = torch.sigmoid(logits.float()).reshape(-1).cpu().numpy()
        attn, err = self.hook.weights()  # (B, heads, m, 256)
        self.worst_err = max(self.worst_err, err)
        attn_mean = attn.mean(dim=(1, 2)).float().cpu().numpy()  # over heads and the m latent queries
        items = []
        for i, img in enumerate(images):
            hp, wp = batch["spatial_shapes"][i].tolist()
            canvas, ids = padded_canvas(batch["pixel_values"][i], hp, wp)
            items.append({
                "width": img.size[0], "height": img.size[1], "hp": hp, "wp": wp, "canvas": canvas, "ids": ids,
                "attn": attn_mean[i], "prob": float(probs[i]), "n_pad": attn.shape[-1] - hp * wp,
            })
        return items


# ----------------------------------------------------------------------------- figure
def pair_figure(items, names, sizes0, stem, variant, plan, detector, args, heat_name="head-mean attention",
                heat_desc=None):
    """2 x 2: top = canvases with the per-token heat (items[i]["attn"], a (256,) vector), bottom = canvases alone.
    heat_name labels the panels, heat_desc the suptitle (default: the QFormer wording)."""
    heat_desc = heat_desc or f"final QFormer cross-attention, mean of {detector.n_heads} heads"
    ring, ring_label = getattr(detector, "ring", 0), getattr(detector, "ring_label", "the outer ring")
    cmap = plt.get_cmap(args.cmap)
    aspects = [it["ids"].shape[0] / it["wp"] for it in items]
    fig_w, title_h, head_h = 13.0, 0.55, 0.75
    row_h = fig_w / 2 * max(aspects) + title_h
    fig_h = head_h + 2 * row_h
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.01, right=0.94, top=1 - head_h / fig_h, bottom=0.005, wspace=0.05,
                          hspace=0.13)
    vmax = None if args.own_scale else max(float(it["attn"].max()) for it in items)
    top_axes, mappable = [], None
    for col, (it, name, size0, label) in enumerate(zip(items, names, sizes0, LABELS)):
        hp, wp, rows = it["hp"], it["wp"], it["ids"].shape[0]
        extent = (0, wp * PATCH, rows * PATCH, 0)
        xs, ys = np.arange(wp + 1) * PATCH, np.arange(rows + 1) * PATCH
        tops = top_patches(it["attn"], hp, wp, args.top_k)
        b_mass, b_share = border_mass(it["attn"], hp, wp, ring)
        size_txt = (f"{size0[0]}x{size0[1]}" if size0 == (it["width"], it["height"])
                    else f"{size0[0]}x{size0[1]} -> aug {it['width']}x{it['height']}")

        ax = fig.add_subplot(gs[0, col])
        heat = heat_in_token_space(it["attn"], it["ids"])
        mappable = overlay(ax, it["canvas"], heat, extent, cmap, args.alpha, vmax, args.dim)
        if not args.no_grid:
            draw_grid(ax, xs, ys)
        ax.axhline(hp * PATCH, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
        draw_boxes(ax, [(c * PATCH, r * PATCH, PATCH, PATCH, i + 1) for i, (_, r, c, _) in enumerate(tops)])
        if it.get("keep_box"):  # tokens outside this dashed box were dropped from the visualisation softmax
            r0, c0, r1, c1 = it["keep_box"]
            ax.add_patch(mpatches.Rectangle((c0 * PATCH, r0 * PATCH), (c1 - c0) * PATCH, (r1 - r0) * PATCH,
                                            fill=False, edgecolor=INK, linewidth=1.3, linestyle=(0, (4, 3))))
        tidy(ax, f"{label}: {name}   {size_txt} px -> {wp * PATCH}x{hp * PATCH} canvas, {hp}x{wp} patches"
                 f"{' + %d pad' % it['n_pad'] if it['n_pad'] else ''}\n"
                 f"p(AI) = {it['prob']:.3f}   {heat_name}: max {it['attn'].max():.3f}, "
                 f"{b_mass:.0%} on {ring_label} ({b_share:.0%} of patches), boxes = top {len(tops)}")
        top_axes.append(ax)

        ax2 = fig.add_subplot(gs[1, col])
        ax2.imshow(it["canvas"], extent=extent, interpolation="nearest")
        if not args.no_grid:
            draw_grid(ax2, xs, ys)
        ax2.axhline(hp * PATCH, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
        if rows > hp:
            ax2.text(wp * PATCH / 2, hp * PATCH + (rows - hp) * PATCH / 2, f"{it['n_pad']} padding tokens",
                     ha="center", va="center", fontsize=8, color=INK)
        tidy(ax2, f"{label}: the same canvas without the overlay")
        if args.own_scale:
            cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.01)
            cbar.ax.tick_params(labelsize=7, colors=INK_MUTED)
    if not args.own_scale:
        pos = top_axes[-1].get_position()
        cax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.012, pos.height])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label("attention weight, scale shared by the pair", fontsize=7, color=INK_MUTED)
        cbar.ax.tick_params(labelsize=7, colors=INK_MUTED)

    fig.suptitle(
        f"{stem}   {variant}: {plan_label(plan)}\n"
        f"{Path(args.real_dir).name}/{names[0]} vs {Path(args.ai_dir).name}/{names[1]}; "
        f"{heat_desc}, token space; "
        f"{Path(detector.info['ckpt_path']).parent.name}/{Path(detector.info['ckpt_path']).name}",
        fontsize=10, color=INK, x=0.01, y=0.995, ha="left", va="top")
    return fig


# ----------------------------------------------------------------------------- main
def build_parser(description=__doc__, out=DEFAULT_OUT):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--pairs", type=int, default=10, help="how many pairs to draw")
    parser.add_argument("--stems", nargs="*", default=None, help="use these stems instead of drawing")
    parser.add_argument("--seed", type=int, default=0, help="which pairs are drawn")
    parser.add_argument("--n-aug", type=int, default=3, help="augmented figures per pair")
    parser.add_argument("--aug-seed", type=int, default=None,
                        help="seed of the augmentation plans (default: online_augment.seed of the checkpoint)")
    parser.add_argument("--real-dir", default=DEFAULT_REAL)
    parser.add_argument("--ai-dir", default=DEFAULT_AI)
    parser.add_argument("--out", type=Path, default=out)
    parser.add_argument("--ckpt", default=None, help="Lightning checkpoint (default: model.ckpt_path of --eval-config)")
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CFG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CFG,
                        help="online_augment block fallback when the checkpoint stores none")
    parser.add_argument("--device", default="auto", help="cpu | cuda:N | N | auto (CUDA device with most free memory)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cmap", default="Oranges")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--dim", type=float, default=0.4, help="darken the canvas under the heat by this fraction")
    parser.add_argument("--own-scale", action="store_true", help="scale each heat panel to its own max")
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--dpi", type=int, default=100)
    return parser


def run_pairs(args, make_detector=Detector, heat_name="head-mean attention", heat_desc=None):
    """The whole pipeline: draw the pairs, build the detector (make_detector(ckpt, device) -> Detector-like: .run,
    .close, .hparams, .info, .worst_err, optional .note), one figure per pair and variant, summary.csv.
    heat_desc may be a string or a function of the detector. Returns the summary rows (any "extra" dict a
    detector puts in its items is flattened into them)."""
    if args.ckpt is None:
        with open(args.eval_config) as f:
            args.ckpt = yaml.safe_load(f)["model"]["ckpt_path"]
    all_pairs = find_pairs(args.real_dir, args.ai_dir)
    pairs = choose_pairs(all_pairs, args.pairs, args.seed, args.stems)
    print(f"{len(pairs)} pairs from {len(all_pairs)} shared stems: " + " ".join(s for s, _ in pairs))

    device = resolve_device(str(pick_device(args.device)) if args.device == "auto" else args.device)
    t0 = time.perf_counter()
    detector = make_detector(args.ckpt, device)
    print(f"model: {describe_model(detector.hparams)}\nckpt:  {detector.info['ckpt_path']} "
          f"(epoch {detector.info['epoch']}) on {device} in {time.perf_counter() - t0:.1f}s")
    if getattr(detector, "note", ""):
        print(detector.note)
    if callable(heat_desc):
        heat_desc = heat_desc(detector)
    aug_block = detector.hparams.get("online_augment")
    if not aug_block:
        with open(args.training_config) as f:
            aug_block = yaml.safe_load(f)["online_augment"]
        print(f"checkpoint has no online_augment block: plans drawn from {args.training_config}")
    augmenter, aug_cfg = build_online_augmenter(aug_block)
    aug_seed = aug_cfg.seed if args.aug_seed is None else args.aug_seed
    print(f"augmentations: {augmenter.describe()}, plan seed {aug_seed}, identity plans re-drawn")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem, (real_path, ai_path) in pairs:
        originals = [Image.open(p).convert("RGB") for p in (real_path, ai_path)]
        names, sizes0 = [real_path.name, ai_path.name], [img.size for img in originals]
        variants = [("clean", [])] + [(f"aug{k}", draw_plan(augmenter, stem, k, aug_seed)) for k in range(1, args.n_aug + 1)]
        print(f"\n{stem}   real {sizes0[0][0]}x{sizes0[0][1]}   AI {sizes0[1][0]}x{sizes0[1][1]}")
        for variant, plan in variants:
            images = [augmenter.apply(img, plan) for img in originals] if plan else originals
            items = detector.run(images)
            fig = pair_figure(items, names, sizes0, stem, variant, plan, detector, args, heat_name, heat_desc)
            fig.savefig(args.out / f"{stem}__{variant}.png", dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            line = f"  {variant:<6} {plan_label(plan)[:70]:<70}"
            for it, label in zip(items, LABELS):
                t, r, c, s = top_patches(it["attn"], it["hp"], it["wp"], 1)[0]
                b_mass, b_share = border_mass(it["attn"], it["hp"], it["wp"], getattr(detector, "ring", 0))
                line += f" | {label:<4} p(AI) {it['prob']:.3f} grid {it['hp']:>2}x{it['wp']:<2} ring {b_mass:>4.0%}"
                rows.append({
                    "stem": stem, "variant": variant, "plan": plan_label(plan), "label": label, "file": names[LABELS.index(label)],
                    "width": it["width"], "height": it["height"], "grid_h": it["hp"], "grid_w": it["wp"],
                    "p_ai": f"{it['prob']:.6f}", "attn_max": f"{it['attn'].max():.6f}", "top_token": t,
                    "top_row": r, "top_col": c, "border_mass": f"{b_mass:.4f}", "border_share": f"{b_share:.4f}",
                    **it.get("extra", {}),
                })
            print(line)
    detector.close()

    with open(args.out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for label in LABELS:
        sel = [r for r in rows if r["label"] == label]
        print(f"\n{label}: mean p(AI) {np.mean([float(r['p_ai']) for r in sel]):.3f} over {len(sel)} images; "
              f"attention on {getattr(detector, 'ring_label', 'the outer ring')} "
              f"{np.mean([float(r['border_mass']) for r in sel]):.1%} "
              f"(ring = {np.mean([float(r['border_share']) for r in sel]):.1%} of patches)")
    print(f"recomputation check (recomputed attention vs the module's own output): max rel err {detector.worst_err:.1e}")
    print(f"wrote {len(pairs) * (1 + args.n_aug)} figures + summary.csv to {args.out}")
    return rows


def main():
    run_pairs(build_parser().parse_args(), Detector)
    return 0


if __name__ == "__main__":
    sys.exit(main())
