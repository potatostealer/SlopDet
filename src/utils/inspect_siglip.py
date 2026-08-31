#!/usr/bin/env python3
"""How the training data path and the SigLIP2 NaFlex processor handle images of varying resolution.

What the code says (src/experiments/train_aug_ddp.py -> train_aug_single.build_aug_loaders ->
src/dataset/{dataloader,collate,online_augment}.py, and in transformers 5.15
models/siglip2/{image_processing_siglip2,modeling_siglip2}.py, image_processing_backends.py):

  1. Dataset side. BinaryImageDataset.__getitem__ returns (path, label) only -- no decode, no resize, so the
     DataLoader never has to stack images of different sizes. Everything happens in the collate_fn, i.e. in the
     DataLoader workers: Siglip2Collate opens each file with PIL and .convert("RGB"); Siglip2AugmentCollate first
     applies OnlineAugmenter to the PIL image (crop shrinks the canvas to 80% per side, resize goes down and back
     UP to the original size, jpeg / blur / noise / jitter keep the size), and both then call
     Siglip2ImageProcessor(images=<list of PIL>, return_tensors="pt"). Under DDP the sampler is swapped for a
     DistributedSampler, which changes which files a rank gets, not how they are processed.

  2. Processor side (NaFlex). Every image is handled INDEPENDENTLY, with no batch-wide target size and no crop:
       a. get_image_size_for_max_num_patches binary-searches ONE scale s such that
              H' = max(16, ceil(s*H/16)*16),  W' = max(16, ceil(s*W/16)*16)
          has (H'/16)*(W'/16) <= max_num_patches = 256 patches, and takes the largest such s. So the image is
          scaled -- DOWN if it is big, UP if it is small -- to the largest multiple-of-16 canvas with at most
          256 patches; the aspect ratio is preserved up to the ceil-to-16 rounding of each side.
       b. torchvision bilinear resize with antialias=True on the uint8 (3, H, W) tensor, then the fused
          rescale + normalise: (x - 127.5) / 127.5  (mean = std = 0.5 after /255).
       c. patchify: (3, H', W') -> (h_p * w_p, 16 * 16 * 3 = 768), rows in raster order over the patch grid,
          each row ordered (patch_row, patch_col, channel).
       d. zero-pad the rows to exactly 256; pixel_attention_mask is 1 for the h_p * w_p real rows and 0 for the
          padding (always at the end); spatial_shapes = (h_p, w_p).
     => whatever the input resolutions, a batch is pixel_values (B, 256, 768), pixel_attention_mask (B, 256),
        spatial_shapes (B, 2). Fixed shapes, so no bucketing / per-batch resizing is needed and an image's
        tensors do not depend on what else is in its batch.

  3. Model side. Siglip2VisionEmbeddings applies a Linear(768 -> 1152) to each row and adds the learned 16x16
     position grid, bilinearly interpolated (antialias) to each image's (h_p, w_p) and padded with its first
     entry. pixel_attention_mask becomes a bidirectional attention mask, so real tokens never attend to padding.
     The padded rows still come out of last_hidden_state (as garbage), and LoraQFormerDetector passes
     pixel_attention_mask.bool() -- True = real token -- as the QFormer key_padding_mask, which is exactly the
     torch scaled_dot_product_attention boolean convention (True = may attend), so the head ignores them too.

What this script checks, on the files sample_dims.py put in dim_samples/ (and on buckets.csv for the whole set):
  [A] the resize rule: for every sample, the predicted canvas / patch grid / padding / scale factors / aspect
      distortion, and two independent properties of the grid (<= 256 patches, and maximal: one more patch on
      the tighter side would overflow), compared with the processor's actual spatial_shapes and mask.
  [B] batch independence: each image processed alone == the same image inside the full / reversed batch.
  [C] reconstruction: un-patchify + de-normalise pixel_values and compare with a direct torchvision resize of
      the original; the canvases the model actually sees are saved to dim_samples/_siglip_view/.
  [D] the real collates: Siglip2Collate == the processor; Siglip2AugmentCollate (deterministic val seed):
      which augmentation each sample got, its size after it, and whether the patch grid changed.
  [E] the model (Siglip2VisionModel, fp32): resized position embeddings; an image alone vs inside a
      mixed-resolution batch; and random noise written into the padded rows must leave the real tokens and the
      pooled output unchanged (it only changes the padded rows themselves).
  [F] dataset-wide, from buckets.csv (formula only): how many images are up / down-scaled, by how much, and how
      many of the 256 tokens they use; plus --also sizes such as the 200x200 real thumbnails.

Run from the repo root (after sample_dims.py):
    python inspect_siglip.py                                  # all sections, model on the freest GPU
    python inspect_siglip.py --device cpu --limit 6           # smoke test
    python inspect_siglip.py --skip-model --also 200x200 1920x1080 100x100
"""

import argparse
import csv
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms.v2 import functional as tvF
from transformers import Siglip2ImageProcessor
from transformers.models.siglip2.image_processing_siglip2 import get_image_size_for_max_num_patches

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # `python inspect_siglip.py` finds src/ like `python -m` would
from src.dataset.collate import Siglip2AugmentCollate, Siglip2Collate  # noqa: E402
from src.dataset.online_augment import build_online_augmenter, plan_label, val_rng  # noqa: E402

DEFAULT_SAMPLES = HERE / "dim_samples"
DEFAULT_TRAINING_CFG = HERE / "src" / "configs" / "training.yml"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# ----------------------------------------------------------------------------- bookkeeping
class Checks:
    def __init__(self):
        self.results = []

    def __call__(self, ok, label, detail=""):
        ok = bool(ok)
        self.results.append((ok, label))
        print(f"  [{'OK' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
        return ok

    def summary(self):
        failed = [label for ok, label in self.results if not ok]
        print("\n" + "=" * 100)
        print(f"{len(self.results) - len(failed)}/{len(self.results)} checks passed")
        for label in failed:
            print(f"  FAILED: {label}")
        return not failed


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ----------------------------------------------------------------------------- the resize rule
def analyse(width, height, patch=16, max_patches=256):
    """What the processor does to a W x H image, plus the derived numbers we want to look at."""
    new_h, new_w = get_image_size_for_max_num_patches(height, width, patch, max_patches)
    hp, wp = new_h // patch, new_w // patch
    n = hp * wp
    return {
        "width": width, "height": height, "new_w": new_w, "new_h": new_h, "hp": hp, "wp": wp,
        "patches": n, "pad": max_patches - n,
        "scale_w": new_w / width, "scale_h": new_h / height,
        "scale": math.sqrt(new_w * new_h / (width * height)),  # linear scale, geometric mean of the sides
        "aspect_distortion": (new_w / new_h) / (width / height) - 1,  # 0 = aspect ratio kept exactly
    }


def grid_at(scale, width, height, patch=16):
    return max(1, math.ceil(height * scale / patch)), max(1, math.ceil(width * scale / patch))


def grid_is_maximal(width, height, hp, wp, patch=16, max_patches=256):
    """Independent of the binary search: the grid fits, and the smallest larger scale (the one at which the
    tighter side gains a patch) overflows max_patches -- so no aspect-preserving canvas with more patches exists."""
    if hp * wp > max_patches:
        return False
    next_scale = min(hp * patch / height, wp * patch / width) * (1 + 1e-9) + 1e-12
    nh, nw = grid_at(next_scale, width, height, patch)
    return nh * nw > max_patches


def direction(row):
    area = row["scale"]
    return "up" if area > 1.0005 else "down" if area < 0.9995 else "same"


def print_rows(rows, counts=None):
    print(f"{'resolution':>11} {'n':>4}  {'-> canvas':>11} {'grid h x w':>10} {'patches':>8} {'pad':>4} "
          f"{'scale w':>8} {'scale h':>8} {'dir':>5} {'aspect':>8}")
    for row in rows:
        n = counts.get((row["width"], row["height"]), "") if counts else ""
        print(f"{row['width']:>5}x{row['height']:<5} {n:>4}  {row['new_w']:>5}x{row['new_h']:<5} "
              f"{row['hp']:>4} x {row['wp']:<3} {row['patches']:>8} {row['pad']:>4} "
              f"{row['scale_w']:>8.3f} {row['scale_h']:>8.3f} {direction(row):>5} {row['aspect_distortion']:>+8.2%}")


# ----------------------------------------------------------------------------- tensors <-> images
def unpatchify(rows, hp, wp, patch=16, channels=3):
    """Inverse of convert_image_to_patches: (hp*wp, patch*patch*channels) -> (channels, hp*patch, wp*patch)."""
    x = rows.reshape(hp, wp, patch, patch, channels)  # (hp, wp, py, px, c)
    return x.permute(4, 0, 2, 1, 3).reshape(channels, hp * patch, wp * patch)


def denormalise(x):
    """(x - 127.5) / 127.5 undone, back to uint8."""
    return (x * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)


def reference_resize(img, new_h, new_w):
    """The processor's own resize call, applied directly: torchvision bilinear + antialias on the uint8 tensor."""
    return tvF.resize(tvF.pil_to_tensor(img), [new_h, new_w], interpolation=tvF.InterpolationMode.BILINEAR,
                      antialias=True)


# ----------------------------------------------------------------------------- inputs
def list_samples(samples_dir, limit=None):
    paths = sorted(p for p in samples_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"no sample images in {samples_dir}; run sample_dims.py first")
    return paths[:limit] if limit else paths


def load_buckets(samples_dir):
    path = samples_dir / "buckets.csv"
    if not path.is_file():
        return None
    with open(path, newline="") as f:
        return [(int(r["width"]), int(r["height"]), int(r["count"])) for r in csv.DictReader(f)]


def parse_size(text):
    w, h = text.lower().split("x")
    return int(w), int(h)


def pick_device(spec):
    if spec != "auto":
        return torch.device(spec)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{max(free)[1]}")


# ----------------------------------------------------------------------------- sections
def section_a(paths, images, processor, enc, checks):
    section("[A] the resize rule, predicted from the formula vs what the processor did")
    max_patches, patch = processor.max_num_patches, processor.patch_size
    per_res, counts = {}, Counter()
    for img in images:
        w, h = img.size
        counts[(w, h)] += 1
        per_res.setdefault((w, h), analyse(w, h, patch, max_patches))
    rows = [per_res[k] for k in sorted(per_res, key=lambda k: -counts[k])]
    print(f"patch_size={patch} max_num_patches={max_patches} resample={processor.resample} "
          f"(bilinear, antialias) mean={processor.image_mean} std={processor.image_std}")
    print(f"{len(images)} samples, {len(rows)} distinct resolutions\n")
    print_rows(rows, counts)

    pv, mask, shapes = enc["pixel_values"], enc["pixel_attention_mask"], enc["spatial_shapes"]
    print()
    checks(tuple(pv.shape) == (len(images), max_patches, patch * patch * 3), "pixel_values is (B, 256, 768) for every batch",
           f"{tuple(pv.shape)}")
    checks(tuple(mask.shape) == (len(images), max_patches) and tuple(shapes.shape) == (len(images), 2),
           "pixel_attention_mask (B, 256), spatial_shapes (B, 2)", f"{tuple(mask.shape)}, {tuple(shapes.shape)}")
    ok_pred = ok_mask = ok_tail = ok_fit = ok_max = True
    for i, img in enumerate(images):
        w, h = img.size
        row = per_res[(w, h)]
        hp, wp = shapes[i].tolist()
        n = int(mask[i].sum())
        ok_pred &= (hp, wp) == (row["hp"], row["wp"])
        ok_mask &= n == hp * wp
        ok_tail &= bool(mask[i, :n].all()) and not bool(mask[i, n:].any()) and not bool(pv[i, n:].any())
        ok_fit &= hp * wp <= max_patches
        ok_max &= grid_is_maximal(w, h, hp, wp, patch, max_patches)
    checks(ok_pred, "spatial_shapes == get_image_size_for_max_num_patches(H, W) // 16 for every sample")
    checks(ok_mask, "mask.sum() == h_p * w_p for every sample")
    checks(ok_tail, "mask is 1 for the first h_p*w_p rows, 0 after; padded rows of pixel_values are all zero")
    checks(ok_fit, "every grid has <= 256 patches")
    checks(ok_max, "every grid is maximal: one more patch on the tighter side would exceed 256")
    return per_res


def section_b(images, processor, enc, checks):
    section("[B] batch independence: alone vs in the batch vs in the reversed batch")
    ok_alone, worst = True, 0.0
    for i, img in enumerate(images):
        one = processor(images=[img], return_tensors="pt")
        same = (torch.equal(one["pixel_values"][0], enc["pixel_values"][i])
                and torch.equal(one["pixel_attention_mask"][0], enc["pixel_attention_mask"][i])
                and torch.equal(one["spatial_shapes"][0], enc["spatial_shapes"][i]))
        worst = max(worst, float((one["pixel_values"][0] - enc["pixel_values"][i]).abs().max()))
        ok_alone &= same
    checks(ok_alone, f"each of the {len(images)} images processed alone is bit-identical to its row in the batch",
           f"max |diff| {worst:.2e}")
    rev = processor(images=images[::-1], return_tensors="pt")
    checks(torch.equal(rev["pixel_values"].flip(0), enc["pixel_values"])
           and torch.equal(rev["spatial_shapes"].flip(0), enc["spatial_shapes"]),
           "reversing the batch order only permutes the rows")


def section_c(paths, images, enc, per_res, views_dir, checks):
    section("[C] reconstruction: un-patchify + de-normalise == a direct torchvision resize of the original")
    if views_dir:
        views_dir.mkdir(parents=True, exist_ok=True)
    worst, ok = 0, True
    for i, (path, img) in enumerate(zip(paths, images)):
        hp, wp = enc["spatial_shapes"][i].tolist()
        canvas = denormalise(unpatchify(enc["pixel_values"][i, : hp * wp], hp, wp))
        ref = reference_resize(img, hp * 16, wp * 16)
        diff = int((canvas.int() - ref.int()).abs().max())
        worst = max(worst, diff)
        ok &= diff == 0
        if views_dir:
            out = views_dir / f"{path.stem}__seen_as_{wp * 16}x{hp * 16}_grid{hp}x{wp}.png"
            tvF.to_pil_image(canvas).save(out)
    checks(ok, "un-patchified pixel_values reproduce tvF.resize(uint8, bilinear, antialias) exactly",
           f"max abs pixel diff {worst}")
    if views_dir:
        print(f"  the canvases the model sees are in {views_dir}/ (<sample>__seen_as_<W'>x<H'>_grid<h_p>x<w_p>.png)")


def section_d(paths, images, enc, processor, checkpoint, training_cfg, checks):
    section("[D] the collates used by train_aug_ddp.py (src/dataset/collate.py)")
    batch = [(str(p), 1) for p in paths]
    plain = Siglip2Collate(checkpoint)(batch)
    checks(torch.equal(plain["pixel_values"], enc["pixel_values"])
           and torch.equal(plain["pixel_attention_mask"], enc["pixel_attention_mask"])
           and torch.equal(plain["spatial_shapes"], enc["spatial_shapes"])
           and plain["labels"].dtype == torch.float32,
           "Siglip2Collate(paths) == processor(PIL RGB images): open + convert('RGB') + processor, nothing else")

    aug_block = training_cfg.get("online_augment")
    if not aug_block:
        print("  training.yml has no online_augment block: augment collate skipped")
        return
    augmenter, aug_cfg = build_online_augmenter(aug_block)
    print(f"  online augment: {augmenter.describe()}; val_mode={aug_cfg.val_mode} seed={aug_cfg.seed}")
    print("  deterministic val collate (plan keyed on blake2b(seed:stem), same as val in training):")
    collate = Siglip2AugmentCollate(checkpoint, augmenter, deterministic_seed=aug_cfg.seed)
    out = collate(batch)
    ok_grid, changed = True, 0
    print(f"  {'sample':<42} {'original':>10} {'augmented':>10} {'grid':>8} {'clean grid':>10}  plan")
    for i, (path, img) in enumerate(zip(paths, images)):
        plan = augmenter.sample_plan(val_rng(path, aug_cfg.seed))  # the same draw the collate made
        aug_img = augmenter.apply(img, plan)
        expected = analyse(*aug_img.size, processor.patch_size, processor.max_num_patches)
        hp, wp = out["spatial_shapes"][i].tolist()
        chp, cwp = enc["spatial_shapes"][i].tolist()
        ok_grid &= (hp, wp) == (expected["hp"], expected["wp"])
        changed += (hp, wp) != (chp, cwp)
        name = path.name if len(path.name) <= 40 else path.name[:37] + "..."
        print(f"  {name:<42} {f'{img.size[0]}x{img.size[1]}':>10} {f'{aug_img.size[0]}x{aug_img.size[1]}':>10} "
              f"{f'{hp}x{wp}':>8} {f'{chp}x{cwp}':>10}  {plan_label(plan)}")
    checks(ok_grid, "augment collate grids == formula applied to the augmented (post-crop) size")
    print(f"  grid changed by the augmentation for {changed}/{len(paths)} samples "
          f"(crop keeps the aspect ratio, so the canvas is the same and only the up/down-scale factor changes; "
          f"resize goes back to the original size)")


def section_e(images, enc, checkpoint, device, batch_size, checks):
    section(f"[E] Siglip2VisionModel on {device} (fp32): position embeddings, batching, padding")
    import transformers
    from transformers import Siglip2VisionModel

    transformers.logging.set_verbosity_error()  # the checkpoint also holds the text tower: silence the "UNEXPECTED keys" report
    t0 = time.perf_counter()
    model = Siglip2VisionModel.from_pretrained(checkpoint, attn_implementation="sdpa").eval().to(device)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s; hidden={model.config.hidden_size} "
          f"num_patches={model.config.num_patches} -> position grid "
          f"{model.embeddings.position_embedding_size}x{model.embeddings.position_embedding_size}")
    pv = enc["pixel_values"].to(device)
    mask = enc["pixel_attention_mask"].to(device)
    shapes = enc["spatial_shapes"].to(device)

    # position embeddings
    emb = model.embeddings
    grid = emb.position_embedding.weight.reshape(emb.position_embedding_size, emb.position_embedding_size, -1)
    with torch.no_grad():
        pos = emb.resize_positional_embeddings(grid, shapes, max_length=pv.shape[1])
    checks(tuple(pos.shape) == (len(images), pv.shape[1], model.config.hidden_size),
           "resized position embeddings are (B, 256, hidden)", f"{tuple(pos.shape)}")
    ok_full = ok_pad = True
    full_idx = [i for i in range(len(images)) if tuple(shapes[i].tolist()) == (16, 16)]
    part_idx = [i for i in range(len(images)) if i not in full_idx]
    for i in full_idx:
        ok_full &= torch.allclose(pos[i], emb.position_embedding.weight, atol=1e-5)
    for i in part_idx:
        n = int(shapes[i].prod())
        ok_pad &= torch.equal(pos[i, n:], pos[i, 0].expand(pv.shape[1] - n, -1))
    if full_idx:
        checks(ok_full, f"a 16x16 grid gets the stored position embeddings unchanged ({len(full_idx)} samples)")
    if part_idx:
        checks(ok_pad, f"padded positions get the interpolated grid's first entry ({len(part_idx)} samples)")

    def forward(pv_, mask_, shapes_):
        outs = []
        with torch.no_grad():
            for s in range(0, pv_.shape[0], batch_size):
                o = model(pixel_values=pv_[s:s + batch_size], pixel_attention_mask=mask_[s:s + batch_size],
                          spatial_shapes=shapes_[s:s + batch_size])
                outs.append((o.last_hidden_state, o.pooler_output))
        return torch.cat([h for h, _ in outs]), torch.cat([p for _, p in outs])

    t0 = time.perf_counter()
    hidden, pooled = forward(pv, mask, shapes)
    print(f"  batch forward: last_hidden_state {tuple(hidden.shape)} pooler_output {tuple(pooled.shape)} "
          f"in {time.perf_counter() - t0:.1f}s")

    # alone vs in a mixed-resolution batch (first sample of each distinct grid)
    seen, probe = set(), []
    for i in range(len(images)):
        key = tuple(shapes[i].tolist())
        if key not in seen:
            seen.add(key)
            probe.append(i)
    worst_h = worst_p = 0.0
    for i in probe:
        h1, p1 = forward(pv[i:i + 1], mask[i:i + 1], shapes[i:i + 1])
        n = int(shapes[i].prod())
        worst_h = max(worst_h, float((h1[0, :n] - hidden[i, :n]).abs().max() / hidden[i, :n].abs().max()))
        worst_p = max(worst_p, float((p1[0] - pooled[i]).abs().max() / pooled[i].abs().max()))
    checks(worst_h < 1e-3 and worst_p < 1e-3,
           f"real-token features and pooled output of an image alone == inside the batch ({len(probe)} grids probed)",
           f"max rel diff hidden {worst_h:.1e}, pooled {worst_p:.1e}")

    # padding immunity: noise into the padded rows
    if part_idx:
        noisy = pv.clone()
        gen = torch.Generator(device="cpu").manual_seed(0)
        for i in part_idx:
            n = int(shapes[i].prod())
            noisy[i, n:] = torch.randn(pv.shape[1] - n, pv.shape[2], generator=gen).to(device)
        hidden_n, pooled_n = forward(noisy, mask, shapes)
        worst_real = worst_pool = 0.0
        pad_moved = 0
        for i in part_idx:
            n = int(shapes[i].prod())
            worst_real = max(worst_real, float((hidden_n[i, :n] - hidden[i, :n]).abs().max() / hidden[i, :n].abs().max()))
            worst_pool = max(worst_pool, float((pooled_n[i] - pooled[i]).abs().max() / pooled[i].abs().max()))
            pad_moved += int(not torch.allclose(hidden_n[i, n:], hidden[i, n:], rtol=1e-3, atol=1e-3))
        checks(worst_real < 1e-4 and worst_pool < 1e-4,
               f"random noise in the padded rows leaves real tokens and pooled output unchanged ({len(part_idx)} padded samples)",
               f"max rel diff real tokens {worst_real:.1e}, pooled {worst_pool:.1e}")
        checks(pad_moved == len(part_idx),
               "... while the padded rows' own outputs do change (they are garbage the mask must hide)",
               f"{pad_moved}/{len(part_idx)}")
    else:
        print("  every sample fills all 256 patches: no padded rows to test")

    # the QFormer's key_padding_mask convention (src/modules/attention.py passes pixel_attention_mask.bool())
    q = torch.randn(1, 2, 3, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    keep = torch.tensor([True, True, True, False, False])
    masked = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[None, None, None, :])
    only_real = F.scaled_dot_product_attention(q, k[:, :, :3], v[:, :, :3])
    checks(torch.allclose(masked, only_real, atol=1e-6),
           "SDPA boolean mask: True = attend, so key_padding_mask=pixel_attention_mask.bool() drops the padding in the QFormer")


def section_f(buckets, also, patch, max_patches):
    section("[F] dataset-wide, formula only")
    if buckets:
        total = sum(c for _, _, c in buckets)
        rows = [(analyse(w, h, patch, max_patches), c) for w, h, c in buckets]
        dirs = Counter()
        grids, patches = Counter(), Counter()
        for row, c in rows:
            dirs[direction(row)] += c
            grids[(row["hp"], row["wp"])] += c
            patches[row["patches"]] += c
        print(f"buckets.csv: {total} images, {len(buckets)} distinct resolutions")
        print("  scaling:   " + "   ".join(f"{k} {v / total:.1%}" for k, v in dirs.most_common()))
        srt = sorted(((row["scale"], c) for row, c in rows))
        acc, out = 0, {}
        for p in (5, 25, 50, 75, 95):
            target = p / 100 * total
            acc = 0
            for s, c in srt:
                acc += c
                if acc >= target:
                    out[p] = s
                    break
        print("  linear scale factor (side' / side): " + "  ".join(f"p{p} {out[p]:.3f}" for p in sorted(out))
              + f"  min {srt[0][0]:.3f}  max {srt[-1][0]:.3f}")
        full = patches.get(max_patches, 0)
        others = sorted(((c, n) for n, c in patches.items() if n != max_patches), reverse=True)
        print(f"  tokens used: all {max_patches}: {full / total:.1%}; "
              + "  ".join(f"{n}: {c / total:.1%}" for c, n in others[:8])
              + (f"  (+{len(others) - 8} rarer counts)" if len(others) > 8 else ""))
        print("  most common patch grids (h_p x w_p):")
        for (hp, wp), c in grids.most_common(8):
            print(f"    {hp:>2} x {wp:<2}  {c:>7}  {c / total:6.1%}")
        extremes = sorted(rows, key=lambda rc: rc[0]["scale"])
        print("  extremes of the scale factor (3 smallest, 3 largest):")
        print_rows([r for r, _ in extremes[:3]] + [r for r, _ in extremes[-3:]], {(w, h): c for w, h, c in buckets})
    else:
        print("  no buckets.csv next to the samples (older sample_dims.py run?): dataset-wide part skipped")
    if also:
        print("\n--also sizes (no files needed):")
        print_rows([analyse(w, h, patch, max_patches) for w, h in also])


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES, help="dir written by sample_dims.py")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CFG,
                        help="training.yml: model.checkpoint_path and the online_augment block")
    parser.add_argument("--checkpoint", default=None, help="override model.checkpoint_path")
    parser.add_argument("--limit", type=int, default=None, help="only the first N sample files")
    parser.add_argument("--device", default="auto", help="cpu | cuda:N | auto (the CUDA device with most free memory)")
    parser.add_argument("--batch-size", type=int, default=16, help="model forward chunk size")
    parser.add_argument("--skip-model", action="store_true", help="skip section [E] (no 1.7 GB model load)")
    parser.add_argument("--no-views", action="store_true", help="do not write dim_samples/_siglip_view/")
    parser.add_argument("--also", nargs="*", default=["200x200"], type=parse_size,
                        help="extra WxH sizes for the formula-only table (default: the 200x200 real thumbnails)")
    args = parser.parse_args()

    with open(args.config) as f:
        training_cfg = yaml.safe_load(f)
    checkpoint = args.checkpoint or training_cfg["model"]["checkpoint_path"]
    checks = Checks()

    paths = list_samples(args.samples, args.limit)
    images = [Image.open(p).convert("RGB") for p in paths]  # exactly what Siglip2Collate does
    processor = Siglip2ImageProcessor.from_pretrained(checkpoint)
    print(f"samples: {len(paths)} files in {args.samples}   processor: {checkpoint}")
    t0 = time.perf_counter()
    enc = processor(images=images, return_tensors="pt")
    print(f"processor(images=<{len(images)} PIL>) took {time.perf_counter() - t0:.2f}s")

    per_res = section_a(paths, images, processor, enc, checks)
    section_b(images, processor, enc, checks)
    section_c(paths, images, enc, per_res, None if args.no_views else args.samples / "_siglip_view", checks)
    section_d(paths, images, enc, processor, checkpoint, training_cfg, checks)
    if not args.skip_model:
        section_e(images, enc, checkpoint, pick_device(args.device), args.batch_size, checks)
    section_f(load_buckets(args.samples), args.also, processor.patch_size, processor.max_num_patches)
    return 0 if checks.summary() else 1


if __name__ == "__main__":
    sys.exit(main())
