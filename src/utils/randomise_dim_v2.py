#!/usr/bin/env python
"""Build a dimension-matched AI-gen set with a less lossy crop than randomise_dim.py.

As in randomise_dim.py, the (width, height) of every image in --real-dir gives the
empirical joint distribution of dimensions, --num images are drawn (without
replacement) from --src-dir, and each is converted to a (width, height) sampled
from that distribution and saved under its original filename in --out-dir.

Each sampled (w, h) is also jittered: every side is redrawn from a Gaussian centred
on it with std --sigma (15), unless either side is smaller than 4 * sigma, in which
case the pair is used exactly as drawn.

What differs is the conversion. v1 resizes the image to cover the target box and
random-crops it, which can discard most of a picture when the aspect ratios
disagree (a 3:4 portrait squeezed into 16:9 keeps 42% of its pixels). Here:

  1. Orientation. The target aspect ratio r = w / h and its transpose 1 / r are
     both candidates for the crop rectangle. For each, a centred rectangle of
     that shape is grown until it touches the source border, giving the largest
     centred crop of that orientation and the fraction of the source it covers.
     Orientations whose largest crop covers less than --min-cover (50%) are
     rejected and one of the remaining ones is chosen at random. If neither
     reaches --min-cover, a new target size is sampled (up to --max-resample
     times); if that still fails, the least lossy of the tried (target,
     orientation) pairs is used at its maximum coverage and the image is
     counted as "under-covered" in the summary.
  2. Coverage. The centred rectangle is shrunk to cover a fraction of the source
     drawn uniformly from [--min-cover, max coverage].
  3. Scale. The crop is resized (up or down) to exactly the sampled (w, h). When
     the transposed orientation was used (a portrait crop for a landscape target
     or vice versa) the crop is first rotated by 90 degrees so that its long
     side lines up with the target's; --transposed swap instead keeps the pixels
     upright and outputs (h, w).

The only directories written to are --out-dir and, with --save-samples K,
--sample-dir, which receives K before/after pairs plus a side-by-side comparison
with the crop box drawn on the source, for a sanity check. The run is fully
deterministic for a given --seed (a smaller --num is a prefix of a larger one),
and outputs that already exist are skipped, so an interrupted run can simply be
re-launched.
"""

import argparse
import math
import os
import random
import shutil
import warnings
from collections import Counter
from multiprocessing import Pool
from typing import NamedTuple, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

DATASETS = "data"   # dataset root
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# A few real images exceed PIL's 89 MP "decompression bomb" threshold; they are
# legitimate, so keep the limit (errors at 2x) but do not spam warnings.
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


class Job(NamedTuple):
    src: str
    dst: str
    box: Tuple[int, int, int, int]  # centred crop in source pixels (l, t, r, b)
    rotation: Optional[int]  # Image.Transpose member applied after the crop, if any
    size: Tuple[int, int]  # final output (width, height)
    # Bookkeeping for the summary and the sanity-check samples.
    src_size: Tuple[int, int]
    target: Tuple[int, int]  # sampled (width, height), after Gaussian jitter
    transpose: bool  # crop has the transposed target aspect ratio
    cover: float  # fraction of the source area inside `box`
    feasible: bool  # False if no orientation could reach min_cover
    resamples: int  # target sizes discarded before this one


def list_images(directory):
    not_tampered = sorted(
        f for f in os.listdir(directory) if (
            os.path.splitext(f)[1].lower() in IMG_EXTS
            and "tampered" not in f.lower()  # exclude the tampered images in the real set
        )
    )
    # remove images with size < 32 x 32
    return [f for f in not_tampered if read_size(os.path.join(directory, f)) is not None and min(read_size(os.path.join(directory, f))) > 32]


def read_size(path):
    """(width, height) from the header only; None if the file cannot be opened."""
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Crop planning (all randomness lives here, in the main process)
# --------------------------------------------------------------------------- #


def max_coverage(sw, sh, aspect):
    """Fraction of a (sw, sh) image covered by the largest centred rectangle of
    width / height == aspect that fits inside it (i.e. grown until it touches
    the border)."""
    s = sw / sh
    return min(s / aspect, aspect / s)


def plan_crop(src_size, target, rng, min_cover):
    """Choose the crop orientation and coverage for one image.

    Returns (transpose, cover, feasible): `transpose` is True if the crop uses
    the transposed target aspect ratio, `cover` is the fraction of the source it
    should cover and `feasible` whether that fraction reaches min_cover.
    """
    sw, sh = src_size
    tw, th = target
    r = tw / th
    # Both orientations coincide for a square target: do not rotate for nothing.
    aspects = [(False, r)] if tw == th else [(False, r), (True, 1 / r)]
    options = [(t, max_coverage(sw, sh, a)) for t, a in aspects]
    ok = [o for o in options if o[1] >= min_cover]
    if ok:
        transpose, cmax = rng.choice(ok)
        return transpose, rng.uniform(min_cover, cmax), True
    transpose, cmax = max(options, key=lambda o: o[1])
    return transpose, cmax, False


def centred_box(src_size, aspect, cover):
    """Centred (l, t, r, b) box with width / height == aspect covering `cover`
    of the source area (clamped to the source, so at most 1 px off in aspect)."""
    sw, sh = src_size
    cw = min(sw, max(1, round(math.sqrt(cover * sw * sh * aspect))))
    ch = min(sh, max(1, round(cw / aspect)))
    x = (sw - cw) // 2
    y = (sh - ch) // 2
    return (x, y, x + cw, y + ch)


def jitter_target(target, rng, sigma):
    """Sample each of (w, h) from a Gaussian centred on it with std `sigma`.
    Left exactly as drawn if either side is smaller than 4 * sigma (or sigma is 0)."""
    if sigma <= 0 or min(target) < 4 * sigma:
        return target
    return tuple(max(1, round(rng.gauss(v, sigma))) for v in target)


def plan_job(src, dst, src_size, dims, rng, args):
    resamples = 0
    best = None  # least lossy of the rejected candidates, used if none is feasible
    while True:
        target = jitter_target(rng.choice(dims), rng, args.sigma)
        transpose, cover, feasible = plan_crop(src_size, target, rng, args.min_cover)
        if feasible:
            break
        if best is None or cover > best[1]:
            best = (transpose, cover, target)
        if resamples >= args.max_resample:
            transpose, cover, target = best
            break
        resamples += 1

    tw, th = target
    aspect = th / tw if transpose else tw / th
    box = centred_box(src_size, aspect, cover)

    rotation, size = None, (tw, th)
    if transpose:
        if args.transposed == "rotate":
            rotation = rng.choice((Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270))
        else:
            size = (th, tw)
    return Job(
        src, dst, box, rotation, size, src_size, target, transpose, cover, feasible, resamples
    )


def scale_of(job):
    """Linear scale factor from the (rotated) crop to the output."""
    cw, ch = job.box[2] - job.box[0], job.box[3] - job.box[1]
    if job.rotation is not None:
        cw, ch = ch, cw
    return job.size[0] / cw


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #


def convert(im, job):
    """Crop, rotate and scale one loaded image according to `job`."""
    # Palette / 1-bit images are always resized with NEAREST by PIL; promote them.
    if im.mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
    elif im.mode == "1":
        im = im.convert("L")
    out = im.crop(job.box)
    if job.rotation is not None:
        out = out.transpose(job.rotation)
    if out.size != job.size:
        out = out.resize(job.size, Image.LANCZOS)
    return out


def process(job):
    """Worker: returns None on success, else an error string."""
    try:
        with Image.open(job.src) as im:
            im.load()
            out = convert(im, job)
        if os.path.splitext(job.dst)[1].lower() in (".jpg", ".jpeg"):
            if out.mode not in ("RGB", "L"):
                out = out.convert("RGB")
            out.save(job.dst, quality=95)
        else:
            out.save(job.dst)
        return None
    except Exception as e:  # noqa: BLE001 - report and keep going
        return f"{os.path.basename(job.src)}: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Reporting and sanity-check samples
# --------------------------------------------------------------------------- #


def quantiles(xs, fmt=str):
    xs = sorted(xs)
    n = len(xs)
    q = lambda p: xs[min(n - 1, int(p * n))]  # noqa: E731
    return f"min={fmt(xs[0])} p10={fmt(q(.1))} med={fmt(q(.5))} p90={fmt(q(.9))} max={fmt(xs[-1])}"


def describe(dims, label):
    print(f"{label}: n={len(dims)}, unique (w,h) pairs={len(set(dims))}")
    print(f"  width : {quantiles([w for w, _ in dims])}")
    print(f"  height: {quantiles([h for _, h in dims])}")
    top = ", ".join(f"{w}x{h}: {c}" for (w, h), c in Counter(dims).most_common(5))
    print(f"  most common: {top}")


def summarise(jobs, args):
    describe([j.target for j in jobs], "Sampled target sizes")
    if args.transposed == "swap":
        describe([j.size for j in jobs], "Output sizes (targets swapped where transposed)")
    n = len(jobs)
    n_t = sum(j.transpose for j in jobs)
    n_r = sum(j.resamples > 0 for j in jobs)
    n_u = sum(not j.feasible for j in jobs)
    scales = [scale_of(j) for j in jobs]
    n_up = sum(s > 1 for s in scales)
    print(
        f"Crops: n={n}, transposed={n_t} ({n_t / n:.1%}), target resampled={n_r}, "
        f"under-covered (<{args.min_cover:.0%})={n_u}"
    )
    print(f"  coverage: {quantiles([j.cover for j in jobs], lambda v: f'{v:.1%}')}")
    print(f"  scale   : {quantiles(scales, lambda v: f'{v:.2f}x')}, upscaled={n_up} ({n_up / n:.1%})")


def _font():
    try:
        return ImageFont.load_default(size=15)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


PANEL_H = 480


def comparison(job):
    """Source with the crop box drawn, next to the output, with an annotation."""
    with Image.open(job.src) as im:
        src = im.convert("RGB")
    with Image.open(job.dst) as im:
        out = im.convert("RGB")

    def fit(im):
        s = PANEL_H / im.height
        return im.resize((max(1, round(im.width * s)), PANEL_H), Image.BILINEAR, reducing_gap=2.0), s

    left, s = fit(src)
    right, _ = fit(out)
    l, t, r, b = job.box
    ImageDraw.Draw(left).rectangle(
        (l * s, t * s, r * s - 1, b * s - 1), outline=(255, 0, 0), width=3
    )

    gap, header = 16, 48
    canvas = Image.new("RGB", (left.width + gap + right.width, header + PANEL_H), (32, 32, 32))
    canvas.paste(left, (0, header))
    canvas.paste(right, (left.width + gap, header))

    cw, ch = r - l, b - t
    if job.rotation is None:
        rot = "no rotation"
    else:
        rot = "rotated 90 " + ("cw" if job.rotation == Image.Transpose.ROTATE_270 else "ccw")
    notes = []
    if job.resamples:
        notes.append(f"target resampled {job.resamples}x")
    if not job.feasible:
        notes.append("UNDER-COVERED")
    text = (
        f"{os.path.basename(job.src)}\n"
        f"src {job.src_size[0]}x{job.src_size[1]}  ->  crop {cw}x{ch} at ({l},{t}) "
        f"cover {job.cover:.1%}  ->  {rot}  ->  out {job.size[0]}x{job.size[1]} "
        f"(target {job.target[0]}x{job.target[1]}, scale {scale_of(job):.2f}x)"
        + (f"   [{', '.join(notes)}]" if notes else "")
    )
    ImageDraw.Draw(canvas).multiline_text((8, 6), text, fill=(255, 255, 255), font=_font())
    return canvas


def save_samples(jobs, sample_dir, k):
    """Copy K before/after pairs and write a comparison image for each."""
    os.makedirs(sample_dir, exist_ok=True)
    picked = [j for j in jobs if os.path.exists(j.dst)][:k]
    for i, job in enumerate(picked):
        stem, ext = os.path.splitext(os.path.basename(job.src))
        prefix = os.path.join(sample_dir, f"{i:03d}_{stem}")
        shutil.copy2(job.src, f"{prefix}_before{ext}")
        shutil.copy2(job.dst, f"{prefix}_after{ext}")
        comparison(job).save(f"{prefix}_compare.jpg", quality=90)
    print(f"{len(picked)} before/after samples written to {sample_dir}")


# --------------------------------------------------------------------------- #


def scan_sizes(paths, workers, desc):
    with Pool(workers) as pool:
        return list(tqdm(pool.imap(read_size, paths, chunksize=64), total=len(paths), desc=desc))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--real-dir", default=f"{DATASETS}/real_outofdimsample_val",
                    help="images whose (width, height) distribution is matched")
    ap.add_argument("--src-dir", default=f"{DATASETS}/ai_gen_all_val",
                    help="images that are cropped and rescaled")
    ap.add_argument("--out-dir", default=f"{DATASETS}/ai_gen_outofsample_v2_val")
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    ap.add_argument("--sigma", type=float, default=15.0,
                    help="std of the Gaussian jitter applied to each sampled target side; skipped "
                         "when either side is < 4*sigma (0 disables)")
    ap.add_argument("--min-cover", type=float, default=0.5,
                    help="minimum fraction of the source area a crop must keep (default 0.5)")
    ap.add_argument("--max-resample", type=int, default=0,
                    help="target sizes to retry when no orientation reaches --min-cover; "
                         "0 keeps every first draw (exact target distribution) and accepts "
                         "under-covered crops instead")
    ap.add_argument("--transposed", choices=("rotate", "swap"), default="rotate",
                    help="when the crop uses the transposed target aspect: 'rotate' the crop "
                         "90 degrees and output exactly (w, h) [default], or 'swap' and output "
                         "(h, w) with the pixels left upright")
    ap.add_argument("--save-samples", type=int, default=0, metavar="K",
                    help="also write K before/after pairs and comparison images to --sample-dir")
    ap.add_argument("--sample-dir", default="logs/random_samples")
    args = ap.parse_args()
    if not 0 < args.min_cover <= 1:
        raise SystemExit("--min-cover must be in (0, 1]")

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # 1. Empirical joint (w, h) distribution of the real set.
    real_files = list_images(args.real_dir)
    sizes = scan_sizes([os.path.join(args.real_dir, f) for f in real_files], args.workers,
                       "Scanning real dims")
    dims = [s for s in sizes if s is not None]
    if len(dims) < len(sizes):
        print(f"WARNING: {len(sizes) - len(dims)} real images could not be read")
    if not dims:
        raise SystemExit(f"No readable images in {args.real_dir}")
    describe(dims, "Real (target) distribution")

    # 2. Sample source images. Shuffle the full list once, then take a prefix, so
    #    the choice for a small --num is a prefix of the choice for a larger one.
    src_files = list_images(args.src_dir)
    if args.num > len(src_files):
        # warn and then concat the list to itself until it is long enough, so that the prefix is still a prefix of a larger --num
        print(f"WARNING: --num {args.num} > {len(src_files)} images in {args.src_dir}")
        while len(src_files) < args.num:
            src_files.extend(src_files)
    rng.shuffle(src_files)
    chosen = src_files[: args.num]

    # 3. Plan a crop for every chosen image; this needs its size (header only).
    src_sizes = scan_sizes([os.path.join(args.src_dir, f) for f in chosen], args.workers,
                           "Scanning source dims")
    errors = []
    jobs = []
    for f, size in zip(chosen, src_sizes):
        if size is None:
            errors.append(f"{f}: could not read image header")
            continue
        jobs.append(plan_job(os.path.join(args.src_dir, f), os.path.join(args.out_dir, f),
                             size, dims, rng, args))
    summarise(jobs, args)

    todo = [j for j in jobs if not os.path.exists(j.dst)]
    print(f"{len(jobs) - len(todo)} outputs already exist; processing {len(todo)}")

    # 4. Crop + rotate + resize + save.
    with Pool(args.workers) as pool:
        for err in tqdm(pool.imap_unordered(process, todo, chunksize=4), total=len(todo),
                        desc="Crop+resize"):
            if err:
                errors.append(err)

    if errors:
        print(f"{len(errors)} images failed:")
        for e in errors:
            print("  " + e)
    n_out = len(list_images(args.out_dir))
    print(f"Done: {len(todo) - len(errors)} written, {n_out} images now in {args.out_dir}")

    # 5. Optional sanity-check samples.
    if args.save_samples > 0:
        save_samples(jobs, args.sample_dir, args.save_samples)


if __name__ == "__main__":
    main()
