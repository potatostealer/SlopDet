#!/usr/bin/env python3
"""Profile image dimensions across the dataset splits under --root (default: data/).

Every sub-directory of the root is treated as one split (real_all_train,
ai_gen_all_train, real_all_val, ai_gen_all_val). For each split it reports the
file formats / colour modes, width / height / short-side / long-side / aspect
ratio percentiles, orientation shares and the most common resolutions, then a
compact table comparing all splits side by side.

Only the image headers are parsed (PIL's lazy open, no pixel decode), so a full
pass over ~175k files takes well under a minute on local disk with a few dozen
workers.

Run from the repo root:
    python profile_dims.py                       # all splits, all files
    python profile_dims.py --sample 2000         # quick look: 2000 files per split
    python profile_dims.py --splits real_all_val ai_gen_all_val
    python profile_dims.py --csv dims.csv        # also dump one row per image
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

DEFAULT_ROOT = "data"   # dataset root: every sub-directory is one split
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
PERCENTILES = (0, 5, 25, 50, 75, 95, 100)

# We only read headers, so a huge image is worth reporting, not refusing.
Image.MAX_IMAGE_PIXELS = None


def read_header(path):
    """Return (path, width, height, format, mode, error) without decoding pixels."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            return path, width, height, image.format or "?", image.mode, None
    except Exception as exc:  # noqa: BLE001 - any unreadable file is a data point
        return path, None, None, None, None, f"{type(exc).__name__}: {exc}"


def list_images(split_dir, sample=None):
    paths = sorted(
        entry.path for entry in os.scandir(split_dir)
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTS
    )
    if sample and sample < len(paths):
        step = len(paths) / sample  # evenly spaced, deterministic subset
        paths = [paths[int(i * step)] for i in range(sample)]
    return paths


def percentile(sorted_values, pct):
    if not sorted_values:
        return float("nan")
    index = (len(sorted_values) - 1) * pct / 100
    low, high = int(index), min(int(index) + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (index - low)


def summarise(name, rows, errors, top=10):
    print()
    print("=" * 78)
    print(f"{name}   {len(rows)} images   {len(errors)} unreadable")
    print("=" * 78)
    if errors:
        for path, error in errors[:5]:
            print(f"  unreadable: {Path(path).name}: {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    if not rows:
        return None

    formats = Counter(fmt for _, _, _, fmt, _ in rows)
    modes = Counter(mode for _, _, _, _, mode in rows)
    print("formats: " + "  ".join(f"{k} {v}" for k, v in formats.most_common()))
    print("modes:   " + "  ".join(f"{k} {v}" for k, v in modes.most_common()))

    widths = sorted(w for _, w, _, _, _ in rows)
    heights = sorted(h for _, _, h, _, _ in rows)
    shorts = sorted(min(w, h) for _, w, h, _, _ in rows)
    longs = sorted(max(w, h) for _, w, h, _, _ in rows)
    aspects = sorted(w / h for _, w, h, _, _ in rows)
    pixels = sorted(w * h / 1e6 for _, w, h, _, _ in rows)

    print()
    header = f"{'':<12}{'mean':>9}" + "".join(f"{'p' + str(p) if 0 < p < 100 else ('min' if p == 0 else 'max'):>9}" for p in PERCENTILES)
    print(header)
    for label, values, fmt in (
        ("width", widths, "{:9.0f}"),
        ("height", heights, "{:9.0f}"),
        ("short side", shorts, "{:9.0f}"),
        ("long side", longs, "{:9.0f}"),
        ("aspect w/h", aspects, "{:9.3f}"),
        ("megapixels", pixels, "{:9.2f}"),
    ):
        mean = sum(values) / len(values)
        print(f"{label:<12}" + fmt.format(mean) + "".join(fmt.format(percentile(values, p)) for p in PERCENTILES))

    n = len(rows)
    landscape = sum(1 for _, w, h, _, _ in rows if w > h)
    portrait = sum(1 for _, w, h, _, _ in rows if w < h)
    square = n - landscape - portrait
    print()
    print(f"orientation: landscape {landscape / n:6.1%}   portrait {portrait / n:6.1%}   square {square / n:6.1%}")

    resolutions = Counter((w, h) for _, w, h, _, _ in rows)
    print(f"distinct resolutions: {len(resolutions)}   top {min(top, len(resolutions))}:")
    for (w, h), count in resolutions.most_common(top):
        print(f"  {w:>5}x{h:<5} {count:>7}  {count / n:6.1%}")

    return {
        "split": name,
        "n": n,
        "bad": len(errors),
        "w_med": percentile(widths, 50),
        "h_med": percentile(heights, 50),
        "short_min": shorts[0],
        "short_med": percentile(shorts, 50),
        "long_max": longs[-1],
        "square": square / n,
        "distinct": len(resolutions),
        "top": "{}x{} ({:.0%})".format(*resolutions.most_common(1)[0][0], resolutions.most_common(1)[0][1] / n),
    }


def print_comparison(summaries):
    print()
    print("=" * 78)
    print("ALL SPLITS")
    print("=" * 78)
    print(f"{'split':<18}{'images':>8}{'bad':>5}{'med w':>8}{'med h':>8}{'min short':>11}"
          f"{'med short':>11}{'max long':>10}{'square':>8}{'#res':>7}  most common")
    for s in summaries:
        print(f"{s['split']:<18}{s['n']:>8}{s['bad']:>5}{s['w_med']:>8.0f}{s['h_med']:>8.0f}{s['short_min']:>11}"
              f"{s['short_med']:>11.0f}{s['long_max']:>10}{s['square']:>8.1%}{s['distinct']:>7}  {s['top']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--splits", nargs="*", default=None,
                        help="sub-directories of root to profile (default: all of them)")
    parser.add_argument("--sample", type=int, default=None,
                        help="profile only this many evenly spaced files per split")
    parser.add_argument("-j", "--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--top", type=int, default=10, help="how many resolutions to list per split")
    parser.add_argument("--csv", default=None, help="write one row per image (split,path,width,height,format,mode) here")
    args = parser.parse_args()

    root = Path(args.root)
    splits = args.splits or sorted(p.name for p in root.iterdir() if p.is_dir())
    missing = [s for s in splits if not (root / s).is_dir()]
    if missing:
        raise SystemExit(f"no such split directory under {root}: {missing}")
    print(f"root: {root}   splits: {splits}   workers: {args.workers}"
          + (f"   sample: {args.sample}/split" if args.sample else ""))

    writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["split", "path", "width", "height", "format", "mode"])

    summaries = []
    with Pool(args.workers) as pool:
        for split in splits:
            start = time.perf_counter()
            paths = list_images(root / split, args.sample)
            rows, errors = [], []
            for path, w, h, fmt, mode, error in pool.imap_unordered(read_header, paths, chunksize=256):
                if error:
                    errors.append((path, error))
                    continue
                rows.append((path, w, h, fmt, mode))
                if writer:
                    writer.writerow([split, path, w, h, fmt, mode])
            rows.sort()
            print(f"\n[{split}] read {len(paths)} headers in {time.perf_counter() - start:.1f}s", end="")
            summary = summarise(split, rows, errors, args.top)
            if summary:
                summaries.append(summary)

    if writer:
        csv_file.close()
        print(f"\nwrote {args.csv}")
    if len(summaries) > 1:
        print_comparison(summaries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
