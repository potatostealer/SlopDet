#!/usr/bin/env python3
"""Profile image file formats across the dataset splits under --root (default: data/).

Companion to profile_dims.py: every sub-directory of the root is treated as one
split (real_all_train, ai_gen_all_train, real_all_val, ai_gen_all_val, ...). For
each split it reports the file extensions, the container format PIL actually
detects from the bytes (JPEG, PNG, WEBP, ...), the colour modes, any files whose
extension disagrees with their real format, a format x mode crosstab and file
size percentiles per format, then a compact table comparing all splits side by
side.

Only the image headers are parsed (PIL's lazy open, no pixel decode), so a full
pass over ~175k files takes well under a minute on local disk with a few dozen
workers.

Run from the repo root:
    python profile_format.py                       # all splits, all files
    python profile_format.py --sample 2000         # quick look: 2000 files per split
    python profile_format.py --splits real_all_val ai_gen_all_val
    python profile_format.py --csv formats.csv     # also dump one row per image
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

DEFAULT_ROOT = "data"   # dataset root: every sub-directory is one split
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
# The format PIL is expected to detect for each extension (extension -> format mismatches are reported).
EXT_FORMAT = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP", ".bmp": "BMP",
              ".gif": "GIF", ".tif": "TIFF", ".tiff": "TIFF"}
PERCENTILES = (0, 5, 25, 50, 75, 95, 100)

# We only read headers, so a huge image is worth reporting, not refusing.
Image.MAX_IMAGE_PIXELS = None


def read_header(path):
    """Return (path, ext, format, mode, bytes, error) without decoding pixels."""
    ext = os.path.splitext(path)[1].lower()
    try:
        size = os.path.getsize(path)
        with Image.open(path) as image:
            return path, ext, image.format or "?", image.mode, size, None
    except Exception as exc:  # noqa: BLE001 - any unreadable file is a data point
        return path, ext, None, None, None, f"{type(exc).__name__}: {exc}"


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


def counter_line(label, counter, n):
    return f"{label:<12}" + "  ".join(f"{k} {v} ({v / n:.1%})" for k, v in counter.most_common())


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

    n = len(rows)
    exts = Counter(ext for _, ext, _, _, _ in rows)
    formats = Counter(fmt for _, _, fmt, _, _ in rows)
    modes = Counter(mode for _, _, _, mode, _ in rows)
    print(counter_line("extensions:", exts, n))
    print(counter_line("formats:", formats, n))
    print(counter_line("modes:", modes, n))

    # Files whose extension lies about the container format (e.g. a PNG saved as .jpg).
    mismatches = [(path, ext, fmt) for path, ext, fmt, _, _ in rows if EXT_FORMAT.get(ext) != fmt]
    print()
    print(f"extension/format mismatches: {len(mismatches)} ({len(mismatches) / n:.1%})")
    if mismatches:
        pairs = Counter((ext, fmt) for _, ext, fmt in mismatches)
        for (ext, fmt), count in pairs.most_common():
            print(f"  {ext} is really {fmt}: {count}")
        for path, ext, fmt in mismatches[:top]:
            print(f"  e.g. {Path(path).name} -> {fmt}")
        if len(mismatches) > top:
            print(f"  ... and {len(mismatches) - top} more")

    # format x mode crosstab
    combos = Counter((fmt, mode) for _, _, fmt, mode, _ in rows)
    print()
    print(f"{'format':<8}{'mode':<8}{'count':>9}{'share':>8}")
    for (fmt, mode), count in combos.most_common():
        print(f"{fmt:<8}{mode:<8}{count:>9}{count / n:>8.1%}")

    # file size percentiles per format (KB)
    sizes = defaultdict(list)
    for _, _, fmt, _, size in rows:
        sizes[fmt].append(size / 1024)
    print()
    print(f"{'size KB':<12}{'mean':>9}" + "".join(
        f"{'p' + str(p) if 0 < p < 100 else ('min' if p == 0 else 'max'):>9}" for p in PERCENTILES))
    for fmt, _ in formats.most_common():
        values = sorted(sizes[fmt])
        mean = sum(values) / len(values)
        print(f"{fmt:<12}{mean:>9.1f}" + "".join(f"{percentile(values, p):>9.1f}" for p in PERCENTILES))
    all_sizes = sorted(size / 1024 for _, _, _, _, size in rows)
    total_mb = sum(all_sizes) / 1024
    print(f"{'all':<12}{sum(all_sizes) / n:>9.1f}"
          + "".join(f"{percentile(all_sizes, p):>9.1f}" for p in PERCENTILES)
          + f"   total {total_mb:,.0f} MB")

    return {
        "split": name,
        "n": n,
        "bad": len(errors),
        "formats": formats,
        "mismatch": len(mismatches),
        "mode": "{} ({:.0%})".format(modes.most_common(1)[0][0], modes.most_common(1)[0][1] / n),
        "med_kb": percentile(all_sizes, 50),
        "total_mb": total_mb,
    }


def print_comparison(summaries):
    print()
    print("=" * 78)
    print("ALL SPLITS")
    print("=" * 78)
    total = Counter()
    for s in summaries:
        total.update(s["formats"])
    format_columns = [fmt for fmt, _ in total.most_common()]
    print(f"{'split':<26}{'images':>8}{'bad':>5}"
          + "".join(f"{fmt:>8}" for fmt in format_columns)
          + f"{'mismatch':>10}{'med KB':>9}{'total MB':>10}  top mode")
    for s in summaries:
        print(f"{s['split']:<26}{s['n']:>8}{s['bad']:>5}"
              + "".join(f"{s['formats'].get(fmt, 0) / s['n']:>8.1%}" for fmt in format_columns)
              + f"{s['mismatch']:>10}{s['med_kb']:>9.1f}{s['total_mb']:>10,.0f}  {s['mode']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--splits", nargs="*", default=None,
                        help="sub-directories of root to profile (default: all of them)")
    parser.add_argument("--sample", type=int, default=None,
                        help="profile only this many evenly spaced files per split")
    parser.add_argument("-j", "--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--top", type=int, default=10, help="how many mismatched files to name per split")
    parser.add_argument("--csv", default=None, help="write one row per image (split,path,ext,format,mode,bytes) here")
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
        writer.writerow(["split", "path", "ext", "format", "mode", "bytes"])

    summaries = []
    with Pool(args.workers) as pool:
        for split in splits:
            start = time.perf_counter()
            paths = list_images(root / split, args.sample)
            rows, errors = [], []
            for path, ext, fmt, mode, size, error in pool.imap_unordered(read_header, paths, chunksize=256):
                if error:
                    errors.append((path, error))
                    continue
                rows.append((path, ext, fmt, mode, size))
                if writer:
                    writer.writerow([split, path, ext, fmt, mode, size])
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
