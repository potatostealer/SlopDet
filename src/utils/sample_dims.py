#!/usr/bin/env python3
"""Sample a few images from each common resolution of one dataset directory.

Companion to profile_dims.py: the same header-only scan (PIL lazy open, a
multiprocessing pool) buckets every image of one flat directory by its exact
(width, height). Buckets are ranked by how many images they hold, the most
frequent ones are kept until they cover --coverage (default 90%) of the images,
and K (default 2) images are drawn from each kept bucket and copied
byte-for-byte (no re-encoding, so a sample is exactly what the training collate
would open) into --out (default dim_samples/ next to this script).

Samples are named  r<rank>_<W>x<H>_<i>_<original name>  so `ls` sorts them by
bucket frequency. Next to them:
    manifest.csv   one row per sample: rank, width, height, count, share, cum_share, sample_index, file, source
    buckets.csv    one row per distinct resolution of the whole directory (rank, width, height, count, kept)

"The first 90% of distinct resolutions" is read as the head of the
frequency-ranked list that covers 90% of the images (--coverage-of images);
--coverage-of buckets instead keeps the first 90% of the distinct resolutions
themselves (a long tail of near-singletons, i.e. many more samples).

Run from the repo root:
    python sample_dims.py                          # 2 per bucket, buckets covering 90% of the images
    python sample_dims.py -k 3 --coverage 0.95
    python sample_dims.py --dir data/real_train --out dim_samples_real
    python sample_dims.py --dry-run                # print the buckets and the plan, copy nothing
    python sample_dims.py --clean                  # empty --out first (stale samples of an earlier run)
"""

import argparse
import csv
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from utils.profile_dims import list_images, read_header  # noqa: E402  the same header-only scan

DEFAULT_DIR = "data/ai_gen_train"   # directory whose resolutions are profiled
DEFAULT_OUT = HERE / "dim_samples"


def scan(paths, workers):
    """{(width, height): sorted paths}, [(path, error)] without decoding any pixels."""
    buckets, errors = defaultdict(list), []
    with Pool(workers) as pool:
        for path, w, h, _fmt, _mode, error in pool.imap_unordered(read_header, paths, chunksize=256):
            if error:
                errors.append((path, error))
            else:
                buckets[(w, h)].append(path)
    for bucket in buckets.values():
        bucket.sort()
    return buckets, errors


def rank_buckets(buckets):
    """Most frequent first; ties broken by (width, height) so the order is deterministic."""
    return sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0]))


def keep_head(ranked, total, coverage, coverage_of):
    """The leading buckets that make up `coverage` of the images (the bucket crossing
    the threshold is included) or of the distinct resolutions."""
    if coverage_of == "buckets":
        return ranked[: max(1, math.ceil(coverage * len(ranked)))]
    kept, seen = [], 0
    for res, paths in ranked:
        if seen >= coverage * total:
            break
        kept.append((res, paths))
        seen += len(paths)
    return kept


def clean_dir(out):
    entries = list(out.iterdir())
    for entry in entries:
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    print(f"--clean: removed {len(entries)} entries from {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=DEFAULT_DIR, help="flat image directory to sample from")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where the samples, manifest.csv and buckets.csv go")
    parser.add_argument("-k", "--per-bucket", type=int, default=2, help="images copied from each kept bucket")
    parser.add_argument("--coverage", type=float, default=0.9,
                        help="keep the most frequent resolutions until they cover this share (default 0.9)")
    parser.add_argument("--coverage-of", choices=("images", "buckets"), default="images",
                        help="share of the images (default) or of the distinct resolutions")
    parser.add_argument("--seed", type=int, default=0, help="which images are drawn from each bucket")
    parser.add_argument("--sample", type=int, default=None,
                        help="scan only this many evenly spaced files (quick look, as in profile_dims.py)")
    parser.add_argument("-j", "--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--clean", action="store_true", help="delete everything already in --out first")
    parser.add_argument("--dry-run", action="store_true", help="print the buckets and the plan, write nothing")
    args = parser.parse_args()
    if not 0 < args.coverage <= 1:
        parser.error("--coverage must be in (0, 1]")
    if args.per_bucket < 1:
        parser.error("--per-bucket must be >= 1")

    src_dir = Path(args.dir)
    if not src_dir.is_dir():
        raise SystemExit(f"not a directory: {src_dir}")
    paths = list_images(src_dir, args.sample)
    if not paths:
        raise SystemExit(f"no images in {src_dir}")
    print(f"dir: {src_dir}   files: {len(paths)}   workers: {args.workers}"
          + (f"   sample: {args.sample}" if args.sample else ""))

    start = time.perf_counter()
    buckets, errors = scan(paths, args.workers)
    total = sum(len(b) for b in buckets.values())
    print(f"read {total} headers ({len(errors)} unreadable) in {time.perf_counter() - start:.1f}s: "
          f"{len(buckets)} distinct resolutions")
    for path, error in errors[:5]:
        print(f"  unreadable: {Path(path).name}: {error}")
    if len(errors) > 5:
        print(f"  ... and {len(errors) - 5} more")

    ranked = rank_buckets(buckets)
    kept = keep_head(ranked, total, args.coverage, args.coverage_of)
    covered = sum(len(b) for _, b in kept)
    print(f"\nkeeping the {len(kept)} most frequent resolutions = {covered / total:.1%} of the images "
          f"(--coverage {args.coverage:g} of {args.coverage_of}); "
          f"tail: {len(ranked) - len(kept)} resolutions, {total - covered} images ({(total - covered) / total:.1%})")

    rng = random.Random(args.seed)
    rank_width = max(2, len(str(len(kept))))
    rows, cum = [], 0  # manifest rows
    print(f"\n{'rank':>4} {'resolution':>11} {'count':>7} {'share':>7} {'cum':>7}  samples")
    for rank, ((w, h), bucket) in enumerate(kept, start=1):
        cum += len(bucket)
        chosen = rng.sample(bucket, min(args.per_bucket, len(bucket)))
        names = []
        for i, src in enumerate(chosen):
            name = f"r{rank:0{rank_width}d}_{w}x{h}_{i}_{Path(src).name}"
            names.append(Path(src).name)
            rows.append({
                "rank": rank, "width": w, "height": h, "count": len(bucket),
                "share": f"{len(bucket) / total:.6f}", "cum_share": f"{cum / total:.6f}",
                "sample_index": i, "file": name, "source": src,
            })
        print(f"{rank:>4} {f'{w}x{h}':>11} {len(bucket):>7} {len(bucket) / total:>7.2%} {cum / total:>7.2%}  "
              + "  ".join(names))

    if args.dry_run:
        print(f"\n--dry-run: would write {len(rows)} samples + manifest.csv + buckets.csv to {args.out}")
        return 0

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_dir(out)
    elif any(out.iterdir()):
        print(f"\nnote: {out} is not empty; existing files are kept (use --clean to start from scratch)")

    for row in rows:
        shutil.copyfile(row["source"], out / row["file"])
    with open(out / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    kept_set = {res for res, _ in kept}
    with open(out / "buckets.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "width", "height", "count", "kept"])
        for rank, ((w, h), bucket) in enumerate(ranked, start=1):
            writer.writerow([rank, w, h, len(bucket), int((w, h) in kept_set)])
    print(f"\nwrote {len(rows)} samples from {len(kept)} resolutions + manifest.csv + buckets.csv "
          f"({len(ranked)} resolutions) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
