#!/usr/bin/env python3
"""Sample Open Images photos whose resolution is far from the real training set's.

Builds an out-of-dimension validation set: photos from the Open Images V7 parquet
shards on NFS (--parquet-dir: 1.9M JPEGs in 1,206 shards, one `image` struct
column of {bytes, path}) whose (width, height) lies where the real training set
--ref-dir (real_all_toremove_train) has (almost) no images.

Criterion.  Two resolutions are "within tol" of each other when both sides
differ by at most a factor of 2**tol (Chebyshev distance in log2 space;
tol 0.2 = +-15 %). A candidate QUALIFIES when at most --max-ref-share of the
reference images (default 0.5 % = ~345 of 69k) lie within --tol of its
resolution. Its `radius` is the half-width at which that neighbourhood first
holds more than --max-ref-share of the reference (larger = further from the
training profile; qualifying <=> radius > tol). The reference is ~80 % long
side 1024 (1024x768 21 %, 1024x683 13 %, 768x1024, 1024x1024, ...) plus 9 %
200x200 COCO thumbnails, and the parquet photos are ~99 % long side 1024 as
well, so the defaults keep the ~1 % that is genuinely elsewhere: panoramas and
tall crops at 1024 (w/h >= 2 or <= 0.5), 768x768 squares, the few full-size
originals (2560..4608 px) -- about 22k of the 1.9M before the exclusions below.

Two stages, one pass over NFS.
  1. scan    Every shard is read once (header-only parse of each JPEG, no pixel
             decode; NFS-bound at ~140 MB/s, so the full pass takes ~70 min
             whatever -j). One CSV per shard (<scan-dir>/shards/<shard>.csv,
             dims of every image) is written atomically, so a killed run resumes
             where it stopped and nothing is scanned twice. The bytes of every
             image passing a LOOSER pre-filter (at most --pool-share of the
             reference within --pool-tol; default 1 % within +-7 %, ~4 % of the
             photos = ~80k files / ~23 GB) are stashed in <scan-dir>/pool/ so
             that stage 2 never touches the parquet again (a chosen image would
             otherwise cost its whole ~34 MB row group, i.e. a second full pass).
  2. select  Apply the real criterion (--tol / --max-ref-share), drop every stem
             already used by a dataset under --exclude-root (real
             AND ai_gen: the AI images were generated from these very photos and
             share their stems), draw --n images at random (--seed) or the
             farthest ones (--pick farthest), and print the reference /
             candidate / selected resolution profiles plus a (tol, share) grid
             of how many images every other setting would give. Dry run by
             default: manifest.csv + summary.json go to --scan-dir and nothing
             is written to --out. With --execute the chosen files are copied
             byte-for-byte from the pool into --out (<stem>.jpg, atomic, header
             re-verified; files already there are kept, so re-running is safe).

Run from the repo root:
    python sample_outofdim.py --max-shards 6 --n 50 --scan-dir /scratch/s --out /scratch/o   # smoke test, dry run
    python sample_outofdim.py --max-shards 6 --n 50 --scan-dir /scratch/s --out /scratch/o --execute
    python sample_outofdim.py                          # full scan (~70 min, resumable) + plan for 20k
    python sample_outofdim.py --execute                # copy the 20k into real_outofdimsample_val
    python sample_outofdim.py --tol 0.3 --max-ref-share 0.01     # another criterion: seconds, no rescan
A criterion outside the pool pre-filter (tol < --pool-tol or max-ref-share >
--pool-share) needs a rescan with a wider pool: pass wider --pool-tol /
--pool-share and a fresh --scan-dir.
"""

import argparse
import csv
import io
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from utils.profile_dims import PERCENTILES, list_images, percentile, read_header  # noqa: E402  same header-only scan

DEFAULT_REF = "data/real_train"   # the split whose resolution profile is avoided
DEFAULT_PARQUET = "/path/to/open-images-v7-subset-splits/data"   # the Open Images parquet shards
DEFAULT_OUT = "data/real_outofdim_val"
DEFAULT_EXCLUDE_ROOT = "data"   # every dataset directory in here is excluded by stem
DEFAULT_SCAN_DIR = "/path/to/outofdim_scan"   # the ~23 GB candidate pool lives here: use a large disk
SPLITS = ("train", "validation", "test")
SCAN_FIELDS = ["split", "shard", "row_group", "row", "stem", "ext", "width", "height", "format", "nbytes", "pooled", "error"]
TOL_GRID = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
SHARE_GRID = (0.001, 0.002, 0.005, 0.01, 0.02)
AUG_SUFFIX = re.compile(r"_a\d+$")   # <stem>_a<k>.png in the *_with_aug_test directories
LONG_ORDER = ("long<512", "512<=long<1024", "long=1024", "1024<long<2048", "long>=2048")
ASPECT_ORDER = ("w/h<=0.5", "portrait", "square", "landscape", "w/h>=2")
KEY = 1 << 20   # (width, height) -> width * KEY + height, one integer per resolution

Image.MAX_IMAGE_PIXELS = None


# ----------------------------------------------------------------------------- resolutions

def long_bucket(w, h):
    longest = max(w, h)
    if longest < 512:
        return LONG_ORDER[0]
    if longest < 1024:
        return LONG_ORDER[1]
    if longest == 1024:
        return LONG_ORDER[2]
    if longest < 2048:
        return LONG_ORDER[3]
    return LONG_ORDER[4]


def aspect_bucket(w, h):
    aspect = w / h
    if aspect <= 0.5:
        return ASPECT_ORDER[0]
    if aspect < 1:
        return ASPECT_ORDER[1]
    if aspect == 1:
        return ASPECT_ORDER[2]
    if aspect < 2:
        return ASPECT_ORDER[3]
    return ASPECT_ORDER[4]


def describe(title, counter, top=8):
    """Print the resolution profile of {(w, h): count} the way profile_dims.py does and return it as a dict."""
    n = sum(counter.values())
    print(f"\n{title}: {n} images, {len(counter)} distinct resolutions")
    if not n:
        return {"images": 0, "resolutions": 0}
    items = list(counter.items())

    def expanded(fn):
        return sorted(v for (w, h), c in items for v in [fn(w, h)] * c)

    print(f"{'':<12}{'mean':>9}"
          + "".join(f"{'p' + str(p) if 0 < p < 100 else ('min' if p == 0 else 'max'):>9}" for p in PERCENTILES))
    for label, fn, fmt in (
        ("width", lambda w, h: w, "{:9.0f}"),
        ("height", lambda w, h: h, "{:9.0f}"),
        ("short side", min, "{:9.0f}"),
        ("long side", max, "{:9.0f}"),
        ("aspect w/h", lambda w, h: w / h, "{:9.3f}"),
        ("megapixels", lambda w, h: w * h / 1e6, "{:9.2f}"),
    ):
        values = expanded(fn)
        print(f"{label:<12}" + fmt.format(sum(values) / n) + "".join(fmt.format(percentile(values, p)) for p in PERCENTILES))
    longs, aspects = Counter(), Counter()
    for (w, h), c in items:
        longs[long_bucket(w, h)] += c
        aspects[aspect_bucket(w, h)] += c
    print("long side:  " + "  ".join(f"{b} {longs[b] / n:.1%}" for b in LONG_ORDER))
    print("aspect:     " + "  ".join(f"{b} {aspects[b] / n:.1%}" for b in ASPECT_ORDER))
    print(f"top {min(top, len(counter))}:      " + "  ".join(f"{w}x{h} {c / n:.1%}" for (w, h), c in counter.most_common(top)))
    return {
        "images": n, "resolutions": len(counter),
        "long_side": {b: longs[b] for b in LONG_ORDER}, "aspect": {b: aspects[b] for b in ASPECT_ORDER},
        "top": [[w, h, c] for (w, h), c in counter.most_common(top)],
    }


class Reference:
    """The distinct resolutions of the reference directory with their counts, in log2 space."""

    def __init__(self, counter):
        self.counter = counter
        self.total = sum(counter.values())
        keys = sorted(counter)
        self.log_w = np.log2([w for w, _ in keys])
        self.log_h = np.log2([h for _, h in keys])
        self.counts = np.array([counter[k] for k in keys], dtype=np.float64)

    def distances(self, log_w, log_h):
        """(candidates x reference) Chebyshev distance in log2 space."""
        return np.maximum(np.abs(log_w[:, None] - self.log_w[None, :]), np.abs(log_h[:, None] - self.log_h[None, :]))

    def share_within(self, tol, log_w, log_h):
        """Share of the reference images within tol of each candidate."""
        return ((self.distances(log_w, log_h) <= tol).astype(np.float64) @ self.counts) / self.total

    def profile(self, log_w, log_h, tols, max_share, chunk=1024):
        """Per candidate: the reference share within each tol of `tols`, and the radius at which
        the neighbourhood first holds more than max_share of the reference."""
        n = len(log_w)
        shares, radius = np.empty((n, len(tols))), np.empty(n)
        for start in range(0, n, chunk):
            block = slice(start, start + chunk)
            d = self.distances(log_w[block], log_h[block])
            for t, tol in enumerate(tols):
                shares[block, t] = ((d <= tol).astype(np.float64) @ self.counts) / self.total
            order = np.argsort(d, axis=1, kind="stable")
            cumulative = np.cumsum(self.counts[order], axis=1) / self.total
            first = (cumulative > max_share).argmax(axis=1)   # exists: the full cumulative share is 1
            radius[block] = np.take_along_axis(d, order, axis=1)[np.arange(d.shape[0]), first]
        return shares, radius


def reference_profile(ref_dir, workers):
    paths = list_images(ref_dir)
    if not paths:
        raise SystemExit(f"no images in {ref_dir}")
    counter, bad = Counter(), 0
    with Pool(workers) as pool:
        for _, w, h, _, _, error in pool.imap_unordered(read_header, paths, chunksize=256):
            if error:
                bad += 1
            else:
                counter[(w, h)] += 1
    return counter, bad


# ----------------------------------------------------------------------------- stage 1: scan

_WORKER = {}


def _init_scan_worker(reference, pool_tol, pool_share, shards_dir, pool_dir):
    _WORKER.update(reference=reference, pool_tol=pool_tol, pool_share=pool_share,
                   shards_dir=shards_dir, pool_dir=pool_dir, cache={})


def _pooled(w, h):
    """Does resolution (w, h) pass the pool pre-filter? Memoised per worker process."""
    cache = _WORKER["cache"]
    if (w, h) not in cache:
        share = _WORKER["reference"].share_within(_WORKER["pool_tol"], np.log2([float(w)]), np.log2([float(h)]))[0]
        cache[(w, h)] = bool(share <= _WORKER["pool_share"])
    return cache[(w, h)]


def read_header_bytes(data):
    """(width, height, format, error) of an encoded image without decoding its pixels."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            w, h = image.size
            return w, h, image.format or "?", None
    except Exception as exc:  # noqa: BLE001 - any unreadable image is a data point
        return None, None, None, f"{type(exc).__name__}: {exc}"


def row_group_images(parquet_file, row_group):
    """([bytes], [path]) of one row group of the `image` struct column."""
    image = parquet_file.read_row_group(row_group, columns=["image"]).column("image").combine_chunks()
    return image.field("bytes").to_pylist(), image.field("path").to_pylist()


def write_atomic(target, data):
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, target)


def scan_shard(shard_path):
    """Header-parse every image of one shard into <shards_dir>/<shard>.csv and stash the pre-filtered
    ones in the pool. Returns (shard, images, unreadable, pooled, bytes read, seconds, error)."""
    import pyarrow.parquet as pq

    shard, split = shard_path.stem, shard_path.stem.split("-")[0]
    target = _WORKER["shards_dir"] / f"{shard}.csv"
    tmp = target.with_suffix(".csv.tmp")
    start = time.perf_counter()
    images = bad = pooled = nbytes = 0
    try:
        parquet_file = pq.ParquetFile(shard_path)
        with open(tmp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(SCAN_FIELDS)
            for row_group in range(parquet_file.metadata.num_row_groups):
                for row, (blob, path) in enumerate(zip(*row_group_images(parquet_file, row_group))):
                    stem, ext = os.path.splitext(path)
                    w, h, fmt, error = read_header_bytes(blob)
                    keep = bool(w) and _pooled(w, h)
                    if keep:
                        write_atomic(_WORKER["pool_dir"] / f"{stem}{ext}", blob)
                    writer.writerow([split, shard, row_group, row, stem, ext, w or "", h or "", fmt or "",
                                     len(blob), int(keep), error or ""])
                    images += 1
                    bad += 1 if error else 0
                    pooled += 1 if keep else 0
                    nbytes += len(blob)
        os.replace(tmp, target)
        return shard, images, bad, pooled, nbytes, time.perf_counter() - start, None
    except Exception as exc:  # noqa: BLE001 - reported, the shard is retried by the next run
        tmp.unlink(missing_ok=True)
        return shard, images, bad, pooled, nbytes, time.perf_counter() - start, f"{type(exc).__name__}: {exc}"


def list_shards(parquet_dir, splits, max_shards):
    shards = sorted(p for p in Path(parquet_dir).glob("*.parquet") if p.name.split("-")[0] in splits)
    if max_shards and max_shards < len(shards):
        step = len(shards) / max_shards   # evenly spaced, deterministic subset (as profile_dims --sample)
        shards = [shards[int(i * step)] for i in range(max_shards)]
    return shards


def scan(shards, reference, args, scan_dir):
    shards_dir, pool_dir = scan_dir / "shards", scan_dir / "pool"
    shards_dir.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(exist_ok=True)
    stale = [p for d in (shards_dir, pool_dir) for p in d.glob("*.tmp")]
    for p in stale:
        p.unlink()
    todo = [s for s in shards if not (shards_dir / f"{s.stem}.csv").exists()]
    print(f"\nscan: {len(shards)} shards requested, {len(shards) - len(todo)} already in {shards_dir}, "
          f"{len(todo)} to read with {args.workers} workers" + (f", {len(stale)} stale .tmp removed" if stale else ""))
    failed = []
    if not todo:
        return failed
    start = time.perf_counter()
    done = images = pooled = nbytes = 0
    init = (reference, args.pool_tol, args.pool_share, shards_dir, pool_dir)
    with Pool(args.workers, initializer=_init_scan_worker, initargs=init) as pool:
        for shard, n, bad, npool, nb, _, error in pool.imap_unordered(scan_shard, todo):
            done += 1
            images += n
            pooled += npool
            nbytes += nb
            if error:
                failed.append((shard, error))
                print(f"  FAILED {shard}: {error}", flush=True)
            if done % 10 == 0 or done == len(todo):
                elapsed = time.perf_counter() - start
                print(f"  [{done}/{len(todo)}] {images} images, {pooled} pooled, {nbytes / 1e9:.1f} GB, "
                      f"{nbytes / 1e6 / elapsed:.0f} MB/s, {images / elapsed:.0f} img/s, "
                      f"ETA {(len(todo) - done) * elapsed / done / 60:.0f} min", flush=True)
    if failed:
        print(f"  {len(failed)} shards failed (no CSV written; the next run retries them)")
    return failed


def check_pool_settings(scan_dir, args, reference):
    """The per-shard CSVs and the pool are only comparable when every run used the same pre-filter."""
    meta_path = scan_dir / "pool.json"
    meta = {"pool_tol": args.pool_tol, "pool_share": args.pool_share, "ref_dir": str(args.ref_dir),
            "ref_images": reference.total, "ref_resolutions": len(reference.counter)}
    if meta_path.exists():
        with open(meta_path) as f:
            old = json.load(f)
        if (old["pool_tol"], old["pool_share"]) != (meta["pool_tol"], meta["pool_share"]):
            raise SystemExit(f"{meta_path} was written with --pool-tol {old['pool_tol']} --pool-share {old['pool_share']}; "
                             f"this run asks for {args.pool_tol} / {args.pool_share}. Use the same values or a fresh --scan-dir.")
        if old != meta:
            print(f"warning: the pool was pre-filtered against {old['ref_dir']} ({old['ref_images']} images, "
                  f"{old['ref_resolutions']} resolutions), the reference is now {reference.total} images / "
                  f"{len(reference.counter)} resolutions; qualifying images missing from the pool are reported below")
    else:
        scan_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    if args.tol < args.pool_tol or args.max_ref_share > args.pool_share:
        raise SystemExit(f"--tol {args.tol} / --max-ref-share {args.max_ref_share} is looser than the pool pre-filter "
                         f"(--pool-tol {args.pool_tol} / --pool-share {args.pool_share}): its images were not stashed. "
                         f"Rescan into a fresh --scan-dir with a wider --pool-tol / --pool-share.")


# ----------------------------------------------------------------------------- stage 2: select

def load_scan(shards, shards_dir):
    """Every readable image of the requested shards' CSVs as columns; (data, unreadable, missing shards)."""
    columns = {k: [] for k in ("shard", "stem", "ext", "width", "height", "nbytes", "pooled")}
    bad, missing = 0, []
    for index, shard in enumerate(shards):
        path = shards_dir / f"{shard.stem}.csv"
        if not path.exists():
            missing.append(shard.stem)
            continue
        with open(path, newline="") as f:
            reader = csv.reader(f)
            if next(reader) != SCAN_FIELDS:
                raise SystemExit(f"{path} has another column layout; use a fresh --scan-dir")
            for _, _, _, _, stem, ext, w, h, _, nbytes, pooled, error in reader:
                if error:
                    bad += 1
                    continue
                columns["shard"].append(index)
                columns["stem"].append(stem)
                columns["ext"].append(ext)
                columns["width"].append(int(w))
                columns["height"].append(int(h))
                columns["nbytes"].append(int(nbytes))
                columns["pooled"].append(pooled == "1")
    data = {
        "shard": np.array(columns["shard"], dtype=np.int64), "stem": columns["stem"], "ext": columns["ext"],
        "width": np.array(columns["width"], dtype=np.int64), "height": np.array(columns["height"], dtype=np.int64),
        "nbytes": np.array(columns["nbytes"], dtype=np.int64), "pooled": np.array(columns["pooled"], dtype=bool),
    }
    return data, bad, missing


def used_stems(dirs):
    """Stems (extension and _a<k> augmentation suffix stripped) of every file in the given directories."""
    stems = set()
    for d in dirs:
        if not d.is_dir():
            print(f"  warning: not a directory, nothing excluded: {d}")
            continue
        before = len(stems)
        with os.scandir(d) as entries:
            for entry in entries:
                if entry.is_file():
                    stems.add(AUG_SUFFIX.sub("", os.path.splitext(entry.name)[0]))
        print(f"  {d}: {len(stems) - before} new stems")
    return stems


def select(data, reference, used, args, tols):
    """Qualify every scanned image against the reference, drop used stems, draw --n of the rest."""
    keys, inverse = np.unique(data["width"] * KEY + data["height"], return_inverse=True)
    res_w, res_h = keys // KEY, keys % KEY
    shares, radius = reference.profile(np.log2(res_w.astype(np.float64)), np.log2(res_h.astype(np.float64)),
                                       tols, args.max_ref_share)
    t = tols.index(args.tol)
    image_share, image_radius = shares[inverse, t], radius[inverse]
    excluded = np.fromiter((s in used for s in data["stem"]), dtype=bool, count=len(data["stem"]))
    qualifying = image_share <= args.max_ref_share
    eligible = qualifying & ~excluded
    not_pooled = eligible & ~data["pooled"]
    candidates = np.flatnonzero(eligible & data["pooled"]).tolist()

    rng = random.Random(args.seed)
    if args.pick == "random":
        chosen = sorted(rng.sample(candidates, min(args.n, len(candidates))))
    else:   # farthest from the training profile first, ties broken at random
        chosen = sorted(candidates, key=lambda i: (-image_radius[i], rng.random()))[:args.n]

    # how many images every other setting would give (after the exclusions)
    eligible_per_res = np.bincount(inverse[~excluded], minlength=len(keys))
    grid = {f"tol={tol:g}": {f"share<={s:g}": int(eligible_per_res[shares[:, ti] <= s].sum()) for s in SHARE_GRID}
            for ti, tol in enumerate(tols)}
    return {
        "chosen": chosen, "share": image_share, "radius": image_radius, "excluded": int(excluded.sum()),
        "qualifying": int(qualifying.sum()), "eligible": int(eligible.sum()), "eligible_not_pooled": int(not_pooled.sum()),
        "grid": grid,
    }


def print_grid(grid, tols, args):
    print("\nimages each setting would give (after the exclusions); * = this run, "
          "+ = outside the pool pre-filter (bytes not stashed, would need a rescan)")
    print(f"{'tol':>8} {'(+-%)':>6}" + "".join(f"{'share<=' + format(s, 'g'):>13}" for s in SHARE_GRID))
    for tol in tols:
        cells = []
        for s in SHARE_GRID:
            mark = "*" if (tol == args.tol and s == args.max_ref_share) else ""
            mark += "+" if (tol < args.pool_tol or s > args.pool_share) else ""
            cells.append(f"{grid[f'tol={tol:g}'][f'share<={s:g}']}{mark}")
        print(f"{tol:>8g} {2 ** tol - 1:>6.0%}" + "".join(f"{c:>13}" for c in cells))


def copy_selected(rows, pool_dir, out):
    """Copy the chosen pool files into `out` (atomic, header re-verified); rows get a status."""
    out.mkdir(parents=True, exist_ok=True)
    present = sum(1 for e in os.scandir(out) if e.is_file())
    if present:
        print(f"note: {out} already holds {present} files; same-named files of the right size are kept")
    counts = Counter()
    for row in rows:
        target = out / row["file"]
        try:
            if target.exists() and target.stat().st_size == row["nbytes"]:
                row["status"] = "kept"
            else:
                tmp = out / (row["file"] + ".tmp")
                shutil.copyfile(pool_dir / row["file"], tmp)
                _, w, h, _, _, error = read_header(tmp)
                if error or (w, h) != (row["width"], row["height"]):
                    tmp.unlink()
                    raise RuntimeError(error or f"copy is {w}x{h}, expected {row['width']}x{row['height']}")
                os.replace(tmp, target)
                row["status"] = "written"
        except Exception as exc:  # noqa: BLE001 - recorded per file, the run goes on
            row["status"] = f"error: {type(exc).__name__}: {exc}"
        counts[row["status"].split(":")[0]] += 1
    return counts


# ----------------------------------------------------------------------------- main

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_REF, help="the training images to be far from")
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET, help="Open Images parquet shards")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS, help="parquet splits to scan")
    parser.add_argument("--max-shards", type=int, default=None, help="scan only this many evenly spaced shards (smoke test)")
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR,
                        help="per-shard CSVs, the pool of pre-filtered images, manifest.csv and summary.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where the sampled images go (with --execute)")
    parser.add_argument("--exclude-dirs", nargs="*", type=Path, default=None,
                        help=f"directories whose file stems must not be sampled (default: every directory in "
                             f"{DEFAULT_EXCLUDE_ROOT} other than --out / --scan-dir; pass nothing to exclude none)")
    parser.add_argument("--tol", type=float, default=0.2,
                        help="neighbourhood half-width in log2 units per side (0.2 = +-15 %%, default)")
    parser.add_argument("--max-ref-share", type=float, default=0.005,
                        help="qualify when at most this share of the reference is within --tol (default 0.005)")
    parser.add_argument("--pool-tol", type=float, default=0.1, help="pre-filter half-width used while scanning (default 0.1)")
    parser.add_argument("--pool-share", type=float, default=0.01, help="pre-filter share used while scanning (default 0.01)")
    parser.add_argument("--n", type=int, default=20000, help="how many images to sample (default 20000)")
    parser.add_argument("--pick", choices=("random", "farthest"), default="random",
                        help="draw at random among the qualifying images (default) or take the largest radius first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-j", "--workers", type=int, default=16, help="scan workers (NFS-bound above ~8)")
    parser.add_argument("--execute", action="store_true", help="copy the chosen images into --out (default: dry run)")
    args = parser.parse_args()
    for name in ("tol", "pool_tol"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    for name in ("max_ref_share", "pool_share"):
        if not 0 <= getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1)")
    if args.n < 1:
        parser.error("--n must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def main():
    args = parse_args()
    started = time.perf_counter()
    scan_dir, out = args.scan_dir, args.out
    if args.exclude_dirs is None:
        skip = {out.resolve(), scan_dir.resolve()}
        args.exclude_dirs = [p for p in sorted(Path(DEFAULT_EXCLUDE_ROOT).iterdir()) if p.is_dir() and p.resolve() not in skip]
    if not args.ref_dir.is_dir():
        raise SystemExit(f"not a directory: {args.ref_dir}")
    shards = list_shards(args.parquet_dir, args.splits, args.max_shards)
    if not shards:
        raise SystemExit(f"no {args.splits} shards in {args.parquet_dir}")
    tols = sorted(set(TOL_GRID) | {args.tol})
    print(f"ref: {args.ref_dir}\nparquet: {args.parquet_dir}   splits: {args.splits}   shards: {len(shards)}"
          + (f" (--max-shards {args.max_shards})" if args.max_shards else "")
          + f"\nscan dir: {scan_dir}\nout: {out}   {'EXECUTE' if args.execute else 'dry run'}"
          f"\ncriterion: <= {args.max_ref_share:g} of the reference within tol {args.tol:g} (+-{2 ** args.tol - 1:.0%}); "
          f"pool pre-filter: <= {args.pool_share:g} within {args.pool_tol:g}; n {args.n} pick {args.pick} seed {args.seed}")

    # 1. the reference profile
    t = time.perf_counter()
    counter, ref_bad = reference_profile(args.ref_dir, args.workers)
    reference = Reference(counter)
    print(f"\nread {reference.total} reference headers ({ref_bad} unreadable) in {time.perf_counter() - t:.1f}s")
    ref_summary = describe(f"reference {args.ref_dir.name}", counter)
    check_pool_settings(scan_dir, args, reference)

    # 2. scan the shards (cached per shard)
    failed = scan(shards, reference, args, scan_dir)
    t = time.perf_counter()
    data, scan_bad, missing = load_scan(shards, scan_dir / "shards")
    print(f"\nloaded {len(data['stem'])} scanned images ({scan_bad} unreadable) from {len(shards) - len(missing)} shard CSVs "
          f"in {time.perf_counter() - t:.1f}s; {int(data['pooled'].sum())} in the pool"
          + (f"; {len(missing)} shards missing: {missing[:5]}{' ...' if len(missing) > 5 else ''}" if missing else ""))
    if not data["stem"]:
        raise SystemExit("nothing scanned")
    cand_summary = describe("candidates (every scanned image)", Counter(zip(data["width"].tolist(), data["height"].tolist())))

    # 3. select
    print(f"\nexcluding the stems of {len(args.exclude_dirs)} directories:")
    used = used_stems(args.exclude_dirs)
    result = select(data, reference, used, args, tols)
    chosen = result["chosen"]
    print(f"\nqualifying: {result['qualifying']} images   excluded as used: {result['excluded']} images (of {len(used)} stems)   "
          f"eligible: {result['eligible']}   in the pool: {result['eligible'] - result['eligible_not_pooled']}")
    if result["eligible_not_pooled"]:
        print(f"warning: {result['eligible_not_pooled']} eligible images are not in the pool (reference changed since the scan?) "
              f"and cannot be copied without a rescan")
    print_grid(result["grid"], tols, args)
    print(f"\nselected {len(chosen)} of {args.n} requested ({args.pick}, seed {args.seed})")
    if len(chosen) < args.n:
        print(f"  shortfall of {args.n - len(chosen)}: relax --tol / --max-ref-share (see the grid) or scan more shards")
    sel_summary = {}
    if chosen:
        sel_summary = describe("selected", Counter((int(data["width"][i]), int(data["height"][i])) for i in chosen))
        radii = sorted(float(result["radius"][i]) for i in chosen)
        sel_summary["radius"] = {str(p): percentile(radii, p) for p in PERCENTILES}
        print("radius:     " + "  ".join(f"p{p} {percentile(radii, p):.2f}" for p in PERCENTILES))

    # 4. manifest (+ copy)
    rows = [{
        "file": f"{data['stem'][i]}{data['ext'][i]}", "stem": data["stem"][i], "split": shards[data["shard"][i]].stem.split("-")[0],
        "shard": shards[data["shard"][i]].stem, "width": int(data["width"][i]), "height": int(data["height"][i]),
        "nbytes": int(data["nbytes"][i]), "ref_share": f"{result['share'][i]:.6f}", "radius": f"{result['radius'][i]:.4f}",
        "status": "planned",
    } for i in chosen]
    copy_counts = None
    if args.execute and rows:
        t = time.perf_counter()
        copy_counts = copy_selected(rows, scan_dir / "pool", out)
        print(f"\ncopied into {out} in {time.perf_counter() - t:.1f}s: " + "  ".join(f"{k} {v}" for k, v in sorted(copy_counts.items())))
    scan_dir.mkdir(parents=True, exist_ok=True)
    manifest = scan_dir / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "stem", "split", "shard", "width", "height", "nbytes", "ref_share", "radius", "status"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "settings": {k: (str(v) if isinstance(v, Path) else [str(p) for p in v] if isinstance(v, list) and v and isinstance(v[0], Path) else v)
                     for k, v in vars(args).items()},
        "shards": len(shards), "shards_missing": missing, "shards_failed": failed,
        "reference": {**ref_summary, "unreadable": ref_bad},
        "scanned": {**cand_summary, "unreadable": scan_bad, "pooled": int(data["pooled"].sum())},
        "used_stems": len(used), "excluded": result["excluded"], "qualifying": result["qualifying"],
        "eligible": result["eligible"], "eligible_not_pooled": result["eligible_not_pooled"],
        "selected": len(chosen), "selected_profile": sel_summary, "grid": result["grid"],
        "copy": dict(copy_counts) if copy_counts is not None else None,
        "seconds": round(time.perf_counter() - started, 1),
    }
    with open(scan_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {manifest} ({len(rows)} rows) and summary.json in {time.perf_counter() - started:.0f}s")
    if not args.execute:
        print(f"dry run: nothing written to {out}; add --execute to copy the {len(rows)} images")
    elif copy_counts and copy_counts.get("error"):
        print(f"{copy_counts['error']} copies failed, see the status column of the manifest")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
