#!/usr/bin/env python3
"""Merge the per-source dataset dirs into the combined ai_gen_all / real_all dirs.

For each (prefix, split) pair the images from every ``<prefix>_<source>_<split>``
directory are copied into ``<prefix>_all_<split>``, keeping their filenames.
"""

import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm

ROOT = Path("/path/to/dataset/sources")   # holds the per-source <prefix>_<source>_<split> directories
GROUPS = [("ai_gen", "train"), ("ai_gen", "val"), ("real", "train"), ("real", "val")]
GROUPS = [("real", "train"), ("real", "val")]


def sources_for(prefix, split, dst):
    return sorted(
        p
        for p in ROOT.glob(f"{prefix}_*_{split}")
        if p.is_dir() and p != dst
    )


def merge(prefix, split, dry_run, link):
    # dst = ROOT / f"{prefix}_all_{split}"
    dst_root = Path("data")
    dst = dst_root / f"{prefix}_all_{split}"
    srcs = sources_for(prefix, split, dst)
    if not srcs:
        print(f"{dst.name}: no source directories found")
        return

    files = [(p, src) for src in srcs for p in sorted(src.iterdir()) if p.is_file()]
    print(f"{dst.name}: {len(files)} images from {', '.join(s.name for s in srcs)}")
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    copied = skipped = 0
    taken = set()
    for path, _src in tqdm(files, desc=dst.name, unit="img"):
        target = dst / path.name
        if path.name in taken or target.exists():
            skipped += 1
            continue
        taken.add(path.name)
        if not dry_run:
            if link:
                os.link(path, target)
            else:
                shutil.copy2(path, target)
        copied += 1

    verb = "would copy" if dry_run else "copied"
    print(f"{dst.name}: {verb} {copied}, skipped {skipped} (name already present)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--link", action="store_true", help="hardlink instead of copy")
    args = parser.parse_args()

    for prefix, split in GROUPS:
        merge(prefix, split, args.dry_run, args.link)


if __name__ == "__main__":
    main()
