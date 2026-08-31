#!/usr/bin/env python3
"""Move a random validation split out of the ai_gen / real training dirs.

Images are paired by filename stem (ai_gen uses .png, real uses .jpg), so a
stem is only eligible if it exists in both directories.
"""

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path("data")   # holds the ai_gen / real train directories the split is moved out of


def index_by_stem(directory):
    return {p.stem: p for p in directory.iterdir() if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ai_src, real_src = ROOT / "ai_gen_hidream_train", ROOT / "real_open_subset_train"
    ai_dst, real_dst = ROOT / "ai_gen_hidream_val", ROOT / "real_open_subset_val"

    ai_files, real_files = index_by_stem(ai_src), index_by_stem(real_src)
    paired = sorted(set(ai_files) & set(real_files))
    if len(paired) < args.num:
        raise SystemExit(f"only {len(paired)} paired images available, need {args.num}")

    random.seed(args.seed)
    picked = random.sample(paired, args.num)

    ai_dst.mkdir(parents=True, exist_ok=True)
    real_dst.mkdir(parents=True, exist_ok=True)

    for stem in picked:
        for src, dst in ((ai_files[stem], ai_dst), (real_files[stem], real_dst)):
            target = dst / src.name
            if args.dry_run:
                print(f"{src} -> {target}")
            else:
                shutil.move(str(src), str(target))

    verb = "would move" if args.dry_run else "moved"
    print(f"{verb} {len(picked)} pairs ({2 * len(picked)} files)")


if __name__ == "__main__":
    main()
