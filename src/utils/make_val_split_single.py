#!/usr/bin/env python3
"""Move a random validation split out of the ai_gen / real training dirs.

Images are paired by filename stem (ai_gen uses .png, real uses .jpg), so a
stem is only eligible if it exists in both directories.
"""

import argparse
import random
import shutil
from pathlib import Path


def index_by_stem(directory):
    return {p.stem: p for p in directory.iterdir() if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = Path("/path/to/gemini_image_outputs")   # <-- the generated images to split
    dst = Path("data/ai_gen_gemini_val")

    files = index_by_stem(src)
    if len(files) < args.num:
        raise SystemExit(f"only {len(files)} images available, need {args.num}")

    random.seed(args.seed)
    picked = random.sample(list(files.keys()), args.num)

    dst.mkdir(parents=True, exist_ok=True)

    for stem in picked:
        src_path = files[stem]
        target = dst / src_path.name
        if args.dry_run:
            print(f"{src_path} -> {target}")
        else:
            shutil.move(str(src_path), str(target))

    verb = "would move" if args.dry_run else "moved"
    print(f"{verb} {len(picked)} pairs ({2 * len(picked)} files)")


if __name__ == "__main__":
    main()
