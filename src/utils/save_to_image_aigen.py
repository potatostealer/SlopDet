#!/usr/bin/env python3
"""Dump images out of the Defactify arrow shards into a flat directory of files.

By default it does what the name says: every AI generated image (Label_A=1) of
the train split, written as-is into ai_gen_detactify_train.

    python save_to_image_aigen.py                          # aigen, train
    python save_to_image_aigen.py --split val              # aigen, validation
    python save_to_image_aigen.py --label real             # real, train
    python save_to_image_aigen.py --label real --split val # real, validation

The output directory defaults to OUT_ROOT/{ai_gen,real}_detactify_{split}, i.e.
the paths src/configs/dataset.yml points at; override it with --out-dir.
Images are written byte for byte in their original encoding (never re-encoded),
with the extension taken from the actual file header.
"""

import argparse
import io
from collections import Counter
from pathlib import Path

import pyarrow.compute as pc
from PIL import Image
from tqdm import tqdm

from explore import DEFAULT_ROOT, find_shards, read_shard

OUT_ROOT = Path("data")   # output root for the extracted image directories

SPLIT_ALIASES = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
LABEL_A = {"aigen": 1, "real": 0}
DIR_PREFIX = {"aigen": "ai_gen", "real": "real"}
EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp"}


def default_out_dir(label, alias):
    return OUT_ROOT / f"{DIR_PREFIX[label]}_detactify_{alias}"


def extension_of(raw):
    with Image.open(io.BytesIO(raw)) as image:
        return EXTENSIONS.get(image.format, f".{image.format.lower()}")


def count_matching(shards, label_a, label_b=None):
    total = 0
    for shard in shards:
        table = read_shard(shard, ["Label_A", "Label_B"])
        total += sum(
            a == label_a and (label_b is None or b == label_b)
            for a, b in zip(table.column("Label_A").to_pylist(), table.column("Label_B").to_pylist())
        )
    return total


def dump(shards, out_dir, alias, label_a, label_b=None, limit=None, overwrite=False, dry_run=False):
    formats = Counter()
    written = skipped = 0
    row = 0
    total = count_matching(shards, label_a, label_b)
    if limit is not None:
        total = min(total, limit)
    progress = tqdm(total=total, unit="img", desc=out_dir.name)

    for shard in shards:
        table = read_shard(shard)
        labels_a = table.column("Label_A").to_pylist()
        labels_b = table.column("Label_B").to_pylist()
        image_bytes = pc.struct_field(table.column("Image"), "bytes")

        for index, (a, b) in enumerate(zip(labels_a, labels_b)):
            row += 1
            if a != label_a or (label_b is not None and b != label_b):
                continue
            if limit is not None and written + skipped >= limit:
                progress.close()
                return written, skipped, formats

            raw = image_bytes[index].as_py()
            target = out_dir / f"{alias}_{row - 1:06d}_b{b}{extension_of(raw)}"
            formats[target.suffix] += 1
            if target.exists() and not overwrite:
                skipped += 1
            elif dry_run:
                written += 1
            else:
                target.write_bytes(raw)
                written += 1
            progress.update(1)

    progress.close()
    return written, skipped, formats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT, help="directory holding the arrow shards")
    parser.add_argument("--split", default="train", choices=sorted(SPLIT_ALIASES))
    parser.add_argument("--label", default="aigen", choices=sorted(LABEL_A),
                        help="which class to dump (Label_A=1 vs Label_A=0)")
    parser.add_argument("--label-b", type=int, default=None,
                        help="keep only this generator (Label_B), e.g. 3")
    parser.add_argument("--out-dir", default=None, help="defaults to OUT_ROOT/<label>_detactify_<split>")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many images")
    parser.add_argument("--overwrite", action="store_true", help="rewrite files that already exist")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    alias = "val" if args.split in ("val", "validation") else args.split
    split = SPLIT_ALIASES[args.split]
    label_a = LABEL_A[args.label]
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(args.label, alias)

    shards = find_shards(args.root)
    if split not in shards:
        raise SystemExit(f"no {split!r} shards under {args.root}, available: {sorted(shards)}")

    selector = f"Label_A={label_a} ({args.label})"
    if args.label_b is not None:
        selector += f", Label_B={args.label_b}"
    print(f"{split} split, {len(shards[split])} shards -> {out_dir}")
    print(f"selecting {selector}{', dry run' if args.dry_run else ''}")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped, formats = dump(
        shards[split], out_dir, alias, label_a, args.label_b,
        args.limit, args.overwrite, args.dry_run,
    )

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} {written} images, skipped {skipped} already present")
    print(f"formats: {dict(formats)}")


if __name__ == "__main__":
    main()
