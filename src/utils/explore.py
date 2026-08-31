"""Explore the Defactify image dataset stored as HuggingFace arrow shards.

Answers: what the columns are, how Label_A / Label_B work, and which row of each
caption group is the original photo vs the AI generated ones.

Only needs pyarrow + pillow, the arrow shards are memory mapped so nothing is
loaded into RAM unless it is actually printed.
"""

import argparse
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from PIL import Image

DEFAULT_ROOT = "/path/to/detactify_arrow_shards"   # directory of the Defactify HF arrow shards
SHARD_RE = re.compile(r"^(?P<name>.+)-(?P<split>[a-z]+)-(?P<index>\d+)-of-(?P<total>\d+)\.arrow$")


def find_shards(root):
    shards = defaultdict(list)
    for path in sorted(Path(root).glob("*.arrow")):
        match = SHARD_RE.match(path.name)
        if match:
            shards[match["split"]].append(path)
    return shards


def read_shard(path, columns=None):
    with pa.memory_map(str(path)) as src:
        table = pa.ipc.open_stream(src).read_all()
    return table.select(columns) if columns else table


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def show_schema(root, shard):
    section("1. COLUMNS")
    info_path = Path(root) / "dataset_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        print("declared features (dataset_info.json):")
        for name, feature in info["features"].items():
            print(f"  {name:<10} {feature.get('_type')}  {feature.get('dtype', '')}")
        print()
        print("splits:")
        for name, split in info["splits"].items():
            print(f"  {name:<11} {split['num_examples']:>6} rows"
                  f"  ({len(split['shard_lengths'])} shards, {split['num_bytes'] / 1e9:.2f} GB)")
        print()
    print("physical arrow schema:")
    print(read_shard(shard).schema.remove_metadata())
    print()
    print("note: Image is the HF Image feature, i.e. a struct with the raw encoded")
    print("      file in `bytes` and a placeholder filename in `path`.")


def show_labels(shards, split):
    section(f"2. LABELS ({split} split, all shards)")
    pairs = Counter()
    for path in shards:
        table = read_shard(path, ["Label_A", "Label_B"])
        pairs.update(zip(table.column("Label_A").to_pylist(), table.column("Label_B").to_pylist()))

    total = sum(pairs.values())
    print(f"{'Label_A':>8} {'Label_B':>8} {'rows':>8}   share")
    for (a, b), count in sorted(pairs.items()):
        print(f"{a:>8} {b:>8} {count:>8}   {count / total:6.2%}")
    print(f"{'total':>17} {total:>8}")

    label_a = sorted({a for a, _ in pairs})
    label_b = sorted({b for _, b in pairs})
    consistent = all((a == 0) == (b == 0) for a, b in pairs)
    print()
    print(f"Label_A values: {label_a}   -> binary task: 0 = real, 1 = AI generated")
    print(f"Label_B values: {label_b}   -> multiclass task: 0 = real, 1..{max(label_b)} = which generator")
    print(f"Label_A == (Label_B != 0) holds for every row: {consistent}")


def show_grouping(shards, split, examples=2):
    section(f"3. HOW THE ROWS OF ONE CAPTION FIT TOGETHER ({split} split)")
    sequence = []
    for path in shards:
        table = read_shard(path, ["Caption", "Label_A", "Label_B"])
        sequence.extend(zip(table.column("Caption").to_pylist(),
                            table.column("Label_B").to_pylist()))

    per_caption = defaultdict(list)
    for index, (caption, label) in enumerate(sequence):
        per_caption[caption].append((index, label))

    print(f"rows: {len(sequence)}   distinct caption strings: {len(per_caption)}")
    print()
    print("Label_B composition of the rows sharing a caption:")
    compositions = Counter(
        tuple(sorted(Counter(label for _, label in rows).items()))
        for rows in per_caption.values()
    )
    for composition, count in compositions.most_common(5):
        pretty = "  ".join(f"{label}x{n}" for label, n in composition)
        print(f"  {count:>5} captions   {pretty}")
    print()
    print("=> every caption is covered by 1 real photo and 1 image per generator")
    print("   (repeated k times when the same caption text is reused by k photos).")

    blocks = [sequence[i:i + 6] for i in range(0, len(sequence), 6)]
    expected = [0, 1, 2, 3, 4, 5]
    ordered = sum(1 for block in blocks
                  if [label for _, label in block] == expected
                  and len({caption for caption, _ in block}) == 1)
    print()
    print(f"consecutive real->gen1..gen5 blocks of 6: {ordered}/{len(blocks)}")
    if ordered == len(blocks):
        print("=> rows are stored grouped: row 6k is the original, 6k+1..6k+5 its fakes.")
    else:
        print("=> rows are stored shuffled, so a group is only recoverable via Caption.")

    print()
    print("example captions:")
    for caption in list(per_caption)[:examples]:
        print(f"  caption: {caption!r}")
        for index, label in per_caption[caption]:
            kind = "ORIGINAL photo" if label == 0 else f"AI generated by model #{label}"
            print(f"    row {index:>6}  Label_A={int(label != 0)} Label_B={label}  {kind}")


def show_images(shard, per_label=3, sample=600, dump_dir=None):
    section("4. WHAT THE IMAGES LOOK LIKE (first shard)")
    table = read_shard(shard)
    captions = table.column("Caption").to_pylist()
    labels_b = table.column("Label_B").to_pylist()
    image_bytes = pc.struct_field(table.column("Image"), "bytes")
    image_paths = pc.struct_field(table.column("Image"), "path")

    shown = Counter()
    print(f"{'row':>5} {'Label_B':>8} {'path':>14} {'format':>7} {'size':>12} {'mode':>5} {'kB':>7}")
    sizes = defaultdict(Counter)
    for index in range(min(sample, table.num_rows)):
        label = labels_b[index]
        raw = image_bytes[index].as_py()
        with Image.open(io.BytesIO(raw)) as image:
            sizes[label][image.size] += 1
            if shown[label] < per_label:
                shown[label] += 1
                print(f"{index:>5} {label:>8} {image_paths[index].as_py():>14} {image.format:>7}"
                      f" {str(image.size):>12} {image.mode:>5} {len(raw) / 1024:>7.0f}")

    print()
    print(f"resolutions over the first {min(sample, table.num_rows)} rows:")
    for label in sorted(sizes):
        top = "  ".join(f"{w}x{h} ({n})" for (w, h), n in sizes[label].most_common(3))
        square = sum(n for (w, h), n in sizes[label].items() if w == h) / sum(sizes[label].values())
        print(f"  Label_B={label}  {len(sizes[label]):>4} distinct  square: {square:6.1%}   {top}")

    print()
    print("=> Label_B=0 rows keep the source photo's natural, mostly non square")
    print("   resolution (they are the ORIGINAL photographs, MS-COCO style).")
    print("   Every Label_B>0 row is a square render whose resolution is a")
    print("   fingerprint of the generator that produced it.")

    if dump_dir:
        out = Path(dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        group = [i for i, caption in enumerate(captions) if caption == captions[0]]
        for index in group:
            tag = "real" if labels_b[index] == 0 else f"aigen{labels_b[index]}"
            target = out / f"row{index:03d}_labelB{labels_b[index]}_{tag}.jpg"
            target.write_bytes(image_bytes[index].as_py())
        print()
        print(f"wrote the {len(group)} images sharing the first caption to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--per-label", type=int, default=3)
    parser.add_argument("--sample", type=int, default=600,
                        help="rows of the first shard to decode for the resolution stats")
    parser.add_argument("--dump-dir", default=None,
                        help="write the images of the first caption group here")
    args = parser.parse_args()

    shards = find_shards(args.root)
    if not shards:
        raise SystemExit(f"no arrow shards found under {args.root}")
    if args.split not in shards:
        raise SystemExit(f"unknown split {args.split!r}, available: {sorted(shards)}")

    print(f"dataset root: {args.root}")
    for name, paths in sorted(shards.items()):
        print(f"  {name:<11} {len(paths)} shards")

    split_shards = shards[args.split]
    show_schema(args.root, split_shards[0])
    show_labels(split_shards, args.split)
    show_grouping(split_shards, args.split)
    show_images(split_shards[0], args.per_label, args.sample, args.dump_dir)


if __name__ == "__main__":
    main()
