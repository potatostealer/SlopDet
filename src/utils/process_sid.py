"""Split the SID_Set validation split into per-label image folders.

    label 0 (real)      -> data/real_sid_val
    label 1 (AI-gen)    -> data/ai_gen_sid_val
    label 2 (tampered)  -> data/tampered_sig_val

Reads the HF `datasets` arrow cache directly (no hub access needed) and writes
each image's original encoded bytes untouched, so the copies are byte-exact.
Files are named ``<img_id>.<ext>`` with the extension sniffed from the bytes.

Usage:
    python process_sid.py                 # full run (re-runs skip existing files)
    python process_sid.py --limit 20      # smoke test
    python process_sid.py --overwrite     # rewrite files that already exist
"""

import argparse
import glob
import os
import sys
import time
from collections import Counter

import pyarrow as pa

SRC_DIR = (
    "/path/to/hf_datasets_cache/saberzl___sid_set/"   # the HF `datasets` arrow cache of saberzl/SID_Set
    "default/0.0.0/dc03ead57929879319ce30a82bfcfb8d317b10bd"
)
# SPLIT = "validation"
SPLIT = "train"

# OUT_DIRS = {
#     0: "data/real_sid_val",
#     1: "data/ai_gen_sid_val",
#     2: "data/tampered_sig_val",
# }
OUT_DIRS = {
    2: "data/tampered_sig_train",
}


def sniff_ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:2] == b"BM":
        return "bmp"
    return "bin"


def write_atomic(dst: str, data: bytes) -> None:
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="stop after this many images (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="rewrite files that already exist")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(SRC_DIR, f"sid_set-{SPLIT}-*.arrow")))
    if not shards:
        sys.exit(f"no {SPLIT} shards found under {SRC_DIR}")
    for d in OUT_DIRS.values():
        os.makedirs(d, exist_ok=True)

    written = Counter()
    skipped = Counter()
    unknown_labels = Counter()
    seen_ids = set()
    duplicate_ids = 0
    total = 0
    t0 = time.time()

    for si, shard in enumerate(shards):
        with pa.memory_map(shard) as src:
            for batch in pa.ipc.open_stream(src):
                img_ids = batch.column("img_id").to_pylist()
                labels = batch.column("label").to_pylist()
                images = batch.column("image").to_pylist()
                for img_id, label, img in zip(img_ids, labels, images):
                    if args.limit is not None and total >= args.limit:
                        break
                    total += 1
                    if label not in OUT_DIRS:
                        unknown_labels[label] += 1
                        continue
                    if img_id in seen_ids:
                        duplicate_ids += 1
                        print(f"  warning: duplicate img_id {img_id!r} (label {label})", flush=True)
                    seen_ids.add(img_id)

                    data = img["bytes"]
                    dst = os.path.join(OUT_DIRS[label], f"{img_id}.{sniff_ext(data)}")
                    if os.path.exists(dst) and not args.overwrite:
                        skipped[label] += 1
                        continue
                    write_atomic(dst, data)
                    written[label] += 1
        elapsed = time.time() - t0
        print(
            f"[{si + 1}/{len(shards)}] {os.path.basename(shard)}  "
            f"seen={total}  written={sum(written.values())}  skipped={sum(skipped.values())}  "
            f"{elapsed:.0f}s",
            flush=True,
        )
        if args.limit is not None and total >= args.limit:
            break

    print("\ndone")
    for label, d in OUT_DIRS.items():
        print(f"  label {label}: written={written[label]:6d}  skipped(existing)={skipped[label]:6d}  -> {d}")
    if unknown_labels:
        print(f"  unknown labels (not written): {dict(unknown_labels)}")
    if duplicate_ids:
        print(f"  duplicate img_ids encountered: {duplicate_ids}")


if __name__ == "__main__":
    main()
