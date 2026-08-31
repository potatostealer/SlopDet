"""Remove train images that leak into val/test, judged by SigLIP similarity.

Companion to leakage_check.py. For each class (ai_gen, real) the split
<root>/<class>_all_toremove_train is compared against BOTH
<root>/<class>_all_val and <root>/<class>_test with SigLIP2 image embeddings.
Every train image whose cosine similarity with ANY val or test image is larger
than --threshold (default 0.9) is deleted from the *_toremove_train directory.
The val/test directories are only read; nothing else on disk is touched.

Each directory is embedded once on the GPU. Embeddings are cached per directory
in --out-dir (same file names and format as leakage_check.py, so the two scripts
can share a cache dir) and reused on later runs, including partial reuse when
files were added to or removed from a directory. The comparison itself is a
chunked matrix product on the GPU. GPU 0 is preferred; when it is busy another
idle GPU is picked (--gpu auto).

Before anything is deleted, a CSV with every flagged image (its best-matching
val/test image and the cosine) and a summary.json are written to --out-dir.

Smoke test:  python leak_removal.py --limit 200 --dry-run --out-dir /some/scratch/dir
Full run:    python leak_removal.py
"""

import argparse
import json
import os
import subprocess
import time

DATA_ROOT = "data"   # dataset root: the split directories live directly in here
CLASSES = ["ai_gen", "real"]
TRAIN_SUFFIX = "_all_toremove_train"
REF_SUFFIXES = {"val": "_all_val", "test": "_test"}
THRESHOLD = 0.9
PREFERRED_GPU = "0"
GPU_ORDER = ["0", "1", "2", "3", "4", "5", "6", "7"]  # preference order for --gpu auto
CKPT = "model_data/siglip"   # SigLIP2 base model + image processor (see README "Downloads")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--classes", nargs="+", default=CLASSES,
                   help=f"cleans <root>/<class>{TRAIN_SUFFIX} against <root>/<class>"
                        f"{REF_SUFFIXES['val']} and <root>/<class>{REF_SUFFIXES['test']}")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help="remove a train image if cosine with any val/test image is > this")
    p.add_argument("--out-dir", default="leakage_out",
                   help="where the removal CSVs, summary.json and the embedding caches go")
    p.add_argument("--dry-run", action="store_true",
                   help="find and report the images to remove but do not delete anything")
    p.add_argument("--limit", type=int, default=None,
                   help="only use the first N images of every directory (smoke test)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--gpu", default="auto",
                   help=f"CUDA device index, or 'auto' = GPU {PREFERRED_GPU} if idle, else another idle one")
    p.add_argument("--unsafe-dir", action="store_true",
                   help="allow deleting from a train directory whose name does not contain 'toremove'")
    return p.parse_args()


def pick_gpu(choice):
    """Return the CUDA_VISIBLE_DEVICES value: the preferred GPU if idle, else another idle one."""
    if choice != "auto":
        return choice
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
        stats = {}
        for line in out.strip().splitlines():
            idx, mem, util = (x.strip() for x in line.split(","))
            stats[idx] = (int(mem), int(util))
    except Exception as e:
        print(f"[gpu] nvidia-smi failed ({e}), using GPU {PREFERRED_GPU}")
        return PREFERRED_GPU
    idle = [g for g in GPU_ORDER + sorted(set(stats) - set(GPU_ORDER))
            if g in stats and stats[g][0] < 1024 and stats[g][1] < 5]
    if idle:
        g = idle[0]
        why = "idle" if g == PREFERRED_GPU else f"GPU {PREFERRED_GPU} busy " \
            f"({stats.get(PREFERRED_GPU, ('?', '?'))[0]} MiB used), this one is idle"
    else:
        g = min(stats, key=lambda k: stats[k])
        why = "no idle GPU, least loaded"
    print(f"[gpu] using GPU {g} ({why})")
    return g


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = pick_gpu(args.gpu)
# Set HF_HOME in the environment to relocate the HuggingFace cache.

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def list_images(d, limit=None):
    if not os.path.isdir(d):
        return []
    paths = sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if os.path.splitext(f)[1].lower() in EXTS
    )
    return paths[:limit] if limit else paths


# ------------------------------------------------------------------ embedding

class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                return i, im.convert("RGB")
        except Exception as e:  # unreadable/corrupt file: skip it
            print(f"[warn] skipping {self.paths[i]}: {e}")
            return i, None


def make_collate(processor):
    def collate(batch):
        idx = [i for i, im in batch if im is not None]
        imgs = [im for _, im in batch if im is not None]
        if not imgs:
            return None, None
        return idx, processor(images=imgs, return_tensors="pt")

    return collate


def load_model():
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(CKPT, dtype=torch.bfloat16, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(CKPT)
    return model, processor


@torch.no_grad()
def embed(paths, model, processor, tag):
    """Return (L2-normalised fp16 embeddings on the GPU, bool mask of readable images)."""
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=args.batch_size,
        num_workers=min(args.num_workers, len(paths)),
        collate_fn=make_collate(processor),
        pin_memory=True,
    )
    out = torch.zeros(len(paths), model.config.vision_config.hidden_size,
                      dtype=torch.float16, device=model.device)
    ok = torch.zeros(len(paths), dtype=torch.bool)
    pbar = tqdm(total=len(paths), desc=tag, unit="img")
    for idx, inputs in loader:
        if idx is None:
            continue
        inputs = {k: v.to(model.device, non_blocking=True) for k, v in inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = model.get_image_features(**inputs).pooler_output
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        out[idx] = feats.half()
        ok[idx] = True
        pbar.update(len(idx))
    pbar.close()
    return out, ok


def cached_embed(dir_path, paths, model, processor, tag):
    """Embed a directory, reusing cached embeddings for every file name still present."""
    suffix = f"_limit{args.limit}" if args.limit else ""
    cache = os.path.join(args.out_dir, f"emb_{os.path.basename(dir_path.rstrip('/'))}{suffix}.npz")
    names = np.array([os.path.basename(p) for p in paths])
    emb = torch.zeros(len(paths), model.config.vision_config.hidden_size,
                      dtype=torch.float16, device=model.device)
    ok = torch.zeros(len(paths), dtype=torch.bool)
    todo = list(range(len(paths)))
    if os.path.exists(cache):
        z = np.load(cache)
        pos = {n: j for j, n in enumerate(z["names"].tolist())}
        hit = [(i, pos[n]) for i, n in enumerate(names.tolist()) if n in pos]
        if hit:
            dst, src = (list(x) for x in zip(*hit))
            emb[dst] = torch.from_numpy(z["emb"][src]).to(model.device)
            ok[dst] = torch.from_numpy(z["ok"][src])
            todo = sorted(set(todo) - set(dst))
        print(f"[{tag}] {len(hit)} embeddings reused from {cache}, {len(todo)} to compute")
    if todo:
        e, o = embed([paths[i] for i in todo], model, processor, tag)
        emb[todo], ok[todo] = e, o
        np.savez(cache, emb=emb.cpu().numpy(), ok=ok.numpy(), names=names)
        print(f"[{tag}] cached {len(paths)} embeddings to {cache}")
    return emb, ok


# ----------------------------------------------------------------- similarity

@torch.no_grad()
def best_matches(emb_tr, ok_tr, refs):
    """For every train image, its highest cosine against all reference images.

    refs: list of (split_name, emb, ok). Returns (best, best_ref_index, per_split) on the
    CPU where per_split[:, k] is the best cosine against refs[k] alone. Unreadable images
    on either side score -1.
    """
    dev = emb_tr.device
    ref_t = torch.cat([e for _, e, _ in refs]).float().T.contiguous()  # (D, Nref)
    ref_bad = ~torch.cat([o for _, _, o in refs]).to(dev)
    bounds, start = [], 0
    for _, e, _ in refs:
        bounds.append((start, start + len(e)))
        start += len(e)

    n = len(emb_tr)
    best = torch.full((n,), -1.0, device=dev)
    best_j = torch.full((n,), -1, dtype=torch.long, device=dev)
    per_split = torch.full((n, len(refs)), -1.0, device=dev)
    chunk = 2048
    for s in tqdm(range(0, n, chunk), desc="similarity", unit="chunk"):
        sim = emb_tr[s:s + chunk].float() @ ref_t  # cosine == dot of L2-normalised vectors
        sim[~ok_tr[s:s + chunk].to(dev)] = -1.0
        sim[:, ref_bad] = -1.0
        for k, (a, b) in enumerate(bounds):
            per_split[s:s + chunk, k] = sim[:, a:b].max(dim=1).values
        m, j = sim.max(dim=1)
        best[s:s + chunk], best_j[s:s + chunk] = m, j
    return best.cpu(), best_j.cpu(), per_split.cpu()


# ----------------------------------------------------------------------- main

def write_csv(rows, out, splits):
    cols = ["train_image", "cosine_sim", "matched_image", "matched_split"] + [f"best_{s}" for s in splits]
    with open(out, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(x) if not isinstance(x, float) else f"{x:.6f}" for x in r) + "\n")


def delete(paths, train_dir):
    """Delete the given files, refusing anything that is not directly inside train_dir."""
    root = os.path.realpath(train_dir)
    if "toremove" not in os.path.basename(root) and not args.unsafe_dir:
        raise SystemExit(f"refusing to delete from {root}: name has no 'toremove' (use --unsafe-dir)")
    removed = 0
    for p in tqdm(paths, desc="deleting", unit="img"):
        rp = os.path.realpath(p)
        if os.path.dirname(rp) != root:
            raise SystemExit(f"refusing to delete {p}: not inside {root}")
        os.remove(rp)
        removed += 1
    return removed


def clean_class(cls, model, processor):
    tr_dir = os.path.join(args.data_root, f"{cls}{TRAIN_SUFFIX}")
    paths_tr = list_images(tr_dir, args.limit)
    ref_dirs = {s: os.path.join(args.data_root, f"{cls}{suf}") for s, suf in REF_SUFFIXES.items()}
    ref_paths = {s: list_images(d, args.limit) for s, d in ref_dirs.items()}
    print(f"\n=== {cls}: {len(paths_tr)} train images vs "
          + " + ".join(f"{len(p)} {s}" for s, p in ref_paths.items()) + " ===")
    print(f"train: {tr_dir}")
    for s, d in ref_dirs.items():
        print(f"{s:5s}: {d}" + ("" if ref_paths[s] else "  (missing or empty, ignored)"))
    if not paths_tr:
        print(f"[{cls}] train directory missing or empty, skipping")
        return None
    splits = [s for s in ref_paths if ref_paths[s]]
    if not splits:
        print(f"[{cls}] no reference images at all, nothing can be flagged, skipping")
        return None

    emb_tr, ok_tr = cached_embed(tr_dir, paths_tr, model, processor, f"{cls} train")
    refs = []
    for s in splits:
        e, o = cached_embed(ref_dirs[s], ref_paths[s], model, processor, f"{cls} {s}")
        refs.append((s, e, o))
    all_ref_paths = [p for s in splits for p in ref_paths[s]]
    split_of = [s for s in splits for _ in ref_paths[s]]

    best, best_j, per_split = best_matches(emb_tr, ok_tr, refs)
    flagged = (best > args.threshold).nonzero().flatten().tolist()
    rows = [(paths_tr[i], best[i].item(), all_ref_paths[best_j[i]], split_of[best_j[i]],
             *per_split[i].tolist()) for i in flagged]
    rows.sort(key=lambda r: -r[1])
    per_split_hit = {s: int((per_split[flagged, k] > args.threshold).sum()) if flagged else 0
                     for k, s in enumerate(splits)}
    n_both = int(((per_split[flagged] > args.threshold).sum(dim=1) == len(splits)).sum()) if flagged else 0

    csv_out = os.path.join(args.out_dir, f"removed_{cls}.csv")
    write_csv(rows, csv_out, splits)
    n, r, unreadable = len(paths_tr), len(rows), int((~ok_tr).sum())
    print(f"[{cls}] {r} of {n} train images have cosine > {args.threshold} with some val/test image "
          f"({100 * r / n:.2f}%) -> {csv_out}")
    print(f"[{cls}] matched " + ", ".join(f"{s}: {per_split_hit[s]}" for s in splits)
          + (f", both: {n_both}" if len(splits) > 1 else "")
          + f"; unreadable train images kept: {unreadable}")
    for p, score, q, s, *_ in rows[:20]:
        print(f"  {score:.4f}  {os.path.basename(p)}  <->  {s}/{os.path.basename(q)}")

    if args.dry_run:
        print(f"[{cls}] dry run, nothing deleted")
        removed = 0
    else:
        removed = delete([r[0] for r in rows], tr_dir)
        print(f"[{cls}] deleted {removed} file(s) from {tr_dir}")
    left_on_disk = len(list_images(tr_dir))
    print(f"[{cls}] {n - removed} of {n} entries left ({100 * removed / n:.2f}% removed); "
          f"directory now holds {left_on_disk} images")
    return {
        "class": cls, "train_dir": tr_dir, "threshold": args.threshold, "dry_run": args.dry_run,
        "n_train_before": n, "n_flagged": r, "n_removed": removed,
        "proportion_removed": removed / n, "n_left": n - removed, "n_images_on_disk_after": left_on_disk,
        "n_unreadable_train_kept": unreadable,
        "reference_counts": {s: len(ref_paths[s]) for s in splits},
        "flagged_by_split": per_split_hit, "flagged_by_both": n_both,
        "csv": csv_out,
    }


def main():
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    model, processor = load_model()
    summary = [s for s in (clean_class(cls, model, processor) for cls in args.classes) if s]
    out = os.path.join(args.out_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n===== summary ({'DRY RUN, ' if args.dry_run else ''}threshold {args.threshold}) =====")
    for s in summary:
        print(f"{s['class']:7s} before {s['n_train_before']:7d}  removed {s['n_removed']:6d} "
              f"({100 * s['proportion_removed']:.2f}%)  left {s['n_left']:7d}  "
              f"(on disk: {s['n_images_on_disk_after']})")
    print(f"-> {out}   [{(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
