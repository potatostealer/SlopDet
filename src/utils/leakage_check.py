"""Check for train/val leakage between the dataset splits under --data-root.

For each class (ai_gen, real) the train split is compared against its own val
split with two detectors, mirroring exact_check.py and siglip_check.py:

  exact   sha256 over the raw file bytes (byte-exact) and over the decoded RGB
          pixel buffer (pixel-exact, so it still catches the same image
          re-encoded at a different quality or in a different format)
  siglip  SigLIP2 image embeddings; cross-split pairs whose cosine similarity
          exceeds --threshold (default 0.9) are flagged

Each directory is hashed and embedded once. Embeddings are cached per directory
in --out-dir and reused on later runs as long as the directory's file list is
unchanged. Output is one CSV with a row per flagged (train, val) pair labelled
by the strongest evidence: byte > pixel > siglip.

Smoke test:  python leakage_check.py --limit 200 --out-dir /some/scratch/dir
Full run:    python leakage_check.py --gpu 4
"""

import argparse
import hashlib
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

DATA_ROOT = "data"   # dataset root: the split directories live directly in here
CLASSES = ["ai_gen", "real"]
CKPT = "model_data/siglip"   # SigLIP2 base model + image processor (see README "Downloads")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
TYPE_RANK = {"byte": 0, "pixel": 1, "siglip": 2}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--classes", nargs="+", default=CLASSES,
                   help="compares <root>/<class>_all_train against <root>/<class>_all_val")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--out-dir", default="leakage_out",
                   help="where leakage_matches.csv and the embedding caches go")
    p.add_argument("--no-exact", action="store_true", help="skip the byte/pixel hash check")
    p.add_argument("--no-siglip", action="store_true", help="skip the SigLIP similarity check")
    p.add_argument("--limit", type=int, default=None,
                   help="only use the first N images of every directory (smoke test)")
    p.add_argument("--hash-workers", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--gpu", default="4")
    return p.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# Set HF_HOME in the environment to relocate the HuggingFace cache.

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def list_images(d, limit=None):
    paths = sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if os.path.splitext(f)[1].lower() in EXTS
    )
    return paths[:limit] if limit else paths


# ----------------------------------------------------------------- exact check

def hashes(path):
    """Return (path, sha256 of file bytes, sha256 of decoded RGB pixels)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        byte_h = hashlib.sha256(raw).hexdigest()
        with Image.open(path) as im:
            im = im.convert("RGB")
            # bind the hash to the dimensions so buffers can't collide across shapes
            pixel_h = hashlib.sha256(f"{im.size}|".encode() + im.tobytes()).hexdigest()
        return path, byte_h, pixel_h
    except Exception as e:
        print(f"[warn] skipping {path}: {e}")
        return path, None, None


def hash_dir(paths, workers, tag):
    """Return {path: byte_hash} and {pixel_hash: [paths]}."""
    byte_of, by_pixel, bad = {}, defaultdict(list), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path, bh, ph in tqdm(ex.map(hashes, paths, chunksize=32),
                                 total=len(paths), desc=tag, unit="img"):
            if bh is None:
                bad += 1
            else:
                byte_of[path] = bh
                by_pixel[ph].append(path)
    if bad:
        print(f"[{tag}] {bad} unreadable file(s) skipped")
    return byte_of, by_pixel


def exact_matches(paths_tr, paths_va, tag):
    """Return {(train_path, val_path): 'byte' | 'pixel'} for pixel-identical pairs."""
    byte_tr, pix_tr = hash_dir(paths_tr, args.hash_workers, f"{tag} hash train")
    byte_va, pix_va = hash_dir(paths_va, args.hash_workers, f"{tag} hash val")
    # identical pixels is implied by identical bytes, so pixel matches are a superset
    found = {
        (a, b): "byte" if byte_tr[a] == byte_va[b] else "pixel"
        for h in set(pix_tr) & set(pix_va)
        for a in pix_tr[h]
        for b in pix_va[h]
    }
    # duplicates living inside a single split, worth knowing about
    for split, tbl in (("train", pix_tr), ("val", pix_va)):
        dupes = sum(len(v) - 1 for v in tbl.values() if len(v) > 1)
        print(f"[{tag}] {dupes} pixel-exact duplicate(s) within {split} itself")
    return found


# ---------------------------------------------------------------- siglip check

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
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
    """Embed a directory, reusing the on-disk cache if its file list is unchanged."""
    suffix = f"_limit{args.limit}" if args.limit else ""
    cache = os.path.join(args.out_dir, f"emb_{os.path.basename(dir_path.rstrip('/'))}{suffix}.npz")
    names = np.array([os.path.basename(p) for p in paths])
    if os.path.exists(cache):
        z = np.load(cache)
        if z["names"].shape == names.shape and (z["names"] == names).all():
            print(f"[{tag}] loaded {len(paths)} embeddings from {cache}")
            return torch.from_numpy(z["emb"]).to(model.device), torch.from_numpy(z["ok"])
        print(f"[{tag}] file list changed since {cache} was written, re-embedding")
    emb, ok = embed(paths, model, processor, tag)
    np.savez(cache, emb=emb.cpu().numpy(), ok=ok.numpy(), names=names)
    print(f"[{tag}] cached embeddings to {cache}")
    return emb, ok


def siglip_matches(emb_tr, ok_tr, paths_tr, emb_va, ok_va, paths_va, tag):
    """Return {(train_path, val_path): cosine} for every pair above the threshold."""
    dev = emb_tr.device
    # cosine similarity == dot product of the L2-normalized embeddings
    val_t = emb_va.float().T.contiguous()
    bad_va = ~ok_va.to(dev)
    found = {}
    chunk = 256
    for s in tqdm(range(0, len(paths_tr), chunk), desc=f"{tag} sim", unit="chunk"):
        sim = emb_tr[s:s + chunk].float() @ val_t
        sim[~ok_tr[s:s + chunk].to(dev)] = -1.0
        sim[:, bad_va] = -1.0
        hits = (sim > args.threshold).nonzero()
        scores = sim[hits[:, 0], hits[:, 1]].tolist()
        for (i, j), score in zip(hits.tolist(), scores):
            found[(paths_tr[s + i], paths_va[j])] = score
    return found


# ----------------------------------------------------------------------- main

def check_class(cls, model, processor):
    """Return rows (class, match_type, cosine_or_None, train_path, val_path) for one class."""
    tr_dir = os.path.join(args.data_root, f"{cls}_all_train")
    va_dir = os.path.join(args.data_root, f"{cls}_all_val")
    paths_tr, paths_va = list_images(tr_dir, args.limit), list_images(va_dir, args.limit)
    print(f"\n=== {cls}: {len(paths_tr)} train images vs {len(paths_va)} val images ===")
    print(f"train: {tr_dir}\nval:   {va_dir}")
    if not paths_tr or not paths_va:
        print(f"[{cls}] empty directory, skipping")
        return []

    exact = {} if args.no_exact else exact_matches(paths_tr, paths_va, cls)

    sims = {}
    if model is not None:
        emb_tr, ok_tr = cached_embed(tr_dir, paths_tr, model, processor, f"{cls} train")
        emb_va, ok_va = cached_embed(va_dir, paths_va, model, processor, f"{cls} val")
        sims = siglip_matches(emb_tr, ok_tr, paths_tr, emb_va, ok_va, paths_va, cls)
        # make sure every exact pair carries a cosine too (they should all be ~1.0)
        idx_tr = {p: i for i, p in enumerate(paths_tr)}
        idx_va = {p: i for i, p in enumerate(paths_va)}
        for a, b in exact:
            if (a, b) not in sims:
                sims[(a, b)] = (emb_tr[idx_tr[a]].float() @ emb_va[idx_va[b]].float()).item()

    rows = [(cls, exact.get(k, "siglip"), sims.get(k), k[0], k[1]) for k in exact.keys() | sims.keys()]
    rows.sort(key=lambda r: (TYPE_RANK[r[1]], -(r[2] or 0.0), r[3], r[4]))

    n = {t: sum(r[1] == t for r in rows) for t in TYPE_RANK}
    print(f"[{cls}] {len(rows)} leaked pair(s): {n['byte']} byte-exact, {n['pixel']} pixel-exact, "
          f"{n['siglip']} siglip > {args.threshold} only")
    print(f"[{cls}] {len({r[3] for r in rows})} unique train images / "
          f"{len({r[4] for r in rows})} unique val images involved")
    for _, t, score, a, b in rows[:20]:
        s = f"{score:.4f}" if score is not None else "  -   "
        print(f"  [{t:6s}] {s}  {os.path.basename(a)}  <->  {os.path.basename(b)}")
    return rows


def write_csv(rows, out):
    with open(out, "w") as f:
        f.write("class,match_type,cosine_sim,train_image,val_image\n")
        for cls, t, score, a, b in rows:
            f.write(f"{cls},{t},{'' if score is None else f'{score:.6f}'},{a},{b}\n")


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "leakage_matches.csv")
    model, processor = (None, None) if args.no_siglip else load_model()

    rows = []
    for cls in args.classes:
        rows += check_class(cls, model, processor)
        write_csv(rows, out)  # checkpoint after every class

    write_csv(rows, out)
    print(f"\n{len(rows)} flagged pair(s) in total -> {out}")
    for cls in args.classes:
        print(f"  {cls}: {sum(r[0] == cls for r in rows)}")


if __name__ == "__main__":
    main()
