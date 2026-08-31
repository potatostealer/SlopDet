"""Training-free baseline: SigLIP2 image embeddings + cosine K-NN over the train split, scored on the val split.

Run from the repo root:
    python -m src.experiments.training_free [--config src/configs/training_free.yml] [--device cuda:4] [--k 50] [--limit 64]

Nothing is trained. The frozen Siglip2 vision tower (model.checkpoint_path)
embeds every image with its pooled image embedding (the attention-pooling head
output, L2-normalised). The train split of data.dataset_config (dataset.yml) is
the reference set: images under real_img_train_ds_path carry label 0 (real),
images under aigen_img_train_ds_path label 1 (AI generated, the positive class
of precision / recall / F1). Every val image is labelled by a majority vote of
its knn.k most similar reference images (knn.weighting = uniform: one vote each
| similarity: each vote weighted by its cosine). An exact tie is NOT classified:
the image is reported as indecisive.

Embeddings are cached per directory in embeddings.cache_dir (the emb_<dir>.npz
format of leakage_check.py / leak_removal.py, so their caches are reused) and
only files missing from the cache are embedded on later runs.

Prints the confusion matrix (with an indecisive column), coverage, accuracy /
precision / recall / F1 over the decided images, the strict accuracy that counts
indecisive as wrong, and the most confident failures; writes
<output.dir>/<run_name>/<split>/k<K>_<weighting>/metrics.json, failures.csv
(every misclassified image with its label and the vote share for AI generated)
and indecisive.csv (every tie with both vote totals).
eval_training_free.py runs the same procedure on the test set.
"""

import argparse
import contextlib
import csv
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import transformers.utils.logging as hf_logging
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Siglip2ImageProcessor, Siglip2VisionModel

from src.dataset.dataloader import load_config as load_dataset_config
from src.dataset.image_dataset import AIGEN_LABEL, REAL_LABEL, list_images
from src.experiments.eval import (
    LABEL_NAMES,
    RULE,
    THIN,
    collect_failures,
    compute_metrics,
    confusion,
    resolve_device,
    write_failures_csv,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "training_free.yml"

INDECISIVE = -1  # prediction value of an exact vote tie
WEIGHTINGS = ("uniform", "similarity")
# precision name -> dtype of the vision tower weights and (autocast) forward
PRECISIONS = {"bf16": torch.bfloat16, "16": torch.float16, "32": torch.float32}
CACHE_SUFFIX = {"bf16": "", "16": "_fp16", "32": "_fp32"}  # bf16 shares the leakage_out caches
KNN_CHUNK = 1024  # queries per similarity matmul: 1024 x 131k reference fp32 scores = 0.5 GB


# ------------------------------------------------------------------ config

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=str, default=None, help="overrides device (cuda:N | N | cpu)")
    parser.add_argument("--k", type=int, default=None, help="overrides knn.k")
    parser.add_argument("--weighting", choices=WEIGHTINGS, default=None, help="overrides knn.weighting")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke tests: only the first N images of every directory (reference and queries)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="overrides embeddings.cache_dir")
    parser.add_argument("--output-dir", type=Path, default=None, help="overrides output.dir")
    return parser.parse_args()


def load_training_free_config(args) -> dict:
    """Load training_free.yml and apply the CLI overrides."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.device is not None:
        cfg["device"] = args.device
    if args.k is not None:
        cfg["knn"]["k"] = args.k
    if args.weighting is not None:
        cfg["knn"]["weighting"] = args.weighting
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.cache_dir is not None:
        cfg["embeddings"]["cache_dir"] = str(args.cache_dir)
    if args.output_dir is not None:
        cfg["output"]["dir"] = str(args.output_dir)
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be >= 1, got {args.limit}")
    cfg["knn"]["k"] = int(cfg["knn"]["k"])
    if cfg["knn"]["k"] < 1:
        raise ValueError(f"knn.k must be >= 1, got {cfg['knn']['k']}")
    if cfg["knn"]["weighting"] not in WEIGHTINGS:
        raise ValueError(f"knn.weighting must be one of {WEIGHTINGS}, got {cfg['knn']['weighting']!r}")
    precision = str(cfg["precision"])
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
    cfg["precision"] = precision
    return cfg


def query_dirs(cfg: dict, ds_cfg: dict, split: str) -> dict:
    """{'real': dir, 'ai_gen': dir} of the query images: the val split of dataset.yml or the test block."""
    source = ds_cfg if split == "val" else cfg["test"]
    return {
        "real": source[f"real_img_{split}_ds_path"],
        "ai_gen": source[f"aigen_img_{split}_ds_path"],
    }


# --------------------------------------------------------------- embedding

class PathDataset(Dataset):
    """Yields (index, RGB image | None for an unreadable file)."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                return i, im.convert("RGB")
        except Exception as e:  # unreadable/corrupt file: skipped, reported, never voted on
            print(f"[warn] skipping {self.paths[i]}: {e}")
            return i, None


class EmbedCollate:
    """(index, image) pairs -> (indices, processor batch); plain-attribute class so it pickles to workers."""

    def __init__(self, checkpoint_path: str):
        self.processor = Siglip2ImageProcessor.from_pretrained(checkpoint_path)

    def __call__(self, batch: list):
        idx = [i for i, im in batch if im is not None]
        images = [im for _, im in batch if im is not None]
        if not images:
            return idx, None
        return idx, self.processor(images=images, return_tensors="pt")


def load_siglip(model_cfg: dict, dtype, device: torch.device):
    """The frozen Siglip2 vision tower (with its pooling head) in eval mode on device."""
    # The Siglip2 directory holds the full two-tower model, so from_pretrained of
    # the vision tower alone prints a LOAD REPORT listing every text-tower key as
    # UNEXPECTED, plus a weight-loading bar. Silenced: those keys are not needed.
    verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        model = Siglip2VisionModel.from_pretrained(
            model_cfg["checkpoint_path"],
            dtype=dtype,
            attn_implementation=model_cfg["attn_implementation"],
        )
    finally:
        hf_logging.set_verbosity(verbosity)
        hf_logging.enable_progress_bar()
    return model.to(device).eval().requires_grad_(False)


def autocast_context(device: torch.device, dtype):
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.no_grad()
def embed(paths, model, collate, data_cfg, device, dtype, tag):
    """Embed paths -> (L2-normalised fp16 embeddings (N, D) on device, bool mask (N,) of readable images)."""
    loader = DataLoader(
        PathDataset(paths),
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=min(data_cfg["num_workers"], len(paths)),
        collate_fn=collate,
        pin_memory=data_cfg["pin_memory"] and device.type == "cuda",
    )
    out = torch.zeros(len(paths), model.config.hidden_size, dtype=torch.float16, device=device)
    ok = torch.zeros(len(paths), dtype=torch.bool)
    bar = tqdm(total=len(paths), desc=tag, unit="img", dynamic_ncols=True)
    for idx, inputs in loader:
        if inputs is None:
            continue
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        with autocast_context(device, dtype):
            feats = model(**inputs).pooler_output  # (B, D): the SigLIP image embedding
        out[idx] = torch.nn.functional.normalize(feats.float(), dim=-1).half()
        ok[idx] = True
        bar.update(len(idx))
    bar.close()
    return out, ok


def cache_files(cache_dir: Path, directory: str, precision: str, limit):
    """(files to read from, in order; file to write). A --limit run reads the full
    cache too but writes its own _limit<N> file, so a smoke test never shrinks a full cache."""
    base = f"emb_{Path(directory).name}{CACHE_SUFFIX[precision]}"
    full = cache_dir / f"{base}.npz"
    if limit is None:
        return [full], full
    limited = cache_dir / f"{base}_limit{limit}.npz"
    return [full, limited], limited


def embed_dir(directory, model, collate, cfg, device, dtype, limit, tag):
    """List a directory (first --limit files only, if set), embed it with cache reuse.

    Returns (paths, embeddings (N, D) fp16 on device, ok mask (N,) bool on CPU, stats dict).
    Cached entries are matched by file name; only missing files are embedded, and
    the cache is rewritten to cover every current file.
    """
    paths = [str(p) for p in list_images(directory)]
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"no images found in {directory}")
    names = np.array([Path(p).name for p in paths])
    emb = torch.zeros(len(paths), model.config.hidden_size, dtype=torch.float16, device=device)
    ok = torch.zeros(len(paths), dtype=torch.bool)
    filled = torch.zeros(len(paths), dtype=torch.bool)
    cache_dir = Path(cfg["embeddings"]["cache_dir"])
    readable, target = cache_files(cache_dir, directory, cfg["precision"], limit)
    for cache in readable:
        if filled.all() or not cache.is_file():
            continue
        z = np.load(cache)
        pos = {n: j for j, n in enumerate(z["names"].tolist())}
        hit = [(i, pos[n]) for i, n in enumerate(names.tolist()) if not filled[i] and n in pos]
        if not hit:
            continue
        dst, src = (list(x) for x in zip(*hit))
        emb[dst] = torch.from_numpy(z["emb"][src]).to(device)
        ok[dst] = torch.from_numpy(z["ok"][src])
        filled[dst] = True
        print(f"[{tag}] {len(hit):,} embeddings reused from {cache}")
    todo = (~filled).nonzero(as_tuple=True)[0].tolist()
    start = time.perf_counter()
    if todo:
        print(f"[{tag}] embedding {len(todo):,} of {len(paths):,} images ...")
        e, o = embed([paths[i] for i in todo], model, collate, cfg["data"], device, dtype, tag)
        emb[todo], ok[todo] = e, o
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(target, emb=emb.cpu().numpy(), ok=ok.numpy(), names=names)
        print(f"[{tag}] cached {len(paths):,} embeddings to {target}")
    stats = {
        "dir": str(directory),
        "images": len(paths),
        "cached": len(paths) - len(todo),
        "embedded": len(todo),
        "unreadable": int((~ok).sum()),
        "embed_seconds": time.perf_counter() - start,
    }
    return paths, emb, ok, stats


def embed_labelled(dirs: dict, model, collate, cfg, device, dtype, limit, tag):
    """Embed the real (0) and ai_gen (1) directories -> (paths, emb, ok, labels (N,) long, per-dir stats)."""
    paths, embs, oks, labels, stats = [], [], [], [], {}
    for name, label in (("real", REAL_LABEL), ("ai_gen", AIGEN_LABEL)):
        p, e, o, s = embed_dir(dirs[name], model, collate, cfg, device, dtype, limit, f"{tag} {name}")
        paths += p
        embs.append(e)
        oks.append(o)
        labels.append(torch.full((len(p),), label, dtype=torch.long))
        stats[name] = s
    return paths, torch.cat(embs), torch.cat(oks), torch.cat(labels), stats


# --------------------------------------------------------------------- K-NN

@torch.no_grad()
def knn_votes(ref_emb, ref_labels, query_emb, k: int, weighting: str):
    """Vote totals of the k most similar reference images for every query.

    ref_emb (N, D) and query_emb (M, D): L2-normalised, on the same device, so
    cosine similarity is the dot product. Returns (votes_real, votes_aigen), two
    (M,) float tensors on the CPU: neighbour counts for uniform weighting,
    sums of the (non-negative) cosines for similarity weighting.
    """
    if k > len(ref_emb):
        raise ValueError(f"k={k} but only {len(ref_emb)} reference images (raise --limit or lower --k)")
    ref_t = ref_emb.float().T.contiguous()  # (D, N)
    ref_labels = ref_labels.to(ref_emb.device)
    votes_real, votes_aigen = [], []
    bar = tqdm(range(0, len(query_emb), KNN_CHUNK), desc=f"knn k={k}", unit="chunk", dynamic_ncols=True)
    for s in bar:
        sim = query_emb[s:s + KNN_CHUNK].float() @ ref_t  # (c, N) cosines
        top_sim, top_idx = sim.topk(k, dim=1)  # (c, k), most similar first
        top_labels = ref_labels[top_idx]
        weights = top_sim.clamp(min=0.0) if weighting == "similarity" else torch.ones_like(top_sim)
        votes_aigen.append((weights * (top_labels == AIGEN_LABEL)).sum(dim=1).cpu())
        votes_real.append((weights * (top_labels == REAL_LABEL)).sum(dim=1).cpu())
    bar.close()
    return torch.cat(votes_real), torch.cat(votes_aigen)


def decide(votes_real, votes_aigen):
    """Majority vote -> (preds (M,) long in {REAL_LABEL, AIGEN_LABEL, INDECISIVE}, share of ai_gen votes (M,) float)."""
    preds = torch.full_like(votes_real, INDECISIVE, dtype=torch.long)
    preds[votes_aigen > votes_real] = AIGEN_LABEL
    preds[votes_real > votes_aigen] = REAL_LABEL
    total = votes_real + votes_aigen
    share = torch.where(total > 0, votes_aigen / total.clamp(min=1e-12), torch.full_like(total, 0.5))
    return preds, share


def confusion_with_ties(labels, preds) -> dict:
    """eval.confusion over the decided images plus the indecisive count of each class."""
    decided = preds != INDECISIVE
    cm = confusion(labels[decided], preds[decided])
    cm["indecisive_real"] = int((~decided & (labels == REAL_LABEL)).sum())
    cm["indecisive_aigen"] = int((~decided & (labels == AIGEN_LABEL)).sum())
    return cm


def compute_metrics_with_ties(cm: dict) -> dict:
    """eval.compute_metrics over the decided images, plus coverage (share decided)
    and the strict accuracy where an indecisive image counts as wrong."""
    metrics = compute_metrics(cm)
    n_decided = cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"]
    n_total = n_decided + cm["indecisive_real"] + cm["indecisive_aigen"]
    metrics["coverage"] = n_decided / n_total if n_total else 0.0
    metrics["strict_accuracy"] = (cm["tp"] + cm["tn"]) / n_total if n_total else 0.0
    return metrics


def collect_indecisive(paths, labels, votes_real, votes_aigen, preds) -> list:
    """Every tie, real images first, in path order."""
    rows = []
    for i in (preds == INDECISIVE).nonzero(as_tuple=True)[0].tolist():
        label = int(labels[i])
        rows.append({
            "path": paths[i],
            "label": label,
            "label_name": LABEL_NAMES[label],
            "votes_real": float(votes_real[i]),
            "votes_aigen": float(votes_aigen[i]),
        })
    rows.sort(key=lambda r: (r["label"], r["path"]))
    return rows


def write_indecisive_csv(rows: list, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "label_name", "votes_real", "votes_aigen"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "votes_real": f"{row['votes_real']:.6f}", "votes_aigen": f"{row['votes_aigen']:.6f}"})


# ------------------------------------------------------------------- report

def describe_method(cfg: dict) -> str:
    knn = cfg["knn"]
    vote = "one vote per neighbour" if knn["weighting"] == "uniform" else "votes weighted by cosine"
    return (f"{Path(cfg['model']['checkpoint_path']).name} pooled image embedding (L2-normalised, {cfg['precision']})"
            f" + cosine {knn['k']}-NN majority vote, {vote}; ties -> indecisive")


def print_report(cfg, split, ref_dirs, ref_stats, query_dirs_, query_stats, n_ref, cm, metrics,
                 failures, indecisive, out_dir, timings, limit):
    data_cfg, knn = cfg["data"], cfg["knn"]
    n_decided = cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"]
    n_indecisive = cm["indecisive_real"] + cm["indecisive_aigen"]
    n_eval = n_decided + n_indecisive
    n_real_eval = cm["tn"] + cm["fp"] + cm["indecisive_real"]
    n_aigen_eval = cm["tp"] + cm["fn"] + cm["indecisive_aigen"]
    n_unreadable = sum(s["unreadable"] for s in query_stats.values())

    print()
    print(RULE)
    print(f" Training-free evaluation: {cfg['run_name']} / {split} split  (K={knn['k']}, {knn['weighting']} votes)")
    print(RULE)
    print(f" method     {describe_method(cfg)}")
    print(f" reference  {n_ref['real']:>9,} real   {ref_dirs['real']}")
    print(f"            {n_ref['ai_gen']:>9,} ai_gen {ref_dirs['ai_gen']}")
    print(f" real   (0) {query_stats['real']['images']:>9,} images  {query_dirs_['real']}")
    print(f" ai_gen (1) {query_stats['ai_gen']['images']:>9,} images  {query_dirs_['ai_gen']}")
    print(f" device {cfg['device']} | precision {cfg['precision']} | batch size {data_cfg['batch_size']}"
          f" | workers {data_cfg['num_workers']} | cache {cfg['embeddings']['cache_dir']}")
    if limit is not None:
        print(f" *** SMOKE TEST: only the first {limit} image(s) of every directory were used "
              f"({n_ref['real'] + n_ref['ai_gen']:,} reference, {n_eval:,} evaluated) ***")
    if n_unreadable:
        print(f" *** {n_unreadable} unreadable query image(s) skipped (not counted below) ***")
    print(THIN)
    print(f" Confusion matrix on {n_eval:,} images (rows = truth, columns = prediction; positive = ai_gen)")
    print(f" {'':>16}{'pred real':>14}{'pred ai_gen':>14}{'indecisive':>14}{'total':>12}")
    print(f" {'true real':>16}{cm['tn']:>14,}{cm['fp']:>14,}{cm['indecisive_real']:>14,}{n_real_eval:>12,}")
    print(f" {'true ai_gen':>16}{cm['fn']:>14,}{cm['tp']:>14,}{cm['indecisive_aigen']:>14,}{n_aigen_eval:>12,}")
    print(THIN)
    print(f" {'Decided':<12}{metrics['coverage']:>8.4f}   (N - indecisive) / N = {n_decided:,} / {n_eval:,}")
    print(f" {'Accuracy':<12}{metrics['accuracy']:>8.4f}   (TP + TN) / decided = {cm['tp'] + cm['tn']:,} / {n_decided:,}")
    print(f" {'Acc strict':<12}{metrics['strict_accuracy']:>8.4f}   (TP + TN) / N = {cm['tp'] + cm['tn']:,} / {n_eval:,}"
          f"   (indecisive counted as wrong)")
    print(f" {'Precision':<12}{metrics['precision']:>8.4f}   TP / (TP + FP) = {cm['tp']:,} / {cm['tp'] + cm['fp']:,}")
    print(f" {'Recall':<12}{metrics['recall']:>8.4f}   TP / (TP + FN) = {cm['tp']:,} / {cm['tp'] + cm['fn']:,}")
    print(f" {'F1':<12}{metrics['f1']:>8.4f}   2PR / (P + R)")
    print(f" {'Real acc':<12}{metrics['real_accuracy']:>8.4f}   TN / (TN + FP) = {cm['tn']:,} / {cm['tn'] + cm['fp']:,}"
          f"   (decided real images)")
    print(f" {'Ai_gen acc':<12}{metrics['aigen_accuracy']:>8.4f}   TP / (TP + FN) = recall   (decided ai_gen images)")
    print(THIN)
    print(f" Indecisive: {n_indecisive:,} of {n_eval:,}  ({cm['indecisive_real']:,} real, {cm['indecisive_aigen']:,} ai_gen)"
          f"  ->  {out_dir / 'indecisive.csv'}")
    print(f" Failures: {len(failures):,} of {n_decided:,} decided  ({cm['fp']:,} real -> ai_gen, "
          f"{cm['fn']:,} ai_gen -> real)  ->  {out_dir / 'failures.csv'}")
    show = int(cfg["output"]["show_failures"])
    if show > 0:
        for label, title in ((REAL_LABEL, "real predicted ai_gen (false positives)"),
                             (AIGEN_LABEL, "ai_gen predicted real (false negatives)")):
            rows = [r for r in failures if r["label"] == label][:show]
            if not rows:
                continue
            print(f" Most confident {title}, ai_gen vote share and file under {Path(rows[0]['path']).parent}:")
            for r in rows:
                print(f"   {r['prob_aigen']:>8.4f}  {Path(r['path']).name}")
    print(THIN)
    ref_total = sum(s["images"] for s in ref_stats.values())
    ref_new = sum(s["embedded"] for s in ref_stats.values())
    query_new = sum(s["embedded"] for s in query_stats.values())
    print(f" Embedding reference {ref_total:,} images ({ref_new:,} embedded, {ref_total - ref_new:,} from cache) "
          f"{timedelta(seconds=round(timings['embed_reference_seconds']))}; "
          f"queries {n_eval + n_unreadable:,} images ({query_new:,} embedded) "
          f"{timedelta(seconds=round(timings['embed_query_seconds']))}")
    rate = n_eval / timings["knn_seconds"] if timings["knn_seconds"] > 0 else float("nan")
    print(f" K-NN {timedelta(seconds=round(timings['knn_seconds']))} for {n_eval:,} queries x "
          f"{n_ref['real'] + n_ref['ai_gen']:,} reference ({rate:,.1f} img/s)  ->  {out_dir / 'metrics.json'}")
    print(RULE)


# --------------------------------------------------------------------- main

def run(cfg: dict, args, split: str):
    """Embed the reference (train) set and the query split, vote, report.

    split = "val": queries are the val split of data.dataset_config (training_free.py);
    split = "test": queries are the test block of the config (eval_training_free.py).
    """
    knn, dtype = cfg["knn"], PRECISIONS[cfg["precision"]]
    device = resolve_device(cfg["device"])
    cfg["device"] = str(device)  # resolved form (cuda:4, not 4) in the report and metrics.json
    ds_cfg = load_dataset_config(cfg["data"]["dataset_config"])
    ref_dirs = {"real": ds_cfg["real_img_train_ds_path"], "ai_gen": ds_cfg["aigen_img_train_ds_path"]}
    q_dirs = query_dirs(cfg, ds_cfg, split)

    print(f"loading {cfg['model']['checkpoint_path']} vision tower ({cfg['precision']}) -> {device} ...")
    model = load_siglip(cfg["model"], dtype, device)
    collate = EmbedCollate(cfg["model"]["checkpoint_path"])
    print(f"method: {describe_method(cfg)}")

    start = time.perf_counter()
    _, ref_emb, ref_ok, ref_labels, ref_stats = embed_labelled(
        ref_dirs, model, collate, cfg, device, dtype, args.limit, "train")
    t_ref = time.perf_counter() - start
    ref_unreadable = int((~ref_ok).sum())
    if ref_unreadable:
        print(f"[train] {ref_unreadable} unreadable reference image(s) excluded from the vote")
    ref_emb, ref_labels = ref_emb[ref_ok.to(device)], ref_labels[ref_ok]
    n_ref = {"real": int((ref_labels == REAL_LABEL).sum()), "ai_gen": int((ref_labels == AIGEN_LABEL).sum())}
    print(f"reference set: {n_ref['real']:,} real + {n_ref['ai_gen']:,} ai_gen = {len(ref_labels):,} images")

    start = time.perf_counter()
    q_paths, q_emb, q_ok, q_labels, q_stats = embed_labelled(
        q_dirs, model, collate, cfg, device, dtype, args.limit, split)
    t_query = time.perf_counter() - start
    keep = q_ok.nonzero(as_tuple=True)[0]
    q_paths = [q_paths[i] for i in keep.tolist()]
    q_emb, q_labels = q_emb[keep.to(device)], q_labels[keep]
    print(f"{split} set: {int((q_labels == REAL_LABEL).sum()):,} real + {int((q_labels == AIGEN_LABEL).sum()):,} ai_gen"
          f" = {len(q_labels):,} images")

    del model  # the tower is not needed for the vote; frees the device for the similarity chunks
    if device.type == "cuda":
        torch.cuda.empty_cache()
    start = time.perf_counter()
    votes_real, votes_aigen = knn_votes(ref_emb, ref_labels, q_emb, knn["k"], knn["weighting"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_knn = time.perf_counter() - start

    preds, share = decide(votes_real, votes_aigen)
    cm = confusion_with_ties(q_labels, preds)
    metrics = compute_metrics_with_ties(cm)
    decided = (preds != INDECISIVE).nonzero(as_tuple=True)[0]
    failures = collect_failures([q_paths[i] for i in decided.tolist()], q_labels[decided], share[decided], preds[decided])
    indecisive = collect_indecisive(q_paths, q_labels, votes_real, votes_aigen, preds)

    out_dir = Path(cfg["output"]["dir"]) / cfg["run_name"] / split / f"k{knn['k']}_{knn['weighting']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_failures_csv(failures, out_dir / "failures.csv")
    write_indecisive_csv(indecisive, out_dir / "indecisive.csv")
    timings = {"embed_reference_seconds": t_ref, "embed_query_seconds": t_query, "knn_seconds": t_knn}
    summary = {
        "run_name": cfg["run_name"],
        "split": split,
        "method": describe_method(cfg),
        "config": cfg,
        "limit": args.limit,
        "reference": {"dirs": ref_dirs, "counts": n_ref, "unreadable": ref_unreadable, "embedding": ref_stats},
        "query": {"dirs": q_dirs, "embedding": q_stats},
        "dataset_counts": {
            "real": q_stats["real"]["images"],
            "ai_gen": q_stats["ai_gen"]["images"],
            "total": q_stats["real"]["images"] + q_stats["ai_gen"]["images"],
        },
        "evaluated": {
            "real": cm["tn"] + cm["fp"] + cm["indecisive_real"],
            "ai_gen": cm["tp"] + cm["fn"] + cm["indecisive_aigen"],
            "total": len(q_labels),
        },
        "confusion": cm,
        "metrics": metrics,
        "num_failures": len(failures),
        "num_indecisive": len(indecisive),
        "failures_csv": str(out_dir / "failures.csv"),
        "indecisive_csv": str(out_dir / "indecisive.csv"),
        "timings": timings,
        "knn_images_per_second": len(q_labels) / t_knn if t_knn > 0 else None,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_report(cfg, split, ref_dirs, ref_stats, q_dirs, q_stats, n_ref, cm, metrics,
                 failures, indecisive, out_dir, timings, args.limit)
    return summary


def main(split: str = "val"):
    args = parse_args()
    cfg = load_training_free_config(args)
    run(cfg, args, split)


if __name__ == "__main__":
    main()
