"""Bucket calibration of a LoRA-Siglip2 + QFormer + MLP checkpoint on the validation split.

Run from the repo root:
    python -m src.experiments.calibration [--config src/configs/calibration.yml] [--device cuda:5]
        [--rounds 10] [--buckets 10] [--limit-per-class 8]

Takes two directories of CLEAN validation images -- data.real_dir (Y = 0, real) and data.ai_gen_dir (Y = 1,
AI generated) -- and runs calibration.num_rounds (N) full passes of one checkpoint over them. Every round augments
every image on the fly with the online mixture of src/dataset/online_augment.py (p_none / p_single / p_multi,
exactly as the train_aug_* scripts apply it); round r seeds each image's plan from blake2b(augment.seed + r - 1 :
stem), so the N rounds see N different augmentations of every image, yet a re-run reproduces the same pixels.

The model output c = sigmoid(logit) of every (image, round) prediction is pooled and dropped into
calibration.num_buckets (K) equal buckets

    S_i = [(i - 1) / K, i / K)   for i = 1..K-1,      S_K = [(K - 1) / K, 1]

(half-open, so an interior boundary c = i/K lands in S_{i+1}; S_1 includes 0 and S_K includes 1). From the two
per-class bucket histograms the script tabulates the likelihoods P(c in S_i | Y = 1) and P(c in S_i | Y = 0)
(after Laplace add-alpha smoothing, calibration.smoothing) and applies Bayes' rule under calibration.prior_aigen
(null = the empirical class proportion of the pooled predictions):

    P(Y = 1 | c in S_i) = P(c in S_i | Y = 1) P(Y = 1)
                          / (P(c in S_i | Y = 1) P(Y = 1) + P(c in S_i | Y = 0) P(Y = 0))

The calibrated prediction of a bucket is whichever class has the larger posterior, with that posterior as its
confidence. To look up a new model output c at inference time: bucket index = min(floor(c * K), K - 1) into the
JSON's "table" list (bucket_of() here is that mapping).

Every round also records the usual threshold metrics (accuracy / precision / recall / F1) as a sanity check. The
JSON is rewritten after every round with the table recomputed from the counts pooled so far, so an interrupted
run keeps a usable (if noisier) table.

Output: <output.dir>/<run_name>/<output.folder_name>/calibration.json -- the per-round metrics and bucket counts,
the pooled counts, and the lookup table -- plus the table on stdout.

--limit-per-class N (smoke tests) evaluates the first N real and the first N AI generated images. It replaces a
--limit-batches flag on purpose: the loader is unshuffled and lists real before AI generated, so the first
batches would be real images only.
"""

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset.collate import Siglip2AugmentCollate
from src.dataset.image_dataset import AIGEN_LABEL, REAL_LABEL
from src.dataset.online_augment import build_online_augmenter
from src.experiments.comprehensive_eval import METRIC_KEYS, build_dataset, dataset_counts, predict
from src.experiments.eval import (
    AUTOCAST_DTYPES,
    compute_metrics,
    confusion,
    describe_model,
    load_model,
    resolve_device,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "calibration.yml"

RULE = "=" * 100
THIN = "-" * 100


# --------------------------------------------------------------------------------------
# Buckets
# --------------------------------------------------------------------------------------


def bucket_of(probs: torch.Tensor, num_buckets: int) -> torch.Tensor:
    """0-based bucket index of each c in [0, 1]: min(floor(c * K), K - 1).

    This realises S_i = [(i-1)/K, i/K) with S_K also containing c = 1.0; use the same mapping to look a new
    prediction up in the saved table.
    """
    return torch.clamp((probs * num_buckets).long(), max=num_buckets - 1)


def bucket_counts(labels: torch.Tensor, probs: torch.Tensor, num_buckets: int) -> dict:
    """Per-class histogram over the buckets -> {"real": [K ints], "ai_gen": [K ints]}."""
    idx = bucket_of(probs, num_buckets)
    return {
        "real": torch.bincount(idx[labels == REAL_LABEL], minlength=num_buckets).tolist(),
        "ai_gen": torch.bincount(idx[labels == AIGEN_LABEL], minlength=num_buckets).tolist(),
    }


def add_counts(total: dict, counts: dict) -> None:
    for key in ("real", "ai_gen"):
        total[key] = [a + b for a, b in zip(total[key], counts[key])]


def build_table(total: dict, num_buckets: int, prior_aigen: Optional[float], smoothing: float) -> tuple:
    """The lookup table from the pooled bucket counts.

    Returns (rows, resolved prior): one row per bucket with the smoothed likelihoods P(c in S_i | Y), the Bayes
    posteriors P(Y | c in S_i), and the calibrated prediction (argmax posterior) with its confidence (the max).
    A bucket whose posterior denominator is 0 (only possible with smoothing 0) reports None there.
    """
    n_real, n_aigen = sum(total["real"]), sum(total["ai_gen"])
    if prior_aigen is None:  # empirical: the class proportion of the pooled predictions
        prior_aigen = n_aigen / (n_real + n_aigen) if n_real + n_aigen else 0.5
    prior_real = 1.0 - prior_aigen

    def likelihood(count: int, n: int) -> float:
        den = n + num_buckets * smoothing
        return (count + smoothing) / den if den else 0.0

    rows = []
    for i in range(num_buckets):
        c_real, c_aigen = total["real"][i], total["ai_gen"][i]
        lik_real = likelihood(c_real, n_real)
        lik_aigen = likelihood(c_aigen, n_aigen)
        evidence = lik_aigen * prior_aigen + lik_real * prior_real
        post_aigen = lik_aigen * prior_aigen / evidence if evidence else None
        post_real = 1.0 - post_aigen if post_aigen is not None else None
        rows.append({
            "bucket": i + 1,  # S_1..S_K, matching the notation above; list position is the 0-based lookup index
            "lower": i / num_buckets,
            "upper": (i + 1) / num_buckets,
            "count_real": c_real,
            "count_aigen": c_aigen,
            "p_c_given_real": lik_real,
            "p_c_given_aigen": lik_aigen,
            "p_real_given_c": post_real,
            "p_aigen_given_c": post_aigen,
            "empirical_p_aigen": c_aigen / (c_real + c_aigen) if c_real + c_aigen else None,  # raw, val prior
            "predicted_label": None if post_aigen is None else ("ai_gen" if post_aigen >= 0.5 else "real"),
            "confidence": None if post_aigen is None else max(post_aigen, post_real),
        })
    return rows, prior_aigen


# --------------------------------------------------------------------------------------
# One round
# --------------------------------------------------------------------------------------


def run_round(round_idx: int, round_seed: int, model, dataset, data_cfg, checkpoint_path, augmenter, num_buckets,
              device, autocast_dtype, threshold) -> dict:
    collate = Siglip2AugmentCollate(checkpoint_path, augmenter, deterministic_seed=round_seed)
    loader = DataLoader(
        dataset, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"],
        drop_last=False, collate_fn=collate, pin_memory=data_cfg["pin_memory"] and device.type == "cuda",
        persistent_workers=False,  # each round is one pass; the workers go away with the loader
    )
    start = time.perf_counter()
    labels, probs = predict(model, loader, device, autocast_dtype, threshold, f"round {round_idx}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    cm = confusion(labels, (probs >= threshold).long())
    n = sum(cm.values())
    return {
        "round": round_idx,
        "augment_seed": round_seed,
        "evaluated": {"real": cm["tn"] + cm["fp"], "ai_gen": cm["tp"] + cm["fn"], "total": n},
        "confusion": cm,
        "metrics": compute_metrics(cm),
        "bucket_counts": bucket_counts(labels, probs, num_buckets),
        "inference_seconds": elapsed,
        "images_per_second": n / elapsed if elapsed > 0 else None,
    }


def mean_metrics(rounds: list) -> Optional[dict]:
    if not rounds:
        return None
    return {k: sum(r["metrics"][k] for r in rounds) / len(rounds) for k in METRIC_KEYS}


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def round_header() -> str:
    return (f" {'round':<6} {'seed':<12} {'accuracy':>9} {'precision':>10} {'recall':>8} {'f1':>8}"
            f" {'real_acc':>9} {'ai_acc':>8}{'N':>10}  {'time':>8}")


def format_round_row(label: str, seed: str, m: dict, n=None, elapsed=None) -> str:
    tail = "" if n is None else f"{n:>10,}"
    if elapsed is not None:
        tail += f"  {elapsed:>7.1f}s"
    return (f" {label:<6} {seed:<12} {m['accuracy']:>9.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}"
            f" {m['f1']:>8.4f} {m['real_accuracy']:>9.4f} {m['aigen_accuracy']:>8.4f}{tail}")


def print_header(cfg, hparams, info, counts, augmenter, limit_per_class):
    data_cfg, cal_cfg = cfg["data"], cfg["calibration"]
    print()
    print(RULE)
    print(f" Calibration: {hparams['run_name']} / {Path(info['ckpt_path']).name}"
          f"  (epoch {info['epoch']}, step {info['global_step']})")
    print(RULE)
    print(f" model      {describe_model(hparams)}")
    print(f" checkpoint {info['ckpt_path']}")
    print(f" real   (0) {counts['real']:>9,} images  {data_cfg['real_dir']}")
    print(f" ai_gen (1) {counts['ai_gen']:>9,} images  {data_cfg['ai_gen_dir']}")
    print(f" augment    {augmenter.describe()}  (round r: per-file seed {cfg['augment']['seed']} + r - 1)")
    print(f" buckets    K={cal_cfg['num_buckets']} over {cal_cfg['num_rounds']} round(s)"
          f" | prior_aigen {cal_cfg['prior_aigen'] if cal_cfg['prior_aigen'] is not None else 'empirical'}"
          f" | smoothing {cal_cfg['smoothing']}")
    print(f" device {cfg['device']} | precision {cfg['precision']} | batch size {data_cfg['batch_size']}"
          f" | workers {data_cfg['num_workers']} | threshold {cfg['threshold']}")
    if limit_per_class is not None:
        print(f" *** SMOKE TEST: only the first {limit_per_class} image(s) of each class are evaluated ***")
    print(THIN)
    print(round_header())
    print(THIN)


def print_table(table: list, prior_aigen: float, rounds_done: int, out_path: Path, total_elapsed: float):
    def fmt(v, spec=".4f"):
        return "-" if v is None else format(v, spec)

    print(THIN)
    print(f" Calibration table over {rounds_done} round(s), prior P(Y=1) = {prior_aigen:.4f}"
          f"  (lookup: bucket = min(floor(c * K), K - 1))")
    print(THIN)
    print(f" {'bucket':<7} {'range':<14} {'n_real':>10} {'n_aigen':>10} {'P(c|Y=0)':>10} {'P(c|Y=1)':>10}"
          f" {'P(Y=1|c)':>10} {'pred':>7} {'conf':>8}")
    print(THIN)
    for row in table:
        close = "]" if row["bucket"] == len(table) else ")"
        rng = f"[{row['lower']:.2f}, {row['upper']:.2f}{close}"
        print(f" S_{row['bucket']:<5} {rng:<14} {row['count_real']:>10,} {row['count_aigen']:>10,}"
              f" {fmt(row['p_c_given_real']):>10} {fmt(row['p_c_given_aigen']):>10}"
              f" {fmt(row['p_aigen_given_c']):>10} {(row['predicted_label'] or '-'):>7}"
              f" {fmt(row['confidence']):>8}")
    print(THIN)
    print(f" Total {timedelta(seconds=round(total_elapsed))}  ->  {out_path}")
    print(RULE)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ckpt", type=Path, default=None, help="overrides model.ckpt_path")
    parser.add_argument("--ai-gen-dir", type=Path, default=None, help="overrides data.ai_gen_dir (label 1)")
    parser.add_argument("--real-dir", type=Path, default=None, help="overrides data.real_dir (label 0)")
    parser.add_argument("--device", type=str, default=None, help="overrides device (cuda:N | N | cpu)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None, help="overrides calibration.num_rounds (N)")
    parser.add_argument("--buckets", type=int, default=None, help="overrides calibration.num_buckets (K)")
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="smoke tests: only the first N real and the first N AI generated images")
    parser.add_argument("--output-dir", type=Path, default=None, help="overrides output.dir")
    return parser.parse_args()


def load_config(args) -> dict:
    """Load calibration.yml and apply the CLI overrides."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.ckpt is not None:
        cfg["model"]["ckpt_path"] = str(args.ckpt)
    if args.ai_gen_dir is not None:
        cfg["data"]["ai_gen_dir"] = str(args.ai_gen_dir)
    if args.real_dir is not None:
        cfg["data"]["real_dir"] = str(args.real_dir)
    if args.device is not None:
        cfg["device"] = args.device
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.rounds is not None:
        cfg["calibration"]["num_rounds"] = args.rounds
    if args.buckets is not None:
        cfg["calibration"]["num_buckets"] = args.buckets
    if args.output_dir is not None:
        cfg["output"]["dir"] = str(args.output_dir)
    if args.limit_per_class is not None and args.limit_per_class < 1:
        raise ValueError(f"--limit-per-class must be >= 1, got {args.limit_per_class}")
    precision = str(cfg["precision"])
    if precision not in AUTOCAST_DTYPES:
        raise ValueError(f"precision must be one of {sorted(AUTOCAST_DTYPES)}, got {precision!r}")
    cfg["precision"] = precision

    cal = cfg["calibration"]
    cal["num_rounds"] = int(cal.get("num_rounds", 10))
    cal["num_buckets"] = int(cal.get("num_buckets", 10))
    cal["smoothing"] = float(cal.get("smoothing", 1.0))
    prior = cal.get("prior_aigen", 0.5)
    cal["prior_aigen"] = None if prior is None else float(prior)
    if cal["num_rounds"] < 1:
        raise ValueError(f"calibration.num_rounds must be >= 1, got {cal['num_rounds']}")
    if cal["num_buckets"] < 2:
        raise ValueError(f"calibration.num_buckets must be >= 2, got {cal['num_buckets']}")
    if cal["smoothing"] < 0:
        raise ValueError(f"calibration.smoothing must be >= 0, got {cal['smoothing']}")
    if cal["prior_aigen"] is not None and not 0 < cal["prior_aigen"] < 1:
        raise ValueError(f"calibration.prior_aigen must be in (0, 1) or null, got {cal['prior_aigen']}")
    cfg["augment"]["seed"] = int(cfg["augment"].get("seed", 1234))
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args)
    data_cfg, cal_cfg, out_cfg = cfg["data"], cfg["calibration"], cfg["output"]
    num_rounds, num_buckets = cal_cfg["num_rounds"], cal_cfg["num_buckets"]
    threshold = float(cfg["threshold"])
    autocast_dtype = AUTOCAST_DTYPES[cfg["precision"]]

    # The same mixture and validation the train_aug_* scripts use; unknown keys in augment: are an error.
    augmenter, aug_cfg = build_online_augmenter(cfg["augment"])

    torch.set_float32_matmul_precision("high")
    device = resolve_device(cfg["device"])
    cfg["device"] = str(device)  # resolved form (cuda:4, not 4) in the report and the JSON

    dataset = build_dataset(data_cfg["real_dir"], data_cfg["ai_gen_dir"], args.limit_per_class)
    counts = dataset_counts(dataset)

    print(f"loading checkpoint {cfg['model']['ckpt_path']} -> {device} ...")
    model, hparams, info = load_model(cfg["model"], device)
    checkpoint_path = hparams["model"]["checkpoint_path"]

    out_dir = Path(out_cfg["dir"]) / hparams["run_name"] / Path(out_cfg.get("folder_name", Path(info["ckpt_path"]).stem))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration.json"

    total = {"real": [0] * num_buckets, "ai_gen": [0] * num_buckets}
    rounds = []
    summary = {
        "checkpoint": info,
        "run_name": hparams["run_name"],
        "model": describe_model(hparams),
        "eval_config": cfg,
        "limit_per_class": args.limit_per_class,
        "dataset": {"real_dir": data_cfg["real_dir"], "ai_gen_dir": data_cfg["ai_gen_dir"], "counts": counts},
        "augmenter": augmenter.describe(),
        "num_buckets": num_buckets,
        "bucket_rule": "S_i = [(i-1)/K, i/K), S_K includes 1.0; lookup index = min(floor(c * K), K - 1)",
        "smoothing": cal_cfg["smoothing"],
        "prior_aigen_config": cal_cfg["prior_aigen"],  # null = empirical; the resolved value is prior_aigen below
        "completed_rounds": 0,
        "total_rounds": num_rounds,
        "rounds": rounds,
        "mean_metrics": None,
        "total_bucket_counts": total,
        "prior_aigen": None,
        "table": None,
    }

    def save():
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    print_header(cfg, hparams, info, counts, augmenter, args.limit_per_class)
    start = time.perf_counter()
    for r in range(1, num_rounds + 1):
        round_seed = aug_cfg.seed + r - 1
        result = run_round(r, round_seed, model, dataset, data_cfg, checkpoint_path, augmenter, num_buckets,
                           device, autocast_dtype, threshold)
        rounds.append(result)
        add_counts(total, result["bucket_counts"])
        table, prior = build_table(total, num_buckets, cal_cfg["prior_aigen"], cal_cfg["smoothing"])
        summary["completed_rounds"] = r
        summary["mean_metrics"] = mean_metrics(rounds)
        summary["prior_aigen"] = prior
        summary["table"] = table
        save()  # rewritten after every round: an interrupted run keeps a table over the rounds it finished
        print(format_round_row(str(r), str(round_seed), result["metrics"], result["evaluated"]["total"],
                               result["inference_seconds"]) + f"   [{r}/{num_rounds}]")
    if num_rounds > 1:
        print(format_round_row("MEAN", "", summary["mean_metrics"]))
    total_elapsed = time.perf_counter() - start
    summary["total_seconds"] = total_elapsed
    save()

    print_table(summary["table"], summary["prior_aigen"], num_rounds, out_path, total_elapsed)


if __name__ == "__main__":
    main()
