"""Comprehensive per-augmentation evaluation of a LoRA-Siglip2 + QFormer + MLP checkpoint.

Run from the repo root:
    python -m src.experiments.comprehensive_eval [--config src/configs/comprehensive_eval.yml] [--device cuda:5]
        [--ai-gen-dir DIR] [--real-dir DIR] [--augments none,jpeg,blur] [--limit-per-class 8]

Takes two directories of CLEAN images -- data.real_dir (ground truth 0, real) and data.ai_gen_dir (ground truth 1,
AI generated, the positive class of precision / recall / F1) -- and evaluates one checkpoint on every single
augmentation of src/dataset/augment.py at every one of its option values, plus the un-augmented images. The grid
is read from the augment.params block of augment.params_config (the file the train_aug_* scripts read too), so it
exhausts exactly the options the augmenters draw from:

    none     the images as they are
    jpeg     one round per quality        augment.params.jpeg.quality
    blur     one round per sigma          augment.params.blur.sigma
    resize   one round per scale          augment.params.resize.scale (down_filter / up_filter as configured)
    noise    one round per sigma          augment.params.noise.sigma (per_channel as configured); the noise field is
                                          seeded per file from augment.seed, so a re-run reproduces the same pixels
    jitter   the factors are continuous in training (each uniform in [1 - x, 1 + x]), so the grid is the range
             ends: brightness / contrast / saturation at 1 - x and 1 + x, one axis at a time with the other two
             factors at 1.0; with augment.jitter_random one more round draws all three factors per file exactly
             as the offline / online augmenters do (seeded from augment.seed)
    crop     one round per fraction       augment.params.crop.fraction (resize_back as configured)

Every round applies its ONE augmentation on the fly (in the collate) to every image, runs the model and records
the confusion matrix and accuracy / precision / recall / F1. No chains. The JSON is rewritten after every round,
so an interrupted run keeps the rounds it finished.

Output: <output.dir>/<run_name>/<checkpoint stem>/comprehensive_eval.json -- the results split by augmentation,
each a list of rounds in grid order with the parameters used, the counts and the metrics, plus the mean over an
augmentation's rounds -- and a summary table on stdout.

--limit-per-class N (smoke tests) evaluates the first N real and the first N AI generated images. It replaces a
--limit-batches flag on purpose: the loader is unshuffled and lists real before AI generated, so the first
batches would be real images only.
"""

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.augment import AUG_ORDER, ParamsCfg, Step, _as_list, apply_step, image_seed, sample_step
from src.dataset.collate import Siglip2Collate
from src.dataset.image_dataset import AIGEN_LABEL, REAL_LABEL, BinaryImageDataset
from src.dataset.online_augment import load_params, val_rng
from src.experiments.eval import (
    AUTOCAST_DTYPES,
    autocast_context,
    compute_metrics,
    confusion,
    describe_model,
    load_model,
    resolve_device,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "comprehensive_eval.yml"

NONE = "none"
AUGMENTS = (NONE,) + AUG_ORDER  # the order of the grid and of the JSON
JITTER_AXES = ("brightness", "contrast", "saturation")
METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "real_accuracy", "aigen_accuracy")

RULE = "=" * 100
THIN = "-" * 100


# --------------------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------------------


@dataclass
class Round:
    """One evaluation round: one augmentation with one fixed choice of its parameters."""

    augment: str  # none | jpeg | blur | resize | noise | jitter | crop
    variant: str  # "quality=50", "brightness=0.8", "random", "identity", ...
    params: dict  # the Step params (per-file seeds are added in the collate); {"random": true} = drawn per file

    @property
    def label(self) -> str:
        return self.augment if self.augment == NONE else f"{self.augment}({self.variant})"


def build_grid(params: ParamsCfg, jitter_random: bool) -> list:
    """Every single augmentation at every option value, in AUGMENTS order, preceded by the identity round."""
    rounds = [Round(NONE, "identity", {})]
    for q in _as_list(params.jpeg.quality):
        rounds.append(Round("jpeg", f"quality={int(q)}", {"quality": int(q)}))
    for s in _as_list(params.blur.sigma):
        rounds.append(Round("blur", f"sigma={float(s)}", {"sigma": float(s)}))
    for s in _as_list(params.resize.scale):
        rounds.append(Round("resize", f"scale={float(s)}", {"scale": float(s)}))
    for s in _as_list(params.noise.sigma):
        rounds.append(Round("noise", f"sigma={float(s)}", {"sigma": float(s)}))
    for axis in JITTER_AXES:
        x = float(getattr(params.jitter, axis))
        if x <= 0:  # 1 - x == 1 + x == 1.0: the identity round already covers it
            continue
        for factor in (round(1 - x, 4), round(1 + x, 4)):
            p = {a: 1.0 for a in JITTER_AXES}
            p[axis] = factor
            rounds.append(Round("jitter", f"{axis}={factor}", p))
    if jitter_random:
        ranges = {a: [round(1 - float(getattr(params.jitter, a)), 4), round(1 + float(getattr(params.jitter, a)), 4)]
                  for a in JITTER_AXES}
        rounds.append(Round("jitter", "random", {"random": True, "range": ranges}))
    for f in _as_list(params.crop.fraction):
        rounds.append(Round("crop", f"fraction={float(f)}", {"fraction": float(f)}))
    return rounds


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


class Siglip2RoundCollate(Siglip2Collate):
    """Siglip2Collate that applies one Round's augmentation to every image before the processor, and carries
    the image paths through the batch. Plain attributes only, so it pickles to DataLoader workers."""

    def __init__(self, checkpoint_path: str, rnd: Round, params: ParamsCfg, seed: int):
        super().__init__(checkpoint_path)
        self.round = rnd
        self.params = params
        self.seed = int(seed)

    def step_for(self, path) -> Optional[Step]:
        r = self.round
        if r.augment == NONE:
            return None
        if r.augment == "noise":  # the noise field depends only on the file name and the seed
            return Step("noise", {**r.params, "seed": image_seed(Path(path).stem, self.seed)})
        if r.augment == "jitter" and r.params.get("random"):
            return sample_step("jitter", val_rng(path, self.seed), self.params)
        return Step(r.augment, dict(r.params))

    def __call__(self, batch: list) -> dict:
        paths, labels = zip(*batch)
        images = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            step = self.step_for(path)
            images.append(apply_step(img, step, self.params) if step is not None else img)
        out = self.encode(images, labels)
        out["paths"] = list(paths)
        return out


def build_dataset(real_dir: str, ai_gen_dir: str, limit_per_class: Optional[int]) -> BinaryImageDataset:
    dataset = BinaryImageDataset(real_dir=real_dir, aigen_dir=ai_gen_dir)
    if limit_per_class is not None:
        by_label = {REAL_LABEL: [], AIGEN_LABEL: []}
        for sample in dataset.samples:
            by_label[sample[1]].append(sample)
        dataset.samples = by_label[REAL_LABEL][:limit_per_class] + by_label[AIGEN_LABEL][:limit_per_class]
    return dataset


def dataset_counts(dataset: BinaryImageDataset) -> dict:
    labels = [label for _, label in dataset.samples]
    return {"real": labels.count(REAL_LABEL), "ai_gen": labels.count(AIGEN_LABEL), "total": len(labels)}


# --------------------------------------------------------------------------------------
# One round
# --------------------------------------------------------------------------------------


@torch.no_grad()
def predict(model, loader, device, autocast_dtype, threshold: float, desc: str):
    """Run the loader through the model -> (labels (N,) long, probs (N,) float of being AI generated)."""
    labels, probs = [], []
    correct = seen = 0
    bar = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=False)
    for batch in bar:
        with autocast_context(device, autocast_dtype):
            logits = model(
                batch["pixel_values"].to(device, non_blocking=True),
                batch["pixel_attention_mask"].to(device, non_blocking=True),
                batch["spatial_shapes"].to(device, non_blocking=True),
            )
        p = torch.sigmoid(logits.float()).cpu()
        y = batch["labels"].long()
        labels.append(y)
        probs.append(p)
        correct += int(((p >= threshold).long() == y).sum())
        seen += len(y)
        bar.set_postfix(acc=f"{correct / seen:.4f}", refresh=False)
    bar.close()
    return torch.cat(labels), torch.cat(probs)


def run_round(rnd: Round, model, dataset, data_cfg, checkpoint_path, params, seed, device, autocast_dtype,
              threshold) -> dict:
    collate = Siglip2RoundCollate(checkpoint_path, rnd, params, seed)
    loader = DataLoader(
        dataset, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"],
        drop_last=False, collate_fn=collate, pin_memory=data_cfg["pin_memory"] and device.type == "cuda",
        persistent_workers=False,  # each round is one pass; the workers go away with the loader
    )
    start = time.perf_counter()
    labels, probs = predict(model, loader, device, autocast_dtype, threshold, rnd.label)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    cm = confusion(labels, (probs >= threshold).long())
    n = sum(cm.values())
    return {
        "augment": rnd.augment,
        "variant": rnd.variant,
        "label": rnd.label,
        "params": rnd.params,
        "evaluated": {"real": cm["tn"] + cm["fp"], "ai_gen": cm["tp"] + cm["fn"], "total": n},
        "confusion": cm,
        "metrics": compute_metrics(cm),
        "mean_prob_aigen": {
            "real": float(probs[labels == REAL_LABEL].mean()) if cm["tn"] + cm["fp"] else None,
            "ai_gen": float(probs[labels == AIGEN_LABEL].mean()) if cm["tp"] + cm["fn"] else None,
        },
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


def format_row(augment: str, variant: str, m: dict, n=None, elapsed=None) -> str:
    tail = "" if n is None else f"{n:>9,}"
    if elapsed is not None:
        tail += f"  {elapsed:>7.1f}s"
    return (f" {augment:<8} {variant:<20} {m['accuracy']:>9.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}"
            f" {m['f1']:>8.4f} {m['real_accuracy']:>9.4f} {m['aigen_accuracy']:>8.4f}{tail}")


def table_header() -> str:
    return (f" {'augment':<8} {'variant':<20} {'accuracy':>9} {'precision':>10} {'recall':>8} {'f1':>8}"
            f" {'real_acc':>9} {'ai_acc':>8}{'N':>9}  {'time':>8}")


def print_header(cfg, hparams, info, counts, grid, limit_per_class):
    data_cfg, aug_cfg = cfg["data"], cfg["augment"]
    print()
    print(RULE)
    print(f" Comprehensive evaluation: {hparams['run_name']} / {Path(info['ckpt_path']).name}"
          f"  (epoch {info['epoch']}, step {info['global_step']})")
    print(RULE)
    print(f" model      {describe_model(hparams)}")
    print(f" checkpoint {info['ckpt_path']}")
    print(f" real   (0) {counts['real']:>9,} images  {data_cfg['real_dir']}")
    print(f" ai_gen (1) {counts['ai_gen']:>9,} images  {data_cfg['ai_gen_dir']}")
    print(f" grid       {len(grid)} rounds from {aug_cfg['params_config']}"
          f"  (jitter_random={aug_cfg['jitter_random']}, seed={aug_cfg['seed']})")
    print(f" device {cfg['device']} | precision {cfg['precision']} | batch size {data_cfg['batch_size']}"
          f" | workers {data_cfg['num_workers']} | threshold {cfg['threshold']}")
    if limit_per_class is not None:
        print(f" *** SMOKE TEST: only the first {limit_per_class} image(s) of each class are evaluated ***")
    print(THIN)
    print(table_header())
    print(THIN)


def print_summary(results: dict, out_path: Path, total_elapsed: float):
    print(THIN)
    print(" Summary (mean over the rounds of each augmentation; none has one round)")
    print(THIN)
    print(table_header())
    print(THIN)
    for augment in AUGMENTS:
        block = results.get(augment)
        if not block or not block["rounds"]:
            continue
        for r in block["rounds"]:
            print(format_row(augment, r["variant"], r["metrics"], r["evaluated"]["total"], r["inference_seconds"]))
        if len(block["rounds"]) > 1:
            print(format_row(augment, "MEAN", block["mean_metrics"]))
        print()
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
    parser.add_argument("--augments", type=str, default=None,
                        help=f"comma-separated subset of {','.join(AUGMENTS)} to run (default: all)")
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="smoke tests: only the first N real and the first N AI generated images")
    parser.add_argument("--output-dir", type=Path, default=None, help="overrides output.dir")
    return parser.parse_args()


def load_config(args) -> dict:
    """Load comprehensive_eval.yml and apply the CLI overrides."""
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
    if args.output_dir is not None:
        cfg["output"]["dir"] = str(args.output_dir)
    if args.limit_per_class is not None and args.limit_per_class < 1:
        raise ValueError(f"--limit-per-class must be >= 1, got {args.limit_per_class}")
    precision = str(cfg["precision"])
    if precision not in AUTOCAST_DTYPES:
        raise ValueError(f"precision must be one of {sorted(AUTOCAST_DTYPES)}, got {precision!r}")
    cfg["precision"] = precision
    aug_cfg = cfg["augment"]
    aug_cfg["jitter_random"] = bool(aug_cfg.get("jitter_random", True))
    aug_cfg["seed"] = int(aug_cfg.get("seed", 1234))
    return cfg


def parse_augments(spec: Optional[str]) -> tuple:
    if spec is None:
        return AUGMENTS
    names = tuple(dict.fromkeys(s.strip() for s in spec.split(",") if s.strip()))
    unknown = [n for n in names if n not in AUGMENTS]
    if unknown or not names:
        raise ValueError(f"--augments: unknown {unknown}; valid names are {','.join(AUGMENTS)}")
    return names


def main():
    args = parse_args()
    cfg = load_config(args)
    data_cfg, aug_cfg, out_cfg = cfg["data"], cfg["augment"], cfg["output"]
    threshold = float(cfg["threshold"])
    autocast_dtype = AUTOCAST_DTYPES[cfg["precision"]]
    selected = parse_augments(args.augments)

    params, _ = load_params(aug_cfg["params_config"])  # the multi block is irrelevant: no chains here
    grid = [r for r in build_grid(params, aug_cfg["jitter_random"]) if r.augment in selected]

    torch.set_float32_matmul_precision("high")
    device = resolve_device(cfg["device"])
    cfg["device"] = str(device)  # resolved form (cuda:4, not 4) in the report and the JSON

    dataset = build_dataset(data_cfg["real_dir"], data_cfg["ai_gen_dir"], args.limit_per_class)
    counts = dataset_counts(dataset)

    print(f"loading checkpoint {cfg['model']['ckpt_path']} -> {device} ...")
    model, hparams, info = load_model(cfg["model"], device)
    checkpoint_path = hparams["model"]["checkpoint_path"]

    # out_dir = Path(out_cfg["dir"]) / hparams["run_name"] / Path(info["ckpt_path"]).stem
    out_dir = Path(out_cfg["dir"]) / hparams["run_name"] / Path(out_cfg.get("folder_name", Path(info["ckpt_path"]).stem))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comprehensive_eval.json"

    results = {a: {"rounds": [], "mean_metrics": None} for a in AUGMENTS if a in selected}
    summary = {
        "checkpoint": info,
        "run_name": hparams["run_name"],
        "model": describe_model(hparams),
        "eval_config": cfg,
        "limit_per_class": args.limit_per_class,
        "dataset": {"real_dir": data_cfg["real_dir"], "ai_gen_dir": data_cfg["ai_gen_dir"], "counts": counts},
        "augment_params": dataclasses.asdict(params),  # the grid's source, incl. filters / per_channel / resize_back
        "grid": [{"augment": r.augment, "variant": r.variant, "params": r.params} for r in grid],
        "completed_rounds": 0,
        "total_rounds": len(grid),
        "results": results,
    }

    def save():
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    print_header(cfg, hparams, info, counts, grid, args.limit_per_class)
    start = time.perf_counter()
    for i, rnd in enumerate(grid, start=1):
        result = run_round(rnd, model, dataset, data_cfg, checkpoint_path, params, aug_cfg["seed"], device,
                           autocast_dtype, threshold)
        block = results[rnd.augment]
        block["rounds"].append(result)
        block["mean_metrics"] = mean_metrics(block["rounds"])
        summary["completed_rounds"] = i
        save()  # rewritten after every round: an interrupted run keeps what it finished
        print(format_row(rnd.augment, rnd.variant, result["metrics"], result["evaluated"]["total"],
                         result["inference_seconds"]) + f"   [{i}/{len(grid)}]")
    total_elapsed = time.perf_counter() - start
    summary["total_seconds"] = total_elapsed
    save()

    print_summary(results, out_path, total_elapsed)


if __name__ == "__main__":
    main()
