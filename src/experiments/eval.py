"""Single-device evaluation of a LoRA-Siglip2 + QFormer + MLP checkpoint on a labelled test set.

Run from the repo root:
    python -m src.experiments.eval [--config src/configs/eval.yml] [--device cuda:4] [--limit-batches 5]

The checkpoint (model.ckpt_path in eval.yml) is a Lightning checkpoint written
by the training scripts. The architecture is rebuilt from the hyper-parameters
stored inside it, so eval.yml only says where the checkpoint, the two test
directories and the device are. Checkpoints of train_aug_classical_single*.py
are recognised by their weights and get the collate that also extracts the
classical forensic features (their standardisation statistics are in the checkpoint). Images under data.real_img_test_ds_path have
ground truth 0 (real); images under data.aigen_img_test_ds_path have ground
truth 1 (AI generated), the positive class of precision / recall / F1.

Prints the confusion matrix, accuracy / precision / recall / F1 and the most
confident failures; writes <output.dir>/<run_name>/<checkpoint stem>/metrics.json
and failures.csv (every misclassified image with its label and the predicted
probability of being AI generated).
"""

import argparse
import contextlib
import csv
import json
import time
from datetime import timedelta
from pathlib import Path

import torch
import transformers.utils.logging as hf_logging
import yaml
from tqdm import tqdm

from src.dataset.classical_collate import Siglip2ClassicalCollate
from src.dataset.collate import Siglip2Collate
from src.dataset.dataloader import build_dataloader
from src.dataset.image_dataset import AIGEN_LABEL, REAL_LABEL
from src.experiments.train_aug_classical_single import (
    LoraQFormerClassicalDetector,
    build_extractor,
    load_classical_cfg,
)
from src.experiments.train_single import LoraQFormerDetector

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "eval.yml"

LABEL_NAMES = {REAL_LABEL: "real", AIGEN_LABEL: "ai_gen"}

# Lightning precision names -> autocast dtype; None = plain fp32 forward.
AUTOCAST_DTYPES = {
    "32": None,
    "32-true": None,
    "bf16-mixed": torch.bfloat16,
    "16-mixed": torch.float16,
}

RULE = "=" * 88
THIN = "-" * 88


class _WithPaths:
    """Mixin: also carry the image paths through the batch, so every prediction
    can be traced back to its file. Must precede the collate class in the MRO."""

    def __call__(self, batch: list) -> dict:
        out = super().__call__(batch)
        out["paths"] = [path for path, _ in batch]
        return out


class Siglip2EvalCollate(_WithPaths, Siglip2Collate):
    """Siglip2Collate + paths."""


class Siglip2ClassicalEvalCollate(_WithPaths, Siglip2ClassicalCollate):
    """Siglip2ClassicalCollate + paths: batch["classical"] for checkpoints that fuse classical forensic features."""


def build_eval_collate(hparams: dict, classical: bool):
    """The collate a checkpoint needs: classical checkpoints (load_model's `classical`) also get the extractor
    configured as in training (the classical: block of the stored hparams)."""
    checkpoint_path = hparams["model"]["checkpoint_path"]
    if classical:
        return Siglip2ClassicalEvalCollate(checkpoint_path, build_extractor(load_classical_cfg(hparams)))
    return Siglip2EvalCollate(checkpoint_path)


def is_classical_checkpoint(state_dict: dict) -> bool:
    """Whether a state_dict belongs to LoraQFormerClassicalDetector. Decided on the weights, not on a
    classical: block in the hparams: training.yml carries that block for every script, so its presence
    says nothing about the architecture."""
    return any(k.startswith("classical_tokenizer.") for k in state_dict)


def model_inputs(batch: dict, device) -> dict:
    """The model's forward kwargs from a collate batch, moved to device (classical features included when present)."""
    inputs = {
        k: batch[k].to(device, non_blocking=True)
        for k in ("pixel_values", "pixel_attention_mask", "spatial_shapes")
    }
    if "classical" in batch:
        inputs["classical"] = {k: v.to(device, non_blocking=True) for k, v in batch["classical"].items()}
    return inputs


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ckpt", type=Path, default=None, help="overrides model.ckpt_path")
    parser.add_argument("--device", type=str, default=None, help="overrides device (cuda:N | N | cpu)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-batches", type=int, default=None, help="smoke tests: stop after N batches")
    parser.add_argument("--output-dir", type=Path, default=None, help="overrides output.dir")
    return parser.parse_args()


def load_eval_config(args) -> dict:
    """Load eval.yml and apply the CLI overrides."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.ckpt is not None:
        cfg["model"]["ckpt_path"] = str(args.ckpt)
    if args.device is not None:
        cfg["device"] = args.device
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.output_dir is not None:
        cfg["output"]["dir"] = str(args.output_dir)
    if args.limit_batches is not None and args.limit_batches < 1:
        raise ValueError(f"--limit-batches must be >= 1, got {args.limit_batches}")
    precision = str(cfg["precision"])
    if precision not in AUTOCAST_DTYPES:
        raise ValueError(f"precision must be one of {sorted(AUTOCAST_DTYPES)}, got {precision!r}")
    cfg["precision"] = precision
    return cfg


def resolve_device(spec) -> torch.device:
    """'cuda:N' / N / 'cpu' -> torch.device, made current so nothing lands on GPU 0 by accident."""
    spec = str(spec)
    device = torch.device(f"cuda:{spec}" if spec.isdigit() else spec)  # a bare index (yml int or --device 4) is a GPU
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {spec!r} requested but CUDA is not available")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"device {spec!r} requested but only {torch.cuda.device_count()} GPU(s) are visible"
            )
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    return device


def load_model(model_cfg: dict, device: torch.device):
    """Rebuild LoraQFormerDetector (or LoraQFormerClassicalDetector, told apart by the weights) from the
    hparams stored in the checkpoint and load its weights.

    Returns (model in eval mode on device, training hparams, checkpoint info); info["classical"] says
    which architecture it is, so the caller can pick the matching collate (build_eval_collate).
    """
    ckpt_path = Path(model_cfg["ckpt_path"])
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]
    classical = is_classical_checkpoint(ckpt["state_dict"])
    if model_cfg.get("siglip_checkpoint_path"):
        hparams["model"]["checkpoint_path"] = model_cfg["siglip_checkpoint_path"]
    # The Siglip2 directory holds the full two-tower model, so from_pretrained
    # of the vision tower alone prints a LOAD REPORT listing every text-tower
    # key as UNEXPECTED, plus a weight-loading bar. Silenced while the model is
    # built: the strict load_state_dict below overwrites every weight anyway.
    verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        model = (LoraQFormerClassicalDetector if classical else LoraQFormerDetector)(hparams)
    finally:
        hf_logging.set_verbosity(verbosity)
        hf_logging.enable_progress_bar()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    info = {
        "ckpt_path": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "classical": classical,
    }
    del ckpt
    # eval(): LoRA dropout off (the constructor switches the vision tower to train mode).
    return model.to(device).eval(), hparams, info


def describe_model(hparams: dict, classical: bool = False) -> str:
    lora, qf, clf = hparams["lora"], hparams["qformer"], hparams["classifier"]
    s = (
        f"{Path(hparams['model']['checkpoint_path']).name} + LoRA(r={lora['r']}, alpha={lora['alpha']}, "
        f"targets={','.join(lora['targets'])}) + QFormer(m={qf['m']}, layers={qf['n_layers']}, k={qf['k']}, "
        f"heads={qf['n_heads']}) + MLP{list(clf['hidden_dims'])}"
    )
    if classical:
        c = load_classical_cfg(hparams)
        s += (
            f" | classical fusion: {','.join(c.families)} x {c.n_rich + c.n_poor} patches of {c.patch} + global, "
            f"d_model={c.d_model or 'siglip'}"
        )
    return s


def autocast_context(device: torch.device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.no_grad()
def predict(model, loader, device, autocast_dtype, threshold: float, limit_batches=None):
    """Run the loader through the model -> (paths, labels (N,) long, probs (N,) float of being AI generated)."""
    n_batches = len(loader) if limit_batches is None else min(limit_batches, len(loader))
    paths, labels, probs = [], [], []
    correct = seen = 0
    bar = tqdm(loader, total=n_batches, desc="eval", unit="batch", dynamic_ncols=True)
    for i, batch in enumerate(bar):
        if i >= n_batches:
            break
        with autocast_context(device, autocast_dtype):
            logits = model(**model_inputs(batch, device))
        p = torch.sigmoid(logits.float()).cpu()
        y = batch["labels"].long()
        paths.extend(batch["paths"])
        labels.append(y)
        probs.append(p)
        correct += int(((p >= threshold).long() == y).sum())
        seen += len(y)
        bar.set_postfix(acc=f"{correct / seen:.4f}", refresh=False)
    bar.close()
    return paths, torch.cat(labels), torch.cat(probs)


def confusion(labels: torch.Tensor, preds: torch.Tensor) -> dict:
    """Counts with AI generated (label 1) as the positive class."""
    return {
        "tp": int(((preds == AIGEN_LABEL) & (labels == AIGEN_LABEL)).sum()),
        "fp": int(((preds == AIGEN_LABEL) & (labels == REAL_LABEL)).sum()),
        "fn": int(((preds == REAL_LABEL) & (labels == AIGEN_LABEL)).sum()),
        "tn": int(((preds == REAL_LABEL) & (labels == REAL_LABEL)).sum()),
    }


def compute_metrics(cm: dict) -> dict:
    """Accuracy / precision / recall / F1 with AI generated as the positive class.

    A ratio with an empty denominator (e.g. precision when nothing was predicted
    AI generated) is reported as 0.0, as sklearn's zero_division=0 does.
    """
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]

    def ratio(num, den):
        return num / den if den else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return {
        "accuracy": ratio(tp + tn, tp + fp + fn + tn),
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
        "real_accuracy": ratio(tn, tn + fp),  # specificity: real images kept as real
        "aigen_accuracy": recall,  # AI generated images caught == recall
    }


def collect_failures(paths, labels, probs, preds) -> list:
    """Misclassified images, most confident wrong prediction first, real (false
    positives) before AI generated (false negatives)."""
    rows = []
    for i in (preds != labels).nonzero(as_tuple=True)[0].tolist():
        label, pred, prob = int(labels[i]), int(preds[i]), float(probs[i])
        rows.append({
            "path": paths[i],
            "label": label,
            "label_name": LABEL_NAMES[label],
            "pred": pred,
            "pred_name": LABEL_NAMES[pred],
            "prob_aigen": prob,
        })
    # confidence in the wrong answer: prob for a false positive, 1 - prob for a false negative
    rows.sort(key=lambda r: (r["label"], -(r["prob_aigen"] if r["pred"] == AIGEN_LABEL else 1 - r["prob_aigen"])))
    return rows


def write_failures_csv(failures: list, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "label", "label_name", "pred", "pred_name", "prob_aigen"]
        )
        writer.writeheader()
        for row in failures:
            writer.writerow({**row, "prob_aigen": f"{row['prob_aigen']:.6f}"})


def print_report(cfg, hparams, info, dataset_counts, cm, metrics, failures, out_dir, elapsed, limit_batches):
    data_cfg = cfg["data"]
    n_eval = sum(cm.values())
    n_real_eval, n_aigen_eval = cm["tn"] + cm["fp"], cm["tp"] + cm["fn"]

    print()
    print(RULE)
    print(f" Evaluation: {hparams['run_name']} / {Path(info['ckpt_path']).name}"
          f"  (epoch {info['epoch']}, step {info['global_step']})")
    print(RULE)
    print(f" model      {describe_model(hparams, info.get('classical', False))}")
    print(f" checkpoint {info['ckpt_path']}")
    print(f" real   (0) {dataset_counts['real']:>9,} images  {data_cfg['real_img_test_ds_path']}")
    print(f" ai_gen (1) {dataset_counts['ai_gen']:>9,} images  {data_cfg['aigen_img_test_ds_path']}")
    print(f" device {cfg['device']} | precision {cfg['precision']} | batch size {data_cfg['batch_size']}"
          f" | workers {data_cfg['num_workers']} | threshold {cfg['threshold']}")
    if limit_batches is not None:
        print(f" *** SMOKE TEST: only the first {limit_batches} batch(es) = {n_eval:,} of "
              f"{dataset_counts['total']:,} images were evaluated ***")
    print(THIN)
    print(f" Confusion matrix on {n_eval:,} images (rows = truth, columns = prediction; positive = ai_gen)")
    print(f" {'':>16}{'pred real':>14}{'pred ai_gen':>14}{'total':>12}")
    print(f" {'true real':>16}{cm['tn']:>14,}{cm['fp']:>14,}{n_real_eval:>12,}")
    print(f" {'true ai_gen':>16}{cm['fn']:>14,}{cm['tp']:>14,}{n_aigen_eval:>12,}")
    print(THIN)
    print(f" {'Accuracy':<12}{metrics['accuracy']:>8.4f}   (TP + TN) / N = {cm['tp'] + cm['tn']:,} / {n_eval:,}")
    print(f" {'Precision':<12}{metrics['precision']:>8.4f}   TP / (TP + FP) = {cm['tp']:,} / {cm['tp'] + cm['fp']:,}")
    print(f" {'Recall':<12}{metrics['recall']:>8.4f}   TP / (TP + FN) = {cm['tp']:,} / {cm['tp'] + cm['fn']:,}")
    print(f" {'F1':<12}{metrics['f1']:>8.4f}   2PR / (P + R)")
    print(f" {'Real acc':<12}{metrics['real_accuracy']:>8.4f}   TN / (TN + FP) = {cm['tn']:,} / {n_real_eval:,}")
    print(f" {'Ai_gen acc':<12}{metrics['aigen_accuracy']:>8.4f}   TP / (TP + FN) = recall")
    print(THIN)
    print(f" Failures: {len(failures):,} of {n_eval:,}  ({cm['fp']:,} real -> ai_gen, {cm['fn']:,} ai_gen -> real)"
          f"  ->  {out_dir / 'failures.csv'}")
    show = int(cfg["output"]["show_failures"])
    if show > 0:
        for label, title in ((REAL_LABEL, "real predicted ai_gen (false positives)"),
                             (AIGEN_LABEL, "ai_gen predicted real (false negatives)")):
            rows = [r for r in failures if r["label"] == label][:show]
            if not rows:
                continue
            print(f" Most confident {title}, p(ai_gen) and file under {Path(rows[0]['path']).parent}:")
            for r in rows:
                print(f"   {r['prob_aigen']:>8.4f}  {Path(r['path']).name}")
    print(THIN)
    rate = n_eval / elapsed if elapsed > 0 else float("nan")
    print(f" Inference {timedelta(seconds=round(elapsed))} for {n_eval:,} images ({rate:,.1f} img/s)"
          f"  ->  {out_dir / 'metrics.json'}")
    print(RULE)


def main():
    args = parse_args()
    cfg = load_eval_config(args)
    data_cfg, out_cfg = cfg["data"], cfg["output"]
    threshold = float(cfg["threshold"])
    autocast_dtype = AUTOCAST_DTYPES[cfg["precision"]]

    torch.set_float32_matmul_precision("high")
    device = resolve_device(cfg["device"])
    cfg["device"] = str(device)  # resolved form (cuda:4, not 4) in the report and metrics.json

    print(f"loading checkpoint {cfg['model']['ckpt_path']} -> {device} ...")
    model, hparams, info = load_model(cfg["model"], device)
    print(f"model: {describe_model(hparams, info['classical'])}")

    collate = build_eval_collate(hparams, info["classical"])
    loader = build_dataloader(
        data_cfg, "test", shuffle=False, num_workers=data_cfg["num_workers"],
        collate_fn=collate, pin_memory=data_cfg["pin_memory"] and device.type == "cuda",
    )
    sample_labels = [label for _, label in loader.dataset.samples]
    dataset_counts = {
        "real": sample_labels.count(REAL_LABEL),
        "ai_gen": sample_labels.count(AIGEN_LABEL),
        "total": len(sample_labels),
    }
    print(f"test set: {dataset_counts['real']:,} real + {dataset_counts['ai_gen']:,} ai_gen = "
          f"{dataset_counts['total']:,} images, {len(loader)} batches of {data_cfg['batch_size']}")

    start = time.perf_counter()
    paths, labels, probs = predict(model, loader, device, autocast_dtype, threshold, args.limit_batches)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    preds = (probs >= threshold).long()
    cm = confusion(labels, preds)
    metrics = compute_metrics(cm)
    failures = collect_failures(paths, labels, probs, preds)

    out_dir = Path(out_cfg["dir"]) / hparams["run_name"] / Path(info["ckpt_path"]).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    write_failures_csv(failures, out_dir / "failures.csv")
    summary = {
        "checkpoint": info,
        "run_name": hparams["run_name"],
        "eval_config": cfg,
        "limit_batches": args.limit_batches,
        "dataset_counts": dataset_counts,
        "evaluated": {
            "real": cm["tn"] + cm["fp"],
            "ai_gen": cm["tp"] + cm["fn"],
            "total": sum(cm.values()),
        },
        "confusion": cm,
        "metrics": metrics,
        "num_failures": len(failures),
        "failures_csv": str(out_dir / "failures.csv"),
        "inference_seconds": elapsed,
        "images_per_second": sum(cm.values()) / elapsed if elapsed > 0 else None,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_report(cfg, hparams, info, dataset_counts, cm, metrics, failures, out_dir, elapsed, args.limit_batches)


if __name__ == "__main__":
    main()
