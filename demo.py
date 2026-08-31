"""Single-image demo: is this image AI generated?

Run from the repo root:
    python demo.py path/to/image.jpg [--config src/configs/demo.yml]

Loads the checkpoint of demo.yml, computes c = sigmoid(logit) and looks it up in the bucket table of
src/experiments/calibration.py: the prediction is the class with the larger posterior P(Y | c in S_i) and the
confidence is that posterior. Writes {"image", "prediction" (1 = AI generated, 0 = real), "confidence"} to
output_path and prints it.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
import re

from src.experiments.eval import (
    AUTOCAST_DTYPES,
    autocast_context,
    build_eval_collate,
    load_model,
    model_inputs,
    resolve_device,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "src" / "configs" / "demo.yml"


def next_indexed_path(out_path: Path) -> Path:
    """logs/demo/prediction.json -> logs/demo/prediction_<six digits>.json, one past the largest index there."""
    pattern = re.compile(rf"{re.escape(out_path.stem)}_(\d{{6}}){re.escape(out_path.suffix)}")
    indices = [int(m.group(1)) for p in out_path.parent.iterdir() if (m := pattern.fullmatch(p.name))]
    index = max(indices) + 1 if indices else 0
    return out_path.with_name(f"{out_path.stem}_{index:06d}{out_path.suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="image to classify")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(cfg["calibration_path"]) as f:
        calibration = json.load(f)

    device = resolve_device(cfg["device"])
    model, hparams, info = load_model(cfg["model"], device)
    collate = build_eval_collate(hparams, info["classical"])

    batch = collate([(str(args.image), 0)])  # the label is unused at inference
    with torch.no_grad(), autocast_context(device, AUTOCAST_DTYPES[str(cfg["precision"])]):
        logit = model(**model_inputs(batch, device))
    prob = float(torch.sigmoid(logit.float()))  # c = p(AI generated)

    num_buckets = calibration["num_buckets"]
    row = calibration["table"][min(int(prob * num_buckets), num_buckets - 1)]  # bucket_of() in calibration.py
    prediction = int(row["predicted_label"] == "ai_gen")
    confidence = row["confidence"]

    result = {"image": str(args.image), "prediction": prediction, "confidence": confidence}
    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = next_indexed_path(out_path)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
