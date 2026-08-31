#!/usr/bin/env python
"""Apply every augmentation of the comprehensive evaluation to ONE image and save each result as its own file.

Run from the repo root:
    python compare_augmentations.py [IMAGE] [--output-dir comparisons_aug] [--config src/configs/comprehensive_eval.yml]

The grid is exactly the one src/experiments/comprehensive_eval.py evaluates (its build_grid(), fed by the
augment block of the eval config: params_config / jitter_random / seed), and each round's per-file step is drawn
by the same rule (Siglip2RoundCollate.step_for: the noise field and the random jitter factors are seeded from
blake2b(seed : stem)), so the pixels written here are the pixels the model sees in the eval. The identity
("none") round is included. Outputs are PNG, so no second lossy pass lands on top of the augmentation:

    <output-dir>/<stem>__<NN>_<augment>_<variant>.png      e.g. 000000462629__03_jpeg_quality=50.png
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml
from PIL import Image

from src.dataset.augment import apply_step, save_image
from src.dataset.online_augment import load_params
from src.experiments.comprehensive_eval import DEFAULT_CONFIG_PATH, Siglip2RoundCollate, build_grid

DEFAULT_IMAGE = Path("/path/to/image.jpg")   # <-- any test image (--image overrides)
DEFAULT_OUTPUT_DIR = "./comparisons_aug"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, nargs="?", default=DEFAULT_IMAGE, help=f"image to augment (default: {DEFAULT_IMAGE})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"where the PNGs go (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="comprehensive_eval.yml whose augment block (params_config, jitter_random, seed) defines the grid")
    return parser.parse_args()


def step_for(rnd, params, seed: int, path: Path):
    """Exactly Siglip2RoundCollate.step_for, without building the Siglip2 image processor the collate needs."""
    return Siglip2RoundCollate.step_for(SimpleNamespace(round=rnd, params=params, seed=seed), path)


def main():
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    with open(args.config) as f:
        aug_cfg = yaml.safe_load(f)["augment"]
    params, _ = load_params(aug_cfg["params_config"])
    jitter_random = bool(aug_cfg.get("jitter_random", True))
    seed = int(aug_cfg.get("seed", 1234))
    grid = build_grid(params, jitter_random)

    img = Image.open(args.image).convert("RGB")  # as the eval's collate opens it
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.image}  {img.size[0]}x{img.size[1]}  ->  {args.output_dir}/  ({len(grid)} rounds, seed={seed})")
    for i, rnd in enumerate(grid):
        step = step_for(rnd, params, seed, args.image)
        out = apply_step(img, step, params) if step is not None else img
        out_path = args.output_dir / f"{args.image.stem}__{i:02d}_{rnd.augment}_{rnd.variant}.png"
        save_image(out_path, out)
        applied = step.label() if step is not None else "identity"
        print(f"  {i:02d}  {rnd.label:<28} {applied:<62} {out.size[0]}x{out.size[1]}  {out_path.name}")


if __name__ == "__main__":
    main()
