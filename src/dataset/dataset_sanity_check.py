"""Plot the first images of a few batches from each dataloader to eyeball labels."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.dataset.dataloader import DEFAULT_CONFIG_PATH, build_dataloaders

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LABEL_NAMES = {0: "real", 1: "aigen"}


def plot_loader(loader, split, num_batches, num_images, out_dir):
    rows = [
        list(zip(paths, labels.tolist()))[:num_images]
        for _, (paths, labels) in zip(range(num_batches), loader)
    ]
    fig, axes = plt.subplots(
        len(rows), num_images, figsize=(2.2 * num_images, 2.6 * len(rows)), squeeze=False
    )
    for r, row in enumerate(rows):
        for c, ax in enumerate(axes[r]):
            ax.axis("off")
            if c >= len(row):
                continue
            path, label = row[c]
            ax.imshow(Image.open(path).convert("RGB"))
            ax.set_title(
                f"[{label}] {LABEL_NAMES[label]}\n{Path(path).name}", fontsize=6
            )
        axes[r][0].text(
            -0.1, 0.5, f"batch {r}", transform=axes[r][0].transAxes,
            rotation=90, va="center", ha="center", fontsize=8,
        )
    fig.suptitle(f"{split}: first {num_images} images of {len(rows)} batches")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dataset_sanity_check_{split}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=LOG_DIR)
    args = parser.parse_args()

    train_loader, val_loader = build_dataloaders(args.config)
    for split, loader in (("train", train_loader), ("val", val_loader)):
        out_path = plot_loader(
            loader, split, args.num_batches, args.num_images, args.out_dir
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
