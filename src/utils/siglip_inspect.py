import os

# set cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import re

import torch
from transformers import AutoModel

parser = argparse.ArgumentParser(description="Inspect the SigLIP architecture layer by layer.")
parser.add_argument(
    "--collapse",
    action="store_true",
    help="Only expand the first entry of each ModuleList (encoder layers are identical).",
)
args = parser.parse_args()

ckpt = "model_data/siglip"  # SigLIP2 base model + image processor (see README "Downloads")
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()


def human(n):
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else str(n)
        n /= 1000
    return f"{n:.1f}T"


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


section("1. CONFIG")
print(model.config)

section("2. FULL MODULE REPR (nn.Module tree)")
print(model)

def is_repeat(name):
    """True for `...layers.N...` with N > 0 -- identical siblings of layers.0."""
    return bool(re.search(r"\.layers\.(?!0\b)\d+", name))


section("3. MODULE TREE (name / class / #params / trainable)")
if args.collapse:
    print("(--collapse: repeated encoder layers past index 0 are summarised in one line)\n")
print(f"{'module path':<62}{'class':<26}{'params':>12}  {'train':>5}")
print("-" * 100)
for name, module in model.named_modules():
    if args.collapse and is_repeat(name):
        if re.fullmatch(r"[\w.]*layers\.1", name):  # summarise once, at the first skipped block
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            depth = name.count(".") + 1
            print("  " * depth + f"... {len(parent) - 1} more identical {type(module).__name__} blocks")
        continue
    own = sum(p.numel() for p in module.parameters(recurse=False))
    total = sum(p.numel() for p in module.parameters())
    trainable = any(p.requires_grad for p in module.parameters(recurse=False))
    depth = name.count(".") + 1 if name else 0
    label = ("  " * depth + (name.split(".")[-1] if name else "<root>"))[:60]
    print(
        f"{label:<62}{type(module).__name__:<26}"
        f"{human(total):>12}  {('yes' if trainable else '-') if own else '':>5}"
    )

section("4. PARAMETERS (name / shape / dtype / device / #elements)")
print(f"{'parameter':<62}{'shape':<24}{'dtype':<16}{'device':<10}{'numel':>12}")
print("-" * 100)
for name, p in model.named_parameters():
    if args.collapse and is_repeat(name):
        continue
    print(
        f"{name[:60]:<62}{str(tuple(p.shape)):<24}{str(p.dtype).replace('torch.',''):<16}"
        f"{str(p.device):<10}{human(p.numel()):>12}"
    )

section("5. BUFFERS (non-parameter tensors)")
buffers = list(model.named_buffers())
if not buffers:
    print("(none)")
for name, b in buffers:
    print(f"{name[:60]:<62}{str(tuple(b.shape)):<24}{str(b.dtype).replace('torch.',''):<16}")

section("6. PARAMETER COUNTS BY TOP-LEVEL SUBMODULE")
total = sum(p.numel() for p in model.parameters())
for name, child in model.named_children():
    n = sum(p.numel() for p in child.parameters())
    print(f"{name:<30}{type(child).__name__:<28}{human(n):>12}  ({100 * n / total:5.1f}%)")
print("-" * 100)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{'TOTAL':<58}{human(total):>12}  ({total:,})")
print(f"{'TRAINABLE':<58}{human(trainable):>12}  ({trainable:,})")
