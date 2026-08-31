#!/usr/bin/env python
"""Image degradation augmentations.

    python raw_augment.py                       # everything from configs/augment.yml next to this file
    python raw_augment.py --dry-run             # print the plan of the first few images, write nothing
    python raw_augment.py --input-dir DIR --output-dir DIR --limit 4      # or edit the config
    python raw_augment.py --config other.yml --set augment.num_multi=5

Point ``augment.input_dir`` in the config at a directory of images and run it. Needs only ``pillow``, ``numpy`` and
``pyyaml`` (``tqdm`` is used for the progress bar if it happens to be installed).

For every image in ``augment.input_dir`` this writes ``{stem}_a{n}{ext}``:

    _a1 jpeg    JPEG compression, quality uniform over [90, 70, 50, 30]
    _a2 blur    Gaussian blur, sigma uniform over [0.5, 1.0, 2.0]
    _a3 resize  downscale by 0.5x / 0.25x, then back up to the original size
    _a4 noise   additive Gaussian noise, sigma (in [0,1] units) uniform over [0.02, 0.05, 0.10]
    _a5 jitter  brightness / contrast / saturation, each an independent factor in [0.8, 1.2]
    _a6 crop    centre crop to 80% of each side (the output keeps the smaller size)
    _a7.._a{6+K} chains of a random subset of >= 2 of the six, in random order

Everything is derived from ``blake2b(seed : stem)``, so the augmentations of an image depend only on its filename and
``augment.seed`` -- not on the worker count, the directory order, or which outputs already exist. Each image's whole
plan (all 6 + K entries) is sampled up front and only the missing files are rendered, so an interrupted run resumes to
byte-identical output.

JPEG compression is applied in memory (encode to a buffer at the sampled quality, decode back) and the result is stored
as PNG: the artefacts are baked into the pixels without a second lossy pass. Inputs already named ``*_a<n>`` are
skipped, so re-running over an in-place output directory never augments its own output.

This is a standalone export of ``mmrs/augment.py``: same behaviour and same config file, no package imports.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import io
import logging
import os
import random
import re
import sys
import typing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter

log = logging.getLogger("raw_augment")

HERE = Path(__file__).resolve().parent
# ``python raw_augment.py`` with no arguments uses the first of these that exists
DEFAULT_CONFIGS = (HERE / "configs" / "augment.yml", HERE / "augment.yml")

# _a1.._a6 are these six, in this order. Never reorder: the suffix is the only record of what was applied.
AUG_ORDER = ("jpeg", "blur", "resize", "noise", "jitter", "crop")

AUGMENTED_RE = re.compile(r"_a\d+$")  # stems of our own output

RESAMPLE = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "box": Image.BOX,
    "hamming": Image.HAMMING,
}


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class JpegCfg:
    quality: Any = field(default_factory=lambda: [90, 70, 50, 30])


@dataclass
class BlurCfg:
    sigma: Any = field(default_factory=lambda: [0.5, 1.0, 2.0])  # Pillow's GaussianBlur radius IS the std deviation


@dataclass
class ResizeCfg:
    scale: Any = field(default_factory=lambda: [0.5, 0.25])
    down_filter: str = "bicubic"
    up_filter: str = "bicubic"


@dataclass
class NoiseCfg:
    sigma: Any = field(default_factory=lambda: [0.02, 0.05, 0.10])  # in [0, 1] units, i.e. sigma * 255 at 8 bit
    per_channel: bool = True  # false = one luminance-like sample shared by R, G and B


@dataclass
class JitterCfg:
    brightness: float = 0.20  # factor drawn from [1 - x, 1 + x]
    contrast: float = 0.20
    saturation: float = 0.20


@dataclass
class CropCfg:
    fraction: Any = 0.80  # of each side, so 0.8 keeps 64% of the area
    resize_back: Any = False  # false = keep the cropped size | true = upscale back to W x H | a filter name for that upscale


@dataclass
class ParamsCfg:
    jpeg: JpegCfg = field(default_factory=JpegCfg)
    blur: BlurCfg = field(default_factory=BlurCfg)
    resize: ResizeCfg = field(default_factory=ResizeCfg)
    noise: NoiseCfg = field(default_factory=NoiseCfg)
    jitter: JitterCfg = field(default_factory=JitterCfg)
    crop: CropCfg = field(default_factory=CropCfg)


@dataclass
class MultiCfg:
    min_size: int = 2  # a chain uses a random subset of this many .. max_size augmentations
    max_size: int = 6
    distinct_subsets: bool = True  # the K chains of one image use K different subsets (order may still repeat)


@dataclass
class AugmentCfg:
    input_dir: str = "<IMAGE DIRECTORY>"  # set in augment.yml or with --input-dir
    output_dir: Optional[str] = None  # null = in place, next to the originals
    seed: int = 1234
    num_multi: int = 3  # K chains per image, on top of the six singles
    extensions: list = field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"])
    output_format: str = "png"  # png | jpg | webp | same (same = keep the input's extension)
    suffix: str = "_a"  # {stem}{suffix}{n}{ext}
    limit: Optional[int] = None  # only the first K source images (smoke tests)
    workers: int = 0  # 0 = os.cpu_count()
    overwrite: bool = False  # false = keep outputs that already exist (resume)
    params: ParamsCfg = field(default_factory=ParamsCfg)
    multi: MultiCfg = field(default_factory=MultiCfg)

    config_path: Optional[str] = None
    overrides: list = field(default_factory=list)

    def __post_init__(self):
        self.input_dir = resolve_path(self.input_dir)
        if self.output_dir:
            self.output_dir = resolve_path(self.output_dir)
        self.extensions = [e if e.startswith(".") else f".{e}" for e in (str(x).lower() for x in self.extensions)]

    @property
    def out_dir(self) -> Path:
        return Path(self.output_dir or self.input_dir)

    @property
    def in_place(self) -> bool:
        return self.out_dir.resolve() == Path(self.input_dir).resolve()

    def out_suffix(self, src: Path) -> str:
        return src.suffix.lower() if self.output_format == "same" else f".{self.output_format}"


def resolve_path(p: str) -> str:
    """Absolute path; a relative value is taken relative to this script, never to the shell's cwd."""
    q = Path(p)
    return str(q if q.is_absolute() else (HERE / q))


def validate_multi(m: MultiCfg) -> None:
    """The chain-size bounds. Shared with the on-the-fly augmenter (src/dataset/online_augment.py)."""
    if not 2 <= m.min_size <= m.max_size <= len(AUG_ORDER):
        raise ValueError(f"augment.multi needs 2 <= min_size <= max_size <= {len(AUG_ORDER)}")


def validate_params(p: ParamsCfg) -> None:
    """The per-augmentation option lists. Shared with the on-the-fly augmenter (src/dataset/online_augment.py)."""
    for name, values in (("jpeg.quality", p.jpeg.quality), ("blur.sigma", p.blur.sigma),
                         ("resize.scale", p.resize.scale), ("noise.sigma", p.noise.sigma),
                         ("crop.fraction", p.crop.fraction)):
        if not _as_list(values):
            raise ValueError(f"augment.params.{name} must not be empty")
    if not all(0 < q <= 100 for q in _as_list(p.jpeg.quality)):
        raise ValueError("augment.params.jpeg.quality must be in (0, 100]")
    if not all(0 < s <= 1 for s in _as_list(p.resize.scale)):
        raise ValueError("augment.params.resize.scale must be in (0, 1]")
    if not all(0 < f <= 1 for f in _as_list(p.crop.fraction)):
        raise ValueError("augment.params.crop.fraction must be in (0, 1]")
    for key in ("down_filter", "up_filter"):
        if getattr(p.resize, key) not in RESAMPLE:
            raise ValueError(f"augment.params.resize.{key} must be one of {sorted(RESAMPLE)}")
    if isinstance(p.crop.resize_back, str) and p.crop.resize_back not in RESAMPLE:
        raise ValueError(f"augment.params.crop.resize_back must be a bool or one of {sorted(RESAMPLE)}")


def validate(cfg: AugmentCfg) -> None:
    if not Path(cfg.input_dir).is_dir():
        raise ValueError(f"augment.input_dir is not a directory: {cfg.input_dir}")
    if cfg.num_multi < 0:
        raise ValueError("augment.num_multi must be >= 0")
    if cfg.output_format not in ("png", "jpg", "jpeg", "webp", "same"):
        raise ValueError("augment.output_format must be png|jpg|jpeg|webp|same")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cfg.suffix):
        raise ValueError(f"augment.suffix must be a plain filename fragment, got {cfg.suffix!r}")
    validate_multi(cfg.multi)
    validate_params(cfg.params)


def _build(cls, d: Any, where: str):
    """Nested dict -> dataclass, rejecting unknown keys so a typo in the YAML is an error, not a silent default."""
    if d is None:
        d = {}
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(d).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)} (valid: {sorted(names)})")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        hint = hints.get(f.name)
        val = d[f.name]
        if hint is not None and is_dataclass(hint) and isinstance(hint, type):
            val = _build(hint, val, f"{where}.{f.name}")
        kwargs[f.name] = val
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}: {e}") from e


def _coerce_like(old: Any, new: Any) -> Any:
    """Keep the type of the existing value when the override parsed as a string."""
    if isinstance(new, str) and old is not None and not isinstance(old, str):
        try:
            if isinstance(old, bool):
                return new.lower() in ("1", "true", "yes", "on")
            if isinstance(old, int):
                return int(new)
            if isinstance(old, float):
                return float(new)
        except ValueError:
            pass
    return new


def apply_overrides(raw: dict, overrides: list) -> dict:
    """``--set augment.params.blur.sigma=[3.0]`` -> the same dotted key in the raw YAML dict."""
    raw = copy.deepcopy(raw)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set expects key=value, got {ov!r}")
        key, _, value = ov.partition("=")
        parts = key.strip().split(".")
        node = raw
        for p in parts[:-1]:
            if p not in node or node[p] is None:
                node[p] = {}
            if not isinstance(node[p], dict):
                raise ValueError(f"--set {key}: {p} is not a mapping")
            node = node[p]
        parsed = yaml.safe_load(value) if value.strip() != "" else ""
        node[parts[-1]] = _coerce_like(node.get(parts[-1]), parsed)
    return raw


def load_augment_config(path: str, overrides: Optional[list] = None) -> AugmentCfg:
    """``configs/augment.yml`` -> AugmentCfg. Only the ``augment:`` block is read; other top-level keys are ignored,
    so this file can sit next to (or inside) a larger config without tripping its schema."""
    overrides = list(overrides or [])
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    raw = apply_overrides(raw, overrides)
    cfg = _build(AugmentCfg, raw.get("augment") or {}, "augment")
    cfg.config_path = str(Path(path).resolve())
    cfg.overrides = overrides
    validate(cfg)
    return cfg


def default_config_path() -> Path:
    for p in DEFAULT_CONFIGS:
        if p.is_file():
            return p
    raise SystemExit(
        "no config found -- expected one of:\n  "
        + "\n  ".join(str(p) for p in DEFAULT_CONFIGS)
        + "\ncopy configs/augment.yml next to this script, or pass --config PATH"
    )


# --------------------------------------------------------------------------------------
# Planning: filename -> the exact augmentations of its 6 + K outputs
# --------------------------------------------------------------------------------------


def _as_list(v: Any) -> list:
    return list(v) if isinstance(v, (list, tuple)) else [v]


def image_seed(stem: str, seed: int) -> int:
    """Per-image RNG seed. Depends only on the filename and augment.seed, never on the scan order."""
    h = hashlib.blake2b(f"{seed}:{stem}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


@dataclass
class Step:
    name: str
    params: dict

    def label(self) -> str:
        if not self.params:
            return self.name
        keep = {k: v for k, v in self.params.items() if k != "seed"}
        return self.name + "(" + ",".join(f"{k}={v}" for k, v in keep.items()) + ")"


@dataclass
class Plan:
    index: int  # 1-based; becomes the _a{index} suffix
    steps: list

    def label(self) -> str:
        return " -> ".join(s.label() for s in self.steps)


def sample_step(name: str, rng: random.Random, p: ParamsCfg) -> Step:
    if name == "jpeg":
        return Step(name, {"quality": int(rng.choice(_as_list(p.jpeg.quality)))})
    if name == "blur":
        return Step(name, {"sigma": float(rng.choice(_as_list(p.blur.sigma)))})
    if name == "resize":
        return Step(name, {"scale": float(rng.choice(_as_list(p.resize.scale)))})
    if name == "noise":
        # the numpy stream is seeded from here, at plan time, so a resumed run reproduces the same pixels
        return Step(name, {"sigma": float(rng.choice(_as_list(p.noise.sigma))), "seed": rng.getrandbits(63)})
    if name == "jitter":
        j = p.jitter
        return Step(name, {
            "brightness": round(rng.uniform(1 - j.brightness, 1 + j.brightness), 4),
            "contrast": round(rng.uniform(1 - j.contrast, 1 + j.contrast), 4),
            "saturation": round(rng.uniform(1 - j.saturation, 1 + j.saturation), 4),
        })
    if name == "crop":
        return Step(name, {"fraction": float(rng.choice(_as_list(p.crop.fraction)))})
    raise ValueError(f"unknown augmentation {name!r}")


def build_plans(stem: str, cfg: AugmentCfg) -> list:
    """The complete plan for one image: _a1.._a6 (one per augmentation, fixed order) then the K chains."""
    rng = random.Random(image_seed(stem, cfg.seed))
    plans = [Plan(i, [sample_step(name, rng, cfg.params)]) for i, name in enumerate(AUG_ORDER, start=1)]

    m = cfg.multi
    seen: set = set()
    for k in range(cfg.num_multi):
        for _ in range(64):  # a retry budget; 57 subsets of size >= 2 exist, so this only bites at absurd num_multi
            size = rng.randint(m.min_size, m.max_size)
            names = rng.sample(AUG_ORDER, size)  # random.sample already returns them in a random order
            if not m.distinct_subsets or frozenset(names) not in seen:
                break
        seen.add(frozenset(names))
        plans.append(Plan(len(AUG_ORDER) + k + 1, [sample_step(n, rng, cfg.params) for n in names]))
    return plans


# --------------------------------------------------------------------------------------
# The augmentations
# --------------------------------------------------------------------------------------


def aug_jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Encode to JPEG in memory and decode back: the artefacts land in the pixels, the file stays whatever we save."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB")


def aug_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def aug_resize(img: Image.Image, scale: float, cfg: ResizeCfg) -> Image.Image:
    w, h = img.size
    small = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(small, RESAMPLE[cfg.down_filter]).resize((w, h), RESAMPLE[cfg.up_filter])


def aug_noise(img: Image.Image, sigma: float, seed: int, cfg: NoiseCfg) -> Image.Image:
    rng = np.random.default_rng(int(seed))
    a = np.asarray(img, dtype=np.float32) / 255.0
    shape = a.shape if cfg.per_channel else a.shape[:2] + (1,)
    a += rng.normal(0.0, float(sigma), size=shape).astype(np.float32)
    return Image.fromarray(np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB")


def aug_jitter(img: Image.Image, brightness: float, contrast: float, saturation: float) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return ImageEnhance.Color(img).enhance(saturation)


def aug_crop(img: Image.Image, fraction: float, cfg: CropCfg) -> Image.Image:
    w, h = img.size
    cw, ch = max(1, round(w * fraction)), max(1, round(h * fraction))
    left, top = (w - cw) // 2, (h - ch) // 2
    out = img.crop((left, top, left + cw, top + ch))
    if not cfg.resize_back:  # the default: the framing is the augmentation, so the output keeps the smaller size
        return out
    return out.resize((w, h), RESAMPLE[cfg.resize_back if isinstance(cfg.resize_back, str) else "bicubic"])


def apply_step(img: Image.Image, step: Step, p: ParamsCfg) -> Image.Image:
    q = step.params
    if step.name == "jpeg":
        return aug_jpeg(img, q["quality"])
    if step.name == "blur":
        return aug_blur(img, q["sigma"])
    if step.name == "resize":
        return aug_resize(img, q["scale"], p.resize)
    if step.name == "noise":
        return aug_noise(img, q["sigma"], q["seed"], p.noise)
    if step.name == "jitter":
        return aug_jitter(img, q["brightness"], q["contrast"], q["saturation"])
    if step.name == "crop":
        return aug_crop(img, q["fraction"], p.crop)
    raise ValueError(f"unknown augmentation {step.name!r}")


def apply_plan(img: Image.Image, plan: Plan, p: ParamsCfg) -> Image.Image:
    for step in plan.steps:
        img = apply_step(img, step, p)
    return img


# --------------------------------------------------------------------------------------
# Driving one image / the directory
# --------------------------------------------------------------------------------------


def save_image(path: Path, img: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}[path.suffix.lstrip(".").lower()]
    img.save(path, format=fmt, **({"quality": 95} if fmt in ("JPEG", "WEBP") else {}))


def list_sources(cfg: AugmentCfg) -> list:
    """Every image in input_dir that is not already one of our outputs, in a stable (sorted) order."""
    root = Path(cfg.input_dir)
    files = [
        p for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in cfg.extensions and not AUGMENTED_RE.search(p.stem)
    ]
    return files[: cfg.limit] if cfg.limit else files


def augment_one(src: Path, cfg: AugmentCfg) -> tuple:
    """Render the missing outputs of one image. Returns (written, skipped)."""
    plans = build_plans(src.stem, cfg)
    ext = cfg.out_suffix(src)
    todo = [(pl, cfg.out_dir / f"{src.stem}{cfg.suffix}{pl.index}{ext}") for pl in plans]
    if not cfg.overwrite:
        pending = [(pl, dst) for pl, dst in todo if not dst.exists()]
    else:
        pending = todo
    if not pending:
        return 0, len(todo)

    with Image.open(src) as im:
        if im.mode != "RGB":
            log.debug("%s: converting %s -> RGB", src.name, im.mode)
        base = im.convert("RGB")
    for pl, dst in pending:
        save_image(dst, apply_plan(base, pl, cfg.params))
        log.debug("%s  <-  %s", dst.name, pl.label())
    return len(pending), len(todo) - len(pending)


_WORKER_CFG: Optional[AugmentCfg] = None


def _worker_init(cfg: AugmentCfg) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _worker(path_str: str) -> tuple:
    src = Path(path_str)
    try:
        written, skipped = augment_one(src, _WORKER_CFG)
        return src.name, written, skipped, ""
    except Exception as e:  # one unreadable file must not take the run down
        return src.name, 0, 0, f"{type(e).__name__}: {e}"


def run(cfg: AugmentCfg, dry_run: bool = False) -> int:
    sources = list_sources(cfg)
    per_image = len(AUG_ORDER) + cfg.num_multi
    log.info("%d source images in %s", len(sources), cfg.input_dir)
    log.info("%d augmentations per image (%d single + %d chains) -> %d outputs in %s%s",
             per_image, len(AUG_ORDER), cfg.num_multi, len(sources) * per_image, cfg.out_dir,
             " (in place)" if cfg.in_place else "")
    if not sources:
        log.warning("nothing to do")
        return 0

    if dry_run:
        for src in sources[:5]:
            print(f"\n{src.name}")
            for pl in build_plans(src.stem, cfg):
                print(f"  {src.stem}{cfg.suffix}{pl.index}{cfg.out_suffix(src)}  <-  {pl.label()}")
        if len(sources) > 5:
            print(f"\n... and {len(sources) - 5} more images")
        return 0

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    workers = cfg.workers or (os.cpu_count() or 1)
    written = skipped = failed = 0
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(sources), unit="img")
    except ImportError:
        bar = None

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(cfg,)) as pool:
        futures = [pool.submit(_worker, str(p)) for p in sources]
        for i, fut in enumerate(as_completed(futures), start=1):
            name, w, s, err = fut.result()
            written += w
            skipped += s
            if err:
                failed += 1
                log.error("%s: %s", name, err)
            if bar is not None:
                bar.update(1)
                bar.set_postfix(written=written, skipped=skipped, failed=failed, refresh=False)
            elif i % 100 == 0 or i == len(sources):  # no tqdm installed: a plain progress line
                log.info("%d/%d images, %d written, %d already present", i, len(sources), written, skipped)
    if bar is not None:
        bar.close()

    log.info("done: %d written, %d already present, %d images failed", written, skipped, failed)
    return 1 if failed else 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="raw_augment.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--config", default=None, help=f"YAML file with an `augment:` block (default: {DEFAULT_CONFIGS[0]})")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value (dotted key)")
    ap.add_argument("--input-dir", default=None, help="override augment.input_dir")
    ap.add_argument("--output-dir", default=None, help="override augment.output_dir (default: in place)")
    ap.add_argument("--num-multi", type=int, default=None, help="override augment.num_multi (K chains per image)")
    ap.add_argument("--limit", type=int, default=None, help="only the first K source images")
    ap.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    ap.add_argument("--seed", type=int, default=None, help="override augment.seed")
    ap.add_argument("--overwrite", action="store_true", help="re-render outputs that already exist")
    ap.add_argument("--dry-run", action="store_true", help="print the plan of the first few images and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every output file and its augmentations")
    return ap


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout, force=True)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    for flag, key in (("input_dir", "augment.input_dir"), ("output_dir", "augment.output_dir"),
                      ("num_multi", "augment.num_multi"), ("limit", "augment.limit"),
                      ("workers", "augment.workers"), ("seed", "augment.seed")):
        v = getattr(args, flag)
        if v is not None:
            args.set.append(f"{key}={v}")
    if args.overwrite:
        args.set.append("augment.overwrite=true")

    config = args.config or default_config_path()
    cfg = load_augment_config(str(config), args.set)
    log.info("config=%s seed=%d overrides=%s", cfg.config_path, cfg.seed, cfg.overrides or "-")
    return run(cfg, dry_run=args.dry_run)


if __name__ == "__main__":  # guard: the ProcessPoolExecutor workers re-import this module
    raise SystemExit(main())
