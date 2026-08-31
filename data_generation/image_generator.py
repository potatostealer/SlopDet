#!/usr/bin/env python3
"""Generate one image per caption JSON with HiDream-O1-Image via the fal.ai API.

Standalone replica of the generate stage of the local ``mmrs`` pipeline: it reads the
per-image JSON files written by ``captioner.py`` (fields used: ``prompt``, ``status``,
``width``, ``height``, ``image_id``), plans a generation size from the original image
dimensions (aspect kept, sides floored to multiples of 32, short side >= 256, long
side <= 2048 — the fal endpoint's own grid), and saves the returned image as
``{image_id}.<ext>`` in the output directory. Unlike the local pipeline, the output is
kept at the generated size (no resize back to the original dimensions).

Dependencies:  pip install fal-client
Auth:          export FAL_KEY=...   (https://fal.ai/dashboard/keys)

Usage:
    python image_generator.py ./prompts -o ./generated
    python image_generator.py ./prompts --dry-run   # no network: show planned requests
    python image_generator.py --selftest            # check the resolution planner
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "fal-ai/hidream-o1-image"
USABLE_STATUSES = ("ok", "truncated")  # same downstream filter as the local pipeline
MIN_SIDE, MAX_LONG_SIDE, MULTIPLE = 256, 2048, 32
FALLBACK_SIZE = (1024, 1024)  # used when a record carries no width/height
EXT_BY_CONTENT_TYPE = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def plan_resolution(orig_w: int, orig_h: int) -> tuple[int, int]:
    """Generation size for an original of ``orig_w x orig_h`` (native mode of the local
    pipeline): scale uniformly so the short side is >= MIN_SIDE and the long side is
    <= MAX_LONG_SIDE, then floor each side to a multiple of 32. Raises ``ValueError``
    for aspect ratios that cannot satisfy both bounds (> 8:1)."""
    w, h = float(orig_w), float(orig_h)
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid original size {orig_w}x{orig_h}")
    scale = max(1.0, MIN_SIDE / min(w, h))
    scale = min(scale, MAX_LONG_SIDE / max(w, h))
    gw = int(w * scale) // MULTIPLE * MULTIPLE
    gh = int(h * scale) // MULTIPLE * MULTIPLE
    if min(gw, gh) < MIN_SIDE:
        raise ValueError(f"aspect ratio of {orig_w}x{orig_h} cannot fit "
                         f"min_side={MIN_SIDE} and max_long_side={MAX_LONG_SIDE} (got {gw}x{gh})")
    return gw, gh


def build_arguments(prompt: str, gw: int, gh: int, args) -> dict:
    """The fal request payload. steps/guidance/seed are sent only when the flag was
    given — the endpoint's defaults (50 steps, guidance 5.0) equal the local ones."""
    arguments = {"prompt": prompt, "image_size": {"width": gw, "height": gh}}
    if args.steps is not None:
        arguments["num_inference_steps"] = args.steps
    if args.guidance is not None:
        arguments["guidance_scale"] = args.guidance
    if args.seed is not None:
        arguments["seed"] = args.seed
    return arguments


def pick_extension(url: str, content_type: str | None) -> str:
    ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        return ext
    return EXT_BY_CONTENT_TYPE.get(content_type or "", ".png")


def existing_output(out_dir: Path, image_id: str) -> Path | None:
    for p in sorted(out_dir.glob(f"{image_id}.*")):
        if p.is_file() and not p.name.endswith(".error.txt"):
            return p
    return None


def record_size(rec: dict, image_id: str) -> tuple[int, int]:
    """Generation size for a caption record; may raise ValueError (aspect > 8:1)."""
    try:
        w, h = int(rec.get("width") or 0), int(rec.get("height") or 0)
    except (TypeError, ValueError):
        w = h = 0
    if w > 0 and h > 0:
        return plan_resolution(w, h)
    print(f"warning: {image_id}: no width/height in record -> using "
          f"{FALLBACK_SIZE[0]}x{FALLBACK_SIZE[1]}", file=sys.stderr)
    return FALLBACK_SIZE


def resolution_selftest() -> int:
    # Cases pinned by the local pipeline's tests/test_resolution.py.
    assert plan_resolution(1024, 768) == (1024, 768)
    assert plan_resolution(1024, 685) == (1024, 672)   # floored to /32, aspect ~kept
    assert plan_resolution(100, 50) == (512, 256)      # upscaled so short side is 256
    assert plan_resolution(4000, 3000) == (2048, 1536)  # long side capped at 2048
    assert plan_resolution(2048, 256) == (2048, 256)   # 8:1 is the limit
    try:
        plan_resolution(3000, 300)  # 10:1 cannot fit both bounds
    except ValueError:
        pass
    else:
        raise AssertionError("plan_resolution(3000, 300) should raise ValueError")
    print("resolution selftest: PASS (6 cases)")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts_dir", nargs="?", help="directory of caption JSON files from captioner.py")
    ap.add_argument("-o", "--output-dir", default="generated", help="where the images go (default: %(default)s)")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="fal endpoint id (default: %(default)s)")
    ap.add_argument("--steps", type=int, default=None,
                    help="num_inference_steps; omitted from the request if not given (endpoint default: 50)")
    ap.add_argument("--guidance", type=float, default=None,
                    help="guidance_scale; omitted from the request if not given (endpoint default: 5.0)")
    ap.add_argument("--seed", type=int, default=None, help="seed; omitted from the request if not given")
    ap.add_argument("--timeout", type=float, default=600,
                    help="total per-request timeout in seconds, queue wait included (default: %(default)s)")
    ap.add_argument("--force", action="store_true", help="regenerate even if an output image exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="no network and no writes: show the request that would be sent per record")
    ap.add_argument("--selftest", action="store_true", help="run the built-in resolution self-test and exit")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.selftest:
        return resolution_selftest()
    if not args.prompts_dir:
        sys.exit("error: prompts_dir is required (see --help)")
    prompts_dir = Path(args.prompts_dir)
    if not prompts_dir.is_dir():
        sys.exit(f"error: {prompts_dir} is not a directory")
    record_paths = sorted(prompts_dir.glob("*.json"))
    if not record_paths:
        sys.exit(f"error: no .json files found in {prompts_dir}")

    out_dir = Path(args.output_dir)
    fal_client = None
    if not args.dry_run:
        if not (os.environ.get("FAL_KEY")
                or (os.environ.get("FAL_KEY_ID") and os.environ.get("FAL_KEY_SECRET"))):
            sys.exit("error: FAL_KEY is not set. Get a key at https://fal.ai/dashboard/keys "
                     "and run: export FAL_KEY=...")
        import fal_client  # pip install fal-client; imported lazily so --dry-run works without it
        out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"generated": 0, "skipped_status": 0, "skipped_done": 0, "error": 0}
    for path in record_paths:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"warning: {path.name}: unreadable JSON ({e!r}) -> skipped", file=sys.stderr)
            counts["error"] += 1
            continue
        image_id = str(rec.get("image_id") or path.stem)
        prompt = (rec.get("prompt") or "").strip()
        if rec.get("status") not in USABLE_STATUSES or not prompt:
            counts["skipped_status"] += 1
            print(f"{image_id}: status={rec.get('status')} -> skipped")
            continue
        if not args.dry_run and not args.force:
            done = existing_output(out_dir, image_id)
            if done is not None:
                counts["skipped_done"] += 1
                print(f"{image_id}: {done.name} exists -> skipped")
                continue
        try:
            gw, gh = record_size(rec, image_id)
        except ValueError as e:
            counts["error"] += 1
            print(f"{image_id}: ERROR {e}", file=sys.stderr)
            if not args.dry_run:
                (out_dir / f"{image_id}.error.txt").write_text(repr(e), encoding="utf-8")
            continue
        arguments = build_arguments(prompt, gw, gh, args)
        if args.dry_run:
            shown = dict(arguments, prompt=prompt[:80] + ("..." if len(prompt) > 80 else ""))
            print(f"{image_id}: {rec.get('width') or '?'}x{rec.get('height') or '?'} -> {gw}x{gh}; "
                  f"{args.endpoint} <- {json.dumps(shown)}")
            continue
        try:
            result = fal_client.subscribe(args.endpoint, arguments=arguments,
                                          client_timeout=args.timeout)
            image = result["images"][0]
            out_path = out_dir / f"{image_id}{pick_extension(image.get('url', ''), image.get('content_type'))}"
            urllib.request.urlretrieve(image["url"], out_path)
            marker = out_dir / f"{image_id}.error.txt"
            if marker.exists():
                marker.unlink()
            counts["generated"] += 1
            print(f"{image_id}: saved {out_path.name} ({gw}x{gh})")
        except Exception as e:  # one failure never aborts the run
            counts["error"] += 1
            (out_dir / f"{image_id}.error.txt").write_text(repr(e), encoding="utf-8")
            print(f"{image_id}: ERROR {e!r}", file=sys.stderr)

    if args.dry_run:
        print(f"dry run: {len(record_paths) - counts['skipped_status'] - counts['error']} "
              f"image(s) would be generated")
        return 0
    print("done: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    attempted = counts["generated"] + counts["error"]
    return 1 if attempted and not counts["generated"] else 0


if __name__ == "__main__":
    sys.exit(main())
