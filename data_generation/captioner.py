#!/usr/bin/env python3
"""Caption every image in a directory with Qwen3.8-27B via the OpenRouter API.

Standalone replica of the caption stage of the local ``mmrs`` pipeline: each image is
sent as a base64 data URL together with the system prompt from
``img2txt_prompt_simplified.md``; the model answers with an ``<analysis>...</analysis>``
block followed by ``<prompt>...</prompt>``, which is parsed into one JSON file per
image. The JSON directory this script produces is the input of ``image_generator.py``.

Dependencies:  pip install openrouter pillow
Auth:          export OPENROUTER_API_KEY=...   (https://openrouter.ai/keys)

Usage:
    python captioner.py ./my_images -o ./prompts
    python captioner.py ./my_images --dry-run    # no network: show what would be sent
    python captioner.py --selftest               # check the output parser, no network
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_USER_TEXT = "Analyze the attached image and produce the output."
DEFAULT_SYSTEM_PROMPT = Path(__file__).resolve().parent / "img2txt_prompt_simplified.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# Output parsing, ported verbatim from mmrs/prompt_parser.py: the model emits
# <analysis>...</analysis> then <prompt>...</prompt>, or <error>NO_IMAGE</error>.
# The prompt is taken from the LAST <prompt> tag because the system prompt shows the
# literal tag and the model occasionally echoes it inside the analysis.
PROMPT_OPEN_RE = re.compile(r"<prompt(?:\s[^>]*)?>", re.I)
PROMPT_CLOSE_RE = re.compile(r"</prompt\s*>", re.I)
ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.S | re.I)
ERROR_RE = re.compile(r"<error>(.*?)</error>", re.S | re.I)
THINK_END_RE = re.compile(r"</think>", re.I)
FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*$", re.M)
WS_RE = re.compile(r"\s+")


def strip_thinking(text: str) -> str:
    """Drop everything up to the last ``</think>``."""
    last = None
    for m in THINK_END_RE.finditer(text):
        last = m
    return text[last.end():] if last else text


def clean_text(text: str) -> str:
    text = FENCE_RE.sub("", text)
    return WS_RE.sub(" ", text).strip()


def parse_output(text: str) -> dict:
    """-> {"prompt": str|None, "analysis": str|None, "error": str|None, "truncated": bool}."""
    body = strip_thinking(text or "")
    am = ANALYSIS_RE.search(body)
    em = ERROR_RE.search(body)
    prompt, truncated = None, False
    opens = list(PROMPT_OPEN_RE.finditer(body))
    if opens:
        tail = body[opens[-1].end():]
        cm = PROMPT_CLOSE_RE.search(tail)
        if cm:
            prompt = clean_text(tail[: cm.start()])
        else:
            prompt = clean_text(tail)
            truncated = True
        if not prompt:
            prompt = None
    return {
        "prompt": prompt,
        "analysis": am.group(1).strip() if am else None,
        "error": em.group(1).strip() if em else None,
        "truncated": truncated,
    }


def status_of(parsed: dict, min_words: int) -> str:
    """``ok`` | ``truncated`` (unclosed tag, still usable) | ``failed`` (no usable prompt)."""
    prompt = parsed["prompt"]
    if not prompt or len(prompt.split()) < int(min_words):
        return "failed"
    return "truncated" if parsed["truncated"] else "ok"


def find_images(images_dir: Path) -> list[Path]:
    images, seen = [], {}
    for p in sorted(images_dir.iterdir()):
        if not p.is_file() or p.name.startswith(".") or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.stem in seen:
            print(f"warning: {p.name} has the same stem as {seen[p.stem].name} -> skipped", file=sys.stderr)
            continue
        seen[p.stem] = p
        images.append(p)
    return images


def encode_image(path: Path, max_pixels: int) -> tuple[str, int, int]:
    """-> (jpeg data URL, original width, original height).

    Same downscale as the local pipeline: the model sees at most ``max_pixels``, the
    recorded width/height stay those of the original image.
    """
    with Image.open(path) as im:
        orig_w, orig_h = im.size
        if im.mode != "RGB":
            im = im.convert("RGB")
        if max_pixels and orig_w * orig_h > max_pixels:
            s = math.sqrt(max_pixels / float(orig_w * orig_h))
            im = im.resize((max(1, int(orig_w * s)), max(1, int(orig_h * s))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", orig_w, orig_h


def build_messages(data_url: str, system_prompt: str, user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": user_text},
        ]},
    ]


def content_text(content) -> str:
    """The assistant content as plain text (it is normally a string)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content if isinstance(content, list) else [content]:
        text = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def call_openrouter(client, data_url: str, system_prompt: str, args) -> str:
    # No stop=["</prompt>"] here on purpose: the local vLLM run kept the stop string in
    # the output, but hosted APIs strip it, which would wrongly mark every caption as
    # truncated. Without a stop the model just emits the closing tag itself.
    response = client.chat.send(
        model=args.model,
        messages=build_messages(data_url, system_prompt, args.user_text),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_ms=int(args.timeout * 1000),
    )
    if not response.choices:
        raise RuntimeError("OpenRouter returned no choices")
    text = content_text(response.choices[0].message.content)
    if not text.strip():
        raise RuntimeError("OpenRouter returned empty content")
    return text


def base_record(img_path: Path, args) -> dict:
    return {
        "image_id": img_path.stem,
        "source_image": str(img_path.resolve()),
        "width": 0,
        "height": 0,
        "status": "error",
        "error": None,
        "prompt": None,
        "analysis": None,
        "truncated": False,
        "n_words": 0,
        "raw_output": "",
        "attempts": 0,
        "seconds": 0.0,
        "model": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generation_params": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }


def caption_one(client, img_path: Path, system_prompt: str, args) -> dict:
    record = base_record(img_path, args)
    t0 = time.time()
    data_url, record["width"], record["height"] = encode_image(img_path, args.max_pixels)
    parsed = {"prompt": None, "analysis": None, "error": None, "truncated": False}
    raw, attempts, transport_error = "", 0, None
    for _ in range(1 + args.retries):
        attempts += 1
        try:
            raw = call_openrouter(client, data_url, system_prompt, args)
        except Exception as e:  # count transport failures as attempts too
            transport_error = e
            continue
        transport_error = None
        parsed = parse_output(raw)
        if status_of(parsed, args.min_words) != "failed":
            break
    if transport_error is not None:
        record["status"], record["error"] = "error", repr(transport_error)
    else:
        record["status"], record["error"] = status_of(parsed, args.min_words), parsed["error"]
    record.update(
        prompt=parsed["prompt"],
        analysis=parsed["analysis"],
        truncated=parsed["truncated"],
        n_words=len(parsed["prompt"].split()) if parsed["prompt"] else 0,
        raw_output=raw,
        attempts=attempts,
        seconds=round(time.time() - t0, 2),
    )
    return record


def parser_selftest() -> int:
    words = " ".join(f"word{i}" for i in range(25))
    ok = parse_output(f"<think>hidden</think><analysis>MEDIUM: photo</analysis>\n<prompt>{words}</prompt>")
    assert status_of(ok, 20) == "ok" and ok["analysis"] == "MEDIUM: photo" and ok["prompt"] == words, ok
    trunc = parse_output(f"<analysis>a</analysis><prompt>{words}")
    assert status_of(trunc, 20) == "truncated" and trunc["prompt"] == words, trunc
    err = parse_output("<error>NO_IMAGE</error>")
    assert status_of(err, 20) == "failed" and err["error"] == "NO_IMAGE" and err["prompt"] is None, err
    echoed = parse_output(
        f"<analysis>end with <prompt>...</prompt> as shown</analysis> <prompt>{words}</prompt>"
    )
    assert status_of(echoed, 20) == "ok" and echoed["prompt"] == words, echoed
    short = parse_output("<prompt>too few words</prompt>")
    assert status_of(short, 20) == "failed", short
    print("parser selftest: PASS (5 cases)")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images_dir", nargs="?", help="directory containing the input images")
    ap.add_argument("-o", "--output-dir", default="prompts", help="where the per-image JSON files go (default: %(default)s)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id (default: %(default)s)")
    ap.add_argument("--system-prompt", default=str(DEFAULT_SYSTEM_PROMPT),
                    help="file with the captioning system prompt (default: %(default)s)")
    ap.add_argument("--user-text", default=DEFAULT_USER_TEXT, help="user message text (default: %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=3000,
                    help="completion token budget; the prompt asks for >= ~2500 (default: %(default)s)")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (default: %(default)s)")
    ap.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling (default: %(default)s)")
    ap.add_argument("--max-pixels", type=int, default=1048576,
                    help="downscale images to at most this many pixels before sending (default: %(default)s)")
    ap.add_argument("--min-words", type=int, default=20,
                    help="prompts shorter than this are marked failed (default: %(default)s)")
    ap.add_argument("--retries", type=int, default=1,
                    help="extra attempts after a failed parse or transport error (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=300, help="per-request timeout in seconds (default: %(default)s)")
    ap.add_argument("--force", action="store_true", help="re-caption images that already have a good JSON")
    ap.add_argument("--dry-run", action="store_true",
                    help="no network and no writes: encode images and show what would be sent")
    ap.add_argument("--selftest", action="store_true", help="run the built-in parser self-test and exit")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.selftest:
        return parser_selftest()
    if not args.images_dir:
        sys.exit("error: images_dir is required (see --help)")
    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        sys.exit(f"error: {images_dir} is not a directory")
    system_prompt_path = Path(args.system_prompt)
    if not system_prompt_path.is_file():
        sys.exit(f"error: system prompt file not found: {system_prompt_path} (pass --system-prompt)")
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    images = find_images(images_dir)
    if not images:
        sys.exit(f"error: no images found in {images_dir} (looked for {', '.join(sorted(IMAGE_EXTS))})")

    out_dir = Path(args.output_dir)
    client_ctx = contextlib.nullcontext(None)  # dry runs never touch the network
    if not args.dry_run:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            sys.exit("error: OPENROUTER_API_KEY is not set. Get a key at https://openrouter.ai/keys "
                     "and run: export OPENROUTER_API_KEY=...")
        from openrouter import OpenRouter  # pip install openrouter
        client_ctx = OpenRouter(api_key=api_key)
        out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "truncated": 0, "failed": 0, "error": 0, "skipped": 0}
    with client_ctx as client:
        for img in images:
            out_json = out_dir / f"{img.stem}.json"
            if not args.force and out_json.exists():
                try:
                    prev_status = json.loads(out_json.read_text(encoding="utf-8")).get("status")
                except Exception:
                    prev_status = None
                if prev_status in ("ok", "truncated"):  # re-runs retry failed/error items
                    counts["skipped"] += 1
                    print(f"{img.name}: already captioned ({prev_status}) -> skipped")
                    continue
            if args.dry_run:
                data_url, w, h = encode_image(img, args.max_pixels)
                n_msgs = len(build_messages(data_url, system_prompt, args.user_text))
                print(f"{img.name}: {w}x{h}, data URL {len(data_url)} chars, {n_msgs} messages "
                      f"(system {len(system_prompt)} chars + user [image_url, text]), "
                      f"model={args.model}, max_tokens={args.max_tokens} -> would write {out_json}")
                continue
            try:
                record = caption_one(client, img, system_prompt, args)
            except Exception as e:  # e.g. unreadable image: record it, keep going
                record = base_record(img, args)
                record["error"] = repr(e)
            out_json.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            counts[record["status"]] += 1
            print(f"{img.name}: {record['status']} ({record['n_words']} words, "
                  f"{record['seconds']}s, {record['attempts']} attempt(s))")

    if args.dry_run:
        print(f"dry run: {len(images) - counts['skipped']} image(s) would be captioned, "
              f"{counts['skipped']} already done")
        return 0
    print("done: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    attempted = sum(counts.values()) - counts["skipped"]
    return 1 if attempted and not (counts["ok"] + counts["truncated"]) else 0


if __name__ == "__main__":
    sys.exit(main())
