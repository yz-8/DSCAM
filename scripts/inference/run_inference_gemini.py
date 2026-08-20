#!/usr/bin/env python3
"""Gemini API inference for InstaBind-Lite question files.

Expected output JSONL rows:
  {"question_id": "...", "prediction": "...", "model": "gemini-3.5-flash", "setting": "full_image"}
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover - only hit on AutoDL dependency issues
    raise SystemExit(
        "Could not import google-genai. Install it with: pip install -U google-genai pillow"
    ) from exc


ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_done_question_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            question_id = row.get("question_id")
            if question_id:
                done.add(question_id)
    return done


def resolve_image_path(image_path: str) -> Path:
    path = ROOT / image_path
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def build_prompt(question: str) -> str:
    if "Final answer:" in question and "Step 1:" in question:
        return question.strip()
    return question.strip() + "\nAnswer with the shortest valid answer only."


def image_to_jpeg_bytes(path: Path, max_image_side: int, jpeg_quality: int) -> bytes:
    with Image.open(path) as image:
        image = image.convert("RGB")
        w, h = image.size
        longest = max(w, h)
        if max_image_side > 0 and longest > max_image_side:
            scale = max_image_side / float(longest)
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        return buf.getvalue()


def response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for cand in candidates:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", []) or []:
            value = getattr(part, "text", None)
            if value:
                parts.append(str(value))
    return "\n".join(parts).strip()


def ask_gemini(
    client: Any,
    model_name: str,
    image_path: Path,
    question: str,
    max_image_side: int,
    jpeg_quality: int,
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
    sleep_base: float,
) -> str:
    image_bytes = image_to_jpeg_bytes(image_path, max_image_side, jpeg_quality)
    prompt = build_prompt(question)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=1.0,
                    max_output_tokens=max_output_tokens,
                ),
            )
            return response_text(response)
        except Exception as exc:  # pragma: no cover - depends on live API behavior
            last_error = exc
            if attempt >= max_retries:
                break
            delay = sleep_base * (2**attempt) + random.uniform(0, 0.5)
            time.sleep(delay)

    raise RuntimeError(f"Gemini API failed after {max_retries + 1} attempts: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--model-name", default="gemini-3.5-flash")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", "--max-output-tokens", dest="max_output_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--sleep-base", type=float, default=2.0)
    parser.add_argument("--sleep-between", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true", help="Append to output and skip completed question_ids.")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env) or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env}=your_api_key before running this script.")

    client = genai.Client(api_key=api_key)
    questions = read_jsonl(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]

    done = read_done_question_ids(args.output) if args.resume else set()
    mode = "a" if args.resume and args.output.exists() else "w"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open(mode, encoding="utf-8") as f:
        for index, row in enumerate(questions, 1):
            question_id = row["question_id"]
            if question_id in done:
                continue

            image_path = resolve_image_path(row["image_path"])
            prediction = ask_gemini(
                client=client,
                model_name=args.model_name,
                image_path=image_path,
                question=row["question"],
                max_image_side=args.max_image_side,
                jpeg_quality=args.jpeg_quality,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
                sleep_base=args.sleep_base,
            )
            out = {
                "question_id": question_id,
                "prediction": prediction,
                "model": args.model_name,
                "setting": args.setting,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{index}/{len(questions)}] {question_id}: {prediction}")
            if args.sleep_between > 0:
                time.sleep(args.sleep_between)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
