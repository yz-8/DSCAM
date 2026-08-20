#!/usr/bin/env python3
"""AutoDL OpenAI-compatible vision inference for InstaBind-Lite.

Expected output JSONL rows:
  {"question_id": "...", "prediction": "...", "model": "qwen3-vl-plus", "setting": "full_image"}
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


class BadRequestError(RuntimeError):
    """Permanent request error that should not be retried."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_done_question_ids(path: Path, redo_empty: bool) -> set[str]:
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
            if redo_empty and not str(row.get("prediction", "")).strip():
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
    return (
        question.strip()
        + "\nReturn exactly one short answer. Do not explain."
        + "\nIf this is a yes/no question, answer exactly Yes or No."
        + "\nIf this is a multiple-choice question, answer exactly A, B, C, or D."
        + "\nIf this asks for a color, answer with one color word."
        + "\nNever return an empty answer."
    )


def image_to_data_url(path: Path, max_image_side: int, jpeg_quality: int, min_image_side: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        w, h = image.size
        longest = max(w, h)
        if max_image_side > 0 and longest > max_image_side:
            scale = max_image_side / float(longest)
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        w, h = image.size
        if min_image_side > 0 and (w < min_image_side or h < min_image_side):
            canvas_w = max(w, min_image_side)
            canvas_h = max(h, min_image_side)
            canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
            canvas.paste(image, ((canvas_w - w) // 2, (canvas_h - h) // 2))
            image = canvas
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


def extract_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if choices:
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    value = part.get("text") or part.get("content")
                    if value:
                        parts.append(str(value))
                elif part is not None:
                    parts.append(str(part))
            return "\n".join(parts).strip()
        if choice.get("text"):
            return str(choice["text"]).strip()

    if data.get("output_text"):
        return str(data["output_text"]).strip()
    if data.get("text"):
        return str(data["text"]).strip()
    return ""


def build_payload(
    model_name: str,
    image_url: str,
    question: str,
    temperature: float,
    max_output_tokens: int,
    disable_thinking: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": build_prompt(question)},
                ],
            }
        ],
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_output_tokens,
        "stream": False,
    }
    if disable_thinking:
        # Qwen-compatible providers commonly accept this; most OpenAI-compatible
        # routers ignore unknown fields. Disable via CLI if the endpoint rejects it.
        payload["enable_thinking"] = False
    return payload


def ask_model(
    session: requests.Session,
    url: str,
    api_key: str,
    model_name: str,
    image_path: Path,
    question: str,
    max_image_side: int,
    min_image_side: int,
    jpeg_quality: int,
    temperature: float,
    max_output_tokens: int,
    timeout: float,
    max_retries: int,
    sleep_base: float,
    disable_thinking: bool,
    debug_jsonl: Path | None,
) -> str:
    image_url = image_to_data_url(image_path, max_image_side, jpeg_quality, min_image_side)
    payload = build_payload(
        model_name=model_name,
        image_url=image_url,
        question=question,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        disable_thinking=disable_thinking,
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = session.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 400 and disable_thinking:
                # Some routers reject provider-specific fields. Retry once without it.
                payload.pop("enable_thinking", None)
                disable_thinking = False
                response = session.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 400:
                message = f"400 Bad Request: {response.text[:1000]}"
                if debug_jsonl is not None:
                    debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
                    with debug_jsonl.open("a", encoding="utf-8") as debug_f:
                        debug_f.write(json.dumps({"status_code": 400, "body": response.text[:2000]}, ensure_ascii=False) + "\n")
                raise BadRequestError(message)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"{response.status_code}: {response.text[:500]}"
            else:
                response.raise_for_status()
                data = response.json()
                text = extract_text(data)
                if text:
                    return text
                compact = {
                    "status_code": response.status_code,
                    "choices": data.get("choices"),
                    "usage": data.get("usage"),
                    "raw_keys": list(data.keys()),
                }
                if debug_jsonl is not None:
                    debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
                    with debug_jsonl.open("a", encoding="utf-8") as debug_f:
                        debug_f.write(json.dumps(compact, ensure_ascii=False) + "\n")
                last_error = f"empty text response: {json.dumps(compact, ensure_ascii=False)[:500]}"
        except BadRequestError:
            raise
        except Exception as exc:  # pragma: no cover - depends on live API behavior
            last_error = str(exc)

        if attempt >= max_retries:
            break
        delay = sleep_base * (2**attempt) + random.uniform(0, 0.5)
        print(f"Retry after error: {last_error}. Sleep {delay:.1f}s", flush=True)
        time.sleep(delay)

    raise RuntimeError(f"AutoDL OpenAI-compatible vision API failed after {max_retries + 1} attempts. Last error: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--model-name", default="qwen3-vl-plus")
    parser.add_argument("--api-base", default="https://www.autodl.art/api/v1")
    parser.add_argument("--api-key-env", default="AUTODL_API_KEY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", "--max-output-tokens", dest="max_output_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--min-image-side", type=int, default=64)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--sleep-base", type=float, default=2.0)
    parser.add_argument("--sleep-between", type=float, default=0.05)
    parser.add_argument("--debug-jsonl", type=Path, default=None)
    parser.add_argument("--redo-empty", action="store_true", help="With --resume, re-run rows whose saved prediction is empty.")
    parser.add_argument("--allow-thinking", action="store_true", help="Do not send enable_thinking=false.")
    parser.add_argument("--resume", action="store_true", help="Append to output and skip completed question_ids.")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env}=your_autodl_api_key before running this script.")

    url = args.api_base.rstrip("/") + "/chat/completions"
    print(f"AutoDL OpenAI-compatible URL: {url}", flush=True)
    print(f"Model: {args.model_name}", flush=True)

    questions = read_jsonl(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]

    done = read_done_question_ids(args.output, args.redo_empty) if args.resume else set()
    mode = "a" if args.resume and args.output.exists() else "w"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    with args.output.open(mode, encoding="utf-8") as f:
        for index, row in enumerate(questions, 1):
            question_id = row["question_id"]
            if question_id in done:
                continue

            image_path = resolve_image_path(row["image_path"])
            print(f"[{index}/{len(questions)}] request {question_id}", flush=True)
            prediction = ask_model(
                session=session,
                url=url,
                api_key=api_key,
                model_name=args.model_name,
                image_path=image_path,
                question=row["question"],
                max_image_side=args.max_image_side,
                min_image_side=args.min_image_side,
                jpeg_quality=args.jpeg_quality,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                sleep_base=args.sleep_base,
                disable_thinking=not args.allow_thinking,
                debug_jsonl=args.debug_jsonl,
            )
            out = {
                "question_id": question_id,
                "prediction": prediction,
                "model": args.model_name,
                "setting": args.setting,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{index}/{len(questions)}] answer {question_id}: {prediction}", flush=True)
            if args.sleep_between > 0:
                time.sleep(args.sleep_between)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
