#!/usr/bin/env python3
"""AutoDL Gemini-compatible REST inference for InstaBind-Lite.

Expected output JSONL rows:
  {"question_id": "...", "prediction": "...", "model": "gemini-3.5-flash", "setting": "full_image"}
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
from urllib.parse import urlparse

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


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


def image_to_base64(path: Path, max_image_side: int, jpeg_quality: int) -> str:
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
        return base64.b64encode(buf.getvalue()).decode("ascii")


def build_generate_url(api_base: str, model_name: str) -> str:
    api_base = api_base.rstrip("/")
    if api_base.endswith("/models"):
        return f"{api_base}/{model_name}:generateContent"
    if api_base.endswith(f"/models/{model_name}"):
        return f"{api_base}:generateContent"
    return f"{api_base}/models/{model_name}:generateContent"


def extract_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for cand in data.get("candidates", []) or []:
        content = cand.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            if "text" in part and part["text"] is not None:
                parts.append(str(part["text"]))
    if parts:
        return "\n".join(parts).strip()
    if "text" in data and data["text"] is not None:
        return str(data["text"]).strip()
    return ""


def auth_variants(api_key: str, auth_mode: str) -> list[tuple[str, dict[str, str], dict[str, str]]]:
    variants = {
        "bearer": ("bearer", {"Authorization": f"Bearer {api_key}"}, {}),
        "query_key": ("query_key", {}, {"key": api_key}),
        "x-api-key": ("x-api-key", {"x-api-key": api_key}, {}),
    }
    if auth_mode != "auto":
        return [variants[auth_mode]]
    return [variants["bearer"], variants["query_key"], variants["x-api-key"]]


def ask_autodl_gemini(
    session: requests.Session,
    url: str,
    api_key: str,
    auth_mode: str,
    image_path: Path,
    question: str,
    max_image_side: int,
    jpeg_quality: int,
    temperature: float,
    max_output_tokens: int,
    timeout: float,
    max_retries: int,
    sleep_base: float,
    thinking_budget: int | None,
    debug_empty_jsonl: Path | None,
) -> tuple[str, str]:
    image_b64 = image_to_base64(image_path, max_image_side, jpeg_quality)
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                    {"text": build_prompt(question)},
                ],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "topP": 1.0,
            "maxOutputTokens": max_output_tokens,
        },
    }
    if thinking_budget is not None:
        payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    last_error: str = ""
    variants = auth_variants(api_key, auth_mode)

    for attempt in range(max_retries + 1):
        for mode_name, headers, params in variants:
            try:
                response = session.post(url, headers=headers, params=params, json=payload, timeout=timeout)
                if response.status_code in {401, 403} and auth_mode == "auto":
                    last_error = f"{response.status_code} with {mode_name}: {response.text[:300]}"
                    continue
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"{response.status_code}: {response.text[:300]}"
                    continue
                response.raise_for_status()
                data = response.json()
                text = extract_text(data)
                if text:
                    return text, mode_name
                compact = {
                    "mode": mode_name,
                    "status_code": response.status_code,
                    "promptFeedback": data.get("promptFeedback"),
                    "candidates": [
                        {
                            "finishReason": cand.get("finishReason"),
                            "safetyRatings": cand.get("safetyRatings"),
                            "content": cand.get("content"),
                        }
                        for cand in (data.get("candidates") or [])[:2]
                    ],
                }
                if debug_empty_jsonl is not None:
                    debug_empty_jsonl.parent.mkdir(parents=True, exist_ok=True)
                    with debug_empty_jsonl.open("a", encoding="utf-8") as debug_f:
                        debug_f.write(json.dumps(compact, ensure_ascii=False) + "\n")
                last_error = f"empty text response with {mode_name}: {json.dumps(compact, ensure_ascii=False)[:500]}"
                continue
            except Exception as exc:  # pragma: no cover - depends on live API behavior
                last_error = str(exc)

        if attempt >= max_retries:
            break
        delay = sleep_base * (2**attempt) + random.uniform(0, 0.5)
        print(f"Retry after error: {last_error}. Sleep {delay:.1f}s", flush=True)
        time.sleep(delay)

    raise RuntimeError(f"AutoDL Gemini API failed after {max_retries + 1} attempts. Last error: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--model-name", default="gemini-3.5-flash")
    parser.add_argument("--api-base", default="https://www.autodl.art/api/v1/gemini/v1beta/models")
    parser.add_argument("--api-key-env", default="AUTODL_API_KEY")
    parser.add_argument("--auth-mode", choices=["auto", "bearer", "query_key", "x-api-key"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", "--max-output-tokens", dest="max_output_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-base", type=float, default=2.0)
    parser.add_argument("--sleep-between", type=float, default=0.0)
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--debug-empty-jsonl", type=Path, default=None)
    parser.add_argument("--redo-empty", action="store_true", help="With --resume, re-run rows whose saved prediction is empty.")
    parser.add_argument("--resume", action="store_true", help="Append to output and skip completed question_ids.")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env) or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env}=your_autodl_api_key before running this script.")

    url = build_generate_url(args.api_base, args.model_name)
    parsed = urlparse(url)
    print(f"AutoDL Gemini URL: {parsed.scheme}://{parsed.netloc}{parsed.path}", flush=True)
    print(f"Auth mode: {args.auth_mode}", flush=True)

    questions = read_jsonl(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]

    done = read_done_question_ids(args.output, args.redo_empty) if args.resume else set()
    mode = "a" if args.resume and args.output.exists() else "w"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    active_auth_mode = args.auth_mode
    with args.output.open(mode, encoding="utf-8") as f:
        for index, row in enumerate(questions, 1):
            question_id = row["question_id"]
            if question_id in done:
                continue

            image_path = resolve_image_path(row["image_path"])
            print(f"[{index}/{len(questions)}] request {question_id}", flush=True)
            prediction, used_auth_mode = ask_autodl_gemini(
                session=session,
                url=url,
                api_key=api_key,
                auth_mode=active_auth_mode,
                image_path=image_path,
                question=row["question"],
                max_image_side=args.max_image_side,
                jpeg_quality=args.jpeg_quality,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                sleep_base=args.sleep_base,
                thinking_budget=args.thinking_budget,
                debug_empty_jsonl=args.debug_empty_jsonl,
            )
            if active_auth_mode == "auto":
                active_auth_mode = used_auth_mode
                print(f"Using auth mode: {active_auth_mode}", flush=True)
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
