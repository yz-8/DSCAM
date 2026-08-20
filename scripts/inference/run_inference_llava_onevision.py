#!/usr/bin/env python3
"""LLaVA-OneVision inference script for InstaBind-Lite question files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


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


def ask_model(
    image_path: Path,
    question: str,
    model: LlavaOnevisionForConditionalGeneration,
    processor: AutoProcessor,
    device: str,
    dtype: torch.dtype,
    max_new_tokens: int,
) -> str:
    image = Image.open(image_path).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": build_prompt(question)},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    encoded = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {}
    for key, value in encoded.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if torch.is_floating_point(value):
                value = value.to(dtype)
        inputs[key] = value

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    generated = output_ids[0][prompt_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--model-path", default="/root/autodl-tmp/models/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument("--model-name", default="llava-onevision-qwen2-7b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    questions = read_jsonl(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]

    done = read_done_question_ids(args.output) if args.resume else set()
    mode = "a" if args.resume and args.output.exists() else "w"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open(mode, encoding="utf-8") as f:
        for index, row in enumerate(questions, 1):
            if row["question_id"] in done:
                continue
            prediction = ask_model(
                image_path=resolve_image_path(row["image_path"]),
                question=row["question"],
                model=model,
                processor=processor,
                device=args.device,
                dtype=dtype,
                max_new_tokens=args.max_new_tokens,
            )
            out = {
                "question_id": row["question_id"],
                "prediction": prediction,
                "model": args.model_name,
                "setting": args.setting,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            if index % 25 == 0:
                print(f"Processed {index}/{len(questions)}")

    print(f"Saved outputs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
