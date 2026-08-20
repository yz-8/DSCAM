#!/usr/bin/env python3
"""Qwen2.5-VL inference script for InstaBind-Lite question files.

Expected output JSONL rows:
  {"question_id": "...", "prediction": "...", "model": "qwen2.5-vl-7b-instruct", "setting": "full_image"}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except ImportError as exc:  # pragma: no cover - only hit on AutoDL dependency issues
    raise SystemExit(
        "Could not import Qwen2_5_VLForConditionalGeneration. "
        "Install a recent transformers build, for example: "
        "pip install -U transformers accelerate qwen-vl-utils"
    ) from exc

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover - only hit on AutoDL dependency issues
    raise SystemExit(
        "Could not import qwen_vl_utils. Install it with: pip install -U qwen-vl-utils"
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


def ask_model(
    image_path: Path,
    question: str,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    device: str,
    max_new_tokens: int,
) -> str:
    # Open once before passing the path to catch corrupt/missing images early.
    with Image.open(image_path) as image:
        image.verify()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": build_prompt(question)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--model-path", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-name", default="qwen2.5-vl-7b-instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attn-implementation", default=None, help="Optional, e.g. flash_attention_2.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Append to output and skip completed question_ids.")
    args = parser.parse_args()

    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map[args.torch_dtype],
        "device_map": "auto",
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_path)

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
            prediction = ask_model(
                image_path=image_path,
                question=row["question"],
                model=model,
                processor=processor,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            out = {
                "question_id": question_id,
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
