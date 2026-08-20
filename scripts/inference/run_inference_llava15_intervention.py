#!/usr/bin/env python3
"""LLaVA-1.5 inference script for intervention question files.

Use on AutoDL after installing/running LLaVA:

  python scripts/run_inference_llava15_intervention.py \
    --questions data/questions.l1_l3.box_guided.jsonl \
    --output model_outputs/box_guided_outputs.jsonl \
    --setting box_guided
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


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


def ask_model(
    image_path: Path,
    question: str,
    model,
    tokenizer,
    image_processor,
    conv_mode: str,
    max_new_tokens: int,
) -> str:
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()
    qs = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--setting",
        required=True,
        choices=[
            "full_image",
            "full_image_l1_l3",
            "box_guided",
            "crop_oracle",
            "context_crop",
            "dim_non_target",
            "instance_first",
        ],
    )
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-name", default="llava-1.5-7b")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="Append to output and skip completed question_ids.")
    args = parser.parse_args()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device="cuda",
    )

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
            image_path = ROOT / row["image_path"]
            if "Final answer:" in row["question"] and "Step 1:" in row["question"]:
                prompt = row["question"]
            else:
                prompt = row["question"] + "\nAnswer with the shortest valid answer only."
            prediction = ask_model(
                image_path=image_path,
                question=prompt,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                conv_mode=args.conv_mode,
                max_new_tokens=args.max_new_tokens,
            )
            out = {
                "question_id": row["question_id"],
                "prediction": prediction,
                "model": args.model_name,
                "setting": args.setting,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if index % 50 == 0:
                print(f"Processed {index}/{len(questions)}")
    print(f"Saved outputs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
