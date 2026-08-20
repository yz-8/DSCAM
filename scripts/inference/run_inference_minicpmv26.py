#!/usr/bin/env python3
"""MiniCPM-V-2.6 inference script for InstaBind-Lite question files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel


ROOT = Path(__file__).resolve().parents[2]


def patch_transformers_tied_weights_compat() -> None:
    """Patch newer transformers expectations for MiniCPM remote model code."""

    def get_all_tied_weights_keys(self) -> dict[str, None]:
        keys = getattr(self, "_all_tied_weights_keys_compat", None)
        if keys is None:
            keys = getattr(self, "_tied_weights_keys", None)
        if not keys:
            return {}
        if isinstance(keys, dict):
            return keys
        return {str(key): None for key in keys}

    def set_all_tied_weights_keys(self, value) -> None:
        self.__dict__["_all_tied_weights_keys_compat"] = value

    PreTrainedModel.all_tied_weights_keys = property(  # type: ignore[attr-defined]
        get_all_tied_weights_keys,
        set_all_tied_weights_keys,
    )


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
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> str:
    image = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [image, build_prompt(question)]}]
    kwargs = {
        "image": None,
        "msgs": msgs,
        "tokenizer": tokenizer,
        "sampling": False,
        "max_new_tokens": max_new_tokens,
    }
    try:
        response = model.chat(**kwargs)
    except TypeError:
        kwargs.pop("max_new_tokens", None)
        try:
            response = model.chat(**kwargs)
        except TypeError:
            kwargs.pop("sampling", None)
            response = model.chat(**kwargs)
    return str(response).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--model-path", default="/root/autodl-tmp/models/MiniCPM-V-2_6")
    parser.add_argument("--model-name", default="minicpm-v-2.6")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    patch_transformers_tied_weights_compat()

    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

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
                tokenizer=tokenizer,
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
