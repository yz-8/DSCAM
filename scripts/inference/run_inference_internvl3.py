#!/usr/bin/env python3
"""InternVL3 inference script for InstaBind-Lite question files.

Expected output JSONL rows:
  {"question_id": "...", "prediction": "...", "model": "internvl3-8b", "setting": "full_image"}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel


ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def patch_transformers_tied_weights_compat() -> None:
    """Patch newer transformers expectations for InternVL remote model code."""
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


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file: Path, input_size: int, max_num: int) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def resolve_image_path(image_path: str) -> Path:
    path = ROOT / image_path
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def build_prompt(question: str) -> str:
    if "Final answer:" in question and "Step 1:" in question:
        return "<image>\n" + question.strip()
    return "<image>\n" + question.strip() + "\nAnswer with the shortest valid answer only."


def ask_model(
    image_path: Path,
    question: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    image_size: int,
    max_tiles: int,
    dtype: torch.dtype,
    max_new_tokens: int,
) -> str:
    pixel_values = load_image(image_path, input_size=image_size, max_num=max_tiles).to(dtype).cuda()
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    response = model.chat(
        tokenizer,
        pixel_values,
        build_prompt(question),
        generation_config,
    )
    return str(response).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--model-path", default="/root/autodl-tmp/models/InternVL3-8B")
    parser.add_argument("--model-name", default="internvl3-8b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-tiles", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    patch_transformers_tied_weights_compat()

    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
        fix_mistral_regex=True,
    )
    if args.load_in_8bit:
        print("Warning: --load-in-8bit is ignored for InternVL3 because this remote model class does not accept it.")
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()

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
            prediction = ask_model(
                image_path=resolve_image_path(row["image_path"]),
                question=row["question"],
                model=model,
                tokenizer=tokenizer,
                image_size=args.image_size,
                max_tiles=args.max_tiles,
                dtype=dtype,
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
