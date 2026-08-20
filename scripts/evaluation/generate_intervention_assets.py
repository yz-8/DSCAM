#!/usr/bin/env python3
"""Generate box-guided and crop-oracle intervention assets.

Run inside the AutoDL export root after `images/` has been populated:

  python scripts/generate_intervention_assets.py

This creates:

- data/questions.l1_l3.full_image.jsonl
- data/questions.l1_l3.box_guided.jsonl
- data/questions.l1_l3.crop_oracle.jsonl
- data/questions.l1_l3.context_crop.jsonl
- data/questions.l1_l3.dim_non_target.jsonl
- interventions/box_guided/*.jpg
- interventions/crop_oracle/*.jpg
- interventions/context_crop/*.jpg
- interventions/dim_non_target/*.jpg
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_filename(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)
    return text.strip("_")


def build_bbox_index(annotations: dict[str, Any]) -> dict[tuple[str, str, str], list[float]]:
    index: dict[tuple[str, str, str], list[float]] = {}
    for image in annotations.get("images", []):
        image_id = image["image_id"]
        for group in image.get("same_class_groups", []):
            group_id = group["group_id"]
            for inst in group.get("instances", []):
                index[(image_id, group_id, inst["instance_id"])] = inst["bbox_xywh"]
    return index


def resize_for_intervention(image: Image.Image, bbox: list[float], max_side: int) -> tuple[Image.Image, list[float]]:
    if max_side <= 0:
        return image.convert("RGB"), bbox
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image.convert("RGB"), bbox
    scale = max_side / float(longest)
    resized = image.convert("RGB").resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    scaled_bbox = [bbox[0] * scale, bbox[1] * scale, bbox[2] * scale, bbox[3] * scale]
    return resized, scaled_bbox


def draw_box(image: Image.Image, bbox: list[float]) -> Image.Image:
    boxed = image.convert("RGB").copy()
    draw = ImageDraw.Draw(boxed)
    x, y, w, h = bbox
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    thickness = max(5, round(min(image.size) * 0.008))
    colors = [(255, 0, 0), (255, 220, 0)]
    for offset in range(thickness):
        color = colors[0] if offset < thickness - 2 else colors[1]
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)
    return boxed


def crop_oracle(image: Image.Image, bbox: list[float], padding_ratio: float) -> Image.Image:
    x, y, w, h = bbox
    pad_x = w * padding_ratio
    pad_y = h * padding_ratio
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(image.width, int(x + w + pad_x))
    y2 = min(image.height, int(y + h + pad_y))
    return image.convert("RGB").crop((x1, y1, x2, y2))


def dim_non_target(image: Image.Image, bbox: list[float], dim_alpha: int) -> Image.Image:
    base = image.convert("RGB")
    dimmed = Image.blend(base, Image.new("RGB", base.size, (0, 0, 0)), dim_alpha / 255.0)
    x, y, w, h = bbox
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(base.width, int(x + w))
    y2 = min(base.height, int(y + h))
    target = base.crop((x1, y1, x2, y2))
    dimmed.paste(target, (x1, y1))
    return draw_box(dimmed, bbox)


def direct_crop_question(row: dict[str, Any]) -> str:
    class_name = row["class_name"]
    attr_key = row["attribute_key"]
    if class_name == "person":
        if attr_key == "lower_clothing_color":
            return "What color is the lower clothing of the person in the image?"
        return "What color is the upper clothing of the person in the image?"
    if attr_key == "material":
        return f"What material is the {class_name} in the image?"
    return f"What color is the {class_name} in the image?"


def context_crop_question(row: dict[str, Any]) -> str:
    class_name = row["class_name"]
    attr_key = row["attribute_key"]
    if class_name == "person":
        if attr_key == "lower_clothing_color":
            return "What color is the lower clothing of the person closest to the center of the image?"
        return "What color is the upper clothing of the person closest to the center of the image?"
    if attr_key == "material":
        return f"What material is the {class_name} closest to the center of the image?"
    return f"What color is the {class_name} closest to the center of the image?"


def box_prompt(row: dict[str, Any]) -> str:
    return row["question"] + "\nThe target instance is highlighted with a red box."


def prepare_question_row(row: dict[str, Any], image_path: str, question: str, intervention: str) -> dict[str, Any]:
    out = dict(row)
    out["image_path"] = image_path
    out["question"] = question
    out["intervention"] = intervention
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "data/annotations/instabind_lite_v0.4.json",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data/questions/instabind_lite_v0.4_questions.jsonl",
    )
    parser.add_argument("--levels", nargs="*", default=["L1", "L3"])
    parser.add_argument("--crop-padding-ratio", type=float, default=0.15)
    parser.add_argument("--context-padding-ratio", type=float, default=0.5)
    parser.add_argument("--dim-alpha", type=int, default=145, help="0 keeps background unchanged; 255 makes it black.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-image-side", type=int, default=1800, help="Resize large intervention source images.")
    parser.add_argument("--metadata-only", action="store_true", help="Write question files without creating images.")
    args = parser.parse_args()

    annotations = read_json(args.annotations)
    questions = read_jsonl(args.questions)
    bbox_index = build_bbox_index(annotations)
    selected = [row for row in questions if row.get("level") in set(args.levels) and row.get("answer_type") == "attribute"]

    full_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    dim_rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    missing_boxes: list[str] = []

    box_dir = ROOT / "interventions/box_guided"
    crop_dir = ROOT / "interventions/crop_oracle"
    context_dir = ROOT / "interventions/context_crop"
    dim_dir = ROOT / "interventions/dim_non_target"
    box_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    dim_dir.mkdir(parents=True, exist_ok=True)

    for row in selected:
        key = (row["image_id"], row["group_id"], row["target_instance_id"])
        bbox = bbox_index.get(key)
        if bbox is None:
            missing_boxes.append(row["question_id"])
            continue

        source_image_path = ROOT / row["image_path"]
        box_rel = f"interventions/box_guided/{safe_filename(row['question_id'])}.jpg"
        crop_rel = f"interventions/crop_oracle/{safe_filename(row['question_id'])}.jpg"
        context_rel = f"interventions/context_crop/{safe_filename(row['question_id'])}.jpg"
        dim_rel = f"interventions/dim_non_target/{safe_filename(row['question_id'])}.jpg"
        box_path = ROOT / box_rel
        crop_path = ROOT / crop_rel
        context_path = ROOT / context_rel
        dim_path = ROOT / dim_rel

        if not args.metadata_only:
            if not source_image_path.exists():
                missing_images.append(str(source_image_path))
                continue
            with Image.open(source_image_path) as image:
                image, working_bbox = resize_for_intervention(image, bbox, args.max_image_side)
                draw_box(image, working_bbox).save(box_path, quality=args.jpeg_quality)
                crop_oracle(image, working_bbox, args.crop_padding_ratio).save(crop_path, quality=args.jpeg_quality)
                crop_oracle(image, working_bbox, args.context_padding_ratio).save(context_path, quality=args.jpeg_quality)
                dim_non_target(image, working_bbox, args.dim_alpha).save(dim_path, quality=args.jpeg_quality)

        full_rows.append(prepare_question_row(row, row["image_path"], row["question"], "full_image_l1_l3"))
        box_rows.append(prepare_question_row(row, box_rel, box_prompt(row), "box_guided"))
        crop_rows.append(prepare_question_row(row, crop_rel, direct_crop_question(row), "crop_oracle"))
        context_rows.append(prepare_question_row(row, context_rel, context_crop_question(row), "context_crop"))
        dim_rows.append(prepare_question_row(row, dim_rel, box_prompt(row), "dim_non_target"))

    if missing_images:
        print("Missing source images. Run scripts/collect_images_local.py before generating interventions.")
        for path in missing_images[:20]:
            print(path)
        return 1
    if missing_boxes:
        print("Missing bbox entries:")
        for question_id in missing_boxes[:20]:
            print(question_id)
        return 1

    write_jsonl(ROOT / "data/questions.l1_l3.full_image.jsonl", full_rows)
    write_jsonl(ROOT / "data/questions.l1_l3.box_guided.jsonl", box_rows)
    write_jsonl(ROOT / "data/questions.l1_l3.crop_oracle.jsonl", crop_rows)
    write_jsonl(ROOT / "data/questions.l1_l3.context_crop.jsonl", context_rows)
    write_jsonl(ROOT / "data/questions.l1_l3.dim_non_target.jsonl", dim_rows)

    summary = {
        "levels": args.levels,
        "questions": len(full_rows),
        "box_guided_images": len(box_rows),
        "crop_oracle_images": len(crop_rows),
        "context_crop_images": len(context_rows),
        "dim_non_target_images": len(dim_rows),
        "crop_padding_ratio": args.crop_padding_ratio,
        "context_padding_ratio": args.context_padding_ratio,
        "dim_alpha": args.dim_alpha,
    }
    (ROOT / "interventions/intervention_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
