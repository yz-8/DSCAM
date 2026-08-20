#!/usr/bin/env python3
"""Convert an approved review manifest into InstaBind-Lite annotations."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


ORDER_LABELS = {
    3: ["leftmost", "middle", "rightmost"],
    4: ["leftmost", "second from left", "third from left", "rightmost"],
    5: ["leftmost", "second from left", "middle", "second from right", "rightmost"],
    6: ["leftmost", "second from left", "third from left", "fourth from left", "second from right", "rightmost"],
}

ALLOWED_ATTRIBUTE_VALUES = {
    "black",
    "white",
    "transparent",
    "gray",
    "silver",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "beige",
    "gold",
    "multicolor",
}

COLOR_ALIASES = {
    "grey": "gray",
    "dark grey": "gray",
    "light grey": "gray",
    "dark gray": "gray",
    "light gray": "gray",
    "navy": "blue",
    "dark blue": "blue",
    "light blue": "blue",
    "dark green": "green",
    "light green": "green",
    "maroon": "red",
    "burgundy": "red",
    "yello": "yellow",
    "balck": "black",
    "tan": "beige",
    "cream": "beige",
    "clear": "transparent",
    "see through": "transparent",
    "see-through": "transparent",
    "colorless": "transparent",
}


def normalize_attribute(value: str) -> str:
    value = value.strip().lower()
    if " and " in value or "&" in value or "+" in value:
        return "multicolor"
    return COLOR_ALIASES.get(value, value)


def split_attributes(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    values = [part.strip().lower() for part in re.split(r"\s*[|,;/]\s*", text) if part.strip()]
    return [normalize_attribute(value) for value in values]


def read_csv(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "gbk", "cp936"):
        try:
            with path.open("r", newline="", encoding=encoding) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8-sig/gbk", b"", 0, 1, f"Could not decode {path}")


def image_size(row: dict[str, str]) -> tuple[int, int]:
    width = row.get("width")
    height = row.get("height")
    if width and height:
        return int(float(width)), int(float(height))
    with Image.open(row["image_path"]) as image:
        return image.width, image.height


def normalized_image_path(row: dict[str, str], image_root: str | None) -> str:
    image_id = row.get("image_id", "")
    if image_id.startswith("gqa_"):
        if image_root:
            source_image_id = row.get("source_image_id", "").strip()
            if source_image_id:
                return str(Path(image_root) / f"{source_image_id}.jpg")
        return row["image_path"]
    if image_id.startswith("vaw_"):
        if image_root:
            source_image_id = row.get("source_image_id", "").strip()
            if source_image_id:
                return str(Path(image_root) / f"{source_image_id}.jpg")
        return row["image_path"]
    if image_root:
        source_image_id = row.get("source_image_id", "").strip()
        if source_image_id.isdigit():
            return str(Path(image_root) / f"COCO_val2014_{int(source_image_id):012d}.jpg")
    return row["image_path"]


def source_dataset(row: dict[str, str]) -> str:
    image_id = row.get("image_id", "")
    if image_id.startswith("gqa_"):
        return "GQA"
    if image_id.startswith("vaw_"):
        return "VAW"
    if image_id.startswith("coco_"):
        return "COCO"
    return row.get("source_dataset") or "unknown"


def center_xy(bbox: list[float]) -> list[float]:
    return [bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0]


def make_instances(class_name: str, attribute_key: str, bboxes: list[list[float]], values: list[str]) -> list[dict[str, Any]]:
    labels = ORDER_LABELS[len(bboxes)]
    instances = []
    for index, (bbox, value) in enumerate(zip(bboxes, values), 1):
        instance_id = f"{class_name}_{index}"
        left = f"{class_name}_{index - 1}" if index > 1 else None
        right = f"{class_name}_{index + 1}" if index < len(bboxes) else None
        instances.append(
            {
                "instance_id": instance_id,
                "order_index": index,
                "order_label": labels[index - 1],
                "bbox_xywh": bbox,
                "center_xy": center_xy(bbox),
                "attributes": {attribute_key: value},
                "neighbors": {"left": left, "right": right},
                "visibility": {
                    "attribute_visible": True,
                    "bbox_quality": "acceptable",
                    "occlusion": "none",
                    "blur": "none",
                },
                "ambiguity_flag": False,
            }
        )
    return instances


def convert(
    rows: list[dict[str, str]],
    dataset_version: str,
    image_root: str | None,
    require_unique_attributes: bool,
    require_known_attributes: bool,
    exclude_review_ids: set[str],
    skip_incomplete_attributes: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    approved = [row for row in rows if row.get("decision", "").strip().lower() == "approve"]
    images: dict[str, dict[str, Any]] = {}
    group_counts: defaultdict[str, int] = defaultdict(int)
    skipped_non_unique: list[dict[str, str]] = []
    skipped_invalid_attributes: list[dict[str, str]] = []
    skipped_excluded_review_ids: list[dict[str, str]] = []
    skipped_incomplete_attributes: list[dict[str, str]] = []

    for row in approved:
        review_id = row.get("review_id", "").strip()
        if review_id in exclude_review_ids:
            skipped_excluded_review_ids.append(
                {
                    "review_id": review_id,
                    "class_name": row.get("class_name", ""),
                    "reason": "manual exclusion",
                }
            )
            continue
        attrs = split_attributes(row.get("attributes_left_to_right", ""))
        instance_count = int(row["instance_count"])
        if len(attrs) != instance_count:
            if skip_incomplete_attributes:
                skipped_incomplete_attributes.append(
                    {
                        "review_id": review_id,
                        "class_name": row.get("class_name", ""),
                        "instance_count": str(instance_count),
                        "attribute_count": str(len(attrs)),
                        "attributes_left_to_right": "|".join(attrs),
                    }
                )
                continue
            raise ValueError(
                f"{row['review_id']}: expected {instance_count} attributes, got {len(attrs)}. "
                "Use a separator such as black|red|white."
            )
        bboxes = json.loads(row["bbox_xywh_json"])
        if len(bboxes) != instance_count:
            raise ValueError(f"{row['review_id']}: bbox count does not match instance_count.")
        invalid_attrs = [value for value in attrs if value not in ALLOWED_ATTRIBUTE_VALUES]
        if require_known_attributes and invalid_attrs:
            skipped_invalid_attributes.append(
                {
                    "review_id": review_id,
                    "class_name": row.get("class_name", ""),
                    "attributes_left_to_right": "|".join(attrs),
                    "invalid_attributes": "|".join(invalid_attrs),
                }
            )
            continue
        if require_unique_attributes and len(set(attrs)) != len(attrs):
            skipped_non_unique.append(
                {
                    "review_id": row.get("review_id", ""),
                    "class_name": row.get("class_name", ""),
                    "attributes_left_to_right": "|".join(attrs),
                }
            )
            continue

        image_id = row["image_id"]
        width, height = image_size(row)
        if image_id not in images:
            images[image_id] = {
                "image_id": image_id,
                "split": "dev",
                "source": {
                    "dataset": source_dataset(row),
                    "source_image_id": row.get("source_image_id", ""),
                    "license": "",
                    "url": "",
                },
                "image_path": normalized_image_path(row, image_root),
                "width": width,
                "height": height,
                "same_class_groups": [],
                "quality": {
                    "same_class_count_ok": True,
                    "spatial_order_clear": True,
                    "attribute_contrast_ok": len(set(attrs)) > 1,
                    "no_severe_occlusion": True,
                    "no_major_reflection_or_blur": True,
                    "l2_unique_attribute_possible": len(set(attrs)) == len(attrs),
                    "approved": True,
                },
                "notes": "",
            }

        class_name = row["class_name"]
        attribute_key = row.get("attribute_focus") or ("upper_clothing_color" if class_name == "person" else "color")
        group_counts[image_id] += 1
        group_id = f"{image_id}_{class_name}_{group_counts[image_id]}"
        images[image_id]["same_class_groups"].append(
            {
                "group_id": group_id,
                "class_name": class_name,
                "attribute_focus": attribute_key,
                "spatial_axis": "left_to_right",
                "instances": make_instances(class_name, attribute_key, bboxes, attrs),
            }
        )

    data = {
        "dataset_name": "InstaBind-Lite",
        "dataset_version": dataset_version,
        "images": list(images.values()),
    }
    summary = {
        "approved_rows": len(approved),
        "approved_images": len(data["images"]),
        "skipped_incomplete_attribute_count": len(skipped_incomplete_attributes),
        "skipped_incomplete_attributes": skipped_incomplete_attributes,
        "skipped_excluded_review_id_count": len(skipped_excluded_review_ids),
        "skipped_excluded_review_ids": skipped_excluded_review_ids,
        "skipped_invalid_attribute_count": len(skipped_invalid_attributes),
        "skipped_invalid_attributes": skipped_invalid_attributes,
        "skipped_non_unique_count": len(skipped_non_unique),
        "skipped_non_unique": skipped_non_unique,
    }
    return data, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_manifest_csv", type=Path)
    parser.add_argument("output_annotations_json", type=Path)
    parser.add_argument("--dataset-version", default="0.1.0")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--require-unique-attributes", action="store_true")
    parser.add_argument("--require-known-attributes", action="store_true")
    parser.add_argument("--exclude-review-id", action="append", default=[])
    parser.add_argument("--skip-incomplete-attributes", action="store_true")
    args = parser.parse_args()

    data, summary = convert(
        read_csv(args.review_manifest_csv),
        args.dataset_version,
        args.image_root,
        args.require_unique_attributes,
        args.require_known_attributes,
        set(args.exclude_review_id),
        args.skip_incomplete_attributes,
    )
    args.output_annotations_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_annotations_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
