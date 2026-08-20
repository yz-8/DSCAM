#!/usr/bin/env python3
"""Validate InstaBind-Lite annotations.

Usage:
  python scripts/validate_annotations.py data/annotations/instabind_lite_annotations.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALLOWED_CLASSES = {
    "animal",
    "bag",
    "basin",
    "bicycle",
    "bottle",
    "box",
    "bus",
    "car",
    "chair",
    "cup",
    "hat",
    "motorcycle",
    "pen",
    "person",
    "suitcase",
    "truck",
    "umbrella",
}
ALLOWED_OCCLUSION = {"none", "minor", "moderate", "severe"}
ALLOWED_BBOX_QUALITY = {"tight", "acceptable", "loose", "bad"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def attr_value(inst: dict[str, Any], key: str) -> str | None:
    value = inst.get("attributes", {}).get(key)
    return value if isinstance(value, str) and value else None


def bbox_in_bounds(bbox: list[Any], width: int, height: int) -> bool:
    if len(bbox) != 4:
        return False
    x, y, w, h = bbox
    if min(x, y, w, h) < 0:
        return False
    if w <= 0 or h <= 0:
        return False
    return x + w <= width and y + h <= height


def validate(data: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {
        "images": 0,
        "approved_images": 0,
        "groups": 0,
        "instances": 0,
        "classes": Counter(),
        "attribute_keys": Counter(),
        "l2_unique_groups": 0,
    }

    if data.get("dataset_name") != "InstaBind-Lite":
        errors.append("dataset_name must be 'InstaBind-Lite'.")
    images = data.get("images")
    if not isinstance(images, list):
        errors.append("Top-level 'images' must be a list.")
        return errors, warnings, stats

    stats["images"] = len(images)
    seen_images: set[str] = set()

    for img_i, image in enumerate(images):
        prefix = f"images[{img_i}]"
        image_id = image.get("image_id")
        if not image_id:
            errors.append(f"{prefix}: missing image_id.")
            image_id = f"<missing:{img_i}>"
        if image_id in seen_images:
            errors.append(f"{prefix}: duplicate image_id '{image_id}'.")
        seen_images.add(image_id)

        width = image.get("width")
        height = image.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            errors.append(f"{prefix}: width and height must be positive integers.")
            width, height = 0, 0

        quality = image.get("quality", {})
        if quality.get("approved") is True:
            stats["approved_images"] += 1
        else:
            warnings.append(f"{prefix}: image is not approved.")

        groups = image.get("same_class_groups", [])
        if not isinstance(groups, list) or not groups:
            errors.append(f"{prefix}: same_class_groups must be a non-empty list.")
            continue

        seen_groups: set[str] = set()
        for group_i, group in enumerate(groups):
            gprefix = f"{prefix}.same_class_groups[{group_i}]"
            group_id = group.get("group_id")
            if not group_id:
                errors.append(f"{gprefix}: missing group_id.")
                group_id = f"<missing:{group_i}>"
            if group_id in seen_groups:
                errors.append(f"{gprefix}: duplicate group_id '{group_id}' within image.")
            seen_groups.add(group_id)

            class_name = group.get("class_name")
            if class_name not in ALLOWED_CLASSES:
                errors.append(f"{gprefix}: unsupported class_name '{class_name}'.")
            stats["classes"][class_name] += 1

            attr_key = group.get("attribute_focus")
            if not attr_key:
                errors.append(f"{gprefix}: missing attribute_focus.")
            else:
                stats["attribute_keys"][attr_key] += 1

            instances = group.get("instances", [])
            if not isinstance(instances, list) or not (3 <= len(instances) <= 6):
                errors.append(f"{gprefix}: instances must contain 3-6 records.")
                continue

            stats["groups"] += 1
            stats["instances"] += len(instances)

            ids: set[str] = set()
            order_indices: list[int] = []
            values: list[str] = []
            ordered_instances = sorted(instances, key=lambda x: x.get("order_index", 999))

            for inst_i, inst in enumerate(instances):
                iprefix = f"{gprefix}.instances[{inst_i}]"
                inst_id = inst.get("instance_id")
                if not inst_id:
                    errors.append(f"{iprefix}: missing instance_id.")
                    inst_id = f"<missing:{inst_i}>"
                if inst_id in ids:
                    errors.append(f"{iprefix}: duplicate instance_id '{inst_id}'.")
                ids.add(inst_id)

                order_index = inst.get("order_index")
                if not isinstance(order_index, int):
                    errors.append(f"{iprefix}: order_index must be an integer.")
                else:
                    order_indices.append(order_index)

                bbox = inst.get("bbox_xywh")
                if not isinstance(bbox, list) or not bbox_in_bounds(bbox, width, height):
                    errors.append(f"{iprefix}: bbox_xywh is missing, invalid, or outside image bounds.")

                value = attr_value(inst, attr_key)
                if value is None:
                    errors.append(f"{iprefix}: missing queried attribute '{attr_key}'.")
                else:
                    values.append(value)

                if inst.get("ambiguity_flag") is True:
                    warnings.append(f"{iprefix}: ambiguity_flag is true; avoid final test questions.")

                visibility = inst.get("visibility", {})
                if visibility.get("attribute_visible") is not True:
                    errors.append(f"{iprefix}: attribute_visible must be true.")
                if visibility.get("bbox_quality") not in ALLOWED_BBOX_QUALITY:
                    errors.append(f"{iprefix}: invalid bbox_quality.")
                if visibility.get("occlusion") not in ALLOWED_OCCLUSION:
                    errors.append(f"{iprefix}: invalid occlusion label.")

            expected = list(range(1, len(instances) + 1))
            if sorted(order_indices) != expected:
                errors.append(f"{gprefix}: order_index values must be exactly {expected}.")

            id_to_inst = {inst.get("instance_id"): inst for inst in instances}
            for idx, inst in enumerate(ordered_instances):
                inst_id = inst.get("instance_id")
                neighbors = inst.get("neighbors", {})
                expected_left = ordered_instances[idx - 1].get("instance_id") if idx > 0 else None
                expected_right = ordered_instances[idx + 1].get("instance_id") if idx < len(ordered_instances) - 1 else None
                if neighbors.get("left") != expected_left:
                    errors.append(f"{gprefix}.{inst_id}: left neighbor should be {expected_left}.")
                if neighbors.get("right") != expected_right:
                    errors.append(f"{gprefix}.{inst_id}: right neighbor should be {expected_right}.")
                for direction in ("left", "right"):
                    neighbor_id = neighbors.get(direction)
                    if neighbor_id is not None and neighbor_id not in id_to_inst:
                        errors.append(f"{gprefix}.{inst_id}: unknown {direction} neighbor '{neighbor_id}'.")

            if len(set(values)) == len(values):
                stats["l2_unique_groups"] += 1
            elif len(set(values)) < len(values):
                warnings.append(f"{gprefix}: repeated '{attr_key}' values; not all instances support L2.")

    stats["classes"] = dict(stats["classes"])
    stats["attribute_keys"] = dict(stats["attribute_keys"])
    return errors, warnings, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    data = load_json(args.annotations)
    errors, warnings, stats = validate(data)

    result = {
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "stats": stats,
        "errors": errors,
        "warnings": warnings,
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
