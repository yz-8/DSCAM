#!/usr/bin/env python3
"""Extract same-class color-binding candidates from GQA scene graphs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


CLASS_ALIASES = {
    "person": {
        "person",
        "man",
        "woman",
        "boy",
        "girl",
        "child",
        "kid",
        "people",
        "player",
        "skier",
        "rider",
    },
    "car": {"car", "cars", "vehicle", "vehicles", "taxi"},
    "chair": {"chair", "chairs", "seat", "seats"},
    "cup": {"cup", "cups", "mug", "mugs", "glass", "glasses"},
    "bottle": {"bottle", "bottles"},
    "umbrella": {"umbrella", "umbrellas"},
    "bag": {"bag", "bags", "backpack", "backpacks", "handbag", "handbags", "purse", "purses"},
}

COLOR_ALIASES = {
    "black": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
    "silver": "gray",
    "red": "red",
    "orange": "orange",
    "yellow": "yellow",
    "green": "green",
    "blue": "blue",
    "purple": "purple",
    "pink": "pink",
    "brown": "brown",
    "tan": "beige",
    "beige": "beige",
    "cream": "beige",
    "clear": "transparent",
    "transparent": "transparent",
    "colorless": "transparent",
    "multi-colored": "multicolor",
    "multicolored": "multicolor",
    "multicolor": "multicolor",
    "colorful": "multicolor",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def canonical_class(name: str) -> str | None:
    normalized = normalize_name(name)
    for class_name, aliases in CLASS_ALIASES.items():
        if normalized in aliases:
            return class_name
    return None


def canonical_color(attrs: list[str]) -> str | None:
    for attr in attrs:
        normalized = normalize_name(attr).replace(" ", "-")
        if normalized in COLOR_ALIASES:
            return COLOR_ALIASES[normalized]
    return None


def bbox_area(bbox: list[float]) -> float:
    return float(bbox[2]) * float(bbox[3])


def center_x(bbox: list[float]) -> float:
    return float(bbox[0]) + float(bbox[2]) / 2.0


def center_y(bbox: list[float]) -> float:
    return float(bbox[1]) + float(bbox[3]) / 2.0


def intersection_area(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = map(float, a)
    bx1, by1, bw, bh = map(float, b)
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    x_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    y_overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
    return x_overlap * y_overlap


def pairwise_iou(a: list[float], b: list[float]) -> float:
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union else 0.0


def horizontal_overlap_ratio(a: list[float], b: list[float]) -> float:
    ax1, _, aw, _ = map(float, a)
    bx1, _, bw, _ = map(float, b)
    ax2 = ax1 + aw
    bx2 = bx1 + bw
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    return overlap / min(aw, bw) if min(aw, bw) else 0.0


def overlap_metrics(boxes: list[list[float]]) -> dict[str, float]:
    max_iou = 0.0
    max_horizontal_overlap = 0.0
    for i, box_a in enumerate(boxes):
        for box_b in boxes[i + 1 :]:
            max_iou = max(max_iou, pairwise_iou(box_a, box_b))
            max_horizontal_overlap = max(max_horizontal_overlap, horizontal_overlap_ratio(box_a, box_b))
    return {
        "max_pairwise_iou": max_iou,
        "max_horizontal_overlap_ratio": max_horizontal_overlap,
    }


def order_likely_clear(boxes: list[list[float]], image_width: int, min_center_gap_ratio: float) -> bool:
    centers = sorted(center_x(box) for box in boxes)
    if len(centers) < 2:
        return False
    min_gap = min(b - a for a, b in zip(centers, centers[1:]))
    return min_gap >= image_width * min_center_gap_ratio


def resolve_image_path(images_root: Path, image_id: str) -> Path:
    candidates = [
        images_root / f"{image_id}.jpg",
        images_root / f"{image_id}.jpeg",
        images_root / f"{image_id}.png",
        images_root / "images" / f"{image_id}.jpg",
        images_root / "images" / f"{image_id}.jpeg",
        images_root / "images" / f"{image_id}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def image_size(image_data: dict[str, Any], image_path: Path) -> tuple[int, int]:
    width = image_data.get("width")
    height = image_data.get("height")
    if width and height:
        return int(width), int(height)
    if Image is not None and image_path.exists():
        with Image.open(image_path) as image:
            return image.width, image.height
    return 0, 0


def object_bbox(obj: dict[str, Any]) -> list[float] | None:
    keys = ("x", "y", "w", "h")
    if not all(key in obj for key in keys):
        return None
    bbox = [float(obj["x"]), float(obj["y"]), float(obj["w"]), float(obj["h"])]
    if bbox[2] <= 0 or bbox[3] <= 0:
        return None
    return bbox


def extract_candidates(
    scene_graphs: dict[str, Any],
    *,
    images_root: Path,
    classes: set[str],
    min_instances: int,
    max_instances: int,
    min_area_ratio: float,
    min_center_gap_ratio: float,
    max_pairwise_iou: float,
    max_horizontal_overlap_ratio: float,
    require_color_attributes: bool,
    require_unique_colors: bool,
    max_images: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    class_counts: Counter = Counter()
    skipped: Counter = Counter()

    for image_id, image_data in scene_graphs.items():
        image_path = resolve_image_path(images_root, str(image_id))
        width, height = image_size(image_data, image_path)
        if not width or not height:
            skipped["missing_image_size"] += 1
            continue

        objects_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for object_id, obj in image_data.get("objects", {}).items():
            class_name = canonical_class(obj.get("name", ""))
            if class_name is None or class_name not in classes:
                continue
            bbox = object_bbox(obj)
            if bbox is None:
                skipped["missing_bbox"] += 1
                continue
            if bbox_area(bbox) < width * height * min_area_ratio:
                skipped["too_small"] += 1
                continue
            color = canonical_color(obj.get("attributes", []))
            if require_color_attributes and color is None:
                skipped["missing_color_attribute"] += 1
                continue
            objects_by_class[class_name].append(
                {
                    "object_id": str(object_id),
                    "bbox": bbox,
                    "color": color,
                    "raw_name": obj.get("name", ""),
                    "raw_attributes": obj.get("attributes", []),
                }
            )

        candidate_groups = []
        for class_name, objects in objects_by_class.items():
            if not (min_instances <= len(objects) <= max_instances):
                continue
            ordered = sorted(objects, key=lambda item: center_x(item["bbox"]))
            boxes = [item["bbox"] for item in ordered]
            if not order_likely_clear(boxes, width, min_center_gap_ratio):
                skipped["unclear_order"] += 1
                continue
            metrics = overlap_metrics(boxes)
            if metrics["max_pairwise_iou"] > max_pairwise_iou:
                skipped["high_iou"] += 1
                continue
            if metrics["max_horizontal_overlap_ratio"] > max_horizontal_overlap_ratio:
                skipped["high_horizontal_overlap"] += 1
                continue
            colors = [item["color"] for item in ordered]
            if require_color_attributes and any(color is None for color in colors):
                skipped["incomplete_colors"] += 1
                continue
            if require_unique_colors and len(set(colors)) != len(colors):
                skipped["non_unique_colors"] += 1
                continue

            group = {
                "class_name": class_name,
                "instance_count": len(ordered),
                "bbox_xywh": boxes,
                "source_annotation_ids": [item["object_id"] for item in ordered],
                "auto_order_axis": "left_to_right",
                "auto_quality_flags": {
                    "count_ok": True,
                    "box_size_ok": True,
                    "order_likely_clear": True,
                    "overlap_ok": True,
                    "color_attributes_available": all(color is not None for color in colors),
                    "unique_colors": len(set(colors)) == len(colors),
                },
                "auto_quality_metrics": {
                    **metrics,
                    "min_area_ratio": min(bbox_area(box) / (width * height) for box in boxes),
                    "mean_area_ratio": sum(bbox_area(box) / (width * height) for box in boxes) / len(boxes),
                },
                "center_xy": [[center_x(item["bbox"]), center_y(item["bbox"])] for item in ordered],
                "suggested_attributes_left_to_right": "|".join(color for color in colors if color is not None),
                "suggested_notes": "GQA suggested colors; verify visually before approving.",
            }
            candidate_groups.append(group)
            class_counts[class_name] += 1

        if candidate_groups:
            candidates.append(
                {
                    "image_id": f"gqa_{image_id}",
                    "source": {
                        "dataset": "GQA",
                        "source_image_id": str(image_id),
                        "license": "",
                        "url": "",
                    },
                    "image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "candidate_groups": candidate_groups,
                }
            )
            if max_images and len(candidates) >= max_images:
                break

    summary = {
        "candidate_images": len(candidates),
        "candidate_groups": sum(len(row["candidate_groups"]) for row in candidates),
        "class_group_counts": dict(class_counts),
        "skipped": dict(skipped),
    }
    return candidates, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_graph_json", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--classes", nargs="*", default=["cup", "bottle", "chair", "car", "person", "umbrella", "bag"])
    parser.add_argument("--min-instances", type=int, default=3)
    parser.add_argument("--max-instances", type=int, default=6)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--min-center-gap-ratio", type=float, default=0.025)
    parser.add_argument("--max-pairwise-iou", type=float, default=0.12)
    parser.add_argument("--max-horizontal-overlap-ratio", type=float, default=0.65)
    parser.add_argument("--require-color-attributes", action="store_true")
    parser.add_argument("--require-unique-colors", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    data = load_json(args.scene_graph_json)
    candidates, summary = extract_candidates(
        data,
        images_root=args.images_root,
        classes=set(args.classes),
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        min_area_ratio=args.min_area_ratio,
        min_center_gap_ratio=args.min_center_gap_ratio,
        max_pairwise_iou=args.max_pairwise_iou,
        max_horizontal_overlap_ratio=args.max_horizontal_overlap_ratio,
        require_color_attributes=args.require_color_attributes,
        require_unique_colors=args.require_unique_colors,
        max_images=args.max_images,
    )

    result = {
        "source": "GQA scene graphs",
        "summary": summary,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
