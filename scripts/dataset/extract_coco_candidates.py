#!/usr/bin/env python3
"""Extract same-class candidate groups from a local COCO instances file.

Example:
  python scripts/extract_coco_candidates.py instances_val2017.json data/candidates/coco_val_candidates.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CLASSES = {"car", "person", "chair", "cup", "bottle", "umbrella", "backpack", "handbag"}
CLASS_REMAP = {
    "backpack": "bag",
    "handbag": "bag",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def extract_candidates(
    coco: dict[str, Any],
    *,
    image_root: str,
    classes: set[str],
    min_instances: int,
    max_instances: int,
    min_area_ratio: float,
    min_center_gap_ratio: float,
    max_pairwise_iou: float,
    max_horizontal_overlap_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image_class: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)

    for ann in coco.get("annotations", []):
        if ann.get("iscrowd"):
            continue
        cat_name = categories.get(ann.get("category_id"))
        if cat_name not in classes:
            continue
        image = images.get(ann.get("image_id"))
        if not image:
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        image_area = image["width"] * image["height"]
        if bbox_area(bbox) < image_area * min_area_ratio:
            continue
        class_name = CLASS_REMAP.get(cat_name, cat_name)
        anns_by_image_class[(image["id"], class_name)].append(ann)

    candidates_by_image: dict[int, dict[str, Any]] = {}
    class_counts: Counter = Counter()

    for (image_id, class_name), anns in anns_by_image_class.items():
        if not (min_instances <= len(anns) <= max_instances):
            continue
        image = images[image_id]
        boxes = [ann["bbox"] for ann in anns]
        clear_order = order_likely_clear(boxes, image["width"], min_center_gap_ratio)
        if not clear_order:
            continue
        metrics = overlap_metrics(boxes)
        if metrics["max_pairwise_iou"] > max_pairwise_iou:
            continue
        if metrics["max_horizontal_overlap_ratio"] > max_horizontal_overlap_ratio:
            continue

        if image_id not in candidates_by_image:
            candidates_by_image[image_id] = {
                "image_id": f"coco_{image_id}",
                "source": {
                    "dataset": "COCO",
                    "source_image_id": str(image_id),
                    "license": str(image.get("license", "")),
                    "url": image.get("coco_url", ""),
                },
                "image_path": str(Path(image_root) / image.get("file_name", f"{image_id}.jpg")),
                "width": image["width"],
                "height": image["height"],
                "candidate_groups": [],
            }

        ordered = sorted(anns, key=lambda ann: center_x(ann["bbox"]))
        candidates_by_image[image_id]["candidate_groups"].append(
            {
                "class_name": class_name,
                "instance_count": len(ordered),
                "bbox_xywh": [ann["bbox"] for ann in ordered],
                "source_annotation_ids": [str(ann["id"]) for ann in ordered],
                "auto_order_axis": "left_to_right",
                "auto_quality_flags": {
                    "count_ok": True,
                    "box_size_ok": True,
                    "order_likely_clear": clear_order,
                    "overlap_ok": True,
                },
                "auto_quality_metrics": {
                    **metrics,
                    "min_area_ratio": min(bbox_area(box) / (image["width"] * image["height"]) for box in boxes),
                    "mean_area_ratio": sum(bbox_area(box) / (image["width"] * image["height"]) for box in boxes)
                    / len(boxes),
                },
                "center_xy": [[center_x(ann["bbox"]), center_y(ann["bbox"])] for ann in ordered],
            }
        )
        class_counts[class_name] += 1

    candidates = list(candidates_by_image.values())
    summary = {
        "candidate_images": len(candidates),
        "candidate_groups": sum(len(row["candidate_groups"]) for row in candidates),
        "class_group_counts": dict(class_counts),
    }
    return candidates, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coco_instances_json", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--image-root", default="images")
    parser.add_argument("--classes", nargs="*", default=sorted(DEFAULT_CLASSES))
    parser.add_argument("--min-instances", type=int, default=3)
    parser.add_argument("--max-instances", type=int, default=6)
    parser.add_argument("--min-area-ratio", type=float, default=0.002)
    parser.add_argument("--min-center-gap-ratio", type=float, default=0.035)
    parser.add_argument("--max-pairwise-iou", type=float, default=1.0)
    parser.add_argument("--max-horizontal-overlap-ratio", type=float, default=1.0)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    coco = load_json(args.coco_instances_json)
    candidates, summary = extract_candidates(
        coco,
        image_root=args.image_root,
        classes=set(args.classes),
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        min_area_ratio=args.min_area_ratio,
        min_center_gap_ratio=args.min_center_gap_ratio,
        max_pairwise_iou=args.max_pairwise_iou,
        max_horizontal_overlap_ratio=args.max_horizontal_overlap_ratio,
    )

    output = {
        "dataset_name": "InstaBind-Lite",
        "dataset_version": "0.1.0",
        "candidates": candidates,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
