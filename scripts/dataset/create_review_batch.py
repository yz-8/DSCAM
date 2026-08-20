#!/usr/bin/env python3
"""Create a human review batch with boxed thumbnails from candidate groups."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CLASS_ORDER = ["car", "person", "chair", "cup", "bottle", "bag", "umbrella"]
BOX_COLORS = [
    (230, 57, 70),
    (29, 105, 150),
    (42, 157, 143),
    (244, 162, 97),
    (131, 56, 236),
    (255, 190, 11),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_any_encoding(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "gbk", "cp936"):
        try:
            with path.open("r", newline="", encoding=encoding) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8-sig/gbk", b"", 0, 1, f"Could not decode {path}")


def parse_class_counts(text: str | None, default_per_class: int) -> dict[str, int]:
    if not text:
        return {class_name: default_per_class for class_name in CLASS_ORDER}
    counts = {class_name: 0 for class_name in CLASS_ORDER}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid class count '{item}'. Use class=count, such as car=20.")
        class_name, count_text = [part.strip() for part in item.split("=", 1)]
        if class_name not in counts:
            raise ValueError(f"Unsupported class '{class_name}'.")
        counts[class_name] = int(count_text)
    return counts


def excluded_keys_from_manifests(paths: list[Path]) -> set[tuple[str, str, int]]:
    excluded: set[tuple[str, str, int]] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in read_csv_any_encoding(path):
            review_id = row.get("review_id", "")
            match = re.search(r"_g(\d+)$", review_id)
            group_index = int(match.group(1)) if match else 0
            image_id = row.get("image_id", "")
            class_name = row.get("class_name", "")
            if image_id and class_name:
                excluded.add((image_id, class_name, group_index))
    return excluded


def bbox_signature(bboxes: list[list[float]]) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(tuple(round(float(value), 2) for value in bbox) for bbox in bboxes)


def excluded_signatures_from_annotations(paths: list[Path]) -> set[tuple[str, str, tuple[tuple[float, float, float, float], ...]]]:
    excluded: set[tuple[str, str, tuple[tuple[float, float, float, float], ...]]] = set()
    for path in paths:
        if not path.exists():
            continue
        data = load_json(path)
        for image in data.get("images", []):
            image_id = image.get("image_id", "")
            for group in image.get("same_class_groups", []):
                class_name = group.get("class_name", "")
                bboxes = [inst.get("bbox_xywh", []) for inst in group.get("instances", [])]
                if image_id and class_name and bboxes:
                    excluded.add((image_id, class_name, bbox_signature(bboxes)))
    return excluded


def safe_name(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def flatten_groups(
    data: dict[str, Any],
    excluded: set[tuple[str, str, int]],
    excluded_signatures: set[tuple[str, str, tuple[tuple[float, float, float, float], ...]]],
) -> list[dict[str, Any]]:
    rows = []
    for image in data.get("candidates", []):
        for group_index, group in enumerate(image.get("candidate_groups", [])):
            key = (image["image_id"], group["class_name"], group_index)
            if key in excluded:
                continue
            signature = (image["image_id"], group["class_name"], bbox_signature(group["bbox_xywh"]))
            if signature in excluded_signatures:
                continue
            rows.append(
                {
                    "image_id": image["image_id"],
                    "source": image.get("source", {}),
                    "image_path": image["image_path"],
                    "width": image["width"],
                    "height": image["height"],
                    "group_index": group_index,
                    "class_name": group["class_name"],
                    "instance_count": group["instance_count"],
                    "bbox_xywh": group["bbox_xywh"],
                    "center_xy": group.get("center_xy", []),
                    "auto_quality_metrics": group.get("auto_quality_metrics", {}),
                    "suggested_attributes_left_to_right": group.get("suggested_attributes_left_to_right", ""),
                    "suggested_notes": group.get("suggested_notes", ""),
                }
            )
    return rows


def quality_score(row: dict[str, Any]) -> tuple[float, float, float]:
    metrics = row.get("auto_quality_metrics", {})
    mean_area = float(metrics.get("mean_area_ratio", 0.0))
    max_iou = float(metrics.get("max_pairwise_iou", 1.0))
    max_h_overlap = float(metrics.get("max_horizontal_overlap_ratio", 1.0))
    return (-mean_area, max_iou, max_h_overlap)


def select_balanced(
    rows: list[dict[str, Any]],
    class_counts: dict[str, int],
    seed: int,
    max_total: int | None,
    selection: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["class_name"]].append(row)

    selected: list[dict[str, Any]] = []
    for class_name in CLASS_ORDER:
        candidates = by_class.get(class_name, [])
        if selection == "quality":
            candidates.sort(key=quality_score)
        else:
            rng.shuffle(candidates)
        selected.extend(candidates[: class_counts.get(class_name, 0)])

    if max_total is not None and len(selected) > max_total:
        selected = selected[:max_total]
    return selected


def draw_overlay(row: dict[str, Any], output_path: Path, max_width: int) -> None:
    image_path = Path(row["image_path"])
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        scale = min(1.0, max_width / image.width)
        if scale < 1.0:
            image = image.resize((int(image.width * scale), int(image.height * scale)))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for i, bbox in enumerate(row["bbox_xywh"], 1):
            x, y, w, h = bbox
            x1 = int(x * scale)
            y1 = int(y * scale)
            x2 = int((x + w) * scale)
            y2 = int((y + h) * scale)
            color = BOX_COLORS[(i - 1) % len(BOX_COLORS)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            label = f"{i}"
            label_box = draw.textbbox((x1 + 4, y1 + 4), label, font=font)
            pad = 4
            draw.rectangle(
                [label_box[0] - pad, label_box[1] - pad, label_box[2] + pad, label_box[3] + pad],
                fill=color,
            )
            draw.text((x1 + 4, y1 + 4), label, fill=(255, 255, 255), font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=90)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "image_id",
        "source_image_id",
        "width",
        "height",
        "class_name",
        "instance_count",
        "image_path",
        "overlay_path",
        "bbox_xywh_json",
        "decision",
        "attribute_focus",
        "attributes_left_to_right",
        "reject_reason",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"review_items": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        rel_overlay = Path(row["overlay_path"]).name
        cards.append(
            f"""
      <article class="card">
        <img src="overlays/{html.escape(rel_overlay)}" alt="{html.escape(row['review_id'])}">
        <div class="meta">
          <strong>{html.escape(row['review_id'])}</strong>
          <span>{html.escape(row['class_name'])} · {row['instance_count']} instances</span>
          <span>source: {html.escape(row['source_image_id'])}</span>
        </div>
        <div class="fields">
          <div>Decision: approve only; leave blank to skip</div>
          <div>Attributes left-to-right: __________________________</div>
          <div>Optional reject reason / notes: ____________________</div>
        </div>
      </article>
"""
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>InstaBind-Lite Review Batch</title>
  <style>
    body {{
      margin: 24px;
      font-family: Arial, sans-serif;
      color: #1f2933;
      background: #f7f7f4;
    }}
    h1 {{
      font-size: 22px;
      margin: 0 0 8px;
    }}
    .hint {{
      margin: 0 0 20px;
      color: #52616b;
      line-height: 1.45;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: white;
      border: 1px solid #d9e2ec;
      border-radius: 6px;
      overflow: hidden;
    }}
    img {{
      width: 100%;
      display: block;
      background: #e4e7eb;
    }}
    .meta, .fields {{
      padding: 10px 12px;
      display: grid;
      gap: 4px;
      font-size: 13px;
    }}
    .meta {{
      border-top: 1px solid #eef2f5;
      border-bottom: 1px solid #eef2f5;
    }}
    .fields {{
      color: #334e68;
    }}
  </style>
</head>
<body>
  <h1>InstaBind-Lite Review Batch</h1>
  <p class="hint">Approve only clean cases with 3-6 same-class instances, clear left-to-right order, visible attributes, and no severe ambiguity. Leave weak candidates blank. Numbered boxes show candidate instances in left-to-right order.</p>
  <section class="grid">
    {"".join(cards)}
  </section>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument(
        "--class-counts",
        default=None,
        help="Comma-separated class counts, such as person=35,bag=25,car=25,umbrella=15.",
    )
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--exclude-annotations", type=Path, action="append", default=[])
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-width", type=int, default=720)
    parser.add_argument("--selection", choices=["quality", "random"], default="quality")
    args = parser.parse_args()

    data = load_json(args.candidates_json)
    excluded = excluded_keys_from_manifests(args.exclude_manifest)
    excluded_signatures = excluded_signatures_from_annotations(args.exclude_annotations)
    class_counts = parse_class_counts(args.class_counts, args.per_class)
    rows = flatten_groups(data, excluded, excluded_signatures)
    selected = select_balanced(rows, class_counts, args.seed, args.max_total, args.selection)

    review_rows: list[dict[str, Any]] = []
    overlay_dir = args.output_dir / "overlays"
    for index, row in enumerate(selected, 1):
        review_id = f"review_{index:04d}_{safe_name(row['class_name'])}_{safe_name(row['image_id'])}_g{row['group_index']}"
        overlay_path = overlay_dir / f"{review_id}.jpg"
        draw_overlay(row, overlay_path, args.max_width)
        review_rows.append(
            {
                "review_id": review_id,
                "image_id": row["image_id"],
                "source_image_id": row.get("source", {}).get("source_image_id", ""),
                "width": row["width"],
                "height": row["height"],
                "class_name": row["class_name"],
                "instance_count": row["instance_count"],
                "image_path": row["image_path"],
                "overlay_path": str(overlay_path),
                "bbox_xywh_json": json.dumps(row["bbox_xywh"], ensure_ascii=False),
                "decision": "",
                "attribute_focus": "upper_clothing_color" if row["class_name"] == "person" else "color",
                "attributes_left_to_right": row.get("suggested_attributes_left_to_right", ""),
                "reject_reason": "",
                "notes": row.get("suggested_notes", ""),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "review_manifest.csv", review_rows)
    write_json(args.output_dir / "review_manifest.json", review_rows)
    write_html(args.output_dir / "review_sheet.html", review_rows)

    summary = {
        "review_items": len(review_rows),
        "by_class": {class_name: sum(1 for row in review_rows if row["class_name"] == class_name) for class_name in CLASS_ORDER},
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
